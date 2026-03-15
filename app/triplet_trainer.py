"""
Triplet Trainer for CBSA Behavioral Authentication
Loads per-user behavioral logs, constructs triplet training pairs,
trains the GAT with triplet loss, and saves a user profile vector.
"""

import json
import logging
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import hashlib
import random

from app.config import settings

logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
BEHAVIORAL_LOG_DIR = DATA_DIR / "behavioral_logs"
PROFILES_DIR = DATA_DIR / "profiles"
CHECKPOINT_PATH = DATA_DIR / "checkpoints" / "gat_checkpoint.pt"

INPUT_DIM = 56   # 48 behavioral + 8 event-type embedding
HIDDEN_DIM = 64
OUTPUT_DIM = 64
NUM_HEADS = 4
DROPOUT = 0.1
TEMPORAL_DIM = 16
TRIPLET_MARGIN = 0.5
LEARNING_RATE = 0.001
MIN_EVENTS_FOR_SESSION = 5    # minimum events to form a session graph
WINDOW_SECONDS = 20           # seconds per session window
WINDOW_STRIDE_SECONDS = 2    # sliding-window stride (90% overlap)

# Blob name for the shared GAT checkpoint
_CHECKPOINT_BLOB_NAME = "gat_checkpoint.pt"


def _event_type_embedding(event_type: str) -> List[float]:
    digest = hashlib.sha256(str(event_type).encode("utf-8")).digest()
    return [b / 255.0 for b in digest[:8]]


def _extract_vector(event: dict) -> List[float]:
    """Extract 56-D vector from a raw logged event."""
    event_data = event.get("event_data") or {}
    base = list(event_data.get("vector") or [0.0] * 48)
    base = (base + [0.0] * 48)[:48]
    base = [float(v) if v is not None else 0.0 for v in base]
    event_type = event.get("event_type", "unknown")
    embedding = _event_type_embedding(event_type)
    return (base + embedding)[:56]


def _normalize_event(raw: dict) -> dict:
    """Normalize a raw Cosmos / local event into the format the trainer expects.

    The ``behaviour-logs`` Cosmos container may hold documents written by two
    different code-paths:

    1. ``behavioral_logger.log_event``  – spreads the WebSocket payload so the
       document already contains ``timestamp``, ``event_type``,
       ``event_data.vector``.
    2. ``cosmos_prototype_store.insert_behaviour_log`` – stores
       ``eventTimestamp``, ``eventType``, ``vectorJson`` (a JSON string).

    This helper detects which format a document is in and returns a dict that
    always has ``timestamp``, ``event_type`` and ``event_data.vector``.
    """
    ev = dict(raw)  # shallow copy so we don't mutate the original

    # ---- timestamp --------------------------------------------------------
    if "timestamp" not in ev or ev["timestamp"] is None:
        ev["timestamp"] = (
            ev.get("eventTimestamp")
            or ev.get("loggedAt")
            or ev.get("logged_at")
            or 0.0
        )
    ev["timestamp"] = float(ev["timestamp"])

    # ---- event_type -------------------------------------------------------
    if "event_type" not in ev or ev["event_type"] is None:
        ev["event_type"] = ev.get("eventType", "unknown")

    # ---- event_data / vector ----------------------------------------------
    if "event_data" not in ev or not isinstance(ev.get("event_data"), dict):
        vector_json = ev.get("vectorJson")
        vector: List[float] = []
        if vector_json:
            try:
                parsed = json.loads(vector_json) if isinstance(vector_json, str) else vector_json
                vector = [float(v) for v in parsed]
            except (json.JSONDecodeError, ValueError, TypeError):
                pass
        ev["event_data"] = {"vector": vector}

    return ev


def _load_user_events(user_id: str) -> List[dict]:
    """Load user events from Cosmos DB (via behavioral_logger) or local JSONL fallback."""
    from app.behavioral_logger import behavioral_logger

    logger.info("[load] Fetching events for user '%s' from Cosmos DB …", user_id)
    events = behavioral_logger.load_user_events(user_id)
    if events:
        logger.info("[load] Fetched %d raw events for user '%s' from Cosmos DB", len(events), user_id)
        normalized = [_normalize_event(e) for e in events]
        sorted_events = sorted(normalized, key=lambda e: e.get("timestamp", 0))
        if sorted_events:
            ts_min = sorted_events[0].get("timestamp", 0)
            ts_max = sorted_events[-1].get("timestamp", 0)
            logger.info(
                "[load] User '%s': %d events, timestamp range %.3f → %.3f (span %.1fs)",
                user_id, len(sorted_events), ts_min, ts_max, ts_max - ts_min,
            )
        return sorted_events

    logger.info("[load] No Cosmos events for user '%s', trying local JSONL fallback …", user_id)

    # Legacy direct-file fallback (for backwards compatibility)
    path = BEHAVIORAL_LOG_DIR / f"{user_id}.jsonl"
    if not path.exists():
        logger.warning("[load] No local file found for user '%s' at %s", user_id, path)
        return []
    file_events = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    file_events.append(json.loads(line))
                except Exception:
                    pass
    logger.info("[load] Loaded %d events from local file for user '%s'", len(file_events), user_id)
    normalized = [_normalize_event(e) for e in file_events]
    return sorted(normalized, key=lambda e: e.get("timestamp", 0))


def _split_into_windows(
    events: List[dict],
    window_sec: float = WINDOW_SECONDS,
    stride_sec: float = WINDOW_STRIDE_SECONDS,
) -> List[List[dict]]:
    """Split a sorted event list into sliding time windows.

    Each window spans ``window_sec`` seconds.  The next window starts
    ``stride_sec`` seconds after the previous one, producing a 50% overlap
    when stride_sec == window_sec / 2.
    """
    if not events:
        logger.warning("[window] No events to window")
        return []

    first_ts = events[0].get("timestamp", 0) or 0.0
    last_ts = events[-1].get("timestamp", 0) or 0.0
    logger.info(
        "[window] Windowing %d events: first_ts=%.3f, last_ts=%.3f, span=%.1fs, "
        "window_sec=%.1f, stride_sec=%.1f",
        len(events), first_ts, last_ts, last_ts - first_ts, window_sec, stride_sec,
    )

    windows = []
    win_start = first_ts
    while win_start <= last_ts:
        win_end = win_start + window_sec
        window = [
            ev for ev in events
            if win_start <= (ev.get("timestamp", 0) or 0.0) < win_end
        ]
        if len(window) >= MIN_EVENTS_FOR_SESSION:
            windows.append(window)
        win_start += stride_sec

    logger.info("[window] Created %d session windows (min_events_per_window=%d)", len(windows), MIN_EVENTS_FOR_SESSION)
    return windows


def _window_to_matrix(window: List[dict]):
    """Convert a window of events into a (N, 56) numpy array."""
    import numpy as np
    vecs = [_extract_vector(e) for e in window]
    return np.array(vecs, dtype=np.float32)


class TripletTrainer:
    """
    Trains a shared GAT (Graph Attention Network) model on user behavioral data
    using triplet loss with anchor-positive-negative triplets constructed from
    all existing users.  Falls back to a numpy-based approach when PyTorch/PyG
    are unavailable, producing a mean-vector profile instead.
    """

    def __init__(self):
        self._model = None  # Shared trained GAT model (SiameseGATNetwork)
        if settings.DEBUG_MODE:
            PROFILES_DIR.mkdir(parents=True, exist_ok=True)
            CHECKPOINT_PATH.parent.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------ #
    # Public API                                                           #
    # ------------------------------------------------------------------ #

    def train_user(self, user_id: str, force: bool = False) -> dict:
        """
        Generate a profile for a single user using the shared trained GAT model.
        If no trained model exists, triggers train_all() first so the GAT is
        trained on anchor-positive-negative triplets from *all* users.
        Returns a result dict with keys: user_id, status, message, profile_saved.
        """
        logger.info("[train_user] === Starting training for user '%s' (force=%s) ===", user_id, force)
        from app.cosmos_profile_store import cosmos_profile_store

        if cosmos_profile_store.has_profile(user_id) and not force:
            logger.info("[train_user] Profile already exists for '%s', skipping", user_id)
            return {
                "user_id": user_id,
                "status": "skipped",
                "message": "Profile already exists. Use force=True to retrain.",
                "profile_saved": True,
            }

        events = _load_user_events(user_id)
        if not events:
            logger.warning("[train_user] No events found for user '%s'", user_id)
            return {
                "user_id": user_id,
                "status": "error",
                "message": "No behavioral data found",
                "profile_saved": False,
            }

        windows = _split_into_windows(events)
        if len(windows) < 2:
            logger.warning(
                "[train_user] Insufficient session windows for '%s': got %d, need >= 2",
                user_id, len(windows),
            )
            return {
                "user_id": user_id,
                "status": "error",
                "message": f"Need at least 2 session windows, found {len(windows)}",
                "profile_saved": False,
            }

        logger.info("[train_user] Generating profile for '%s' with %d session windows", user_id, len(windows))

        try:
            import torch  # noqa: F401
        except ImportError:
            logger.warning("PyTorch not available, using numpy fallback for profile creation")
            return self._train_numpy_fallback(user_id, windows)

        # Try to load an existing trained GAT model
        model = self._model or self._load_model()

        if model is not None:
            return self._generate_user_profile_from_model(user_id, windows, model)

        # No trained model — train on all available users' data first
        logger.info("[train_user] No trained GAT model found – triggering train_all for '%s' …", user_id)
        results = self.train_all(force=True)

        # Find this user's result in the training output
        for r in results:
            if r.get("user_id") == user_id:
                logger.info("[train_user] === Training complete for user '%s': %s ===", user_id, r.get("status"))
                return r

        logger.warning("[train_user] === Training failed for user '%s': data insufficient ===", user_id)
        return {
            "user_id": user_id,
            "status": "error",
            "message": "User data insufficient or training failed",
            "profile_saved": False,
        }

    def train_all(self, force: bool = False) -> List[dict]:
        """Train a shared GAT model on triplets from all users, then generate per-user profiles.

        User discovery order:
          1. Cosmos DB via behavioral_logger.list_users() (queries behaviour-logs container)
          2. Local JSONL files in data/behavioral_logs/ (fallback / debug)
        The two sources are merged so users present in either place are included.
        """
        logger.info("[train_all] === Starting train_all (force=%s) ===", force)
        from app.behavioral_logger import behavioral_logger

        cosmos_users = behavioral_logger.list_users()
        local_users = [p.stem for p in BEHAVIORAL_LOG_DIR.glob("*.jsonl")]
        # Merge, preserve order, deduplicate
        users = list(dict.fromkeys(cosmos_users + local_users))

        if not users:
            logger.warning("[train_all] No users found in Cosmos or local files")
            return [{"status": "error", "message": "No behavioral data found for any user"}]

        logger.info("[train_all] Discovered %d users (%d from Cosmos, %d local): %s", len(users), len(cosmos_users), len(local_users), users)

        try:
            import torch  # noqa: F401
        except ImportError:
            logger.error("[train_all] PyTorch not available – cannot train")
            return [{"status": "error", "message": "Server error: PyTorch is required for training"}]

        if not torch.cuda.is_available():
            logger.error("[train_all] CUDA is not available – cannot train")
            return [{"status": "error", "message": "Server error: CUDA is not available for training"}]

        return self._train_gat_all_users(users, force)

    # ------------------------------------------------------------------ #
    # Numpy fallback (mean-vector profile, no triplet loss)               #
    # ------------------------------------------------------------------ #

    def _train_numpy_fallback(self, user_id: str, windows: List[List[dict]]) -> dict:
        import numpy as np
        logger.info("[numpy_fallback] Building mean-pool profile for '%s' with %d windows", user_id, len(windows))

        all_vecs = []
        for w in windows:
            mat = _window_to_matrix(w)
            all_vecs.append(mat.mean(axis=0))  # mean-pool each window

        profile_vector = np.array(all_vecs).mean(axis=0).tolist()
        self._save_profile(user_id, profile_vector, method="mean_pool_fallback", sessions=len(windows))
        return {
            "user_id": user_id,
            "status": "success",
            "message": "Profile created with numpy mean-pool (no PyTorch)",
            "profile_saved": True,
            "sessions_used": len(windows),
        }

    # ------------------------------------------------------------------ #
    # GAT-based triplet training (all users)                               #
    # ------------------------------------------------------------------ #

    def _train_gat_all_users(self, users: List[str], force: bool) -> List[dict]:
        """Train shared GAT model with triplets from all users, then generate profiles."""
        logger.info("[gat_train] === Starting GAT triplet training for %d users ===", len(users))
        import torch
        from types import SimpleNamespace
        from app.gat.gat_network import SiameseGATNetwork, GATTrainer
        from app.gat.data_processor import BehavioralDataProcessor, PyTorchDataConverter

        device = torch.device("cuda")

        dp_config = {
            'time_window_seconds': WINDOW_SECONDS,
            'min_events_per_window': MIN_EVENTS_FOR_SESSION,
            'max_events_per_window': 100,
            'distinct_event_connections': 4,
        }
        processor = BehavioralDataProcessor(dp_config)
        converter = PyTorchDataConverter()

        # ---- 1. Build per-user graph data --------------------------------
        user_graphs: Dict[str, list] = {}
        skipped: List[dict] = []

        for uid in users:
            logger.info("[gat_train] Loading & windowing events for user '%s' …", uid)
            events = _load_user_events(uid)
            windows = _split_into_windows(events)
            if len(windows) < 2:
                logger.warning("[gat_train] User '%s' skipped: %d windows (need >= 2)", uid, len(windows))
                skipped.append({
                    "user_id": uid, "status": "error",
                    "message": f"Need at least 2 session windows, found {len(windows)}",
                    "profile_saved": False,
                })
                continue

            graphs = []
            for i, w in enumerate(windows):
                try:
                    session_id = f"{uid}_session_{i}"
                    tg = processor.process_behavioral_data(w, uid, session_id)
                    gd_dict = converter.convert_to_pytorch(tg)
                    gd = SimpleNamespace(
                        x=gd_dict['x'],
                        edge_index=gd_dict['edge_index'],
                        temporal_features=gd_dict['temporal_features'],
                        batch=gd_dict['batch'],
                    )
                    graphs.append(gd)
                except Exception as e:
                    logger.warning("[gat_train] Failed to convert window %d for '%s': %s", i, uid, e)

            if len(graphs) >= 2:
                user_graphs[uid] = graphs
                logger.info("[gat_train] User '%s': %d valid graphs from %d windows", uid, len(graphs), len(windows))
            else:
                logger.warning("[gat_train] User '%s' skipped: only %d valid graphs", uid, len(graphs))
                skipped.append({
                    "user_id": uid, "status": "error",
                    "message": "Not enough valid session graphs",
                    "profile_saved": False,
                })

        if not user_graphs:
            logger.warning("[gat_train] No users with sufficient data – aborting training")
            return skipped + [{"status": "error", "message": "No users with sufficient data for training"}]

        logger.info("[gat_train] %d users ready for training, %d skipped", len(user_graphs), len(skipped))

        # ---- 2. Create GAT model -----------------------------------------
        gat_config = {
            'input_dim': INPUT_DIM,
            'hidden_dim': HIDDEN_DIM,
            'output_dim': OUTPUT_DIM,
            'num_heads': NUM_HEADS,
            'dropout': DROPOUT,
            'temporal_dim': TEMPORAL_DIM,
        }
        model = SiameseGATNetwork(gat_config)
        trainer = GATTrainer(
            model, learning_rate=LEARNING_RATE,
            margin=TRIPLET_MARGIN, device=str(device),
        )

        # ---- 3. Train with triplets from all users -----------------------
        all_user_ids = list(user_graphs.keys())
        n_epochs = 30
        t0 = time.time()
        logger.info("[gat_train] Starting %d epochs of triplet training on %d users …", n_epochs, len(all_user_ids))

        for epoch in range(n_epochs):
            epoch_loss = 0.0
            count = 0

            for uid in all_user_ids:
                graphs = user_graphs[uid]
                other_users = [u for u in all_user_ids if u != uid]

                for i in range(len(graphs)):
                    for j in range(len(graphs)):
                        if i == j:
                            continue

                        anchor = graphs[i]
                        positive = graphs[j]

                        if other_users:
                            neg_uid = random.choice(other_users)
                            negative = random.choice(user_graphs[neg_uid])
                        else:
                            # Single user: synthetic negative via additive noise
                            noisy_x = anchor.x + torch.randn_like(anchor.x) * 0.5
                            negative = SimpleNamespace(
                                x=noisy_x.clamp(0.0, 1.0),
                                edge_index=anchor.edge_index,
                                temporal_features=anchor.temporal_features,
                                batch=anchor.batch,
                            )

                        loss = trainer.train_batch(anchor, positive, negative)
                        epoch_loss += loss
                        count += 1

            if (epoch + 1) % 5 == 0 or epoch == 0:
                avg_loss = epoch_loss / max(count, 1)
                logger.info("[gat_train] Epoch %d/%d – avg triplet loss: %.4f (%d batches)", epoch + 1, n_epochs, avg_loss, count)

        elapsed = time.time() - t0
        logger.info("[gat_train] GAT triplet training complete in %.1fs", elapsed)

        # ---- 4. Save model checkpoint ------------------------------------
        self._save_checkpoint(model)
        self._model = model

        # ---- 5. Generate per-user profiles --------------------------------
        logger.info("[gat_train] Generating per-user profiles …")
        results: List[dict] = list(skipped)
        for uid, graphs in user_graphs.items():
            profile_vector = self._generate_profile_vector(model, graphs)
            self._save_profile(
                uid, profile_vector,
                method="gat_triplet", sessions=len(graphs), training_time=elapsed,
            )
            logger.info("[gat_train] Profile saved for user '%s' (%d sessions)", uid, len(graphs))
            results.append({
                "user_id": uid,
                "status": "success",
                "message": f"GAT triplet training complete in {elapsed:.1f}s",
                "profile_saved": True,
                "sessions_used": len(graphs),
                "training_time_seconds": elapsed,
            })

        logger.info("[gat_train] === GAT training complete: %d profiles saved, %d skipped ===", len(user_graphs), len(skipped))
        return results

    # ------------------------------------------------------------------ #
    # GAT model management                                                #
    # ------------------------------------------------------------------ #

    def _load_model(self):
        """Load trained GAT model from checkpoint (blob storage or local disk)."""
        try:
            import torch
            from app.gat.gat_network import SiameseGATNetwork
        except ImportError:
            return None

        gat_config = {
            'input_dim': INPUT_DIM,
            'hidden_dim': HIDDEN_DIM,
            'output_dim': OUTPUT_DIM,
            'num_heads': NUM_HEADS,
            'dropout': DROPOUT,
            'temporal_dim': TEMPORAL_DIM,
        }
        model = SiameseGATNetwork(gat_config)

        # Try blob storage first
        model_path = None
        try:
            from app.blob_model_store import blob_model_store
            if blob_model_store.enabled:
                import tempfile
                tmp_path = str(Path(tempfile.gettempdir()) / "gat_trainer_checkpoint.pt")
                if blob_model_store.download_model(_CHECKPOINT_BLOB_NAME, tmp_path):
                    model_path = tmp_path
        except Exception:
            pass

        # Fall back to local checkpoint
        if model_path is None and CHECKPOINT_PATH.exists():
            model_path = str(CHECKPOINT_PATH)

        if model_path is not None:
            state = torch.load(model_path, map_location="cpu")
            model.load_state_dict(state)
            logger.info(f"Loaded GAT model from {model_path}")
            self._model = model
            return model

        return None

    def _save_checkpoint(self, model):
        """Save GAT model checkpoint to disk and (optionally) blob storage."""
        import torch

        CHECKPOINT_PATH.parent.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), str(CHECKPOINT_PATH))
        logger.info(f"GAT model checkpoint saved to {CHECKPOINT_PATH}")

        try:
            from app.blob_model_store import blob_model_store
            if blob_model_store.enabled:
                blob_model_store.upload_model(str(CHECKPOINT_PATH), _CHECKPOINT_BLOB_NAME)
        except Exception as e:
            logger.debug(f"Blob checkpoint upload skipped: {e}")

    def _generate_profile_vector(self, model, graphs) -> List[float]:
        """Mean-pool GAT session embeddings into a 64-D profile vector."""
        import torch

        if not graphs:
            return [0.0] * OUTPUT_DIM

        model.eval()
        embeddings = []
        with torch.no_grad():
            for gd in graphs:
                emb = model.forward_once(
                    gd.x, gd.edge_index, gd.temporal_features, gd.batch,
                )
                embeddings.append(emb)

        profile = torch.stack(embeddings).mean(dim=0)
        return profile.cpu().numpy().tolist()

    def _generate_user_profile_from_model(
        self, user_id: str, windows: List[List[dict]], model,
    ) -> dict:
        """Generate a profile for one user using an already-trained GAT model."""
        from types import SimpleNamespace
        from app.gat.data_processor import BehavioralDataProcessor, PyTorchDataConverter

        dp_config = {
            'time_window_seconds': WINDOW_SECONDS,
            'min_events_per_window': MIN_EVENTS_FOR_SESSION,
            'max_events_per_window': 100,
            'distinct_event_connections': 4,
        }
        processor = BehavioralDataProcessor(dp_config)
        converter = PyTorchDataConverter()

        graphs = []
        for i, w in enumerate(windows):
            try:
                session_id = f"{user_id}_session_{i}"
                tg = processor.process_behavioral_data(w, user_id, session_id)
                gd_dict = converter.convert_to_pytorch(tg)
                gd = SimpleNamespace(
                    x=gd_dict['x'],
                    edge_index=gd_dict['edge_index'],
                    temporal_features=gd_dict['temporal_features'],
                    batch=gd_dict['batch'],
                )
                graphs.append(gd)
            except Exception as e:
                logger.warning(f"Failed to convert window {i} for {user_id}: {e}")

        if len(graphs) < 2:
            return {
                "user_id": user_id,
                "status": "error",
                "message": "Not enough valid session graphs",
                "profile_saved": False,
            }

        profile_vector = self._generate_profile_vector(model, graphs)
        self._save_profile(
            user_id, profile_vector,
            method="gat_triplet", sessions=len(graphs),
        )

        return {
            "user_id": user_id,
            "status": "success",
            "message": "Profile generated using trained GAT model",
            "profile_saved": True,
            "sessions_used": len(graphs),
        }

    # ------------------------------------------------------------------ #
    # Profile persistence                                                  #
    # ------------------------------------------------------------------ #

    def _save_profile(
        self,
        user_id: str,
        vector: List[float],
        method: str = "unknown",
        sessions: int = 0,
        training_time: float = 0.0,
    ):
        from app.cosmos_profile_store import cosmos_profile_store

        cosmos_profile_store.save_profile(
            user_id=user_id,
            vector=vector,
            method=method,
            sessions=sessions,
            training_time=training_time,
        )
        logger.info(f"Profile saved for {user_id}")

    def load_profile(self, user_id: str) -> Optional[List[float]]:
        from app.cosmos_profile_store import cosmos_profile_store

        return cosmos_profile_store.load_profile(user_id)


# Singleton
triplet_trainer = TripletTrainer()
