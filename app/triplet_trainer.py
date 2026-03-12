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


def _load_user_events(user_id: str) -> List[dict]:
    """Load user events from Cosmos DB (via behavioral_logger) or local JSONL fallback."""
    from app.behavioral_logger import behavioral_logger

    events = behavioral_logger.load_user_events(user_id)
    if events:
        return sorted(events, key=lambda e: e.get("timestamp", 0))

    # Legacy direct-file fallback (for backwards compatibility)
    path = BEHAVIORAL_LOG_DIR / f"{user_id}.jsonl"
    if not path.exists():
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
    return sorted(file_events, key=lambda e: e.get("timestamp", 0))


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
        return []

    first_ts = events[0].get("timestamp", 0) or 0.0
    last_ts = events[-1].get("timestamp", 0) or 0.0

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
        from app.cosmos_profile_store import cosmos_profile_store

        if cosmos_profile_store.has_profile(user_id) and not force:
            return {
                "user_id": user_id,
                "status": "skipped",
                "message": "Profile already exists. Use force=True to retrain.",
                "profile_saved": True,
            }

        events = _load_user_events(user_id)
        if not events:
            return {
                "user_id": user_id,
                "status": "error",
                "message": "No behavioral data found",
                "profile_saved": False,
            }

        windows = _split_into_windows(events)
        if len(windows) < 2:
            return {
                "user_id": user_id,
                "status": "error",
                "message": f"Need at least 2 session windows, found {len(windows)}",
                "profile_saved": False,
            }

        logger.info(f"Generating profile for {user_id} with {len(windows)} session windows")

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
        logger.info("No trained GAT model found. Training on all available users...")
        results = self.train_all(force=True)

        # Find this user's result in the training output
        for r in results:
            if r.get("user_id") == user_id:
                return r

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
        from app.behavioral_logger import behavioral_logger

        cosmos_users = behavioral_logger.list_users()
        local_users = [p.stem for p in BEHAVIORAL_LOG_DIR.glob("*.jsonl")]
        # Merge, preserve order, deduplicate
        users = list(dict.fromkeys(cosmos_users + local_users))

        if not users:
            return [{"status": "error", "message": "No behavioral data found for any user"}]

        logger.info("train_all: discovered %d users (%d from Cosmos, %d local)", len(users), len(cosmos_users), len(local_users))

        try:
            import torch  # noqa: F401
        except ImportError:
            logger.warning("PyTorch not available, using numpy fallback")
            results: List[dict] = []
            for uid in users:
                events = _load_user_events(uid)
                windows = _split_into_windows(events)
                if len(windows) >= 2:
                    results.append(self._train_numpy_fallback(uid, windows))
                else:
                    results.append({
                        "user_id": uid, "status": "error",
                        "message": f"Need at least 2 session windows, found {len(windows)}",
                        "profile_saved": False,
                    })
            return results

        return self._train_gat_all_users(users, force)

    # ------------------------------------------------------------------ #
    # Numpy fallback (mean-vector profile, no triplet loss)               #
    # ------------------------------------------------------------------ #

    def _train_numpy_fallback(self, user_id: str, windows: List[List[dict]]) -> dict:
        import numpy as np

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
        import torch
        from types import SimpleNamespace
        from app.gat.gat_network import SiameseGATNetwork, GATTrainer
        from app.gat.data_processor import BehavioralDataProcessor, PyTorchDataConverter

        device = torch.device("cpu")

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
            events = _load_user_events(uid)
            windows = _split_into_windows(events)
            if len(windows) < 2:
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
                    logger.warning(f"Failed to convert window {i} for {uid}: {e}")

            if len(graphs) >= 2:
                user_graphs[uid] = graphs
            else:
                skipped.append({
                    "user_id": uid, "status": "error",
                    "message": "Not enough valid session graphs",
                    "profile_saved": False,
                })

        if not user_graphs:
            return skipped + [{"status": "error", "message": "No users with sufficient data for training"}]

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

            if (epoch + 1) % 10 == 0:
                avg_loss = epoch_loss / max(count, 1)
                logger.info(f"Epoch {epoch+1}/{n_epochs}, avg triplet loss: {avg_loss:.4f}")

        elapsed = time.time() - t0
        logger.info(f"GAT triplet training complete in {elapsed:.1f}s")

        # ---- 4. Save model checkpoint ------------------------------------
        self._save_checkpoint(model)
        self._model = model

        # ---- 5. Generate per-user profiles --------------------------------
        results: List[dict] = list(skipped)
        for uid, graphs in user_graphs.items():
            profile_vector = self._generate_profile_vector(model, graphs)
            self._save_profile(
                uid, profile_vector,
                method="gat_triplet", sessions=len(graphs), training_time=elapsed,
            )
            results.append({
                "user_id": uid,
                "status": "success",
                "message": f"GAT triplet training complete in {elapsed:.1f}s",
                "profile_saved": True,
                "sessions_used": len(graphs),
                "training_time_seconds": elapsed,
            })

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
