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
WINDOW_SECONDS = 30           # seconds per session window

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
    path = BEHAVIORAL_LOG_DIR / f"{user_id}.jsonl"
    if not path.exists():
        return []
    events = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    events.append(json.loads(line))
                except Exception:
                    pass
    return sorted(events, key=lambda e: e.get("timestamp", 0))


def _split_into_windows(events: List[dict], window_sec: float = WINDOW_SECONDS) -> List[List[dict]]:
    """Split a sorted event list into non-overlapping time windows."""
    if not events:
        return []
    windows = []
    window: List[dict] = []
    window_start = events[0].get("timestamp", 0) or 0.0
    for ev in events:
        ts = ev.get("timestamp", 0) or 0.0
        if ts - window_start <= window_sec:
            window.append(ev)
        else:
            if len(window) >= MIN_EVENTS_FOR_SESSION:
                windows.append(window)
            window = [ev]
            window_start = ts
    if len(window) >= MIN_EVENTS_FOR_SESSION:
        windows.append(window)
    return windows


def _window_to_matrix(window: List[dict]):
    """Convert a window of events into a (N, 56) numpy array."""
    import numpy as np
    vecs = [_extract_vector(e) for e in window]
    return np.array(vecs, dtype=np.float32)


class TripletTrainer:
    """
    Trains a simple metric-learning model on user behavioral data.
    Uses in-house user as anchor/positive and other users as negative.
    Falls back to a numpy-based approach when PyTorch/PyG are unavailable,
    producing a mean-vector profile instead.
    """

    def __init__(self):
        if settings.DEBUG_MODE:
            PROFILES_DIR.mkdir(parents=True, exist_ok=True)
            CHECKPOINT_PATH.parent.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------ #
    # Public API                                                           #
    # ------------------------------------------------------------------ #

    def train_user(self, user_id: str, force: bool = False) -> dict:
        """
        Train a profile for a single user.
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

        logger.info(f"Training profile for {user_id} with {len(windows)} session windows")

        try:
            import torch
            result = self._train_pytorch(user_id, windows, force)
        except ImportError:
            logger.warning("PyTorch not available, using numpy fallback for profile creation")
            result = self._train_numpy_fallback(user_id, windows)

        return result

    def train_all(self, force: bool = False) -> List[dict]:
        """Train profiles for every user who has behavioral data."""
        users = [p.stem for p in BEHAVIORAL_LOG_DIR.glob("*.jsonl")]
        if not users:
            return [{"status": "error", "message": "No behavioral data found for any user"}]
        results = []
        for uid in users:
            logger.info(f"Training user: {uid}")
            results.append(self.train_user(uid, force=force))
        return results

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
    # PyTorch triplet training                                             #
    # ------------------------------------------------------------------ #

    def _train_pytorch(self, user_id: str, windows: List[List[dict]], force: bool) -> dict:
        import torch
        import torch.nn as nn
        import numpy as np

        device = torch.device("cpu")

        # Build session embeddings for anchor user (mean-pool each window)
        user_tensors = [
            torch.FloatTensor(_window_to_matrix(w).mean(axis=0)).to(device)
            for w in windows
        ]

        # Gather negative samples from other users (if any)
        all_users = [p.stem for p in BEHAVIORAL_LOG_DIR.glob("*.jsonl") if p.stem != user_id]
        neg_tensors: List[torch.Tensor] = []
        for other in all_users:
            other_events = _load_user_events(other)
            other_windows = _split_into_windows(other_events)
            for w in other_windows:
                mat = _window_to_matrix(w)
                neg_tensors.append(torch.FloatTensor(mat.mean(axis=0)).to(device))

        if not neg_tensors:
            # No other users — synthetic negatives via additive Gaussian noise
            logger.info(f"No other users found; generating synthetic negatives for {user_id}")
            for t in user_tensors:
                noise = torch.randn_like(t) * 0.5
                neg_tensors.append((t + noise).clamp(0.0, 1.0))

        # Simple MLP encoder: 56-D → 64-D profile embedding
        encoder = nn.Sequential(
            nn.Linear(INPUT_DIM, HIDDEN_DIM),
            nn.ReLU(),
            nn.Dropout(DROPOUT),
            nn.Linear(HIDDEN_DIM, OUTPUT_DIM),
        ).to(device)

        optimizer = torch.optim.Adam(encoder.parameters(), lr=LEARNING_RATE)
        triplet_loss_fn = nn.TripletMarginLoss(margin=TRIPLET_MARGIN)

        n_epochs = 30
        t0 = time.time()
        for epoch in range(n_epochs):
            epoch_loss = 0.0
            count = 0
            # Build triplets: every pair of anchor-positive windows with random negative
            for i in range(len(user_tensors)):
                for j in range(len(user_tensors)):
                    if i == j:
                        continue
                    anchor = encoder(user_tensors[i].unsqueeze(0))
                    positive = encoder(user_tensors[j].unsqueeze(0))
                    neg_idx = int(torch.randint(0, len(neg_tensors), (1,)).item())
                    negative = encoder(neg_tensors[neg_idx].unsqueeze(0))
                    loss = triplet_loss_fn(anchor, positive, negative)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    epoch_loss += loss.item()
                    count += 1
            if (epoch + 1) % 10 == 0:
                logger.info(
                    f"[{user_id}] Epoch {epoch+1}/{n_epochs}, "
                    f"avg loss: {epoch_loss/max(count,1):.4f}"
                )

        # Create profile vector: mean of all encoded windows
        encoder.eval()
        with torch.no_grad():
            encoded = [encoder(t.unsqueeze(0)).squeeze(0) for t in user_tensors]
            profile_tensor = torch.stack(encoded).mean(dim=0)
        profile_vector: List[float] = profile_tensor.cpu().numpy().tolist()  # type: ignore[assignment]

        elapsed = time.time() - t0
        self._save_profile(
            user_id, profile_vector,
            method="triplet_mlp",
            sessions=len(windows),
            training_time=elapsed,
        )

        return {
            "user_id": user_id,
            "status": "success",
            "message": f"Triplet training complete in {elapsed:.1f}s",
            "profile_saved": True,
            "sessions_used": len(windows),
            "training_time_seconds": elapsed,
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
