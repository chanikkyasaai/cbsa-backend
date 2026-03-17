"""
Enrollment Session Store
Tracks enrollment state per user across sessions.
Enrollment window = 5 minutes of total ACTIVE time (accumulated across sessions).

Stores enrollment state in Cosmos DB with local file fallback in DEBUG_MODE.
"""

import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, Optional

from app.config import settings

logger = logging.getLogger(__name__)

ENROLLMENT_DURATION_SECONDS = 5 * 60  # 5 minutes

DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data"
ENROLLMENT_FILE = DATA_DIR / "enrollment_store.json"

# ---------------------------------------------------------------------------
# Lazy import of azure-cosmos
# ---------------------------------------------------------------------------
try:
    from azure.cosmos import CosmosClient, PartitionKey  # type: ignore[import]
    from azure.cosmos.exceptions import CosmosResourceNotFoundError  # type: ignore[import]

    _COSMOS_SDK_AVAILABLE = True
except ImportError:
    _COSMOS_SDK_AVAILABLE = False
    logger.warning(
        "azure-cosmos package not installed – Cosmos DB enrollment store disabled"
    )


class UserEnrollmentState:
    """In-memory representation of a user's enrollment state."""

    def __init__(
        self,
        user_id: str,
        accumulated_seconds: float = 0.0,
        enrolled: bool = False,
        active_session_start: Optional[float] = None,
    ):
        self.user_id = user_id
        self.accumulated_seconds = accumulated_seconds  # Total seconds collected across sessions
        self.enrolled = enrolled
        self.active_session_start: Optional[float] = active_session_start  # epoch time when current session started

    def to_dict(self) -> dict:
        return {
            "user_id": self.user_id,
            "accumulated_seconds": self.accumulated_seconds,
            "enrolled": self.enrolled,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "UserEnrollmentState":
        return cls(
            user_id=d["user_id"],
            accumulated_seconds=d.get("accumulated_seconds", 0.0),
            enrolled=d.get("enrolled", False),
        )

    def seconds_remaining(self) -> float:
        """Seconds left to complete enrollment including current active session."""
        in_session = (time.time() - self.active_session_start) if self.active_session_start else 0.0
        total = self.accumulated_seconds + in_session
        return max(0.0, ENROLLMENT_DURATION_SECONDS - total)

    def is_enrollment_complete(self) -> bool:
        if self.enrolled:
            return True
        return self.seconds_remaining() <= 0.0


class EnrollmentStore:
    """Persistent store for enrollment state across users.
    
    In production (DEBUG_MODE=False): Cloud only
    In development (DEBUG_MODE=True): Read from cloud first, write to both
    """

    def __init__(self):
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        self._states: Dict[str, UserEnrollmentState] = {}
        self._container = None
        self._enabled = False
        self._try_connect()
        self._load()

    def _try_connect(self) -> None:
        """Connect to Cosmos DB enrollment container."""
        if not _COSMOS_SDK_AVAILABLE:
            return

        endpoint = settings.COSMOS_ENDPOINT.strip()
        key = settings.COSMOS_KEY.strip()
        if not endpoint or not key:
            logger.info(
                "COSMOS_ENDPOINT / COSMOS_KEY not set – "
                "Cosmos DB enrollment store disabled"
            )
            return

        database_name = settings.COSMOS_DATABASE
        container_name = settings.COSMOS_ENROLLMENT_CONTAINER

        try:
            client = CosmosClient(endpoint, credential=key)
            database = client.create_database_if_not_exists(id=database_name)
            self._container = database.create_container_if_not_exists(
                id=container_name,
                partition_key=PartitionKey(path="/userId"),
                offer_throughput=400,
            )
            self._enabled = True
            logger.info(
                "Cosmos DB enrollment store connected: database=%s container=%s",
                database_name,
                container_name,
            )
        except Exception as exc:
            logger.error("Failed to connect Cosmos enrollment store: %s", exc)

    def _load(self):
        """Load enrollment state from Cosmos DB (cloud first) or local file."""
        # Try Cosmos DB first
        if self._enabled and self._container is not None:
            try:
                items = list(
                    self._container.query_items(
                        query="SELECT * FROM c",
                        enable_cross_partition_query=True,
                    )
                )
                for item in items:
                    user_id = item.get("userId")
                    if user_id:
                        self._states[user_id] = UserEnrollmentState.from_dict(item)
                logger.info(f"Loaded enrollment state for {len(self._states)} users from Cosmos DB")
                return
            except Exception as e:
                logger.error(f"Failed to load enrollment store from Cosmos DB: {e}")

        # Fallback to local file in dev mode or if Cosmos fails
        if settings.DEBUG_MODE or not self._enabled:
            self._load_local()

    def _load_local(self):
        """Load enrollment state from local JSON file."""
        if ENROLLMENT_FILE.exists():
            try:
                with ENROLLMENT_FILE.open("r", encoding="utf-8") as f:
                    raw = json.load(f)
                for user_id, d in raw.items():
                    self._states[user_id] = UserEnrollmentState.from_dict(d)
                logger.info(f"Loaded enrollment state for {len(self._states)} users from local file")
            except Exception as e:
                logger.error(f"Failed to load local enrollment store: {e}")

    def _save(self):
        """Save enrollment state to Cosmos DB and (in dev mode) local file."""
        # Save to Cosmos DB
        if self._enabled and self._container is not None:
            try:
                for user_id, state in self._states.items():
                    document: Dict[str, Any] = {
                        "id": user_id,
                        "userId": user_id,
                        **state.to_dict(),
                    }
                    self._container.upsert_item(document)
            except Exception as e:
                logger.error(f"Failed to save enrollment store to Cosmos DB: {e}")

        # Save to local file in dev mode
        if settings.DEBUG_MODE:
            self._save_local()

    def _save_local(self):
        """Save enrollment state to local JSON file."""
        try:
            with ENROLLMENT_FILE.open("w", encoding="utf-8") as f:
                json.dump(
                    {uid: s.to_dict() for uid, s in self._states.items()},
                    f,
                    indent=2,
                )
        except Exception as e:
            logger.error(f"Failed to save local enrollment store: {e}")

    def get_or_create(self, user_id: str) -> UserEnrollmentState:
        if user_id not in self._states:
            self._states[user_id] = UserEnrollmentState(user_id=user_id)
            self._save()
        return self._states[user_id]

    def start_session(self, user_id: str):
        """Called when a user logs in / opens websocket."""
        state = self.get_or_create(user_id)
        if not state.enrolled:
            state.active_session_start = time.time()
            logger.info(
                f"Enrollment session started for {user_id}. "
                f"Accumulated so far: {state.accumulated_seconds:.1f}s, "
                f"remaining: {state.seconds_remaining():.1f}s"
            )

    def end_session(self, user_id: str):
        """Called when a user logs out / websocket disconnects."""
        state = self.get_or_create(user_id)
        if state.active_session_start is not None:
            elapsed = time.time() - state.active_session_start
            state.accumulated_seconds += elapsed
            state.active_session_start = None
            # Cap at enrollment duration
            if state.accumulated_seconds >= ENROLLMENT_DURATION_SECONDS and not state.enrolled:
                state.accumulated_seconds = ENROLLMENT_DURATION_SECONDS
            logger.info(
                f"Enrollment session ended for {user_id}. "
                f"Accumulated: {state.accumulated_seconds:.1f}s"
            )
            self._save()

    def mark_enrolled(self, user_id: str):
        """Mark user as fully enrolled."""
        state = self.get_or_create(user_id)
        state.enrolled = True
        state.accumulated_seconds = ENROLLMENT_DURATION_SECONDS
        state.active_session_start = None
        self._save()
        logger.info(f"User {user_id} marked as enrolled")

    def has_profile(self, user_id: str) -> bool:
        """Returns True if a trained profile exists for this user."""
        from app.azure.cosmos_profile_store import cosmos_profile_store

        return cosmos_profile_store.has_profile(user_id)

    def get_enrollment_status(self, user_id: str) -> dict:
        """
        Returns dict with:
          status: 'no_profile' | 'enrolling' | 'enrolled'
          seconds_remaining: float (only when enrolling)
          message: human-readable string
        """
        state = self.get_or_create(user_id)
        if state.enrolled or self.has_profile(user_id):
            return {"status": "enrolled", "message": "Enrollment complete"}

        remaining = state.seconds_remaining()
        return {
            "status": "enrolling",
            "seconds_remaining": remaining,
            "accumulated_seconds": state.accumulated_seconds,
            "total_seconds": ENROLLMENT_DURATION_SECONDS,
            "message": f"Enrolling... {remaining:.0f}s remaining",
        }


# Singleton
enrollment_store = EnrollmentStore()
