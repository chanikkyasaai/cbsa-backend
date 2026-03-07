"""
Cosmos DB User Profile Store

Persists user baseline profiles (64-D vectors) in a dedicated Cosmos DB
container, separate from the computation-log container.

Configuration is read from the centralised ``app.config.settings`` object
which in turn reads from environment variables:
  COSMOS_ENDPOINT            – Cosmos DB account URI  (required)
  COSMOS_KEY                 – Cosmos DB primary/secondary key (required)
  COSMOS_DATABASE            – database name   (default: "cbsa-logs")
  COSMOS_PROFILES_CONTAINER  – container name  (default: "user-profiles")

The container uses "/userId" as the partition key.
If the environment variables are absent the store operates as a no-op and
falls back to local-disk JSON files so the app still works offline.
"""

import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from app.config import settings

logger = logging.getLogger(__name__)

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
        "azure-cosmos package not installed – Cosmos DB profile store disabled"
    )

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
PROFILES_DIR = DATA_DIR / "profiles"


class CosmosProfileStore:
    """
    Manages user baseline profiles (64-D vectors) in Azure Cosmos DB.

    Each document has the shape::

        {
            "id":              "<user_id>",
            "userId":          "<user_id>",
            "profileVector":   [float x 64],
            "vectorDim":       64,
            "method":          "triplet_mlp",
            "sessionsUsed":    5,
            "trainingTimeSec": 12.3,
            "createdAt":       1709301234.567,
            "updatedAt":       1709301234.567
        }
    """

    def __init__(self) -> None:
        self._container = None
        self._enabled = False
        self._try_connect()

    # ------------------------------------------------------------------
    # Connection
    # ------------------------------------------------------------------

    def _try_connect(self) -> None:
        if not _COSMOS_SDK_AVAILABLE:
            return

        endpoint = settings.COSMOS_ENDPOINT.strip()
        key = settings.COSMOS_KEY.strip()
        if not endpoint or not key:
            logger.info(
                "COSMOS_ENDPOINT / COSMOS_KEY not set – "
                "Cosmos DB profile store disabled"
            )
            return

        database_name = settings.COSMOS_DATABASE
        container_name = settings.COSMOS_PROFILES_CONTAINER

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
                "Cosmos DB profile store connected: database=%s container=%s",
                database_name,
                container_name,
            )
        except Exception as exc:
            logger.error("Failed to connect Cosmos profile store: %s", exc)

    # ------------------------------------------------------------------
    # CRUD
    # ------------------------------------------------------------------

    def save_profile(
        self,
        user_id: str,
        vector: List[float],
        method: str = "unknown",
        sessions: int = 0,
        training_time: float = 0.0,
    ) -> None:
        """Save or update a user profile.

        In production (``DEBUG_MODE=False``) only Cosmos DB is written.
        In development the profile is also persisted to local disk as a
        fallback so the app works without Azure credentials.
        """
        now = time.time()
        document: Dict[str, Any] = {
            "id": user_id,
            "userId": user_id,
            "profileVector": vector,
            "vectorDim": len(vector),
            "method": method,
            "sessionsUsed": sessions,
            "trainingTimeSec": training_time,
            "createdAt": now,
            "updatedAt": now,
        }

        # Persist to Cosmos DB
        if self._enabled and self._container is not None:
            try:
                self._container.upsert_item(document)
                logger.info("Profile for %s saved to Cosmos DB", user_id)
            except Exception as exc:
                logger.error(
                    "Failed to save profile for %s to Cosmos DB: %s",
                    user_id,
                    exc,
                )

        # Local-disk fallback (dev / debug only)
        if settings.DEBUG_MODE:
            self._save_local(user_id, vector, method, sessions, training_time, now)

    def load_profile(self, user_id: str) -> Optional[List[float]]:
        """Load a profile vector.

        Tries Cosmos first.  Falls back to local disk only in dev mode.
        """
        if self._enabled and self._container is not None:
            try:
                item = self._container.read_item(
                    item=user_id, partition_key=user_id
                )
                vec = item.get("profileVector")
                if vec:
                    return vec
            except Exception:
                pass  # fall through

        if settings.DEBUG_MODE:
            return self._load_local(user_id)
        return None

    def has_profile(self, user_id: str) -> bool:
        """Check whether a profile exists.

        In production only Cosmos is checked; in dev local disk is also checked.
        """
        if self._enabled and self._container is not None:
            try:
                self._container.read_item(
                    item=user_id, partition_key=user_id
                )
                return True
            except Exception:
                pass
        if settings.DEBUG_MODE:
            return self._has_local(user_id)
        return False

    def delete_profile(self, user_id: str) -> bool:
        """Delete a user profile from Cosmos and (in dev mode) local disk."""
        deleted = False
        if self._enabled and self._container is not None:
            try:
                self._container.delete_item(
                    item=user_id, partition_key=user_id
                )
                deleted = True
            except Exception:
                pass

        if settings.DEBUG_MODE:
            local_path = PROFILES_DIR / f"{user_id}_profile.json"
            if local_path.exists():
                local_path.unlink()
                deleted = True
        return deleted

    def delete_all_profiles(self) -> int:
        """Delete every profile in Cosmos and (in dev mode) local disk.  Returns count."""
        count = 0

        if self._enabled and self._container is not None:
            try:
                items = list(
                    self._container.query_items(
                        query="SELECT c.id, c.userId FROM c",
                        enable_cross_partition_query=True,
                    )
                )
                for item in items:
                    try:
                        self._container.delete_item(
                            item=item["id"],
                            partition_key=item["userId"],
                        )
                        count += 1
                    except Exception:
                        pass
            except Exception as exc:
                logger.error("Failed to truncate Cosmos profiles: %s", exc)

        # Local disk (dev / debug only)
        if settings.DEBUG_MODE and PROFILES_DIR.exists():
            for p in PROFILES_DIR.glob("*_profile.json"):
                p.unlink(missing_ok=True)
                count += 1

        return count

    # ------------------------------------------------------------------
    # Local-disk helpers
    # ------------------------------------------------------------------

    def _save_local(
        self,
        user_id: str,
        vector: List[float],
        method: str,
        sessions: int,
        training_time: float,
        created_at: float,
    ) -> None:
        PROFILES_DIR.mkdir(parents=True, exist_ok=True)
        profile = {
            "user_id": user_id,
            "profile_vector": vector,
            "vector_dim": len(vector),
            "method": method,
            "sessions_used": sessions,
            "training_time_seconds": training_time,
            "created_at": created_at,
        }
        path = PROFILES_DIR / f"{user_id}_profile.json"
        with path.open("w", encoding="utf-8") as f:
            json.dump(profile, f, indent=2)

    def _load_local(self, user_id: str) -> Optional[List[float]]:
        path = PROFILES_DIR / f"{user_id}_profile.json"
        if not path.exists():
            return None
        try:
            with path.open("r", encoding="utf-8") as f:
                data = json.load(f)
            return data.get("profile_vector")
        except Exception as e:
            logger.error("Failed to load local profile for %s: %s", user_id, e)
            return None

    def _has_local(self, user_id: str) -> bool:
        return (PROFILES_DIR / f"{user_id}_profile.json").exists()


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------
cosmos_profile_store = CosmosProfileStore()
