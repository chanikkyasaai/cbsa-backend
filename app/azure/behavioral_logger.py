"""
Behavioral Data Logger

Logs raw behavioral events to Azure Cosmos DB (and to per-user JSONL files
when DEBUG_MODE is enabled).

Behaviour follows the pattern used by the rest of the codebase:
  - Production  (DEBUG_MODE=False): write to Cosmos DB only.
  - Development (DEBUG_MODE=True):  write to both Cosmos DB and local JSONL files.
  - If Cosmos DB is not configured:  fall back to local JSONL files only.
"""

import json
import logging

# Suppress Azure SDK logs
logging.getLogger("azure").setLevel(logging.WARNING)
logging.getLogger("azure.core.pipeline.policies.http_logging_policy").setLevel(logging.WARNING)
logging.getLogger("azure.cosmos._cosmos_http_logging_policy").setLevel(logging.WARNING)
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List

from app.config import settings

logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data"
BEHAVIORAL_LOG_DIR = DATA_DIR / "behavioral_logs"

# ---------------------------------------------------------------------------
# Lazy import of azure-cosmos
# ---------------------------------------------------------------------------
try:
    from azure.cosmos import CosmosClient, PartitionKey  # type: ignore[import]

    _COSMOS_SDK_AVAILABLE = True
except ImportError:
    _COSMOS_SDK_AVAILABLE = False
    logger.warning(
        "azure-cosmos package not installed – Cosmos DB behavioral logger disabled"
    )


class BehavioralFileLogger:
    """
    Logs raw behavioral events.

    In production writes to Cosmos DB only.
    In DEBUG_MODE writes to both Cosmos DB and local JSONL files.
    If Cosmos DB is not configured, falls back to local JSONL files.
    """

    def __init__(self):
        BEHAVIORAL_LOG_DIR.mkdir(parents=True, exist_ok=True)
        self._container = None
        self._enabled = False
        self._try_connect()
        logger.info("BehavioralFileLogger initialized. Local logs at: %s", BEHAVIORAL_LOG_DIR)

    # ------------------------------------------------------------------
    # Cosmos connection
    # ------------------------------------------------------------------

    def _try_connect(self) -> None:
        if not _COSMOS_SDK_AVAILABLE:
            return

        endpoint = settings.COSMOS_ENDPOINT.strip()
        key = settings.COSMOS_KEY.strip()
        if not endpoint or not key:
            logger.info(
                "COSMOS_ENDPOINT / COSMOS_KEY not set – "
                "Cosmos DB behavioral logger disabled"
            )
            return

        db_name = settings.COSMOS_DATABASE
        container_name = settings.COSMOS_BEHAVIOUR_LOGS_CONTAINER

        try:
            client = CosmosClient(endpoint, credential=key)
            database = client.create_database_if_not_exists(id=db_name)
            self._container = database.create_container_if_not_exists(
                id=container_name,
                partition_key=PartitionKey(path="/userId"),
                offer_throughput=400,
            )
            self._enabled = True
            logger.info(
                "Cosmos DB behavioral logger connected: database=%s container=%s",
                db_name,
                container_name,
            )
        except Exception as exc:
            logger.error("Failed to connect Cosmos behavioral logger: %s", exc)

    # ------------------------------------------------------------------
    # Local-file helpers
    # ------------------------------------------------------------------

    def _log_path(self, user_id: str) -> Path:
        safe_uid = user_id.replace("/", "_").replace("\\", "_")
        return BEHAVIORAL_LOG_DIR / f"{safe_uid}.jsonl"

    def _write_local(self, user_id: str, record: Dict[str, Any]) -> None:
        path = self._log_path(user_id)
        try:
            with path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(record) + "\n")
        except Exception as e:
            logger.error("Failed to write local behavioral log for user %s: %s", user_id, e)

    def _read_local(self, user_id: str) -> List[Dict[str, Any]]:
        path = self._log_path(user_id)
        if not path.exists():
            return []
        events: List[Dict[str, Any]] = []
        try:
            with path.open("r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        try:
                            events.append(json.loads(line))
                        except json.JSONDecodeError:
                            pass
        except Exception as e:
            logger.error("Failed to load local events for user %s: %s", user_id, e)
        return events

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def log_event(self, user_id: str, session_id: str, event_data: Dict[str, Any]) -> None:
        """Append a single event to the behavioral log."""
        record = {
            "logged_at": time.time(),
            "user_id": user_id,
            "session_id": session_id,
            **event_data,
        }

        # Write to Cosmos DB
        if self._enabled and self._container is not None:
            try:
                self._container.upsert_item(
                    {
                        "id": str(uuid.uuid4()),
                        "userId": user_id,
                        "sessionId": session_id,
                        "loggedAt": record["logged_at"],
                        **event_data,
                    }
                )
            except Exception as exc:
                logger.error("Failed to write behavioral event to Cosmos for user %s: %s", user_id, exc)

        # Write to local file in debug mode or when Cosmos is not available
        if settings.DEBUG_MODE or not self._enabled:
            self._write_local(user_id, record)

    def load_user_events(self, user_id: str) -> List[Dict[str, Any]]:
        """Load all logged events for a user. Queries Cosmos first, falls back to local."""
        if self._enabled and self._container is not None:
            try:
                items = list(
                    self._container.query_items(
                        query="SELECT * FROM c WHERE c.userId = @uid ORDER BY c.loggedAt ASC",
                        parameters=[{"name": "@uid", "value": user_id}],
                        partition_key=user_id,
                    )
                )
                if items:
                    return items
            except Exception as exc:
                logger.error("Failed to load Cosmos behavioral events for user %s: %s", user_id, exc)

        # Fallback to local in debug or when Cosmos unavailable
        if settings.DEBUG_MODE or not self._enabled:
            return self._read_local(user_id)
        return []

    def list_users(self) -> List[str]:
        """Return list of user IDs that have logged data."""
        if self._enabled and self._container is not None:
            try:
                items = list(
                    self._container.query_items(
                        query="SELECT DISTINCT c.userId FROM c",
                        enable_cross_partition_query=True,
                    )
                )
                return [item["userId"] for item in items if item.get("userId")]
            except Exception as exc:
                logger.error("Failed to list Cosmos behavioral users: %s", exc)

        # Fallback to local JSONL files
        return [p.stem for p in BEHAVIORAL_LOG_DIR.glob("*.jsonl")]

    def delete_user_log(self, user_id: str) -> None:
        """Delete all logged events for a user."""
        if self._enabled and self._container is not None:
            try:
                items = list(
                    self._container.query_items(
                        query="SELECT c.id FROM c WHERE c.userId = @uid",
                        parameters=[{"name": "@uid", "value": user_id}],
                        partition_key=user_id,
                    )
                )
                for item in items:
                    try:
                        self._container.delete_item(item=item["id"], partition_key=user_id)
                    except Exception:
                        pass
            except Exception as exc:
                logger.error("Failed to delete Cosmos behavioral log for user %s: %s", user_id, exc)

        # Always clean up local file too
        log_path = self._log_path(user_id)
        if log_path.exists():
            log_path.unlink(missing_ok=True)

    def delete_all_logs(self) -> int:
        """Delete all behavioral logs. Returns count of Cosmos docs deleted."""
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
                            item=item["id"], partition_key=item["userId"]
                        )
                        count += 1
                    except Exception:
                        pass
            except Exception as exc:
                logger.error("Failed to truncate Cosmos behavioral logs: %s", exc)

        # Always clean up local files too
        if BEHAVIORAL_LOG_DIR.exists():
            for p in BEHAVIORAL_LOG_DIR.glob("*.jsonl"):
                p.unlink(missing_ok=True)

        return count


# Singleton
behavioral_logger = BehavioralFileLogger()
