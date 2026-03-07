"""
Cosmos DB Computation Logger

Logs every processed event's computation results (engine metrics + GAT scores)
as a JSON document in Azure Cosmos DB.

Configuration is read from the centralised ``app.config.settings`` object
which in turn reads from environment variables:
  COSMOS_ENDPOINT   – Cosmos DB account URI  (required to enable logging)
  COSMOS_KEY        – Cosmos DB primary/secondary key (required)
  COSMOS_DATABASE   – database name (default: "cbsa-logs")
  COSMOS_CONTAINER  – container name (default: "computation-logs")

The container should be created with "/userId" as the partition key.
If the environment variables are absent, logging is silently skipped.
"""

import logging
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, Optional

from app.config import settings

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Lazy import of azure-cosmos so the app starts even if the package is absent
# ---------------------------------------------------------------------------
try:
    from azure.cosmos import CosmosClient, PartitionKey  # type: ignore[import]
    from azure.cosmos.exceptions import CosmosResourceNotFoundError  # type: ignore[import]

    _COSMOS_SDK_AVAILABLE = True
except ImportError:
    _COSMOS_SDK_AVAILABLE = False
    logger.warning(
        "azure-cosmos package not installed – Cosmos DB logging is disabled. "
        "Install it with: pip install azure-cosmos"
    )


class CosmosComputationLogger:
    """
    Thread-safe logger that writes computation records to Azure Cosmos DB.

    Each document has the shape:

    {
        "id": "<uuid>",
        "userId": "<user_id>",
        "timestamp": "2026-03-06 09:22:39 UTC",   # human-readable
        "sessionId": "<session_id>",
        "eventType": "<event_type>",
        "engineMetrics": {
            "similarityScore": 0.92,
            "shortDrift": 0.03,
            "longDrift": 0.01,
            "stabilityScore": 0.95,
            "matchedPrototypeId": 2
        },
        "gatResult": {
            "similarityScore": 0.88,
            "sessionVector": [...],
            "processingTimeMs": 45.2
        }
    }
    """

    def __init__(self):
        self._container = None
        self._enabled = False
        self._try_connect()

    def _try_connect(self):
        if not _COSMOS_SDK_AVAILABLE:
            return

        endpoint = settings.COSMOS_ENDPOINT.strip()
        key = settings.COSMOS_KEY.strip()

        if not endpoint or not key:
            logger.info(
                "COSMOS_ENDPOINT / COSMOS_KEY not set – Cosmos DB logging disabled"
            )
            return

        database_name = settings.COSMOS_DATABASE
        container_name = settings.COSMOS_CONTAINER

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
                "Cosmos DB logger connected: database=%s container=%s",
                database_name,
                container_name,
            )
        except Exception as exc:
            logger.error("Failed to connect to Cosmos DB: %s", exc)

    def log_computation(
        self,
        user_id: str,
        session_id: str,
        event_type: str,
        engine_metrics: Optional[Dict[str, Any]] = None,
        gat_result: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Write a computation record to Cosmos DB.

        This is a synchronous call; callers in the async WebSocket handler
        should wrap it with asyncio.to_thread() to avoid blocking the event loop.

        If Cosmos DB is not configured / unavailable the call is a no-op.
        """
        if not self._enabled or self._container is None:
            return

        now_utc = datetime.now(tz=timezone.utc)
        document: Dict[str, Any] = {
            "id": str(uuid.uuid4()),
            "userId": user_id,
            # ISO 8601 (sortable/queryable) and a human-readable display string
            "timestamp": now_utc.isoformat(),
            "timestampDisplay": now_utc.strftime("%Y-%m-%d %H:%M:%S UTC"),
            "sessionId": session_id,
            "eventType": event_type,
            "engineMetrics": engine_metrics or {},
            "gatResult": gat_result or {},
            "hasGatData": bool(gat_result),
        }

        try:
            self._container.upsert_item(document)
        except Exception as exc:
            logger.error("Failed to write computation log to Cosmos DB: %s", exc)


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------
cosmos_logger = CosmosComputationLogger()
