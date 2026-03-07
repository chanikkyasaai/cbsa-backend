import logging
import os
from typing import Literal


class Settings:
    APP_NAME: str = "CBSA Backend"
    VERSION: str = "1.0.0"
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    LOG_LEVEL: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = "INFO"
    WEBSOCKET_ENDPOINT: str = "/ws/behaviour"

    # Layer 3 GAT settings (cloud endpoint no longer used; GAT is now in-process)
    DEBUG_MODE: bool = True  # Set to False in production
    GAT_CLOUD_ENDPOINT: str = ""  # Kept for backwards compatibility; unused
    GAT_WINDOW_SIZE: int = 32  # Deprecated: event count window (kept for compatibility)
    GAT_WINDOW_SECONDS: int = 20  # Temporal graph window in seconds
    GAT_NODE_FEATURE_DIM: int = 56  # 48 behavioral + 8 event-type embedding (device info removed)
    GAT_EDGE_DISTINCT_TARGET: int = 4  # Distinct event types to reach per node
    GAT_ESCALATION_THRESHOLD: float = 0.5  # Assume Layer 2 escalates at this threshold
    GAT_INFERENCE_INTERVAL_SECONDS: float = 5.0  # Interval between GAT inference calls

    # Azure Cosmos DB – shared connection settings for all containers.
    # Used by: cosmos_logger (computation-logs), cosmos_profile_store (user-profiles).
    # Values are read from environment variables at runtime so secrets are
    # never hard-coded in source.
    COSMOS_ENDPOINT: str = os.environ.get("COSMOS_ENDPOINT", "")
    COSMOS_KEY: str = os.environ.get("COSMOS_KEY", "")
    COSMOS_DATABASE: str = os.environ.get("COSMOS_DATABASE", "cbsa-logs")
    COSMOS_CONTAINER: str = os.environ.get("COSMOS_CONTAINER", "computation-logs")
    COSMOS_PROFILES_CONTAINER: str = os.environ.get("COSMOS_PROFILES_CONTAINER", "user-profiles")

    # Azure Blob Storage – model checkpoint files (.pth).
    # Used by: blob_model_store, gat_engine (download on startup).
    AZURE_STORAGE_CONNECTION_STRING: str = os.environ.get("AZURE_STORAGE_CONNECTION_STRING", "")
    AZURE_STORAGE_CONTAINER: str = os.environ.get("AZURE_STORAGE_CONTAINER", "cbsa-models")


settings = Settings()


def configure_logging():
    logging.basicConfig(
        level=getattr(logging, settings.LOG_LEVEL),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
