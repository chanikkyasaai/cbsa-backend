import logging
from typing import Literal


class Settings:
    APP_NAME: str = "CBSA Backend"
    VERSION: str = "1.0.0"
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    LOG_LEVEL: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = "INFO"
    WEBSOCKET_ENDPOINT: str = "/ws/behaviour"
    
    # Layer 3 GAT settings
    DEBUG_MODE: bool = True  # Set to False in production
    GAT_CLOUD_ENDPOINT: str = "http://localhost:8001"
    GAT_WINDOW_SIZE: int = 32  # Deprecated: event count window (kept for compatibility)
    GAT_WINDOW_SECONDS: int = 20  # Temporal graph window in seconds
    GAT_NODE_FEATURE_DIM: int = 56  # 48 behavioral + 8 event-type embedding (device info removed)
    GAT_EDGE_DISTINCT_TARGET: int = 4  # Distinct event types to reach per node
    GAT_ESCALATION_THRESHOLD: float = 0.5  # Assume Layer 2 escalates at this threshold


settings = Settings()


def configure_logging():
    logging.basicConfig(
        level=getattr(logging, settings.LOG_LEVEL),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
