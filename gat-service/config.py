"""
GAT Service Configuration
"""
from typing import Literal


class GATSettings:
    # Service settings
    APP_NAME: str = "GAT Processing Service"
    VERSION: str = "1.0.0"
    HOST: str = "0.0.0.0"
    PORT: int = 8001
    LOG_LEVEL: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = "INFO"
    
    # GAT Model settings
    INPUT_DIM: int = 56  # Node feature dimension (48 behavioral + 8 event-type embedding; device info removed)
    HIDDEN_DIM: int = 128  # Hidden layer size
    OUTPUT_DIM: int = 64  # Final embedding dimension
    NUM_HEADS: int = 8  # Multi-head attention
    NUM_LAYERS: int = 3  # Number of GAT layers
    DROPOUT: float = 0.1
    
    # Training settings
    LEARNING_RATE: float = 0.001
    BATCH_SIZE: int = 32
    TRIPLET_MARGIN: float = 0.5  # Margin for triplet loss
    SIMILARITY_THRESHOLD: float = 0.85
    
    # Processing settings
    MAX_NODES: int = 64  # Maximum nodes per graph
    EDGE_THRESHOLD: float = 3.0  # Max time delta for edges (seconds)
    TIME_WINDOW_SECONDS: int = 20  # Sliding window size in seconds
    MIN_EVENTS_PER_WINDOW: int = 5
    MAX_EVENTS_PER_WINDOW: int = 100
    DISTINCT_EVENT_CONNECTIONS: int = 4  # Connect across 4 distinct event types
    DEVICE: str = "cpu"  # "cuda" if GPU available
    
    # Model persistence
    MODEL_SAVE_PATH: str = "./models/gat_model.pth"
    CHECKPOINT_INTERVAL: int = 100  # Save every N processed graphs


settings = GATSettings()