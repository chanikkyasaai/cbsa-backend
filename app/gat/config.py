"""
GAT Service Configuration
"""


class GATSettings:
    # Service settings
    app_name: str = "GAT Processing Service"
    version: str = "1.0.0"
    host: str = "0.0.0.0"
    port: int = 8001
    log_level: str = "INFO"
    
    # GAT Model settings
    input_dim: int = 56  # Node feature dimension (48 behavioral + 8 event-type embedding; device info removed)
    hidden_dim: int = 128  # Hidden layer size
    output_dim: int = 64  # Final embedding dimension
    num_attention_heads: int = 8  # Multi-head attention
    num_layers: int = 3  # Number of GAT layers
    dropout_rate: float = 0.1
    temporal_dim: int = 8  # Temporal encoding dimension
    
    # Training settings
    learning_rate: float = 0.001
    batch_size: int = 32
    triplet_margin: float = 0.5  # Margin for triplet loss
    similarity_threshold: float = 0.85
    
    # Processing settings
    max_nodes: int = 64  # Maximum nodes per graph
    edge_threshold: float = 3.0  # Max time delta for edges (seconds)
    time_window_seconds: int = 20  # Sliding window size in seconds
    min_events_per_window: int = 5
    max_events_per_window: int = 100
    distinct_event_connections: int = 4  # Connect across 4 distinct event types
    device: str = "cpu"  # "cuda" if GPU available
    
    # Model persistence
    model_save_path: str = "./models/gat_model.pth"
    checkpoint_interval: int = 100  # Save every N processed graphs


settings = GATSettings()


def get_gat_settings() -> GATSettings:
    return settings