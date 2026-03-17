"""
Layer 3 GAT Processing Models and Data Structures
Implements the data models for Graph Attention Network processing
"""
from typing import List, Dict, Optional, Tuple
from pydantic import BaseModel
import numpy as np
from datetime import datetime


class GATEventNode(BaseModel):
    """Single event node for GAT graph representation"""
    node_id: int
    timestamp: float
    event_type: str

    # 56-dimensional node features (48 behavioral + 8 event-type embedding; device info removed)
    behavioral_vector: List[float]

    # Signature for attestation
    signature: str
    nonce: str


class GATTemporalEdge(BaseModel):
    """Temporal edge between two consecutive events"""
    source_node_id: int
    target_node_id: int

    # Temporal relationship features
    time_delta: float  # Time difference in seconds
    event_transition: str  # e.g., "PAGE_ENTER_HOME->TOUCH_BALANCE_TOGGLE"

    # Edge weight (computed by attention mechanism)
    attention_weight: Optional[float] = None


class GATGraph(BaseModel):
    """Complete temporal graph for GAT processing"""
    session_id: str
    user_id: Optional[str]

    # Graph structure
    nodes: List[GATEventNode]
    edges: List[GATTemporalEdge]

    # Metadata
    window_start: float
    window_end: float
    total_events: int

    # Graph-level features
    session_duration: float
    event_diversity: int  # Number of unique event types
    avg_time_between_events: float


class GATProcessingRequest(BaseModel):
    """Request to cloud GAT service"""
    graph: GATGraph
    user_profile_vector: Optional[List[float]] = None  # 64-dim if available
    processing_mode: str = "inference"  # "inference" or "enrollment"

    # Processing parameters
    attention_heads: int = 8
    embedding_dim: int = 64
    similarity_threshold: float = 0.85


class GATProcessingResponse(BaseModel):
    """Response from GAT service — raw scores only, no auth decisions"""
    session_vector: List[float]  # 64-dimensional output embedding
    similarity_score: Optional[float] = None  # Cosine similarity to user profile

    # Debug information
    processing_time_ms: float
    attention_weights: Optional[List[List[float]]] = None
    node_embeddings: Optional[List[List[float]]] = None


class UserProfile(BaseModel):
    """User behavioral profile stored in database"""
    user_id: str
    profile_vector: List[float]  # 64-dimensional master profile

    # Profile metadata
    enrollment_sessions: int
    last_updated: datetime
    profile_confidence: float

    # Adaptive parameters
    consistency_threshold_low: float = 0.3
    consistency_threshold_high: float = 0.7
    similarity_threshold: float = 0.85
