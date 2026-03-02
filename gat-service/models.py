"""
GAT Service Data Models
Pydantic models for API communication
"""
from pydantic import BaseModel, Field
from typing import List, Dict, Optional, Any
from datetime import datetime


class EventNode(BaseModel):
    """Single event node in temporal graph"""
    node_id: int
    timestamp: float
    event_type: str
    behavioral_vector: List[float] = Field(..., min_length=56, max_length=56)
    signature: str = ""
    nonce: str = ""


class TemporalEdge(BaseModel):
    """Temporal edge between nodes"""
    source_node_id: int
    target_node_id: int
    time_delta: float
    event_transition: str
    attention_weight: Optional[float] = None


class TemporalGraph(BaseModel):
    """Complete temporal graph for GAT processing"""
    session_id: str
    user_id: Optional[str] = None
    nodes: List[EventNode]
    edges: List[TemporalEdge]
    window_start: float
    window_end: float
    total_events: int
    session_duration: float
    event_diversity: int
    avg_time_between_events: float


class GATProcessingRequest(BaseModel):
    """Request to GAT service"""
    graph: TemporalGraph
    user_profile_vector: Optional[List[float]] = Field(None, min_length=64, max_length=64)
    processing_mode: str = Field("inference", pattern="^(inference|enrollment|training)$")
    attention_heads: int = Field(8, ge=1, le=16)
    embedding_dim: int = Field(64, ge=32, le=256)
    similarity_threshold: float = Field(0.85, ge=0.0, le=1.0)


class GATProcessingResponse(BaseModel):
    """Response from GAT service"""
    session_vector: List[float] = Field(..., min_length=64, max_length=64)
    similarity_score: Optional[float] = Field(None, ge=0.0, le=1.0)
    auth_decision: str = Field(..., pattern="^(ALLOW|BLOCK|UNCERTAIN)$")
    confidence: float = Field(..., ge=0.0, le=1.0)
    processing_time_ms: float
    attention_weights: Optional[List[List[float]]] = None
    node_embeddings: Optional[List[List[float]]] = None
    model_info: Dict[str, Any] = Field(default_factory=dict)


class TrainingExample(BaseModel):
    """Training example for Siamese network"""
    anchor_graph: TemporalGraph
    positive_graph: TemporalGraph
    negative_graph: TemporalGraph
    user_id: str


class TrainingRequest(BaseModel):
    """Request for model training"""
    training_examples: List[TrainingExample]
    epochs: int = Field(10, ge=1, le=100)
    learning_rate: float = Field(0.001, ge=0.0001, le=0.01)
    batch_size: int = Field(32, ge=1, le=128)


class TrainingResponse(BaseModel):
    """Response from training"""
    training_status: str
    final_loss: float
    epochs_completed: int
    training_time_seconds: float
    model_saved: bool
    model_path: str