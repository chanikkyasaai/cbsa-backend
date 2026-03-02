"""
Simple GAT Service for Testing
Minimal FastAPI service without complex dependencies
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
import logging
import random
import time
from datetime import datetime
import json
import hashlib

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Pydantic models for API
class EventNode(BaseModel):
    node_id: int
    timestamp: float
    event_type: str
    behavioral_vector: List[float] = Field(..., min_length=60, max_length=60)
    device_features: Dict[str, float] = Field(default_factory=dict)
    signature: str = ""
    nonce: str = ""

class TemporalEdge(BaseModel):
    source_node_id: int
    target_node_id: int
    time_delta: float
    event_transition: str
    attention_weight: Optional[float] = None

class TemporalGraph(BaseModel):
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
    graph: TemporalGraph
    user_profile_vector: Optional[List[float]] = Field(None, min_length=64, max_length=64)
    processing_mode: str = Field("inference", pattern="^(inference|enrollment|training)$")
    attention_heads: int = Field(8, ge=1, le=16)
    embedding_dim: int = Field(64, ge=32, le=256)
    similarity_threshold: float = Field(0.85, ge=0.0, le=1.0)

class GATProcessingResponse(BaseModel):
    session_vector: List[float] = Field(..., min_length=64, max_length=64)
    similarity_score: Optional[float] = Field(None, ge=0.0, le=1.0)
    auth_decision: str = Field(..., pattern="^(ALLOW|BLOCK|UNCERTAIN)$")
    confidence: float = Field(..., ge=0.0, le=1.0)
    processing_time_ms: float
    attention_weights: Optional[List[List[float]]] = None
    node_embeddings: Optional[List[List[float]]] = None
    model_info: Dict[str, Any] = Field(default_factory=dict)

# Create FastAPI app
app = FastAPI(
    title="GAT Behavioral Authentication Service",
    description="Graph Attention Network service for continuous behavioral authentication",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "service": "GAT Behavioral Authentication",
        "status": "running",
        "model_loaded": False,
        "mode": "simulation",
        "timestamp": datetime.now().isoformat()
    }

@app.get("/health")
async def health_check():
    """Detailed health check"""
    return {
        "status": "healthy",
        "model_status": "simulation",
        "config": {
            "input_dim": 60,
            "output_dim": 64,
            "num_heads": 8,
            "device": "cpu"
        },
        "timestamp": datetime.now().isoformat()
    }

def extract_behavioral_vector(event: Dict) -> List[float]:
    """Extract 60-dimensional node feature vector from event"""
    vector = []
    
    # Touch dynamics (12 dimensions)
    touch_data = event.get('touch', {})
    vector.extend([
        touch_data.get('pressure', 0.0),
        touch_data.get('size', 0.0),
        touch_data.get('major', 0.0),
        touch_data.get('minor', 0.0),
        touch_data.get('orientation', 0.0),
        touch_data.get('x_position', 0.0),
        touch_data.get('y_position', 0.0),
        touch_data.get('velocity_x', 0.0),
        touch_data.get('velocity_y', 0.0),
        touch_data.get('acceleration_x', 0.0),
        touch_data.get('acceleration_y', 0.0),
        touch_data.get('duration', 0.0)
    ])
    
    # Motion patterns (12 dimensions)
    motion_data = event.get('motion', {})
    vector.extend([
        motion_data.get('acc_x', 0.0),
        motion_data.get('acc_y', 0.0),
        motion_data.get('acc_z', 0.0),
        motion_data.get('gyro_x', 0.0),
        motion_data.get('gyro_y', 0.0),
        motion_data.get('gyro_z', 0.0),
        motion_data.get('mag_x', 0.0),
        motion_data.get('mag_y', 0.0),
        motion_data.get('mag_z', 0.0),
        motion_data.get('linear_acc_x', 0.0),
        motion_data.get('linear_acc_y', 0.0),
        motion_data.get('linear_acc_z', 0.0)
    ])
    
    # Typing patterns (12 dimensions)
    typing_data = event.get('typing', {})
    vector.extend([
        typing_data.get('dwell_time', 0.0),
        typing_data.get('flight_time', 0.0),
        typing_data.get('typing_speed', 0.0),
        typing_data.get('pressure_variance', 0.0),
        typing_data.get('rhythm_score', 0.0),
        typing_data.get('key_hold_variance', 0.0),
        typing_data.get('inter_key_interval', 0.0),
        typing_data.get('typing_cadence', 0.0),
        typing_data.get('error_rate', 0.0),
        typing_data.get('backspace_frequency', 0.0),
        typing_data.get('caps_lock_usage', 0.0),
        typing_data.get('special_char_usage', 0.0)
    ])
    
    # App usage patterns (12 dimensions)
    app_data = event.get('app_usage', {})
    vector.extend([
        app_data.get('transition_time', 0.0),
        app_data.get('interaction_frequency', 0.0),
        app_data.get('session_duration', 0.0),
        app_data.get('touch_frequency', 0.0),
        app_data.get('scroll_speed', 0.0),
        app_data.get('tap_intensity', 0.0),
        app_data.get('multi_touch_usage', 0.0),
        app_data.get('gesture_complexity', 0.0),
        app_data.get('navigation_pattern', 0.0),
        app_data.get('menu_access_frequency', 0.0),
        app_data.get('notification_interaction', 0.0),
        app_data.get('background_app_switches', 0.0)
    ])
    
    # Add event-type embedding (8D)
    vector.extend(event_type_embedding(event.get("type", "unknown")))

    # Add device context (4D)
    vector.extend(device_context_vector(event.get("device", {})))

    # Ensure vector is exactly 60 dimensions
    while len(vector) < 60:
        vector.append(0.0)
    
    return vector[:60]

def event_type_embedding(event_type: str) -> List[float]:
    """Create deterministic 8D embedding for event type"""
    digest = hashlib.sha256(str(event_type).encode("utf-8")).digest()
    return [b / 255.0 for b in digest[:8]]

def device_context_vector(device: Dict) -> List[float]:
    """Extract 4D device context vector"""
    battery = float(device.get('battery', 0.0))
    cpu = float(device.get('cpu', 0.0))
    memory = float(device.get('memory', 0.0))
    network = float(device.get('signal', 0.0))

    return [
        max(min(battery, 1.0), 0.0),
        max(min(cpu, 1.0), 0.0),
        max(min(memory, 1.0), 0.0),
        max(min(network, 1.0), 0.0)
    ]

def process_behavioral_data(raw_data: List[Dict], user_id: str, session_id: str) -> TemporalGraph:
    """Convert raw behavioral data to temporal graph"""
    # Sort events by timestamp
    events = sorted(raw_data, key=lambda x: x.get('timestamp', 0))
    
    # Create nodes
    nodes = []
    for i, event in enumerate(events):
        behavioral_vector = extract_behavioral_vector(event)
        
        node = EventNode(
            node_id=i,
            timestamp=event.get('timestamp', 0),
            event_type=event.get('type', 'unknown'),
            behavioral_vector=behavioral_vector,
            device_features=event.get('device', {}),
            signature=event.get('signature', ''),
            nonce=event.get('nonce', '')
        )
        nodes.append(node)
    
    # Create edges across 4 distinct event types in sequence (keep repeats)
    edges = []
    distinct_target = 4
    for i in range(len(nodes)):
        current_node = nodes[i]
        seen_types = {current_node.event_type}

        for j in range(i + 1, len(nodes)):
            candidate = nodes[j]

            edges.append(
                TemporalEdge(
                    source_node_id=current_node.node_id,
                    target_node_id=candidate.node_id,
                    time_delta=candidate.timestamp - current_node.timestamp,
                    event_transition=f"{current_node.event_type}->{candidate.event_type}"
                )
            )

            if candidate.event_type not in seen_types:
                seen_types.add(candidate.event_type)
                if len(seen_types) >= distinct_target:
                    break
    
    # Calculate metadata
    timestamps = [event.get('timestamp', 0) for event in events]
    event_types = [event.get('type', 'unknown') for event in events]
    
    window_start = min(timestamps) if timestamps else 0
    window_end = max(timestamps) if timestamps else 0
    session_duration = window_end - window_start
    event_diversity = len(set(event_types))
    
    if len(timestamps) > 1:
        time_deltas = [timestamps[i+1] - timestamps[i] for i in range(len(timestamps)-1)]
        avg_time_between_events = sum(time_deltas) / len(time_deltas)
    else:
        avg_time_between_events = 0.0
    
    return TemporalGraph(
        session_id=session_id,
        user_id=user_id,
        nodes=nodes,
        edges=edges,
        window_start=window_start,
        window_end=window_end,
        total_events=len(nodes),
        session_duration=session_duration,
        event_diversity=event_diversity,
        avg_time_between_events=avg_time_between_events
    )

def simulate_gat_processing(graph: TemporalGraph, user_profile: Optional[List[float]] = None) -> Dict[str, Any]:
    """Simulate GAT processing"""
    start_time = time.time()
    
    # Simulate processing delay
    time.sleep(0.05)
    
    # Generate simulated 64-dim embedding
    session_vector = [random.uniform(-1, 1) for _ in range(64)]
    
    # Simulate similarity calculation
    if user_profile:
        similarity = random.uniform(0.7, 0.95)
    else:
        similarity = random.uniform(0.5, 0.8)
    
    # Authentication decision
    threshold = 0.85
    if similarity >= threshold:
        decision = "ALLOW"
        confidence = similarity
    elif similarity < threshold * 0.7:
        decision = "BLOCK"
        confidence = 1.0 - similarity
    else:
        decision = "UNCERTAIN"
        confidence = abs(similarity - threshold) / threshold
    
    processing_time = (time.time() - start_time) * 1000
    
    return {
        "session_vector": session_vector,
        "similarity_score": similarity,
        "auth_decision": decision,
        "confidence": min(confidence, 1.0),
        "processing_time_ms": processing_time,
        "attention_weights": None,
        "node_embeddings": None,
        "model_info": {
            "mode": "simulation",
            "model_version": "simulated_v1.0",
            "num_nodes": len(graph.nodes),
            "num_edges": len(graph.edges)
        }
    }

@app.post("/process", response_model=GATProcessingResponse)
async def process_gat(request: GATProcessingRequest):
    """Process temporal graph with GAT for authentication decision"""
    try:
        logger.info(f"Processing GAT request for session {request.graph.session_id}")
        
        # Simulate GAT processing
        result = simulate_gat_processing(request.graph, request.user_profile_vector)
        
        return GATProcessingResponse(**result)
        
    except Exception as e:
        logger.error(f"Error processing GAT request: {e}")
        raise HTTPException(status_code=500, detail=f"GAT processing failed: {str(e)}")

@app.post("/enroll")
async def enroll_user_profile(request: GATProcessingRequest):
    """Enroll user profile by processing enrollment sessions"""
    try:
        logger.info(f"Enrolling user profile for user {request.graph.user_id}")
        
        if request.processing_mode != "enrollment":
            raise HTTPException(
                status_code=400,
                detail="Processing mode must be 'enrollment' for this endpoint"
            )
        
        # Process enrollment session
        response = await process_gat(request)
        
        return {
            "user_id": request.graph.user_id,
            "profile_vector": response.session_vector,
            "enrollment_confidence": response.confidence,
            "num_events": len(request.graph.nodes),
            "session_duration": request.graph.session_duration,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error enrolling user: {e}")
        raise HTTPException(status_code=500, detail=f"Enrollment failed: {str(e)}")

@app.get("/models/info")
async def get_model_info():
    """Get information about the current model"""
    return {
        "model_loaded": False,
        "model_path": "./models/gat_model.pt",
        "model_exists": False,
        "mode": "simulation",
        "config": {
            "input_dim": 60,
            "hidden_dim": 64,
            "output_dim": 64,
            "num_heads": 8,
            "dropout": 0.1,
            "device": "cpu"
        }
    }

@app.post("/convert/behavioral-data")
async def convert_behavioral_data(raw_data: Dict[str, Any]):
    """Convert raw behavioral data to temporal graph format"""
    try:
        events = raw_data.get('events', [])
        user_id = raw_data.get('user_id', 'unknown')
        session_id = raw_data.get('session_id', 'unknown')
        
        if not events:
            raise HTTPException(status_code=400, detail="No events provided")
        
        # Process data
        temporal_graph = process_behavioral_data(events, user_id, session_id)
        
        return {
            "temporal_graph": temporal_graph.dict(),
            "summary": {
                "num_nodes": len(temporal_graph.nodes),
                "num_edges": len(temporal_graph.edges),
                "session_duration": temporal_graph.session_duration,
                "event_diversity": temporal_graph.event_diversity,
                "avg_time_between_events": temporal_graph.avg_time_between_events
            }
        }
        
    except Exception as e:
        logger.error(f"Error converting behavioral data: {e}")
        raise HTTPException(status_code=500, detail=f"Data conversion failed: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)