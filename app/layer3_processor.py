"""
Layer 3 GAT Data Processor
Converts incoming behavioral event streams into temporal graph format for GAT processing
"""
from typing import List, Dict, Optional, Tuple, Any
import hashlib
import numpy as np
from app.models import BehaviourMessage
from app.layer3_models import (
    GATEventNode, GATTemporalEdge, GATGraph, 
    GATProcessingRequest, GATProcessingResponse
)


class GATDataProcessor:
    """Processes behavioral event streams into GAT-compatible temporal graphs"""
    
    def __init__(self, window_seconds: int = 20, distinct_target: int = 4):
        """
        Initialize GAT data processor
        
        Args:
            window_seconds: Sliding time window (seconds) for temporal graph
            distinct_target: Number of distinct event types to reach per node
        """
        self.window_seconds = window_seconds
        self.distinct_target = max(distinct_target, 1)
        self.feature_normalizer = self._create_normalizer()
    
    def create_temporal_graph(self, event_window: List[BehaviourMessage]) -> GATGraph:
        """
        Convert sliding window of events into temporal graph
        
        Args:
            event_window: List of behavioral events (assumed escalated from Layer 2)
            
        Returns:
            GATGraph: Temporal graph ready for GAT processing
        """
        if not event_window:
            raise ValueError("Event window cannot be empty")

        # Sort events by timestamp
        sorted_events = sorted(event_window, key=lambda x: x.timestamp or 0.0)

        # Apply time-based window (keep repeats)
        latest_ts = sorted_events[-1].timestamp or 0.0
        window_start = latest_ts - self.window_seconds
        sorted_events = [event for event in sorted_events if (event.timestamp or 0.0) >= window_start]

        # Create nodes
        nodes = []
        for i, event in enumerate(sorted_events):
            node = self._create_event_node(i, event)
            nodes.append(node)

        # Create temporal edges
        edges = self._create_temporal_edges(sorted_events, nodes)

        # Calculate graph-level features
        session_duration = (sorted_events[-1].timestamp or 0.0) - (sorted_events[0].timestamp or 0.0)
        unique_events = len(set(self._get_event_type(event) for event in sorted_events))
        avg_time_delta = session_duration / (len(sorted_events) - 1) if len(sorted_events) > 1 else 0

        return GATGraph(
            session_id=event_window[0].session_id or "unknown",
            user_id=event_window[0].user_id,
            nodes=nodes,
            edges=edges,
            window_start=sorted_events[0].timestamp or 0.0,
            window_end=sorted_events[-1].timestamp or 0.0,
            total_events=len(sorted_events),
            session_duration=session_duration,
            event_diversity=unique_events,
            avg_time_between_events=avg_time_delta
        )
    
    def _create_event_node(self, node_id: int, event: BehaviourMessage) -> GATEventNode:
        """Create a single event node from behavioral message"""

        # Extract 48-dimensional vector from event_data
        event_data = self._get_event_data(event)
        behavioral_vector = event_data.get("vector", [0.0] * 48)
        if len(behavioral_vector) != 48:
            # Pad or truncate to 48 dimensions
            behavioral_vector = (behavioral_vector + [0.0] * 48)[:48]

        # Add 8-dimensional event-type embedding
        event_type_embedding = self._event_type_embedding(self._get_event_type(event))

        # 48 + 8 = 56 total (device info removed)
        node_vector = behavioral_vector + event_type_embedding

        return GATEventNode(
            node_id=node_id,
            timestamp=self._get_event_timestamp(event),
            event_type=self._get_event_type(event),
            behavioral_vector=node_vector,
            signature=event_data.get("signature", ""),
            nonce=event_data.get("nonce", "")
        )

    def _event_type_embedding(self, event_type: str) -> List[float]:
        """Create deterministic 8D embedding for event type"""
        digest = hashlib.sha256(event_type.encode("utf-8")).digest()
        return [b / 255.0 for b in digest[:8]]
    
    def _create_temporal_edges(self, events: List[BehaviourMessage], nodes: List[GATEventNode]) -> List[GATTemporalEdge]:
        """Create temporal edges that keep repeats and count distinct types"""
        edges = []
        
        for i in range(len(events)):
            current_event = events[i]
            current_type = self._get_event_type(current_event)
            seen_types = {current_type}
            
            for j in range(i + 1, len(events)):
                next_event = events[j]
                next_type = self._get_event_type(next_event)
                time_delta = self._get_event_timestamp(next_event) - self._get_event_timestamp(current_event)
                event_transition = f"{current_type}->{next_type}"
                
                edges.append(
                    GATTemporalEdge(
                        source_node_id=i,
                        target_node_id=j,
                        time_delta=time_delta,
                        event_transition=event_transition
                    )
                )

                if next_type not in seen_types:
                    seen_types.add(next_type)
                    if len(seen_types) >= self.distinct_target:
                        break
        
        return edges
    
    def _create_normalizer(self):
        """Create feature normalizer for behavioral vectors"""
        # This would typically be trained on historical data
        # For now, we assume vectors are already normalized to [0,1] range
        return lambda x: np.clip(x, 0.0, 1.0)

    def _get_event_type(self, event: BehaviourMessage) -> str:
        return getattr(event, "event_type", "unknown") or "unknown"

    def _get_event_timestamp(self, event: BehaviourMessage) -> float:
        return getattr(event, "timestamp", 0.0) or 0.0

    def _get_event_data(self, event: BehaviourMessage) -> Dict:
        return getattr(event, "event_data", {}) or {}
    
    def prepare_gat_request(
        self, 
        graph: GATGraph, 
        user_profile_vector: Optional[List[float]] = None,
        processing_mode: str = "inference"
    ) -> GATProcessingRequest:
        """Prepare request for cloud GAT service"""
        
        return GATProcessingRequest(
            graph=graph,
            user_profile_vector=user_profile_vector,
            processing_mode=processing_mode,
            attention_heads=8,  # Multi-head attention from spec
            embedding_dim=64,   # Final 64-dim embedding from spec
            similarity_threshold=0.85
        )


class GATResultProcessor:
    """Processes GAT results — passes through raw similarity scores"""
    
    def process_gat_response(self, response: GATProcessingResponse) -> Dict[str, Any]:
        """Return raw GAT scores without making auth decisions"""
        return {
            "similarity_score": response.similarity_score,
            "session_vector": response.session_vector,
            "processing_time_ms": response.processing_time_ms
        }