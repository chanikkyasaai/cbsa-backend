"""
Data Processing Utilities for GAT Service
Converts behavioral data into graph structures
"""

import json
import numpy as np
import hashlib
from typing import List, Dict, Any, Tuple, Optional
from datetime import datetime, timezone
import logging
from models import EventNode, TemporalEdge, TemporalGraph

logger = logging.getLogger(__name__)


class BehavioralDataProcessor:
    """
    Processes raw behavioral data into temporal graphs
    Handles mobile sensor data conversion to graph format
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.time_window = config.get('time_window_seconds', 20)
        self.min_events = config.get('min_events_per_window', 5)
        self.max_events = config.get('max_events_per_window', 100)
        self.distinct_event_connections = config.get('distinct_event_connections', 8)
        
    def process_behavioral_data(
        self,
        raw_data: List[Dict[str, Any]],
        user_id: str,
        session_id: str
    ) -> TemporalGraph:
        """
        Convert raw behavioral data to temporal graph
        
        Args:
            raw_data: List of behavioral events
            user_id: User identifier
            session_id: Session identifier
            
        Returns:
            TemporalGraph ready for GAT processing
        """
        try:
            # Sort events by timestamp
            events = sorted(raw_data, key=lambda x: x.get('timestamp', 0))
            
            # Filter events within time window
            events = self._filter_time_window(events)
            
            # Create event nodes
            nodes = self._create_event_nodes(events)
            
            # Create temporal edges
            edges = self._create_temporal_edges(nodes, events)
            
            # Calculate metadata
            metadata = self._calculate_metadata(events, nodes)
            
            return TemporalGraph(
                session_id=session_id,
                user_id=user_id,
                nodes=nodes,
                edges=edges,
                window_start=metadata['window_start'],
                window_end=metadata['window_end'],
                total_events=len(nodes),
                session_duration=metadata['session_duration'],
                event_diversity=int(metadata['event_diversity']),
                avg_time_between_events=metadata['avg_time_between_events']
            )
            
        except Exception as e:
            logger.error(f"Error processing behavioral data: {e}")
            raise
    
    def _filter_time_window(self, events: List[Dict]) -> List[Dict]:
        """Filter events to current time window"""
        if not events:
            return []
        
        latest_time = max(event.get('timestamp', 0) for event in events)
        window_start = latest_time - self.time_window
        
        filtered = [
            event for event in events 
            if event.get('timestamp', 0) >= window_start
        ]
        
        # Ensure we have minimum events
        if len(filtered) < self.min_events and len(events) >= self.min_events:
            filtered = events[-self.min_events:]
        
        # Limit to maximum events
        if len(filtered) > self.max_events:
            filtered = filtered[-self.max_events:]
        
        return filtered
    
    def _create_event_nodes(self, events: List[Dict]) -> List[EventNode]:
        """
        Create event nodes from raw data
        
        Expected event structure:
        {
            'timestamp': float,
            'event_type': str,
            'event_data': {
                'vector': List[float],  # 48-D behavioral vector
                'nonce': str,
                'signature': str,
                'deviceInfo': {...}
            },
            ...
        }
        """
        nodes = []
        
        for i, event in enumerate(events):
            # Extract 56-dimensional behavioral vector (48 behavioral + 8 event-type; device info removed)
            behavioral_vector = self._extract_behavioral_vector(event)
            
            # Pull metadata from event_data
            event_data = event.get('event_data', {})
            
            node = EventNode(
                node_id=i,
                timestamp=event.get('timestamp', 0),
                event_type=event.get('event_type', 'unknown'),
                behavioral_vector=behavioral_vector,
                signature=event_data.get('signature', ''),
                nonce=event_data.get('nonce', '')
            )
            
            nodes.append(node)
        
        return nodes
    
    def _extract_behavioral_vector(self, event: Dict) -> List[float]:
        """
        Extract 56-dimensional node feature vector from event.

        Output vector composition (56-D):
        - Behavioral features (48-D): from event.event_data.vector
        - Event-type embedding (8-D): SHA256-based deterministic hash of event_type
        Device context has been removed.
        """
        # Extract pre-computed 48-D behavioral vector from event_data
        event_data = event.get('event_data', {})
        base_vector = event_data.get('vector', [])
        
        # Pad or truncate to exactly 48 dimensions
        if len(base_vector) < 48:
            base_vector = list(base_vector) + [0.0] * (48 - len(base_vector))
        else:
            base_vector = list(base_vector[:48])
        
        # Ensure all elements are floats
        base_vector = [float(v) if v is not None else 0.0 for v in base_vector]
        
        # Get event type (fallback to 'unknown' if missing)
        event_type = event.get('event_type', 'unknown')
        
        # Add event-type embedding (8D)
        type_embedding = self._event_type_embedding(event_type)
        
        # Concatenate: 48 + 8 = 56
        vector = base_vector + type_embedding
        
        return vector[:56]

    def _event_type_embedding(self, event_type: str) -> List[float]:
        """Create deterministic 8D embedding for event type"""
        digest = hashlib.sha256(str(event_type).encode("utf-8")).digest()
        return [b / 255.0 for b in digest[:8]]
    
    def _create_temporal_edges(
        self, 
        nodes: List[EventNode], 
        events: List[Dict]
    ) -> List[TemporalEdge]:
        """Create temporal edges that keep repeats and count distinct types"""
        edges = []
        distinct_target = max(self.distinct_event_connections, 1)

        for i in range(len(nodes)):
            current_node = nodes[i]
            seen_types = {current_node.event_type}

            for j in range(i + 1, len(nodes)):
                candidate = nodes[j]
                time_delta = candidate.timestamp - current_node.timestamp
                transition = f"{current_node.event_type}->{candidate.event_type}"

                edges.append(
                    TemporalEdge(
                        source_node_id=current_node.node_id,
                        target_node_id=candidate.node_id,
                        time_delta=time_delta,
                        event_transition=transition
                    )
                )

                if candidate.event_type not in seen_types:
                    seen_types.add(candidate.event_type)
                    if len(seen_types) >= distinct_target:
                        break
        
        return edges
    
    def _calculate_behavioral_similarity(
        self, 
        vector1: List[float], 
        vector2: List[float]
    ) -> float:
        """Calculate cosine similarity between behavioral vectors"""
        v1 = np.array(vector1)
        v2 = np.array(vector2)
        
        norm1 = np.linalg.norm(v1)
        norm2 = np.linalg.norm(v2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return np.dot(v1, v2) / (norm1 * norm2)
    
    def _calculate_metadata(
        self, 
        events: List[Dict], 
        nodes: List[EventNode]
    ) -> Dict[str, float]:
        """Calculate graph metadata"""
        if not events:
            return {
                'window_start': 0.0,
                'window_end': 0.0,
                'session_duration': 0.0,
                'event_diversity': 0,
                'avg_time_between_events': 0.0
            }
        
        timestamps = [event.get('timestamp', 0) for event in events]
        event_types = [event.get('event_type', 'unknown') for event in events]
        
        window_start = min(timestamps)
        window_end = max(timestamps)
        session_duration = window_end - window_start
        event_diversity = len(set(event_types))
        
        if len(timestamps) > 1:
            time_deltas = [timestamps[i+1] - timestamps[i] for i in range(len(timestamps)-1)]
            avg_time_between_events = np.mean(time_deltas)
        else:
            avg_time_between_events = 0.0
        
        return {
            'window_start': float(window_start),
            'window_end': float(window_end),
            'session_duration': float(session_duration),
            'event_diversity': float(event_diversity),
            'avg_time_between_events': float(avg_time_between_events)
        }


class PyTorchDataConverter:
    """
    Converts TemporalGraph to PyTorch Geometric format
    """
    
    def __init__(self):
        pass
    
    def convert_to_pytorch(self, temporal_graph: TemporalGraph) -> Dict[str, Any]:
        """
        Convert TemporalGraph to PyTorch Geometric Data format
        
        Returns:
            Dictionary with torch tensors for GAT processing
        """
        try:
            # Import torch here to avoid import errors if PyTorch not available
            import torch  # type: ignore[reportMissingImports]
            import numpy as np  # type: ignore[reportMissingImports]
            
            nodes = temporal_graph.nodes
            edges = temporal_graph.edges
            
            # Node features matrix [num_nodes, 60]
            node_features = []
            temporal_features = []
            
            for node in nodes:
                node_features.append(node.behavioral_vector)
                # Normalize timestamp to relative time from start
                rel_time = node.timestamp - temporal_graph.window_start
                temporal_features.append([rel_time])
            
            x = torch.FloatTensor(node_features)
            temporal_feats = torch.FloatTensor(temporal_features)
            
            # Edge indices [2, num_edges]
            if edges:
                edge_sources = [edge.source_node_id for edge in edges]
                edge_targets = [edge.target_node_id for edge in edges]
                edge_index = torch.LongTensor([edge_sources, edge_targets])
            else:
                # Create self-loops if no edges
                num_nodes = len(nodes)
                edge_index = torch.LongTensor([[i for i in range(num_nodes)], 
                                              [i for i in range(num_nodes)]])
            
            return {
                'x': x,
                'edge_index': edge_index,
                'temporal_features': temporal_feats,
                'num_nodes': len(nodes),
                'num_edges': len(edges),
                'batch': None  # Single graph
            }
            
        except ImportError:
            # Fallback to numpy if torch not available
            logger.warning("PyTorch not available, returning numpy arrays")
            
            nodes = temporal_graph.nodes
            edges = temporal_graph.edges
            
            node_features = np.array([node.behavioral_vector for node in nodes])
            temporal_features = np.array([[node.timestamp - temporal_graph.window_start] 
                                        for node in nodes])
            
            if edges:
                edge_sources = [edge.source_node_id for edge in edges]
                edge_targets = [edge.target_node_id for edge in edges]
                edge_index = np.array([edge_sources, edge_targets])
            else:
                num_nodes = len(nodes)
                edge_index = np.array([[i for i in range(num_nodes)], 
                                      [i for i in range(num_nodes)]])
            
            return {
                'x': node_features,
                'edge_index': edge_index,
                'temporal_features': temporal_features,
                'num_nodes': len(nodes),
                'num_edges': len(edges),
                'batch': None
            }