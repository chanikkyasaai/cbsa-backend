"""
Layer 3 GAT Integration Manager
Integrates GAT processing into the main WebSocket handler
"""
import asyncio
from typing import List, Dict, Optional, Any
from app.models import BehaviourMessage
from app.layer3_processor import GATDataProcessor, GATResultProcessor
from app.layer3_cloud import GATCloudInterface, UserProfileManager
from app.config import settings
import logging

logger = logging.getLogger(__name__)


class Layer3GATManager:
    """Manages Layer 3 GAT processing integration"""
    
    def __init__(self):
        self.data_processor = GATDataProcessor(
            window_seconds=settings.GAT_WINDOW_SECONDS,
            distinct_target=settings.GAT_EDGE_DISTINCT_TARGET
        )
        self.result_processor = GATResultProcessor()
        self.cloud_interface = GATCloudInterface(settings.GAT_CLOUD_ENDPOINT)
        self.profile_manager = UserProfileManager()
        
        # Session management
        self.session_windows: Dict[str, List[BehaviourMessage]] = {}
        
    async def process_escalated_session(
        self, 
        session_id: str, 
        event_window: List[BehaviourMessage]
    ) -> Dict[str, Any]:
        """
        Process escalated session through Layer 3 GAT
        
        Args:
            session_id: Session identifier
            event_window: Window of behavioral events from Layer 2 escalation
            
        Returns:
            Authentication decision and analysis
        """
        try:
            logger.info(f"Layer 3 processing session {session_id} with {len(event_window)} events")
            
            # Step 1: Convert to temporal graph
            temporal_graph = self.data_processor.create_temporal_graph(event_window)
            
            # Step 2: Get user profile if available
            user_profile = None
            user_profile_vector = None
            if temporal_graph.user_id:
                user_profile = await self.profile_manager.get_user_profile(temporal_graph.user_id)
                if user_profile:
                    user_profile_vector = user_profile.profile_vector
            
            # Step 3: Prepare GAT request
            gat_request = self.data_processor.prepare_gat_request(
                graph=temporal_graph,
                user_profile_vector=user_profile_vector,
                processing_mode="inference"
            )
            
            # Step 4: Send to cloud GAT service
            gat_response = await self.cloud_interface.process_temporal_graph(gat_request)
            
            # Step 5: Process results and make decision
            auth_result = self.result_processor.process_gat_response(gat_response)
            
            # Add metadata
            auth_result.update({
                "session_id": session_id,
                "user_id": temporal_graph.user_id,
                "layer": 3,
                "graph_events": len(temporal_graph.nodes),
                "graph_duration": temporal_graph.session_duration,
                "has_user_profile": user_profile is not None
            })
            
            logger.info(
                f"Layer 3 decision for {session_id}: {auth_result['auth_decision']} "
                f"(similarity: {auth_result.get('similarity_score', 'N/A'):.3f})"
            )
            
            return auth_result
            
        except Exception as e:
            logger.error(f"Layer 3 processing failed for session {session_id}: {e}")
            # Conservative fallback
            return {
                "session_id": session_id,
                "auth_decision": "BLOCK",
                "confidence": 1.0,
                "error": str(e),
                "layer": 3
            }
    
    async def enroll_user_session(
        self, 
        user_id: str, 
        verified_sessions: List[List[BehaviourMessage]]
    ) -> Dict[str, Any]:
        """
        Enroll user profile from verified behavioral sessions
        
        Args:
            user_id: User identifier  
            verified_sessions: List of verified behavioral event sessions
            
        Returns:
            Enrollment result
        """
        try:
            logger.info(f"Enrolling user {user_id} with {len(verified_sessions)} verified sessions")
            
            session_vectors = []
            
            # Process each verified session through GAT
            for i, session_events in enumerate(verified_sessions):
                # Convert to temporal graph
                temporal_graph = self.data_processor.create_temporal_graph(session_events)
                
                # Process through GAT in enrollment mode
                gat_request = self.data_processor.prepare_gat_request(
                    graph=temporal_graph,
                    user_profile_vector=None,  # No existing profile during enrollment
                    processing_mode="enrollment"
                )
                
                gat_response = await self.cloud_interface.process_temporal_graph(gat_request)
                session_vectors.append(gat_response.session_vector)
                
                logger.debug(f"Processed enrollment session {i+1}/{len(verified_sessions)} for user {user_id}")
            
            # Create user profile
            user_profile = await self.profile_manager.create_or_update_profile(
                user_id=user_id,
                session_vectors=session_vectors
            )
            
            return {
                "user_id": user_id,
                "enrollment_status": "success",
                "sessions_processed": len(verified_sessions),
                "profile_confidence": user_profile.profile_confidence,
                "profile_dimensions": len(user_profile.profile_vector)
            }
            
        except Exception as e:
            logger.error(f"User enrollment failed for {user_id}: {e}")
            return {
                "user_id": user_id,
                "enrollment_status": "failed",
                "error": str(e)
            }
    
    def add_event_to_session(self, session_id: str, event: BehaviourMessage):
        """Add event to session window for potential escalation"""
        if session_id not in self.session_windows:
            self.session_windows[session_id] = []
        
        self.session_windows[session_id].append(event)

        # Maintain time-based sliding window (20 seconds)
        self._prune_session_window(session_id)
    
    def get_session_window(self, session_id: str) -> List[BehaviourMessage]:
        """Get current session window for escalation"""
        return self.session_windows.get(session_id, [])
    
    def clear_session_window(self, session_id: str):
        """Clear session window (e.g., after successful auth or session end)"""
        if session_id in self.session_windows:
            del self.session_windows[session_id]

    def _prune_session_window(self, session_id: str):
        """Prune events outside the configured time window"""
        window_seconds = settings.GAT_WINDOW_SECONDS
        events = self.session_windows.get(session_id, [])
        if not events:
            return

        latest_ts = max((event.timestamp or 0.0) for event in events)
        window_start = latest_ts - window_seconds
        self.session_windows[session_id] = [
            event for event in events if (event.timestamp or 0.0) >= window_start
        ]