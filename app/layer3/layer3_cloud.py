"""
Layer 3 GAT Interface
Previously made HTTP calls to the gat-service microservice; now calls the GAT
engine in-process via app.gat.engine.InternalGATEngine.
"""
import asyncio
from typing import Dict, List, Optional
from app.layer3.layer3_models import GATProcessingRequest, GATProcessingResponse, UserProfile
from app.gat.engine import get_internal_engine
import logging

logger = logging.getLogger(__name__)


class GATCloudInterface:
    """
    In-process interface to the GAT engine.

    The public API is unchanged so all callers (Layer3GATManager etc.) work
    without modification.  Internally, requests are now dispatched directly to
    InternalGATEngine instead of an HTTP endpoint.
    """

    def __init__(self, cloud_endpoint: str = ""):
        # cloud_endpoint kept for backwards-compat but is no longer used
        self._engine = get_internal_engine()

    async def process_temporal_graph(self, request: GATProcessingRequest) -> GATProcessingResponse:
        """
        Process a temporal graph through the in-process GAT engine.

        Runs the (potentially blocking) inference in a thread-pool executor so
        that the async event loop is never blocked.
        """
        try:
            return await asyncio.get_event_loop().run_in_executor(
                None, self._engine.process_request, request
            )
        except Exception as exc:
            logger.error("GAT processing failed: %s", exc)
            return GATProcessingResponse(
                session_vector=[0.0] * 64,
                similarity_score=0.0,
                processing_time_ms=0.0,
            )

class UserProfileManager:
    """Manages user behavioral profiles"""

    def __init__(self):
        # In production, this would be a proper database
        self.profiles: Dict[str, UserProfile] = {}

    async def get_user_profile(self, user_id: str) -> Optional[UserProfile]:
        """Retrieve user profile by ID"""
        return self.profiles.get(user_id)

    async def create_or_update_profile(
        self,
        user_id: str,
        session_vectors: List[List[float]]
    ) -> UserProfile:
        """Create or update user profile from enrollment sessions"""
        from datetime import datetime
        import numpy as np

        # Calculate master profile vector (average of session vectors)
        if not session_vectors:
            raise ValueError("No session vectors provided for profile creation")

        profile_vector = np.mean(session_vectors, axis=0).tolist()

        # Calculate profile confidence based on consistency
        consistency = self._calculate_profile_consistency(session_vectors)

        profile = UserProfile(
            user_id=user_id,
            profile_vector=profile_vector,
            enrollment_sessions=len(session_vectors),
            last_updated=datetime.now(),
            profile_confidence=consistency
        )

        self.profiles[user_id] = profile
        logger.info(f"Updated profile for user {user_id}: {len(session_vectors)} sessions, confidence: {consistency:.3f}")

        return profile

    def _calculate_profile_consistency(self, session_vectors: List[List[float]]) -> float:
        """Calculate consistency score for profile confidence"""
        if len(session_vectors) < 2:
            return 1.0

        import numpy as np

        # Calculate pairwise similarities
        similarities = []
        for i in range(len(session_vectors)):
            for j in range(i + 1, len(session_vectors)):
                sim = np.dot(session_vectors[i], session_vectors[j]) / (
                    np.linalg.norm(session_vectors[i]) * np.linalg.norm(session_vectors[j])
                )
                similarities.append(sim)

        # Return average similarity as consistency measure
        return float(np.mean(similarities))
