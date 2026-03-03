"""
Layer 3 GAT Cloud Interface
Handles communication with the GAT processing service (gat-service)
"""
import asyncio
import aiohttp
from typing import Dict, List, Optional
from app.layer3_models import GATProcessingRequest, GATProcessingResponse, UserProfile
import logging

logger = logging.getLogger(__name__)


class GATCloudInterface:
    """Interface to cloud GAT processing service"""
    
    def __init__(self, cloud_endpoint: str = "http://localhost:8001"):  # Mock endpoint for now
        self.cloud_endpoint = cloud_endpoint
        self.session_timeout = 30.0  # 30 second timeout
        self.retry_attempts = 3
        
    async def process_temporal_graph(self, request: GATProcessingRequest) -> GATProcessingResponse:
        """
        Send temporal graph to GAT service for processing.

        The gat-service (localhost:8001 in dev, cloud URL in prod) exposes
        POST /process which runs the real SiameseGATNetwork / GATInferenceEngine.
        """
        try:
            return await self._call_gat_service(request)
        except Exception as e:
            logger.error(f"GAT processing failed: {e}")
            # Fallback — return zero similarity on failure
            return GATProcessingResponse(
                session_vector=[0.0] * 64,
                similarity_score=0.0,
                processing_time_ms=0.0
            )

    async def _call_gat_service(self, request: GATProcessingRequest) -> GATProcessingResponse:
        """
        HTTP call to the real GAT service endpoint (POST /process).
        Retries with exponential back-off on timeout.
        """
        async with aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=self.session_timeout)
        ) as session:
            payload = request.dict()

            for attempt in range(self.retry_attempts):
                try:
                    async with session.post(
                        f"{self.cloud_endpoint}/process",
                        json=payload,
                        headers={"Content-Type": "application/json"},
                    ) as response:
                        if response.status == 200:
                            result_data = await response.json()
                            return GATProcessingResponse(**result_data)
                        else:
                            error_text = await response.text()
                            logger.error(
                                f"GAT service error {response.status}: {error_text}"
                            )
                except asyncio.TimeoutError:
                    logger.warning(
                        f"GAT service timeout, attempt {attempt + 1}/{self.retry_attempts}"
                    )
                    if attempt == self.retry_attempts - 1:
                        raise
                    await asyncio.sleep(1.0 * (attempt + 1))

            raise Exception("GAT service unavailable after retries")
    
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