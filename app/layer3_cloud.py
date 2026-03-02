"""
Layer 3 GAT Cloud Interface
Handles communication with cloud-based GAT processing service
"""
import json
import asyncio
import aiohttp
from typing import Dict, List, Optional
from app.layer3_models import GATProcessingRequest, GATProcessingResponse, UserProfile
from app.config import settings
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
        Send temporal graph to cloud GAT service for processing
        
        Args:
            request: GAT processing request with temporal graph
            
        Returns:
            GATProcessingResponse: 64-dim embedding and similarity score
        """
        try:
            # For development/testing, simulate GAT processing locally
            if self.cloud_endpoint.startswith("http://localhost") or settings.DEBUG_MODE:
                return await self._simulate_gat_processing(request)
            
            # Real cloud GAT processing
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=self.session_timeout)) as session:
                payload = request.dict()
                
                for attempt in range(self.retry_attempts):
                    try:
                        async with session.post(
                            f"{self.cloud_endpoint}/gat/process",
                            json=payload,
                            headers={"Content-Type": "application/json"}
                        ) as response:
                            
                            if response.status == 200:
                                result_data = await response.json()
                                return GATProcessingResponse(**result_data)
                            else:
                                error_text = await response.text()
                                logger.error(f"GAT service error {response.status}: {error_text}")
                                
                    except asyncio.TimeoutError:
                        logger.warning(f"GAT service timeout, attempt {attempt + 1}/{self.retry_attempts}")
                        if attempt == self.retry_attempts - 1:
                            raise
                        await asyncio.sleep(1.0 * (attempt + 1))  # Exponential backoff
                
                # If all retries failed
                raise Exception("GAT service unavailable after retries")
                
        except Exception as e:
            logger.error(f"GAT processing failed: {e}")
            # Fallback to conservative decision
            return GATProcessingResponse(
                session_vector=[0.0] * 64,  # Zero vector
                similarity_score=0.0,  # Force block decision
                auth_decision="BLOCK",
                confidence=1.0,
                processing_time_ms=0.0
            )
    
    async def _simulate_gat_processing(self, request: GATProcessingRequest) -> GATProcessingResponse:
        """
        Simulate GAT processing for development/testing
        Uses simplified heuristics instead of actual neural network
        """
        import time
        import random
        import numpy as np
        
        start_time = time.time()
        
        # Extract features from graph for heuristic analysis
        graph = request.graph
        nodes = graph.nodes
        
        if not nodes:
            return GATProcessingResponse(
                session_vector=[0.0] * 64,
                similarity_score=0.0,
                auth_decision="BLOCK",
                confidence=1.0,
                processing_time_ms=(time.time() - start_time) * 1000
            )
        
        # Simulate attention-based processing
        await asyncio.sleep(0.1)  # Simulate processing delay
        
        # Create pseudo-session vector based on behavioral patterns
        session_features = self._extract_session_features(graph)
        session_vector = self._create_session_embedding(session_features)
        
        # Calculate similarity score if user profile is available
        similarity_score = None
        if request.user_profile_vector:
            similarity_score = self._calculate_cosine_similarity(
                session_vector, 
                request.user_profile_vector
            )
        else:
            # For enrollment mode, assume high similarity
            similarity_score = 0.95 if request.processing_mode == "enrollment" else 0.5
        
        # Make authentication decision
        if similarity_score >= request.similarity_threshold:
            auth_decision = "ALLOW"
            confidence = similarity_score
        elif similarity_score <= 0.3:
            auth_decision = "BLOCK" 
            confidence = 1.0 - similarity_score
        else:
            auth_decision = "UNCERTAIN"
            confidence = 0.5
        
        processing_time = (time.time() - start_time) * 1000
        
        logger.info(f"Simulated GAT processing: {auth_decision} (similarity: {similarity_score:.3f}, time: {processing_time:.1f}ms)")
        
        return GATProcessingResponse(
            session_vector=session_vector,
            similarity_score=similarity_score,
            auth_decision=auth_decision,
            confidence=confidence,
            processing_time_ms=processing_time
        )
    
    def _extract_session_features(self, graph) -> Dict[str, float]:
        """Extract high-level session features for heuristic analysis"""
        nodes = graph.nodes
        edges = graph.edges
        
        if not nodes:
            return {}
        
        # Temporal features
        session_duration = graph.session_duration
        avg_time_between_events = graph.avg_time_between_events
        event_rate = len(nodes) / (session_duration + 1e-6)
        
        # Behavioral consistency
        behavioral_vectors = [node.behavioral_vector for node in nodes]
        behavioral_variance = np.var(behavioral_vectors, axis=0).mean()
        
        # Event diversity
        unique_events = graph.event_diversity
        event_diversity_ratio = unique_events / len(nodes)
        
        # Touch/interaction patterns (based on event types)
        touch_events = sum(1 for node in nodes if "TOUCH" in node.event_type)
        scroll_events = sum(1 for node in nodes if "SCROLL" in node.event_type)
        page_events = sum(1 for node in nodes if "PAGE" in node.event_type)
        
        touch_ratio = touch_events / len(nodes)
        scroll_ratio = scroll_events / len(nodes)
        page_ratio = page_events / len(nodes)
        
        return {
            "session_duration": session_duration,
            "event_rate": event_rate,
            "avg_time_between_events": avg_time_between_events,
            "behavioral_variance": float(behavioral_variance),
            "event_diversity_ratio": event_diversity_ratio,
            "touch_ratio": touch_ratio,
            "scroll_ratio": scroll_ratio,
            "page_ratio": page_ratio,
            "total_events": len(nodes)
        }
    
    def _create_session_embedding(self, features: Dict[str, float]) -> List[float]:
        """Create 64-dimensional session embedding from features"""
        import numpy as np
        
        # Create a deterministic but complex embedding
        base_seed = hash(str(sorted(features.items()))) % (2**31)
        np.random.seed(base_seed)
        
        # Start with feature vector
        feature_values = list(features.values())
        
        # Pad or truncate to base size
        base_dim = 16
        if len(feature_values) > base_dim:
            feature_values = feature_values[:base_dim]
        else:
            feature_values.extend([0.0] * (base_dim - len(feature_values)))
        
        # Generate additional dimensions using transformations
        embedding = []
        for i in range(64):
            if i < len(feature_values):
                # Direct feature
                embedding.append(float(feature_values[i]))
            else:
                # Generated feature using combinations
                idx1 = i % len(feature_values)
                idx2 = (i + 1) % len(feature_values)
                
                # Various transformations to create 64-dim space
                if i % 4 == 0:
                    val = np.tanh(feature_values[idx1] + feature_values[idx2])
                elif i % 4 == 1:
                    val = np.sin(feature_values[idx1] * 3.14159)
                elif i % 4 == 2:
                    val = feature_values[idx1] * feature_values[idx2]
                else:
                    val = np.sqrt(abs(feature_values[idx1]))
                
                embedding.append(float(val))
        
        # Normalize to unit vector (as per GAT output)
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = [x / norm for x in embedding]
        
        return embedding
    
    def _calculate_cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """Calculate cosine similarity between two vectors"""
        import numpy as np
        
        v1 = np.array(vec1)
        v2 = np.array(vec2)
        
        norm1 = np.linalg.norm(v1)
        norm2 = np.linalg.norm(v2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        similarity = np.dot(v1, v2) / (norm1 * norm2)
        return float(similarity)


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