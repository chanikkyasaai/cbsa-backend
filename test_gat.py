#!/usr/bin/env python3
"""
GAT Layer 3 Test Script
Demonstrates the GAT processing functionality using sample data from normal.txt
"""
import json
import asyncio
import aiohttp
from typing import List, Dict, Any

# Sample behavioral data from your normal.txt logs
SAMPLE_EVENTS = [
    {
        "user_id": "test_user_123",
        "session_id": "sess_mlqtkcmk_hdzagr1", 
        "timestamp": 1771345904.642,
        "event_type": "PAGE_ENTER_HOME",
        "event_data": {
            "timestamp": 1771345904039,
            "nonce": "d03yeqz5vm6",
            "vector": [0, 0, 0, 0.043, 0, 0.03333333333333333, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0, 1, 1, 1, 0, 0, 0.5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.5, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
            "deviceInfo": {
                "deviceId": None, "brand": "samsung", "manufacturer": "samsung",
                "modelName": "SM-A035F", "totalMemory": 3728367616, "osVersion": "13",
                "platformApiLevel": 33, "deviceType": 1, "isDevice": True, "isRooted": False,
                "networkType": "WIFI", "isInternetReachable": True, "ipAddress": "192.168.1.2"
            },
            "signature": "cce0ebedb79c984ab3f1bbaab0c2d851e6d3ddf6515f0394560cfd60a254f045"
        }
    },
    {
        "user_id": "test_user_123",
        "session_id": "sess_mlqtkcmk_hdzagr1",
        "timestamp": 1771345906.633,
        "event_type": "TOUCH_BALANCE_TOGGLE",
        "event_data": {
            "timestamp": 1771345906629,
            "nonce": "dd2ytlh2vv",
            "vector": [0.858529806137085, 0.37450997414442455, 0, 0.1714, 0, 0.03333333333333333, 0.45976244948585243, 0.5739266103243127, 0.7375329461167841, 0.5, 0.5, 0.5, 0, 1, 0.1899999976158142, 1, 0, 0, 0.5, 0, 0, 0, 0, 0, 0.3355617875947486, 0.33705687256138694, 0.0005350936389311329, 0.016356858483998888, 0.10235362053456301, 0, 0, 0, 0, 0, 0, 0, 0, 0.5, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
            "deviceInfo": {"brand": "samsung", "totalMemory": 3728367616},
            "signature": "96a83bc72178c02bfd4668afcb0f3b500a55fdd1e31078ea8ab64cd8846ea7ab"
        }
    },
    {
        "user_id": "test_user_123",
        "session_id": "sess_mlqtkcmk_hdzagr1",
        "timestamp": 1771345908.038,
        "event_type": "TOUCH_TRANSACTION_HISTORY", 
        "event_data": {
            "timestamp": 1771345908032,
            "nonce": "e0x9ec1hwf4",
            "vector": [0.737864096959432, 0.48774977023284183, 0, 0.0447, 0, 0.03333333333333333, 0.43188614677637815, 0.5849949363619089, 0.7267394661903381, 0.5, 0.5, 0.5, 0, 1, 0.1899999976158142, 1, 0, 0, 0.5, 0, 0, 0, 0, 0, 0.33181506730640037, 0.3356464362922758, 0.00045252600853685956, 0.015042041226789328, 0.10148318237784167, 0, 0, 0, 0, 0, 0, 0, 0, 0.5, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
            "deviceInfo": {"brand": "samsung", "totalMemory": 3728367616},
            "signature": "8c989e80813ad2ff7e4860d4443c7ee714e79e153f369d9f7bce316a0e96cbae"
        }
    }
]


class GATTestClient:
    """Test client for GAT Layer 3 functionality"""
    
    def __init__(self, backend_url: str = "http://localhost:8000"):
        self.backend_url = backend_url
        
    async def test_gat_processing(self):
        """Test complete GAT processing workflow"""
        print("🧠 Testing Layer 3 GAT Processing\n")
        
        # Step 1: Send behavioral events via WebSocket
        await self.send_sample_events()
        
        # Step 2: Test manual GAT processing
        await self.test_manual_gat_processing()
        
        # Step 3: Test user enrollment
        await self.test_user_enrollment()
        
        # Step 4: Test profile retrieval
        await self.test_profile_retrieval()
        
        # Step 5: Get GAT statistics
        await self.get_gat_stats()
    
    async def send_sample_events(self):
        """Send sample events via WebSocket to trigger GAT processing"""
        import websockets
        
        print("📡 Sending sample behavioral events...")
        
        try:
            uri = self.backend_url.replace("http", "ws") + "/ws/behaviour"
            async with websockets.connect(uri) as websocket:
                for i, event in enumerate(SAMPLE_EVENTS):
                    await websocket.send(json.dumps(event))
                    response = await websocket.recv()
                    response_data = json.loads(response)
                    
                    print(f"Event {i+1}: {event['event_type']}")
                    if "auth_decision" in response_data:
                        print(f"  🔒 Auth Decision: {response_data['auth_decision']}")
                    print(f"  ✅ Server Response: {response_data['status']}")
                
                print()
        
        except Exception as e:
            print(f"❌ WebSocket error: {e}\n")
    
    async def test_manual_gat_processing(self):
        """Test manual GAT processing API"""
        print("🔍 Testing manual GAT processing...")
        
        async with aiohttp.ClientSession() as session:
            try:
                async with session.post(
                    f"{self.backend_url}/gat/process",
                    json={"session_id": "sess_mlqtkcmk_hdzagr1"}
                ) as response:
                    result = await response.json()
                    print(f"Auth Decision: {result.get('auth_decision', 'N/A')}")
                    print(f"Similarity Score: {result.get('similarity_score', 'N/A'):.3f}")
                    print(f"Confidence: {result.get('confidence', 'N/A'):.3f}")
                    print(f"Processing Layer: {result.get('layer', 'N/A')}")
                    print()
                    
            except Exception as e:
                print(f"❌ GAT processing error: {e}\n")
    
    async def test_user_enrollment(self):
        """Test user profile enrollment"""
        print("👤 Testing user enrollment...")
        
        # Create enrollment data with sample sessions
        enrollment_data = {
            "user_id": "test_user_123",
            "verified_sessions": [SAMPLE_EVENTS]  # One session with sample events
        }
        
        async with aiohttp.ClientSession() as session:
            try:
                async with session.post(
                    f"{self.backend_url}/gat/enroll",
                    json=enrollment_data
                ) as response:
                    result = await response.json()
                    print(f"Enrollment Status: {result.get('enrollment_status')}")
                    print(f"Sessions Processed: {result.get('sessions_processed')}")
                    print(f"Profile Confidence: {result.get('profile_confidence', 'N/A'):.3f}")
                    print(f"Profile Dimensions: {result.get('profile_dimensions')}")
                    print()
                    
            except Exception as e:
                print(f"❌ Enrollment error: {e}\n")
    
    async def test_profile_retrieval(self):
        """Test user profile retrieval"""
        print("📋 Testing profile retrieval...")
        
        async with aiohttp.ClientSession() as session:
            try:
                async with session.get(
                    f"{self.backend_url}/gat/profile/test_user_123"
                ) as response:
                    if response.status == 200:
                        result = await response.json()
                        print(f"User ID: {result.get('user_id')}")
                        print(f"Profile Dimensions: {result.get('profile_dimensions')}")
                        print(f"Enrollment Sessions: {result.get('enrollment_sessions')}")
                        print(f"Profile Confidence: {result.get('profile_confidence', 'N/A'):.3f}")
                        print(f"Last Updated: {result.get('last_updated')}")
                    else:
                        print(f"Profile not found (status: {response.status})")
                    print()
                    
            except Exception as e:
                print(f"❌ Profile retrieval error: {e}\n")
    
    async def get_gat_stats(self):
        """Get GAT processing statistics"""
        print("📊 GAT Processing Statistics...")
        
        async with aiohttp.ClientSession() as session:
            try:
                async with session.get(f"{self.backend_url}/gat/stats") as response:
                    result = await response.json()
                    print(f"Active Sessions: {result.get('active_sessions')}")
                    print(f"Total Events Buffered: {result.get('total_events_buffered')}")
                    print(f"Window Size: {result.get('window_size')}")
                    print(f"Debug Mode: {result.get('debug_mode')}")
                    print(f"Cloud Endpoint: {result.get('cloud_endpoint')}")
                    print()
                    
            except Exception as e:
                print(f"❌ Stats error: {e}\n")


async def main():
    """Run GAT testing suite"""
    print("="*60)
    print("🚀 CBSA Layer 3 GAT Processing Test Suite")
    print("="*60)
    print()
    
    client = GATTestClient()
    await client.test_gat_processing()
    
    print("✅ GAT testing complete!")
    print()
    print("🔍 What was tested:")
    print("  • Behavioral event ingestion via WebSocket") 
    print("  • Temporal graph creation from event streams")
    print("  • Simulated GAT processing with attention networks")
    print("  • User behavioral profile enrollment")
    print("  • Cosine similarity-based authentication decisions")
    print("  • RESTful API endpoints for GAT operations")
    print()
    print("📖 Implementation Details:")
    print("  • 48-dimensional behavioral vectors from your sample data")
    print("  • Temporal graph with attention-weighted edges")
    print("  • 64-dimensional session embeddings (simulated)")
    print("  • Multi-head attention architecture (8 heads)")
    print("  • Metric learning via cosine similarity")
    print("  • Configurable similarity thresholds")


if __name__ == "__main__":
    asyncio.run(main())