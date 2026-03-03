import requests
import json
import time

def test_gat_service():
    """Test the GAT service"""
    print("=" * 60)
    print("Testing GAT Behavioral Authentication Service")
    print("=" * 60)
    
    base_url = "http://localhost:8001"
    
    # Test 1: Health check
    print("\n1. Testing Health Check...")
    try:
        response = requests.get(f"{base_url}/health", timeout=5)
        print(f"   Status: {response.status_code}")
        if response.status_code == 200:
            health = response.json()
            print(f"   Model Status: {health.get('model_status', 'unknown')}")
            print(f"   Service Status: {health.get('status', 'unknown')}")
            print("   ✅ Health check passed")
        else:
            print("   ❌ Health check failed")
            return
    except Exception as e:
        print(f"   ❌ Failed to connect: {e}")
        return
    
    # Test 2: Data conversion
    print("\n2. Testing Data Conversion...")
    try:
        # Sample behavioral data
        sample_data = {
            "events": [
                {
                    "timestamp": time.time(),
                    "type": "touch",
                    "touch": {"pressure": 0.8, "size": 0.4, "x_position": 100, "y_position": 200},
                    "motion": {"acc_x": 0.1, "acc_y": -9.8, "acc_z": 0.2},
                    "typing": {"dwell_time": 0.15, "typing_speed": 45.0},
                    "app_usage": {"session_duration": 120.0, "interaction_frequency": 2.5},
                    "device": {"battery": 0.85, "brightness": 0.7}
                },
                {
                    "timestamp": time.time() + 1,
                    "type": "swipe", 
                    "touch": {"pressure": 0.6, "size": 0.3, "x_position": 150, "y_position": 250},
                    "motion": {"acc_x": 0.2, "acc_y": -9.7, "acc_z": 0.1},
                    "typing": {"dwell_time": 0.12, "typing_speed": 48.0},
                    "app_usage": {"session_duration": 121.0, "interaction_frequency": 2.6},
                    "device": {"battery": 0.84, "brightness": 0.7}
                }
            ],
            "user_id": "test_user_123",
            "session_id": "test_session_456"
        }
        
        response = requests.post(f"{base_url}/convert/behavioral-data", json=sample_data, timeout=10)
        print(f"   Status: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            summary = result.get('summary', {})
            print(f"   Nodes: {summary.get('num_nodes', 0)}")
            print(f"   Edges: {summary.get('num_edges', 0)}")
            print(f"   Session Duration: {summary.get('session_duration', 0):.2f}s")
            print("   ✅ Data conversion successful")
            temporal_graph = result.get('temporal_graph')
        else:
            print(f"   ❌ Data conversion failed: {response.text}")
            return
            
    except Exception as e:
        print(f"   ❌ Data conversion error: {e}")
        return
    
    # Test 3: GAT Processing
    print("\n3. Testing GAT Processing...")
    try:
        # User profile vector (64 dimensions)
        user_profile = [0.1] * 64  # Simple profile for testing
        
        gat_request = {
            "graph": temporal_graph,
            "user_profile_vector": user_profile,
            "processing_mode": "inference",
            "attention_heads": 8,
            "embedding_dim": 64,
            "similarity_threshold": 0.85
        }
        
        response = requests.post(f"{base_url}/process", json=gat_request, timeout=10)
        print(f"   Status: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            print(f"   Auth Decision: {result.get('auth_decision', 'unknown')}")
            print(f"   Confidence: {result.get('confidence', 0):.3f}")
            print(f"   Similarity Score: {result.get('similarity_score', 0):.3f}")
            print(f"   Processing Time: {result.get('processing_time_ms', 0):.2f}ms")
            print(f"   Model Mode: {result.get('model_info', {}).get('mode', 'unknown')}")
            print("   ✅ GAT processing successful")
        else:
            print(f"   ❌ GAT processing failed: {response.text}")
            return
            
    except Exception as e:
        print(f"   ❌ GAT processing error: {e}")
        return
    
    # Test 4: User Enrollment
    print("\n4. Testing User Enrollment...")
    try:
        enrollment_request = {
            "graph": temporal_graph,
            "processing_mode": "enrollment",
            "attention_heads": 8,
            "embedding_dim": 64,
            "similarity_threshold": 0.85
        }
        
        response = requests.post(f"{base_url}/enroll", json=enrollment_request, timeout=10)
        print(f"   Status: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            print(f"   User ID: {result.get('user_id', 'unknown')}")
            print(f"   Profile Vector Length: {len(result.get('profile_vector', []))}")
            print(f"   Enrollment Confidence: {result.get('enrollment_confidence', 0):.3f}")
            print("   ✅ User enrollment successful")
        else:
            print(f"   ❌ User enrollment failed: {response.text}")
            return
            
    except Exception as e:
        print(f"   ❌ User enrollment error: {e}")
        return
    
    print("\n" + "=" * 60)
    print("🎉 All GAT Service Tests Passed Successfully!")
    print("✅ GAT service is working correctly")
    print("✅ Ready for integration with CBSA backend")
    print("=" * 60)

if __name__ == "__main__":
    test_gat_service()