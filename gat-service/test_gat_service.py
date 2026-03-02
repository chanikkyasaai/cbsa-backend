"""
Test script for GAT service
Tests both simulation and real PyTorch modes
"""

import json
import requests
import random
import time
from datetime import datetime, timezone
from typing import List, Dict, Any


def generate_sample_behavioral_data(num_events: int = 20) -> List[Dict[str, Any]]:
    """Generate sample behavioral data for testing"""
    events = []
    current_time = time.time()
    
    event_types = ['touch', 'swipe', 'tap', 'scroll', 'type', 'gesture']
    
    for i in range(num_events):
        # Generate realistic behavioral event
        event = {
            'timestamp': current_time + i * random.uniform(0.5, 2.0),
            'type': random.choice(event_types),
            'touch': {
                'pressure': random.uniform(0.3, 1.0),
                'size': random.uniform(0.1, 0.5),
                'major': random.uniform(5, 20),
                'minor': random.uniform(3, 15),
                'orientation': random.uniform(0, 360),
                'x_position': random.uniform(0, 1080),
                'y_position': random.uniform(0, 1920),
                'velocity_x': random.uniform(-100, 100),
                'velocity_y': random.uniform(-100, 100),
                'acceleration_x': random.uniform(-50, 50),
                'acceleration_y': random.uniform(-50, 50),
                'duration': random.uniform(0.05, 0.5)
            },
            'motion': {
                'acc_x': random.uniform(-10, 10),
                'acc_y': random.uniform(-10, 10),
                'acc_z': random.uniform(-10, 10),
                'gyro_x': random.uniform(-5, 5),
                'gyro_y': random.uniform(-5, 5),
                'gyro_z': random.uniform(-5, 5),
                'mag_x': random.uniform(-50, 50),
                'mag_y': random.uniform(-50, 50),
                'mag_z': random.uniform(-50, 50),
                'linear_acc_x': random.uniform(-5, 5),
                'linear_acc_y': random.uniform(-5, 5),
                'linear_acc_z': random.uniform(-5, 5)
            },
            'typing': {
                'dwell_time': random.uniform(0.05, 0.3),
                'flight_time': random.uniform(0.01, 0.1),
                'typing_speed': random.uniform(20, 80),
                'pressure_variance': random.uniform(0.1, 0.4),
                'rhythm_score': random.uniform(0.5, 1.0),
                'key_hold_variance': random.uniform(0.01, 0.1),
                'inter_key_interval': random.uniform(0.05, 0.2),
                'typing_cadence': random.uniform(0.5, 2.0),
                'error_rate': random.uniform(0.0, 0.1),
                'backspace_frequency': random.uniform(0.0, 0.2),
                'caps_lock_usage': random.uniform(0.0, 0.1),
                'special_char_usage': random.uniform(0.0, 0.3)
            },
            'app_usage': {
                'transition_time': random.uniform(0.1, 2.0),
                'interaction_frequency': random.uniform(1, 10),
                'session_duration': random.uniform(10, 300),
                'touch_frequency': random.uniform(0.5, 5.0),
                'scroll_speed': random.uniform(100, 1000),
                'tap_intensity': random.uniform(0.3, 1.0),
                'multi_touch_usage': random.uniform(0.0, 0.5),
                'gesture_complexity': random.uniform(0.1, 1.0),
                'navigation_pattern': random.uniform(0.0, 1.0),
                'menu_access_frequency': random.uniform(0.0, 0.5),
                'notification_interaction': random.uniform(0.0, 0.3),
                'background_app_switches': random.uniform(0.0, 0.2)
            },
            'device': {
                'battery': random.uniform(0.2, 1.0),
                'brightness': random.uniform(0.3, 1.0),
                'signal': random.uniform(0.5, 1.0),
                'cpu': random.uniform(0.1, 0.8),
                'memory': random.uniform(0.3, 0.9),
                'temperature': random.uniform(20, 40),
                'charging': random.choice([True, False]),
                'wifi': random.choice([True, False]),
                'bluetooth': random.choice([True, False]),
                'location': random.choice([True, False])
            },
            'signature': f"sig_{i}_{random.randint(1000, 9999)}",
            'nonce': f"nonce_{i}_{random.randint(10000, 99999)}"
        }
        
        events.append(event)
    
    return events


def test_health_check(base_url: str):
    """Test health check endpoints"""
    print("\n=== Testing Health Check ===")
    
    try:
        response = requests.get(f"{base_url}/")
        print(f"Root endpoint: {response.status_code}")
        if response.status_code == 200:
            print(f"Response: {json.dumps(response.json(), indent=2)}")
        
        response = requests.get(f"{base_url}/health")
        print(f"Health endpoint: {response.status_code}")
        if response.status_code == 200:
            print(f"Response: {json.dumps(response.json(), indent=2)}")
            
    except requests.exceptions.RequestException as e:
        print(f"Health check failed: {e}")


def test_data_conversion(base_url: str):
    """Test behavioral data conversion"""
    print("\n=== Testing Data Conversion ===")
    
    try:
        # Generate sample data
        events = generate_sample_behavioral_data(15)
        
        payload = {
            'events': events,
            'user_id': 'test_user_123',
            'session_id': f'test_session_{int(time.time())}'
        }
        
        response = requests.post(
            f"{base_url}/convert/behavioral-data",
            json=payload
        )
        
        print(f"Conversion endpoint: {response.status_code}")
        if response.status_code == 200:
            result = response.json()
            print(f"Conversion successful:")
            print(f"  - Nodes: {result['summary']['num_nodes']}")
            print(f"  - Edges: {result['summary']['num_edges']}")
            print(f"  - Session duration: {result['summary']['session_duration']:.2f}s")
            print(f"  - Event diversity: {result['summary']['event_diversity']}")
            
            return result['temporal_graph']
        else:
            print(f"Conversion failed: {response.text}")
            return None
            
    except requests.exceptions.RequestException as e:
        print(f"Data conversion test failed: {e}")
        return None


def test_gat_processing(base_url: str, temporal_graph: Dict):
    """Test GAT processing"""
    print("\n=== Testing GAT Processing ===")
    
    if not temporal_graph:
        print("No temporal graph available for testing")
        return
    
    try:
        # Test inference mode
        print("Testing inference mode...")
        
        # Generate sample user profile vector
        user_profile = [random.uniform(-1, 1) for _ in range(64)]
        
        payload = {
            'graph': temporal_graph,
            'user_profile_vector': user_profile,
            'processing_mode': 'inference',
            'attention_heads': 8,
            'embedding_dim': 64,
            'similarity_threshold': 0.85
        }
        
        response = requests.post(
            f"{base_url}/process",
            json=payload
        )
        
        print(f"GAT processing: {response.status_code}")
        if response.status_code == 200:
            result = response.json()
            print(f"Processing successful:")
            print(f"  - Auth decision: {result['auth_decision']}")
            print(f"  - Confidence: {result['confidence']:.3f}")
            print(f"  - Similarity score: {result.get('similarity_score', 'N/A')}")
            print(f"  - Processing time: {result['processing_time_ms']:.2f}ms")
            print(f"  - Model mode: {result['model_info']['mode']}")
            
            return result
        else:
            print(f"GAT processing failed: {response.text}")
            return None
            
    except requests.exceptions.RequestException as e:
        print(f"GAT processing test failed: {e}")
        return None


def test_enrollment(base_url: str, temporal_graph: Dict):
    """Test user enrollment"""
    print("\n=== Testing User Enrollment ===")
    
    if not temporal_graph:
        print("No temporal graph available for testing")
        return
    
    try:
        payload = {
            'graph': temporal_graph,
            'processing_mode': 'enrollment',
            'attention_heads': 8,
            'embedding_dim': 64,
            'similarity_threshold': 0.85
        }
        
        response = requests.post(
            f"{base_url}/enroll",
            json=payload
        )
        
        print(f"User enrollment: {response.status_code}")
        if response.status_code == 200:
            result = response.json()
            print(f"Enrollment successful:")
            print(f"  - User ID: {result['user_id']}")
            print(f"  - Profile vector length: {len(result['profile_vector'])}")
            print(f"  - Enrollment confidence: {result['enrollment_confidence']:.3f}")
            print(f"  - Number of events: {result['num_events']}")
            
            return result
        else:
            print(f"Enrollment failed: {response.text}")
            return None
            
    except requests.exceptions.RequestException as e:
        print(f"Enrollment test failed: {e}")
        return None


def test_model_info(base_url: str):
    """Test model information endpoint"""
    print("\n=== Testing Model Info ===")
    
    try:
        response = requests.get(f"{base_url}/models/info")
        print(f"Model info: {response.status_code}")
        if response.status_code == 200:
            result = response.json()
            print(f"Model information:")
            print(f"  - Model loaded: {result['model_loaded']}")
            print(f"  - Model path: {result['model_path']}")
            print(f"  - Model exists: {result['model_exists']}")
            print(f"  - Config: {json.dumps(result['config'], indent=4)}")
            
    except requests.exceptions.RequestException as e:
        print(f"Model info test failed: {e}")


def run_performance_test(base_url: str, num_requests: int = 10):
    """Run performance test with multiple requests"""
    print(f"\n=== Performance Test ({num_requests} requests) ===")
    
    processing_times = []
    
    for i in range(num_requests):
        print(f"Request {i+1}/{num_requests}...", end=' ')
        
        try:
            # Generate new data for each request
            events = generate_sample_behavioral_data(random.randint(10, 25))
            
            # Convert to temporal graph
            conversion_payload = {
                'events': events,
                'user_id': f'perf_user_{i}',
                'session_id': f'perf_session_{i}_{int(time.time())}'
            }
            
            start_time = time.time()
            
            conv_response = requests.post(
                f"{base_url}/convert/behavioral-data",
                json=conversion_payload
            )
            
            if conv_response.status_code != 200:
                print("CONVERSION FAILED")
                continue
            
            temporal_graph = conv_response.json()['temporal_graph']
            
            # Process with GAT
            user_profile = [random.uniform(-1, 1) for _ in range(64)]
            
            gat_payload = {
                'graph': temporal_graph,
                'user_profile_vector': user_profile,
                'processing_mode': 'inference',
                'attention_heads': 8,
                'embedding_dim': 64,
                'similarity_threshold': 0.85
            }
            
            gat_response = requests.post(
                f"{base_url}/process",
                json=gat_payload
            )
            
            end_time = time.time()
            total_time = (end_time - start_time) * 1000
            
            if gat_response.status_code == 200:
                result = gat_response.json()
                gat_time = result['processing_time_ms']
                processing_times.append(total_time)
                print(f"OK (Total: {total_time:.1f}ms, GAT: {gat_time:.1f}ms)")
            else:
                print("GAT FAILED")
                
        except Exception as e:
            print(f"ERROR: {e}")
    
    if processing_times:
        avg_time = sum(processing_times) / len(processing_times)
        min_time = min(processing_times)
        max_time = max(processing_times)
        
        print(f"\nPerformance Results:")
        print(f"  - Average: {avg_time:.1f}ms")
        print(f"  - Min: {min_time:.1f}ms")
        print(f"  - Max: {max_time:.1f}ms")
        print(f"  - Success rate: {len(processing_times)}/{num_requests} ({len(processing_times)/num_requests*100:.1f}%)")


def main():
    """Main test function"""
    base_url = "http://localhost:8001"  # GAT service port
    
    print("GAT Service Test Suite")
    print("=" * 50)
    
    # Test all endpoints
    test_health_check(base_url)
    
    # Test data conversion and get temporal graph
    temporal_graph = test_data_conversion(base_url)
    
    # Test GAT processing
    gat_result = test_gat_processing(base_url, temporal_graph)
    
    # Test enrollment
    enrollment_result = test_enrollment(base_url, temporal_graph)
    
    # Test model info
    test_model_info(base_url)
    
    # Performance test
    run_performance_test(base_url, 5)
    
    print("\n=== Test Suite Complete ===")


if __name__ == "__main__":
    main()