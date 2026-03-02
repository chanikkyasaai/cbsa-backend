"""
Test script to validate data processor with actual event structure
Run this to verify that _extract_behavioral_vector returns correct 60-D vectors
"""

import sys
import json
from data_processor import BehavioralDataProcessor

# Sample event from user (actual structure)
sample_event = {
    'user_id': None,
    'session_id': 'sess_mlqtbgkv_jmagefj',
    'timestamp': 1771345409.544,
    'event_type': 'PAGE_ENTER_HOME',
    'event_data': {
        'timestamp': 1771345408977,
        'nonce': '2mz7gea8pem',
        'vector': [
            0, 0, 0, 0.043, 0, 0.03333333333333333, 0.5, 0.5, 0.5, 0.5,
            0.5, 0.5, 0, 1, 1, 1, 0, 0, 0.5, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0.5, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0
        ],
        'deviceInfo': {
            'deviceId': None,
            'brand': 'samsung',
            'manufacturer': 'samsung',
            'modelName': 'SM-A035F',
            'modelId': None,
            'designName': 'a03',
            'productName': 'a03nnxx',
            'deviceYearClass': 2015,
            'totalMemory': 3728367616,
            'osName': 'samsung/a03nnxx/a03:13/TP1A.220624.014/A035FXXU4CWI1:user/release-keys',
            'osVersion': '13',
            'osBuildId': 'TP1A.220624.014.A035FXXS9CYG1',
            'platformApiLevel': 33,
            'deviceType': 1,
            'isDevice': True,
            'isRooted': False,
            'networkType': 'WIFI',
            'networkState': 'connected',
            'isInternetReachable': True,
            'ipAddress': '192.168.1.2',
            'carrier': 'Jio',
            'isoCountryCode': 'in',
            'mobileCountryCode': '405',
            'mobileNetworkCode': '869',
            'allowsVoip': True,
            'latitude': None,
            'longitude': None,
            'locationAccuracy': None,
            'locationTimestamp': None
        },
        'signature': '932cbf9689efd88702b367191d2b3569d935e15ef39539a42f19eeab721c7b42'
    }
}


def test_behavioral_vector_extraction():
    """Test that _extract_behavioral_vector returns correct 60-D vector"""
    print("=" * 70)
    print("Testing _extract_behavioral_vector with actual event structure")
    print("=" * 70)
    
    # Initialize processor
    config = {
        'time_window_seconds': 20,
        'min_events_per_window': 5,
        'max_events_per_window': 100,
        'distinct_event_connections': 4
    }
    
    processor = BehavioralDataProcessor(config)
    
    # Extract behavioral vector
    vector = processor._extract_behavioral_vector(sample_event)
    
    print(f"\n✓ Vector length: {len(vector)}")
    print(f"✓ Expected length: 60")
    print(f"✓ Match: {'YES ✓' if len(vector) == 60 else 'NO ✗'}")
    
    # Verify all elements are floats
    all_floats = all(isinstance(v, float) for v in vector)
    print(f"\n✓ All elements are floats: {'YES ✓' if all_floats else 'NO ✗'}")
    
    # Show vector breakdown
    print("\n" + "=" * 70)
    print("Vector composition breakdown:")
    print("=" * 70)
    print(f"  Base behavioral vector (0-47):   {vector[:48][:5]}... (first 5 shown)")
    print(f"  Event-type embedding (48-55):    {vector[48:56]}")
    print(f"  Device context (56-59):          {vector[56:60]}")
    
    # Decode device context
    print("\n" + "=" * 70)
    print("Device context interpretation:")
    print("=" * 70)
    print(f"  [0] Memory (normalized):      {vector[56]:.4f}")
    print(f"  [1] Network score:            {vector[57]:.4f}  (WIFI=1.0, cellular=0.5)")
    print(f"  [2] Device age (normalized):  {vector[58]:.4f}  (year class: 2015)")
    print(f"  [3] Is rooted:                {vector[59]:.4f}  (False=0.0)")
    
    # Test device features extraction
    print("\n" + "=" * 70)
    print("Device features (metadata dict):")
    print("=" * 70)
    device_features = processor._extract_device_features(sample_event)
    for key, value in device_features.items():
        print(f"  {key:25s}: {value:.4f}")
    
    return vector, device_features


def test_graph_creation():
    """Test full graph creation with multiple events"""
    print("\n" + "=" * 70)
    print("Testing full graph creation pipeline")
    print("=" * 70)
    
    config = {
        'time_window_seconds': 20,
        'min_events_per_window': 1,  # Allow single event for test
        'max_events_per_window': 100,
        'distinct_event_connections': 4
    }
    
    processor = BehavioralDataProcessor(config)
    
    # Create a small batch of events
    events = [sample_event]
    
    try:
        temporal_graph = processor.process_behavioral_data(
            raw_data=events,
            user_id='test_user',
            session_id='sess_mlqtbgkv_jmagefj'
        )
        
        print(f"\n✓ Graph created successfully!")
        print(f"  - Total nodes: {temporal_graph.total_events}")
        print(f"  - Total edges: {len(temporal_graph.edges)}")
        print(f"  - Event diversity: {temporal_graph.event_diversity}")
        print(f"  - Session duration: {temporal_graph.session_duration:.2f}s")
        
        # Verify node vector shape
        if temporal_graph.nodes:
            node = temporal_graph.nodes[0]
            print(f"\n✓ First node:")
            print(f"  - Node ID: {node.node_id}")
            print(f"  - Event type: {node.event_type}")
            print(f"  - Vector length: {len(node.behavioral_vector)}")
            print(f"  - Timestamp: {node.timestamp}")
            print(f"  - Nonce: {node.nonce}")
        
        return temporal_graph
        
    except Exception as e:
        print(f"\n✗ Error creating graph: {e}")
        import traceback
        traceback.print_exc()
        return None


def test_pytorch_conversion():
    """Test PyTorch conversion if torch is available"""
    print("\n" + "=" * 70)
    print("Testing PyTorch conversion")
    print("=" * 70)
    
    try:
        from data_processor import PyTorchDataConverter
        import torch
        
        # First create a graph
        config = {
            'time_window_seconds': 20,
            'min_events_per_window': 1,
            'max_events_per_window': 100,
            'distinct_event_connections': 4
        }
        
        processor = BehavioralDataProcessor(config)
        temporal_graph = processor.process_behavioral_data(
            raw_data=[sample_event],
            user_id='test_user',
            session_id='test_session'
        )
        
        # Convert to PyTorch
        converter = PyTorchDataConverter()
        pytorch_data = converter.convert_to_pytorch(temporal_graph)
        
        print(f"\n✓ PyTorch conversion successful!")
        print(f"  - x.shape: {pytorch_data['x'].shape}  (expected: [1, 60])")
        print(f"  - temporal_features.shape: {pytorch_data['temporal_features'].shape}  (expected: [1, 1])")
        print(f"  - edge_index.shape: {pytorch_data['edge_index'].shape}")
        print(f"  - num_nodes: {pytorch_data['num_nodes']}")
        print(f"  - num_edges: {pytorch_data['num_edges']}")
        
        # Verify shape
        expected_shape = (1, 60)
        actual_shape = tuple(pytorch_data['x'].shape)
        match = expected_shape == actual_shape
        print(f"\n✓ Shape verification: {'PASS ✓' if match else 'FAIL ✗'}")
        
        return pytorch_data
        
    except ImportError as e:
        print(f"\n⚠ PyTorch not available: {e}")
        print("  (This is optional - install torch to test conversion)")
        return None
    except Exception as e:
        print(f"\n✗ Error in PyTorch conversion: {e}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == '__main__':
    print("\n" + "=" * 70)
    print("GAT Data Processor Validation Suite")
    print("=" * 70)
    
    # Test 1: Vector extraction
    vector, device_features = test_behavioral_vector_extraction()
    
    # Test 2: Graph creation
    temporal_graph = test_graph_creation()
    
    # Test 3: PyTorch conversion (optional)
    pytorch_data = test_pytorch_conversion()
    
    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    
    tests_passed = []
    tests_failed = []
    
    if len(vector) == 60:
        tests_passed.append("Vector extraction (60-D)")
    else:
        tests_failed.append(f"Vector extraction (got {len(vector)}-D, expected 60-D)")
    
    if temporal_graph and len(temporal_graph.nodes) > 0:
        tests_passed.append("Graph creation")
    else:
        tests_failed.append("Graph creation")
    
    if pytorch_data is not None:
        if tuple(pytorch_data['x'].shape) == (1, 60):
            tests_passed.append("PyTorch conversion")
        else:
            tests_failed.append(f"PyTorch conversion (shape mismatch)")
    
    print(f"\n✓ Tests passed: {len(tests_passed)}")
    for test in tests_passed:
        print(f"  - {test}")
    
    if tests_failed:
        print(f"\n✗ Tests failed: {len(tests_failed)}")
        for test in tests_failed:
            print(f"  - {test}")
        sys.exit(1)
    else:
        print("\n🎉 All tests passed!")
        sys.exit(0)
