"""
Demo: How to Access and Visualize GAT Attention Weights

This script demonstrates:
1. The 2-layer GAT architecture (why 2 layers)
2. How to extract attention weights (4 heads per layer = 8 total attention mechanisms)
3. How to visualize which nodes attend to which neighbors
"""

import torch
import numpy as np
from gat_network import TemporalGraphAttention, SiameseGATNetwork
from data_processor import BehavioralDataProcessor, PyTorchDataConverter

# Sample event data
sample_events = [
    {
        'timestamp': 1000.0,
        'event_type': 'PAGE_ENTER_HOME',
        'event_data': {
            'vector': [0.1] * 48,
            'nonce': 'abc123',
            'signature': 'sig1',
            'deviceInfo': {
                'totalMemory': 4 * (1024**3),
                'deviceYearClass': 2020,
                'networkType': 'WIFI',
                'isRooted': False
            }
        }
    },
    {
        'timestamp': 1005.0,
        'event_type': 'BUTTON_CLICK',
        'event_data': {
            'vector': [0.2] * 48,
            'nonce': 'def456',
            'signature': 'sig2',
            'deviceInfo': {
                'totalMemory': 4 * (1024**3),
                'deviceYearClass': 2020,
                'networkType': 'WIFI',
                'isRooted': False
            }
        }
    },
    {
        'timestamp': 1010.0,
        'event_type': 'SCROLL',
        'event_data': {
            'vector': [0.3] * 48,
            'nonce': 'ghi789',
            'signature': 'sig3',
            'deviceInfo': {
                'totalMemory': 4 * (1024**3),
                'deviceYearClass': 2020,
                'networkType': 'WIFI',
                'isRooted': False
            }
        }
    },
    {
        'timestamp': 1015.0,
        'event_type': 'PAGE_EXIT',
        'event_data': {
            'vector': [0.4] * 48,
            'nonce': 'jkl012',
            'signature': 'sig4',
            'deviceInfo': {
                'totalMemory': 4 * (1024**3),
                'deviceYearClass': 2020,
                'networkType': 'WIFI',
                'isRooted': False
            }
        }
    }
]


def explain_architecture():
    """Explain the 2-layer, 4-head architecture"""
    print("=" * 80)
    print("GAT ARCHITECTURE EXPLANATION")
    print("=" * 80)
    print()
    print("📊 Architecture Overview:")
    print("  - 2 GAT Layers (stacked)")
    print("  - 4 Attention Heads per layer")
    print("  - Total: 8 independent attention mechanisms")
    print()
    print("🔍 Why 2 Layers?")
    print("  Layer 1: Learns immediate neighbor patterns (1-hop)")
    print("    - Node A looks at its direct neighbors")
    print("    - Example: 'BUTTON_CLICK' attends to 'PAGE_ENTER_HOME'")
    print()
    print("  Layer 2: Learns extended patterns (2-hop)")
    print("    - Uses Layer 1 output as input")
    print("    - Node A now 'sees' information from 2-hop neighbors")
    print("    - Example: 'PAGE_EXIT' can see patterns from 'PAGE_ENTER_HOME'")
    print("             even if not directly connected")
    print()
    print("💡 Analogy: Like CNN with multiple conv layers")
    print("  - Early layers: detect edges/simple patterns")
    print("  - Deeper layers: combine into complex objects")
    print()
    print("🎯 Why 4 Heads?")
    print("  Each head learns DIFFERENT attention patterns:")
    print("    Head 1: Might focus on temporal proximity")
    print("    Head 2: Might focus on event type similarity")
    print("    Head 3: Might focus on device state changes")
    print("    Head 4: Might learn user-specific patterns")
    print()
    print("  Think: 4 experts looking at the same graph from different angles")
    print("=" * 80)
    print()


def demo_attention_extraction():
    """Show how to extract attention weights"""
    print("=" * 80)
    print("DEMO: Extracting Attention Weights")
    print("=" * 80)
    print()
    
    # Step 1: Process events into a graph
    print("Step 1: Creating temporal graph from events...")
    config = {
        'time_window_seconds': 20,
        'min_events_per_window': 1,
        'max_events_per_window': 100,
        'distinct_event_connections': 4
    }
    
    processor = BehavioralDataProcessor(config)
    temporal_graph = processor.process_behavioral_data(
        raw_data=sample_events,
        user_id='demo_user',
        session_id='demo_session'
    )
    
    print(f"  ✓ Created graph with {len(temporal_graph.nodes)} nodes and {len(temporal_graph.edges)} edges")
    print(f"    Events: {[n.event_type for n in temporal_graph.nodes]}")
    print()
    
    # Step 2: Convert to PyTorch format
    print("Step 2: Converting to PyTorch format...")
    converter = PyTorchDataConverter()
    pytorch_data = converter.convert_to_pytorch(temporal_graph)
    
    print(f"  ✓ Node features shape: {pytorch_data['x'].shape}")
    print(f"  ✓ Edge index shape: {pytorch_data['edge_index'].shape}")
    print(f"  ✓ Edges: {pytorch_data['edge_index'].t().tolist()}")
    print()
    
    # Step 3: Create GAT model with 4 heads
    print("Step 3: Creating GAT model (4 heads, 2 layers)...")
    gat_config = {
        'input_dim': 60,
        'hidden_dim': 64,
        'output_dim': 64,
        'num_heads': 4,  # 4 HEADS!
        'dropout': 0.1,
        'temporal_dim': 16,
        'return_attention_weights': True
    }
    
    model = TemporalGraphAttention(**gat_config)
    model.eval()
    
    print(f"  ✓ Model created: {model.num_heads} heads per layer")
    print()
    
    # Step 4: Forward pass WITH attention extraction
    print("Step 4: Running forward pass with attention extraction...")
    with torch.no_grad():
        node_emb, graph_emb, attention_dict = model(
            pytorch_data['x'],
            pytorch_data['edge_index'],
            pytorch_data['temporal_features'],
            batch=None,
            return_attention=True  # KEY: This returns attention weights!
        )
    
    print(f"  ✓ Node embeddings shape: {node_emb.shape}")
    print(f"  ✓ Graph embedding shape: {graph_emb.shape}")
    print(f"  ✓ Attention extracted: {attention_dict is not None}")
    print()
    
    # Step 5: Inspect attention weights
    print("=" * 80)
    print("ATTENTION WEIGHTS BREAKDOWN")
    print("=" * 80)
    print()
    
    if attention_dict:
        # Layer 1 attention
        edge_index_l1, attention_l1 = attention_dict['layer1']
        print(f"Layer 1 Attention:")
        print(f"  - Edge index shape: {edge_index_l1.shape}")
        print(f"  - Attention weights shape: {attention_l1.shape}  [num_edges, num_heads=4]")
        print()
        
        # Show attention for each edge
        for i in range(edge_index_l1.shape[1]):
            src = edge_index_l1[0, i].item()
            tgt = edge_index_l1[1, i].item()
            attn_heads = attention_l1[i].cpu().numpy()
            
            src_event = temporal_graph.nodes[src].event_type if src < len(temporal_graph.nodes) else f"Node{src}"
            tgt_event = temporal_graph.nodes[tgt].event_type if tgt < len(temporal_graph.nodes) else f"Node{tgt}"
            
            print(f"  Edge: {src_event} (node {src}) → {tgt_event} (node {tgt})")
            print(f"    Head 1: {attn_heads[0]:.4f}")
            print(f"    Head 2: {attn_heads[1]:.4f}")
            print(f"    Head 3: {attn_heads[2]:.4f}")
            print(f"    Head 4: {attn_heads[3]:.4f}")
            print(f"    Mean attention: {attn_heads.mean():.4f}")
            print()
        
        # Layer 2 attention
        edge_index_l2, attention_l2 = attention_dict['layer2']
        print(f"Layer 2 Attention:")
        print(f"  - Edge index shape: {edge_index_l2.shape}")
        print(f"  - Attention weights shape: {attention_l2.shape}  [num_edges, num_heads=4]")
        print()
        
        print("  (Layer 2 operates on Layer 1's output representations)")
        print("  (Same edge structure, but different learned attention patterns)")
        print()
    
    # Step 6: Use the visualize_attention helper method
    print("=" * 80)
    print("USING visualize_attention() METHOD")
    print("=" * 80)
    print()
    
    # Inspect what node 0 attends to
    attention_viz = model.visualize_attention(node_id=0, layer=1)
    
    if 'error' not in attention_viz:
        print(f"Node {attention_viz['source_node']} ({temporal_graph.nodes[0].event_type}) attention in Layer {attention_viz['layer']}:")
        print(f"  Number of heads: {attention_viz['num_heads']}")
        print(f"  Attends to {len(attention_viz['neighbors'])} neighbors:")
        print()
        
        for i, neighbor_id in enumerate(attention_viz['neighbors']):
            neighbor_event = temporal_graph.nodes[neighbor_id].event_type
            attn_per_head = attention_viz['attention_per_head'][i]
            attn_mean = attention_viz['attention_mean'][i]
            
            print(f"    → Node {neighbor_id} ({neighbor_event})")
            print(f"       Heads: {attn_per_head}")
            print(f"       Mean: {attn_mean:.4f}")
            print()
    
    print("=" * 80)
    print("KEY TAKEAWAYS")
    print("=" * 80)
    print()
    print("✓ Attention weights show HOW MUCH each node focuses on its neighbors")
    print("✓ 4 heads = 4 different 'perspectives' on the graph")
    print("✓ Layer 1 captures local patterns, Layer 2 captures global patterns")
    print("✓ Use return_attention=True to access these weights")
    print("✓ Use model.visualize_attention(node_id, layer) for easy inspection")
    print()
    
    return model, attention_dict


def demo_siamese_with_attention():
    """Show how to get attention in the Siamese wrapper"""
    print("=" * 80)
    print("DEMO: Attention in Siamese Network (for Training)")
    print("=" * 80)
    print()
    
    gat_config = {
        'input_dim': 60,
        'hidden_dim': 64,
        'output_dim': 64,
        'num_heads': 4,
        'dropout': 0.1,
        'temporal_dim': 16,
        'return_attention_weights': True
    }
    
    siamese_model = SiameseGATNetwork(gat_config)
    siamese_model.eval()
    
    print("✓ Siamese model created")
    print()
    print("During training, attention is typically NOT extracted (for speed)")
    print("But you can enable it for debugging/visualization:")
    print()
    print("  # Normal training (fast)")
    print("  embedding = model.forward_once(x, edge_index, temporal, batch)")
    print()
    print("  # With attention (slower, for debugging)")
    print("  embedding, attention_dict = model.forward_once(")
    print("      x, edge_index, temporal, batch, return_attention=True")
    print("  )")
    print()
    print("=" * 80)


if __name__ == '__main__':
    print("\n")
    print("╔" + "=" * 78 + "╗")
    print("║" + " " * 20 + "GAT ATTENTION VISUALIZATION DEMO" + " " * 26 + "║")
    print("╚" + "=" * 78 + "╝")
    print()
    
    # Part 1: Explain architecture
    explain_architecture()
    input("Press Enter to continue to attention extraction demo...")
    print()
    
    # Part 2: Demo attention extraction
    model, attention_dict = demo_attention_extraction()
    print()
    input("Press Enter to continue to Siamese network demo...")
    print()
    
    # Part 3: Siamese wrapper
    demo_siamese_with_attention()
    
    print()
    print("=" * 80)
    print("DEMO COMPLETE!")
    print("=" * 80)
    print()
    print("Summary:")
    print("  • 2 GAT layers for hierarchical learning (1-hop + 2-hop patterns)")
    print("  • 4 attention heads per layer (8 total attention mechanisms)")
    print("  • Attention weights: shape [num_edges, 4] for each layer")
    print("  • Use return_attention=True to extract weights")
    print("  • Use model.visualize_attention(node_id, layer) for inspection")
    print()
