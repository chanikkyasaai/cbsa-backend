# GAT Architecture Deep Dive: 2 Layers, 4 Heads, Attention Vectors

## Quick Answers to Your Questions

### ❓ "There are two GAT layers. What does that mean?"

**Answer**: Two **stacked** GAT layers create a hierarchical learning architecture, similar to having multiple convolutional layers in a CNN.

```
Layer 1: Learns LOCAL patterns (1-hop neighbors)
   ↓
Layer 2: Learns GLOBAL patterns (2-hop neighbors via Layer 1)
   ↓
Output: Rich graph representation
```

**Why stack layers?**
- **Layer 1**: Each node aggregates information from its immediate neighbors
  - Example: `BUTTON_CLICK` node learns from directly connected `PAGE_ENTER` node
  
- **Layer 2**: Operates on Layer 1's output, so nodes now "see" 2-hop neighbors
  - Example: `PAGE_EXIT` can incorporate patterns from `PAGE_ENTER` even if not directly connected
  - The network learns higher-order relationships

**Analogy**:
- 1 layer = seeing only your immediate friends
- 2 layers = seeing friends-of-friends and indirect relationships
- More layers = wider "receptive field" in the graph

---

### ❓ "I wanted 4 heads. Where is attention vector?"

**Current Status**: ✅ FIXED! Now configured to **4 heads per layer**.

Previously: `num_heads=8` (default)  
Now: `num_heads=4` (as requested)

**What are "heads"?**
- Each head learns a **different attention pattern**
- 4 heads = 4 parallel attention mechanisms looking at the same graph from different angles

**Think of it like 4 experts:**
- Head 1: Focuses on **temporal proximity** (events close in time)
- Head 2: Focuses on **event type similarity** (similar actions)
- Head 3: Focuses on **device state** (hardware changes)
- Head 4: Learns **user-specific patterns**

All 4 outputs are concatenated together for a richer representation.

---

## Architecture Breakdown

```
INPUT: Node features [num_nodes, 60]
       Edge index [2, num_edges]
       Temporal features [num_nodes, 1]
         ↓
    [Temporal Encoder]
         ↓ +16D temporal encoding
    [60 + 16 = 76-D combined input]
         ↓
┌────────────────────────────────────────┐
│  GAT Layer 1 (4 heads)                 │
│  Each head: 76-D → 16-D                │
│  Concatenate: 4 × 16-D = 64-D          │
│                                        │
│  🔍 Attention weights (Layer 1):      │
│     Shape: [num_edges, 4]             │
│     One value per (edge, head) pair   │
└────────────────────────────────────────┘
         ↓
    [LayerNorm + ReLU + Dropout]
         ↓
┌────────────────────────────────────────┐
│  GAT Layer 2 (4 heads)                 │
│  Each head: 64-D → 16-D                │
│  Concatenate: 4 × 16-D = 64-D          │
│                                        │
│  🔍 Attention weights (Layer 2):      │
│     Shape: [num_edges, 4]             │
│     Different patterns on Layer 1 output│
└────────────────────────────────────────┘
         ↓
    [LayerNorm + ReLU]
         ↓
    [Global Mean Pool]
         ↓
    [MLP: 64-D → 64-D]
         ↓
OUTPUT: Graph embedding [64]
```

**Total attention mechanisms**: 2 layers × 4 heads = **8 attention weight matrices**

---

## How to Access Attention Vectors

### Method 1: During Forward Pass

```python
from gat_network import TemporalGraphAttention

model = TemporalGraphAttention(
    input_dim=60,
    hidden_dim=64,
    output_dim=64,
    num_heads=4,  # 4 heads!
    return_attention_weights=True  # Enable attention extraction
)

# Run forward with attention enabled
node_emb, graph_emb, attention_dict = model(
    x=node_features,
    edge_index=edges,
    temporal_features=timestamps,
    return_attention=True  # KEY FLAG!
)

# Access attention weights
layer1_attention = attention_dict['layer1']  # (edge_index, attention_weights)
layer2_attention = attention_dict['layer2']

# Unpack
edge_index_l1, attention_weights_l1 = layer1_attention
# attention_weights_l1: Tensor [num_edges, 4]
#   - Row i = attention for edge i
#   - 4 columns = 4 head values
```

### Method 2: Use Built-in Visualizer

```python
# After running forward pass with return_attention=True
attention_info = model.visualize_attention(node_id=0, layer=1)

# Returns:
{
    'source_node': 0,
    'layer': 1,
    'num_heads': 4,
    'neighbors': [1, 2, 3],  # Node IDs this node connects to
    'attention_per_head': [
        [0.25, 0.30, 0.20, 0.25],  # Attention to neighbor 1 (4 heads)
        [0.35, 0.28, 0.22, 0.15],  # Attention to neighbor 2
        [0.40, 0.42, 0.58, 0.60]   # Attention to neighbor 3
    ],
    'attention_mean': [0.25, 0.25, 0.50]  # Average across heads
}
```

**Interpretation**:
- Higher values = node pays more attention to that neighbor
- Each head can focus on different neighbors (diversity!)
- Sum over all neighbors ≈ 1.0 (softmax normalized)

### Method 3: Access Stored Weights

```python
# After forward pass, weights are cached
cached_attention = model.get_attention_weights()

layer1 = cached_attention['layer1']  # (edge_index, weights)
layer2 = cached_attention['layer2']
```

---

## Example: Inspecting Attention on a Real Graph

Say you have a session with 4 events:
```
Node 0: PAGE_ENTER_HOME (t=0s)
Node 1: BUTTON_CLICK (t=5s)
Node 2: SCROLL (t=10s)
Node 3: PAGE_EXIT (t=15s)
```

Edges (based on temporal windowing):
```
0 → 1, 0 → 2, 0 → 3
1 → 2, 1 → 3
2 → 3
```

**Layer 1 Attention** (after running model):
```
Edge 0→1 (PAGE_ENTER → BUTTON_CLICK):
  Head 1: 0.45  (high: temporal proximity)
  Head 2: 0.30  (medium: type correlation)
  Head 3: 0.15  (low: device state unchanged)
  Head 4: 0.10  (low: user pattern mismatch)
  
Edge 0→2 (PAGE_ENTER → SCROLL):
  Head 1: 0.25  (lower: more time gap)
  Head 2: 0.35  (higher: navigation pattern)
  Head 3: 0.20
  Head 4: 0.20
```

**Layer 2 Attention**:
- Operates on Layer 1's refined representations
- Might emphasize different relationships based on learned higher-order patterns

---

## Configuration Summary

### Current Setup (After Fix):

```python
gat_config = {
    'input_dim': 60,           # Node feature dimension
    'hidden_dim': 64,          # Intermediate size
    'output_dim': 64,          # Final embedding size
    'num_heads': 4,            # 4 heads per layer ✅
    'dropout': 0.1,
    'temporal_dim': 16,        # Temporal encoding size
    'return_attention_weights': True  # Enable attention access
}

model = TemporalGraphAttention(**gat_config)
```

### Total Attention Mechanisms:
- **Layer 1**: 4 heads
- **Layer 2**: 4 heads
- **Total**: 8 independent attention patterns

### Shapes Reference:
```
Input:
  - Node features: [num_nodes, 60]
  - Temporal features: [num_nodes, 1]
  
Attention weights:
  - Layer 1: [num_edges, 4]  (4 heads)
  - Layer 2: [num_edges, 4]  (4 heads)
  
Output:
  - Node embeddings: [num_nodes, 64]
  - Graph embedding: [64]
```

---

## Key Changes Made

1. ✅ Changed `num_heads` from 8 → **4** (as requested)
2. ✅ Added `return_attention_weights=True` parameter
3. ✅ Modified `forward()` to optionally return attention dict
4. ✅ Added `get_attention_weights()` method
5. ✅ Added `visualize_attention(node_id, layer)` helper
6. ✅ Updated Siamese wrapper to support attention extraction
7. ✅ Updated inference engine to optionally return attention

---

## Usage in Training vs Inference

### Training (Speed Priority):
```python
# Don't extract attention during training (faster)
emb = model.forward_once(x, edge_index, temporal, batch)
```

### Debugging/Analysis (Insight Priority):
```python
# Extract attention to understand what model learned
emb, attention = model.forward_once(
    x, edge_index, temporal, batch,
    return_attention=True
)
```

### Production Inference (Optional Explanation):
```python
from gat_network import GATInferenceEngine

engine = GATInferenceEngine(model, device='cpu')

# Normal auth (fast)
result = engine.authenticate(session_graph, user_profile)

# Auth with attention (for explainability)
result = engine.authenticate(
    session_graph, 
    user_profile,
    return_attention=True  # Adds 'attention_weights' to response
)
```

---

## Summary

✅ **2 Layers**: Learn hierarchical patterns (local + global)  
✅ **4 Heads**: 4 parallel attention mechanisms per layer  
✅ **Attention Vectors**: Accessible via `return_attention=True` flag  
✅ **Shape**: `[num_edges, 4]` per layer (4 = num_heads)  
✅ **Total**: 8 attention weight matrices (2 layers × 4 heads)

**Where to find attention**:
- Set `return_attention=True` in `forward()`
- Returns dict with `'layer1'` and `'layer2'` keys
- Each contains `(edge_index, attention_weights)` tuple
- Use `model.visualize_attention(node_id, layer)` for easy inspection
