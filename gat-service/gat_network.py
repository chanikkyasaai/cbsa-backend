"""
GAT Neural Network Implementation
Core Graph Attention Network with temporal processing
"""

import torch
import torch.nn as nn 
import torch.nn.functional as F 
from torch_geometric.nn import GATConv 
from typing import List, Tuple, Optional, Dict
import numpy as np
from datetime import datetime


class TemporalGraphAttention(nn.Module):
    """
    Multi-head Graph Attention Network with temporal features
    Processes behavioral authentication data with attention mechanisms
    
    Architecture:
    - 2 stacked GAT layers (learn hierarchical patterns)
    - 4 attention heads per layer (parallel attention mechanisms)
    - Total: 8 attention weight matrices (2 layers × 4 heads)
    
    Why 2 layers?
    - Layer 1: Captures immediate neighbor patterns (1-hop)
    - Layer 2: Captures extended patterns from Layer 1 output (2-hop)
    - Similar to stacking CNN layers for hierarchical feature learning
    """
    
    def __init__(
        self,
        input_dim: int = 56,
        hidden_dim: int = 64,
        output_dim: int = 64,
        num_heads: int = 4,  # Changed from 8 to 4 as requested
        dropout: float = 0.1,
        temporal_dim: int = 16,
        return_attention_weights: bool = True  # Enable attention weight access
    ):
        super(TemporalGraphAttention, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_heads = num_heads
        self.temporal_dim = temporal_dim
        self.return_attention_weights = return_attention_weights
        
        # Storage for attention weights (populated during forward pass)
        self.attention_weights_layer1 = None
        self.attention_weights_layer2 = None
        
        # Temporal feature encoder
        self.temporal_encoder = nn.Sequential(
            nn.Linear(1, temporal_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Combined input dimension
        combined_dim = input_dim + temporal_dim
        
        # Multi-head GAT layers (4 heads each)
        # Layer 1: [combined_dim] → [hidden_dim]
        #   Each head outputs hidden_dim // num_heads dimensions
        #   Concatenated: 4 heads × 16-D = 64-D
        self.gat_layer1 = GATConv(
            combined_dim,
            hidden_dim // num_heads,
            heads=num_heads,
            dropout=dropout,
            concat=True  # Concatenate head outputs
        )
        
        # Layer 2: [hidden_dim] → [output_dim]
        #   Each head outputs output_dim // num_heads dimensions
        #   Concatenated: 4 heads × 16-D = 64-D
        self.gat_layer2 = GATConv(
            hidden_dim,
            output_dim // num_heads,
            heads=num_heads,
            dropout=dropout,
            concat=True
        )
        
        # Graph-level pooling
        self.global_pool = nn.Sequential(
            nn.Linear(output_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim)
        )
        
        # Layer normalization
        self.layer_norm1 = nn.LayerNorm(hidden_dim)
        self.layer_norm2 = nn.LayerNorm(output_dim)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(
        self, 
        x: torch.Tensor, 
        edge_index: torch.Tensor, 
        temporal_features: torch.Tensor,
        batch: Optional[torch.Tensor] = None,
        return_attention: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[Dict]]:
        """
        Forward pass through GAT with optional attention weight extraction
        
        Args:
            x: Node features [num_nodes, input_dim]
            edge_index: Graph connectivity [2, num_edges]
            temporal_features: Temporal data [num_nodes, 1]
            batch: Batch assignment for multiple graphs
            return_attention: If True, return attention weights for analysis
            
        Returns:
            node_embeddings: Node-level embeddings [num_nodes, output_dim]
            graph_embedding: Graph-level embedding [output_dim]
            attention_weights: Dict with layer1/layer2 attention (if return_attention=True)
                - 'layer1': Tuple of (edge_index, attention_values) for layer 1
                - 'layer2': Tuple of (edge_index, attention_values) for layer 2
                - attention_values shape: [num_edges, num_heads]
        """
        # Encode temporal features
        temporal_encoded = self.temporal_encoder(temporal_features)
        
        # Combine behavioral and temporal features
        x_combined = torch.cat([x, temporal_encoded], dim=1)
        
        # First GAT layer - with attention weights
        if return_attention and self.return_attention_weights:
            x1, attention1 = self.gat_layer1(x_combined, edge_index, return_attention_weights=True)
            self.attention_weights_layer1 = attention1  # Store for inspection
        else:
            x1 = self.gat_layer1(x_combined, edge_index)
            attention1 = None
            
        x1 = self.layer_norm1(x1)
        x1 = F.relu(x1)
        x1 = self.dropout(x1)
        
        # Second GAT layer - with attention weights
        if return_attention and self.return_attention_weights:
            x2, attention2 = self.gat_layer2(x1, edge_index, return_attention_weights=True)
            self.attention_weights_layer2 = attention2  # Store for inspection
        else:
            x2 = self.gat_layer2(x1, edge_index)
            attention2 = None
            
        x2 = self.layer_norm2(x2)
        x2 = F.relu(x2)
        
        # Global pooling for graph-level representation
        if batch is not None:
            # Multiple graphs in batch
            from torch_geometric.nn import global_mean_pool  
            graph_embedding = global_mean_pool(x2, batch)
        else:
            # Single graph
            graph_embedding = torch.mean(x2, dim=0, keepdim=True)
        
        graph_embedding = self.global_pool(graph_embedding)
        
        # Prepare attention output
        attention_dict = None
        if return_attention and attention1 is not None and attention2 is not None:
            attention_dict = {
                'layer1': attention1,  # (edge_index, attention_weights)
                'layer2': attention2
            }
        
        return x2, graph_embedding.squeeze(0), attention_dict
    
    def get_attention_weights(self) -> Dict:
        """
        Get stored attention weights from last forward pass
        
        Returns:
            Dictionary with 'layer1' and 'layer2' attention weights
            Each contains (edge_index, attention_values) tuple
            attention_values shape: [num_edges, num_heads=4]
        """
        return {
            'layer1': self.attention_weights_layer1,
            'layer2': self.attention_weights_layer2
        }
    
    def visualize_attention(self, node_id: int, layer: int = 1) -> Dict:
        """
        Get attention weights for a specific node
        
        Args:
            node_id: Target node to inspect
            layer: Which GAT layer (1 or 2)
            
        Returns:
            Dictionary with attention scores from this node to its neighbors
            Format: {
                'source_node': node_id,
                'neighbors': [list of neighbor node IDs],
                'attention_per_head': [num_neighbors, num_heads=4],
                'attention_mean': [num_neighbors] (averaged across heads)
            }
        """
        weights = self.attention_weights_layer1 if layer == 1 else self.attention_weights_layer2
        
        if weights is None:
            return {'error': 'No attention weights available. Run forward() with return_attention=True first'}
        
        edge_index, attention_values = weights
        
        # Find edges where source is node_id
        mask = edge_index[0] == node_id
        neighbor_nodes = edge_index[1][mask].cpu().numpy()
        neighbor_attention = attention_values[mask].cpu().detach().numpy()  # [num_neighbors, num_heads]
        
        return {
            'source_node': node_id,
            'layer': layer,
            'num_heads': self.num_heads,
            'neighbors': neighbor_nodes.tolist(),
            'attention_per_head': neighbor_attention.tolist(),
            'attention_mean': neighbor_attention.mean(axis=1).tolist()
        }


class SiameseGATNetwork(nn.Module):
    """
    Siamese network using GAT for metric learning
    Learns to distinguish legitimate vs anomalous behavioral patterns
    """
    
    def __init__(self, gat_config: Dict):
        super(SiameseGATNetwork, self).__init__()
        
        self.gat_network = TemporalGraphAttention(**gat_config)
        
        # Similarity computation layer
        self.similarity_layer = nn.Sequential(
            nn.Linear(gat_config['output_dim'] * 2, gat_config['hidden_dim']),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(gat_config['hidden_dim'], 1),
            nn.Sigmoid()
        )
        
    def forward_once(self, x, edge_index, temporal_features, batch=None, return_attention=False):
        """
        Single forward pass through GAT
        
        Args:
            return_attention: If True, also return attention weights
            
        Returns:
            graph_emb: Graph-level embedding (Tensor)
            attention_dict: (Optional) Attention weights if return_attention=True
        """
        node_emb, graph_emb, attention_dict = self.gat_network(
            x, edge_index, temporal_features, batch, return_attention=return_attention
        )
        
        if return_attention:
            return graph_emb, attention_dict  # type: ignore
        return graph_emb  # type: ignore
    
    def forward(self, data1, data2, return_attention=False):
        """
        Siamese forward pass
        
        Args:
            data1: First graph data
            data2: Second graph data
            return_attention: If True, also return attention weights
            
        Returns:
            similarity_score: Cosine similarity between embeddings
            emb1, emb2: Graph embeddings
            attention_data: (Optional) Dict with attention weights if return_attention=True
        """
        # Get embeddings for both graphs (no attention during training for speed)
        emb1 = self.forward_once(
            data1.x, data1.edge_index, 
            data1.temporal_features, data1.batch,
            return_attention=False  # Keep False during training
        )
        emb2 = self.forward_once(
            data2.x, data2.edge_index, 
            data2.temporal_features, data2.batch,
            return_attention=False
        )
        
        # Compute cosine similarity
        similarity = F.cosine_similarity(emb1, emb2, dim=-1)
        
        if return_attention:
            # Get attention for first graph only (for visualization)
            _, attention_dict = self.forward_once(
                data1.x, data1.edge_index,
                data1.temporal_features, data1.batch,
                return_attention=True
            )
            return similarity, emb1, emb2, attention_dict
        
        return similarity, emb1, emb2


class GATTrainer:
    """
    Training manager for GAT network
    Implements triplet loss for metric learning
    """
    
    def __init__(
        self,
        model: SiameseGATNetwork,
        learning_rate: float = 0.001,
        margin: float = 0.2,
        device: str = 'cpu'
    ):
        self.model = model
        self.device = device
        self.model.to(device)
        
        self.optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        self.scheduler = torch.optim.lr_scheduler.StepLR(
            self.optimizer, step_size=10, gamma=0.9
        )
        
        self.margin = margin
        self.triplet_loss = nn.TripletMarginLoss(margin=margin)
        
    def triplet_loss_custom(self, anchor, positive, negative):
        """Custom triplet loss with cosine similarity"""
        pos_sim = F.cosine_similarity(anchor, positive, dim=-1)
        neg_sim = F.cosine_similarity(anchor, negative, dim=-1)
        
        loss = torch.clamp(neg_sim - pos_sim + self.margin, min=0.0)
        return loss.mean()
    
    def train_batch(self, anchor_data, positive_data, negative_data):
        """Train on single batch of triplets"""
        self.model.train()
        self.optimizer.zero_grad()
        
        # Get embeddings
        anchor_emb = self.model.forward_once(
            anchor_data.x, anchor_data.edge_index,
            anchor_data.temporal_features, anchor_data.batch
        )
        positive_emb = self.model.forward_once(
            positive_data.x, positive_data.edge_index,
            positive_data.temporal_features, positive_data.batch
        )
        negative_emb = self.model.forward_once(
            negative_data.x, negative_data.edge_index,
            negative_data.temporal_features, negative_data.batch
        )
        
        # Compute triplet loss
        loss = self.triplet_loss_custom(anchor_emb, positive_emb, negative_emb)
        
        # Backward pass
        loss.backward()
        self.optimizer.step()
        
        return loss.item()
    
    def evaluate(self, test_data):
        """Evaluate model on test set"""
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for anchor, positive, negative in test_data:
                anchor_emb = self.model.forward_once(
                    anchor.x, anchor.edge_index,
                    anchor.temporal_features, anchor.batch
                )
                positive_emb = self.model.forward_once(
                    positive.x, positive.edge_index,
                    positive.temporal_features, positive.batch
                )
                negative_emb = self.model.forward_once(
                    negative.x, negative.edge_index,
                    negative.temporal_features, negative.batch
                )
                
                loss = self.triplet_loss_custom(anchor_emb, positive_emb, negative_emb)
                total_loss += loss.item()
                num_batches += 1
        
        return total_loss / max(num_batches, 1)


class GATInferenceEngine:
    """
    Inference engine for behavioral authentication
    """
    
    def __init__(self, model: SiameseGATNetwork, device: str = 'cpu'):
        self.model = model
        self.device = device
        self.model.to(device)
        self.model.eval()
        
    def compute_session_embedding(self, graph_data, return_attention=False):
        """
        Compute embedding for a session graph
        
        Args:
            graph_data: PyTorch Geometric graph data
            return_attention: If True, also return attention weights
            
        Returns:
            embedding: numpy array [output_dim]
            attention_dict: (Optional) if return_attention=True
        """
        with torch.no_grad():
            result = self.model.forward_once(
                graph_data.x, graph_data.edge_index,
                graph_data.temporal_features, graph_data.batch,
                return_attention=return_attention
            )
            
            if return_attention:
                embedding, attention_dict = result  # type: ignore
                return embedding.cpu().numpy(), attention_dict
            else:
                embedding = result  # type: ignore
                return embedding.cpu().numpy()
    
    def authenticate(
        self, 
        session_graph,
        user_profile_vector: np.ndarray,
        threshold: float = 0.85,
        return_attention: bool = False
    ) -> Dict:
        """
        Perform authentication decision
        
        Args:
            session_graph: Current session graph data
            user_profile_vector: User's enrolled profile
            threshold: Authentication threshold
            return_attention: If True, include attention weights in response
            
        Returns:
            Authentication result with decision and confidence
            (Optional) attention_weights if return_attention=True
        """
        start_time = datetime.now()
        
        # Get session embedding (and optionally attention)
        result = self.compute_session_embedding(session_graph, return_attention=return_attention)
        
        if return_attention:
            session_embedding, attention_dict = result  # type: ignore
        else:
            session_embedding = result  # type: ignore
            attention_dict = None
        
        # Compute similarity with profile
        similarity = float(np.dot(session_embedding, user_profile_vector) /   # type: ignore
                          (np.linalg.norm(session_embedding) * np.linalg.norm(user_profile_vector)))
        
        # Authentication decision
        if similarity >= threshold:
            decision = "ALLOW"
            confidence = similarity
        elif similarity < threshold * 0.7:
            decision = "BLOCK"
            confidence = 1.0 - similarity
        else:
            decision = "UNCERTAIN"
            confidence = abs(similarity - threshold) / threshold
        
        processing_time = (datetime.now() - start_time).total_seconds() * 1000
        
        response = {
            "session_vector": session_embedding.tolist(),  # type: ignore
            "similarity_score": similarity,
            "auth_decision": decision,
            "confidence": confidence,
            "processing_time_ms": processing_time
        }
        
        if return_attention and attention_dict is not None:
            response["attention_weights"] = attention_dict  # type: ignore
        
        return response