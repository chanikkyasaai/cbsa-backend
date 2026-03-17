"""
app.gat – Graph Attention Network sub-package

Contains the full GAT model stack:
  - gat_network.py     : SiameseGATNetwork, GATInferenceEngine, GATTrainer
  - data_processor.py  : BehavioralDataProcessor, PyTorchDataConverter
  - config.py          : GAT model hyperparameters
  - models.py          : Internal GAT data models
  - engine.py          : InternalGATEngine — in-process inference with PyTorch fallback
  - trainer.py         : TripletTrainer — triplet-loss GAT training on user behavioral data
"""
from app.gat.engine import InternalGATEngine, get_internal_engine
from app.gat.trainer import TripletTrainer, triplet_trainer, CHECKPOINT_PATH

__all__ = [
    "InternalGATEngine", "get_internal_engine",
    "TripletTrainer", "triplet_trainer", "CHECKPOINT_PATH",
]
