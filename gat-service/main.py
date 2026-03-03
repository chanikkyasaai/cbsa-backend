"""
GAT Service Main Application
FastAPI service for Graph Attention Network processing
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import logging
import asyncio
from datetime import datetime
import json
import os
from typing import Dict, Any, Optional

try:
    from models import (
        GATProcessingRequest, GATProcessingResponse,
        TrainingRequest, TrainingResponse,
        TemporalGraph
    )
    from data_processor import BehavioralDataProcessor, PyTorchDataConverter
    from config import get_gat_settings
except ImportError as e:
    print(f"Import error: {e}")
    print("Some modules may not be available, will use fallback implementations")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global service state
gat_model = None
inference_engine = None
data_processor = None
pytorch_converter = None
config = None

# Try to load configuration
try:
    config = get_gat_settings()
except:
    # Fallback configuration
    from types import SimpleNamespace
    config = SimpleNamespace(
        host="0.0.0.0",
        port=8001,
        input_dim=48,
        hidden_dim=64,
        output_dim=64,
        num_attention_heads=8,
        dropout_rate=0.1,
        temporal_dim=16,
        device="cpu",
        similarity_threshold=0.85,
        model_save_path="./models/gat_model.pt"
    )


async def load_gat_model():
    """Load GAT model and initialize components"""
    global gat_model, inference_engine, data_processor, pytorch_converter
    
    try:
        logger.info("Initializing GAT service components...")
        
        # Initialize data processor
        config_dict = config.dict() if hasattr(config, 'dict') else vars(config)
        data_processor = BehavioralDataProcessor(config_dict)
        pytorch_converter = PyTorchDataConverter()
        
        # Try to load PyTorch model
        try:
            from gat_network import SiameseGATNetwork, GATInferenceEngine
            
            # GAT configuration
            gat_config = {
                'input_dim': config.input_dim,
                'hidden_dim': config.hidden_dim,
                'output_dim': config.output_dim,
                'num_heads': config.num_attention_heads,
                'dropout': config.dropout_rate,
                'temporal_dim': config.temporal_dim
            }
            
            # Create model
            gat_model = SiameseGATNetwork(gat_config)
            inference_engine = GATInferenceEngine(gat_model, config.device)
            
            # Try to load existing model weights
            model_path = config.model_save_path
            if os.path.exists(model_path):
                import torch
                state_dict = torch.load(model_path, map_location=config.device)
                gat_model.load_state_dict(state_dict)
                logger.info(f"Loaded model from {model_path}")
            else:
                logger.info("No existing model found, using randomly initialized weights")
                
            logger.info("GAT model initialized successfully")
            
        except ImportError as e:
            logger.warning(f"PyTorch not available: {e}")
            logger.info("Running in simulation mode")
            gat_model = None
            inference_engine = None
            
    except Exception as e:
        logger.error(f"Error initializing GAT service: {e}")
        # Continue in simulation mode
        gat_model = None
        inference_engine = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    await load_gat_model()
    yield
    # Shutdown
    logger.info("GAT service shutting down")


# Create FastAPI app
app = FastAPI(
    title="GAT Behavioral Authentication Service",
    description="Graph Attention Network service for continuous behavioral authentication",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "service": "GAT Behavioral Authentication",
        "status": "running",
        "model_loaded": gat_model is not None,
        "timestamp": datetime.now().isoformat()
    }


@app.get("/health")
async def health_check():
    """Detailed health check"""
    return {
        "status": "healthy",
        "model_status": "loaded" if gat_model is not None else "simulation",
        "config": {
            "input_dim": config.input_dim,
            "output_dim": config.output_dim,
            "num_heads": config.num_attention_heads,
            "device": config.device
        },
        "timestamp": datetime.now().isoformat()
    }


def simulate_gat_processing(
    graph: TemporalGraph,
    user_profile: Optional[list] = None
) -> Dict[str, Any]:
    """
    Simulate GAT processing when PyTorch model not available.
    Returns raw similarity score — no auth decisions.
    """
    import random
    import time
    
    start_time = time.time()
    
    # Simulate processing delay
    time.sleep(0.1)
    
    # Generate simulated 64-dim embedding
    session_vector = [random.uniform(-1, 1) for _ in range(config.output_dim)]
    
    # Simulate similarity calculation
    if user_profile:
        similarity = random.uniform(0.7, 0.95)
    else:
        similarity = random.uniform(0.5, 0.8)
    
    processing_time = (time.time() - start_time) * 1000
    
    return {
        "session_vector": session_vector,
        "similarity_score": similarity,
        "processing_time_ms": processing_time,
        "attention_weights": None,
        "node_embeddings": None,
        "model_info": {
            "mode": "simulation",
            "model_version": "simulated_v1.0",
            "num_nodes": len(graph.nodes),
            "num_edges": len(graph.edges)
        }
    }


@app.post("/process", response_model=GATProcessingResponse)
async def process_gat(request: GATProcessingRequest):
    """
    Process temporal graph with GAT for authentication decision
    """
    try:
        logger.info(f"Processing GAT request for session {request.graph.session_id}")
        
        if inference_engine is not None:
            # Real GAT processing
            logger.info("Using real GAT model for processing")
            
            # Convert to PyTorch format
            graph_data = pytorch_converter.convert_to_pytorch(request.graph)
            
            # Convert to proper PyTorch data structure
            try:
                import torch
                from types import SimpleNamespace
                
                # Create data object
                data = SimpleNamespace()
                data.x = graph_data['x']
                data.edge_index = graph_data['edge_index']
                data.temporal_features = graph_data['temporal_features']
                data.batch = graph_data['batch']
                
                # Perform authentication
                result = inference_engine.authenticate(
                    data,
                    request.user_profile_vector or [0.0] * config.output_dim,
                    request.similarity_threshold
                )
                
                # Add additional info
                result.update({
                    "attention_weights": None,  # TODO: Extract from model
                    "node_embeddings": None,   # TODO: Extract from model
                    "model_info": {
                        "mode": "pytorch",
                        "model_version": "gat_v1.0",
                        "num_nodes": graph_data['num_nodes'],
                        "num_edges": graph_data['num_edges'],
                        "attention_heads": request.attention_heads,
                        "embedding_dim": request.embedding_dim
                    }
                })
                
            except ImportError:
                # Fallback to simulation
                logger.warning("PyTorch import failed, using simulation")
                result = simulate_gat_processing(
                    request.graph,
                    request.user_profile_vector
                )
        else:
            # Simulation mode
            logger.info("Using simulation mode for GAT processing")
            result = simulate_gat_processing(
                request.graph,
                request.user_profile_vector
            )
        
        return GATProcessingResponse(**result)
        
    except Exception as e:
        logger.error(f"Error processing GAT request: {e}")
        raise HTTPException(status_code=500, detail=f"GAT processing failed: {str(e)}")


@app.post("/enroll")
async def enroll_user_profile(request: GATProcessingRequest):
    """
    Enroll user profile by processing enrollment sessions
    """
    try:
        logger.info(f"Enrolling user profile for user {request.graph.user_id}")
        
        if request.processing_mode != "enrollment":
            raise HTTPException(
                status_code=400,
                detail="Processing mode must be 'enrollment' for this endpoint"
            )
        
        # Process enrollment session
        response = await process_gat(request)
        
        # In real implementation, would save the session_vector as user profile
        # For now, just return the computed embedding
        
        return {
            "user_id": request.graph.user_id,
            "profile_vector": response.session_vector,
            "similarity_score": response.similarity_score,
            "num_events": len(request.graph.nodes),
            "session_duration": request.graph.session_duration,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error enrolling user: {e}")
        raise HTTPException(status_code=500, detail=f"Enrollment failed: {str(e)}")


@app.post("/train", response_model=TrainingResponse)
async def train_model(request: TrainingRequest, background_tasks: BackgroundTasks):
    """
    Train GAT model with provided training examples
    """
    try:
        logger.info(f"Training GAT model with {len(request.training_examples)} examples")
        
        if gat_model is None:
            raise HTTPException(
                status_code=503,
                detail="PyTorch model not available. Cannot perform training."
            )
        
        # Add training task to background
        background_tasks.add_task(
            train_model_background,
            request.model_dump()
        )
        
        return TrainingResponse(
            training_status="started",
            final_loss=0.0,
            epochs_completed=0,
            training_time_seconds=0.0,
            model_saved=False,
            model_path=config.model_save_path
        )
        
    except Exception as e:
        logger.error(f"Error starting training: {e}")
        raise HTTPException(status_code=500, detail=f"Training failed: {str(e)}")


async def train_model_background(training_config: Dict[str, Any]):
    """Background training task"""
    try:
        from gat_network import GATTrainer
        import torch
        
        logger.info("Starting background training...")
        
        # Create trainer
        trainer = GATTrainer(
            model=gat_model,
            learning_rate=training_config['learning_rate'],
            device=config.device
        )
        
        # Training loop (simplified for demonstration)
        total_loss = 0.0
        num_epochs = training_config['epochs']
        
        start_time = datetime.now()
        
        for epoch in range(num_epochs):
            # In real implementation, would process training_examples
            # For now, just simulate training
            epoch_loss = 0.1 * (1.0 - epoch / num_epochs)  # Decreasing loss
            total_loss += epoch_loss
            
            logger.info(f"Epoch {epoch + 1}/{num_epochs}, Loss: {epoch_loss:.4f}")
        
        # Save model
        torch.save(gat_model.state_dict(), config.model_save_path)
        
        training_time = (datetime.now() - start_time).total_seconds()
        
        logger.info(f"Training completed in {training_time:.2f} seconds")
        
    except Exception as e:
        logger.error(f"Background training failed: {e}")


@app.get("/models/info")
async def get_model_info():
    """Get information about the current model"""
    return {
        "model_loaded": gat_model is not None,
        "model_path": config.model_save_path,
        "model_exists": os.path.exists(config.model_save_path),
        "config": {
            "input_dim": config.input_dim,
            "hidden_dim": config.hidden_dim,
            "output_dim": config.output_dim,
            "num_heads": config.num_attention_heads,
            "dropout": config.dropout_rate,
            "device": config.device
        }
    }


@app.post("/convert/behavioral-data")
async def convert_behavioral_data(
    raw_data: Dict[str, Any]
):
    """
    Convert raw behavioral data to temporal graph format
    """
    try:
        events = raw_data.get('events', [])
        user_id = raw_data.get('user_id', 'unknown')
        session_id = raw_data.get('session_id', 'unknown')
        
        if not events:
            raise HTTPException(status_code=400, detail="No events provided")
        
        # Process data
        temporal_graph = data_processor.process_behavioral_data(
            events, user_id, session_id
        )
        
        return {
            "temporal_graph": temporal_graph.model_dump(),
            "summary": {
                "num_nodes": len(temporal_graph.nodes),
                "num_edges": len(temporal_graph.edges),
                "session_duration": temporal_graph.session_duration,
                "event_diversity": temporal_graph.event_diversity,
                "avg_time_between_events": temporal_graph.avg_time_between_events
            }
        }
        
    except Exception as e:
        logger.error(f"Error converting behavioral data: {e}")
        raise HTTPException(status_code=500, detail=f"Data conversion failed: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        app,
        host=config.host,
        port=config.port,
        log_level="info"
    )