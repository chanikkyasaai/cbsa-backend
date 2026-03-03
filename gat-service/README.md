# GAT Behavioral Authentication Service

A Graph Attention Network (GAT) based microservice for continuous behavioral authentication using mobile sensor data and user interaction patterns.

## Overview

This service implements a sophisticated multi-layer behavioral authentication system using Graph Attention Networks to process temporal behavioral data from mobile devices. It provides real-time authentication decisions based on learned behavioral patterns.

## Architecture

### Core Components

1. **GAT Neural Network** (`gat_network.py`)
   - Multi-head Graph Attention Network with temporal processing
   - Siamese network architecture for metric learning
   - PyTorch implementation with torch-geometric

2. **Data Processing** (`data_processor.py`)
  - Converts raw behavioral data to temporal graphs
  - 60-dimensional node features (48 event + 8 event type + 4 device context)
  - Graph edge creation with temporal relationships

3. **API Service** (`main.py`)
   - FastAPI-based REST API
   - Real-time processing endpoints
   - User enrollment and authentication

4. **Configuration** (`config.py`)
   - Service settings and hyperparameters
   - Model configuration parameters

## Features

### Behavioral Authentication
- **Multi-modal sensing**: Touch dynamics, motion patterns, typing rhythms, app usage
- **Temporal graph modeling**: Events as nodes with temporal edges
- **Attention mechanisms**: Multi-head attention for feature importance
- **Continuous authentication**: Real-time decision making

### API Endpoints

- `GET /health` - Service health check
- `POST /process` - GAT processing for authentication
- `POST /enroll` - User profile enrollment
- `POST /train` - Model training (background)
- `POST /convert/behavioral-data` - Raw data to graph conversion

### Data Format

#### Input: Raw Behavioral Events
```json
{
  "events": [
    {
      "timestamp": 1703123456.789,
      "type": "touch",
      "touch": {
        "pressure": 0.75,
        "size": 0.3,
        "x_position": 540,
        "y_position": 960,
        "duration": 0.15
      },
      "motion": {
        "acc_x": 0.2,
        "acc_y": -9.8,
        "acc_z": 0.1,
        "gyro_x": 0.05,
        "gyro_y": -0.02,
        "gyro_z": 0.01
      },
      "typing": {
        "dwell_time": 0.12,
        "flight_time": 0.03,
        "typing_speed": 45.2
      },
      "app_usage": {
        "session_duration": 120.5,
        "interaction_frequency": 3.2
      }
    }
  ],
  "user_id": "user123",
  "session_id": "session456"
}
```

#### Output: Authentication Decision
```json
{
  "session_vector": [0.12, -0.45, 0.78, ...],  // 64-dim embedding
  "similarity_score": 0.89,
  "auth_decision": "ALLOW",
  "confidence": 0.91,
  "processing_time_ms": 45.2,
  "model_info": {
    "mode": "pytorch",
    "num_nodes": 20,
    "num_edges": 35
  }
}
```

## Installation

### Prerequisites
- Python 3.8+
- pip package manager

### Quick Start
```bash
# Clone or copy the service files
cd gat-service

# Run the startup script (installs dependencies and starts service)
python start_service.py
```

### Manual Installation
```bash
# Install dependencies
pip install -r requirements.txt

# For full PyTorch functionality (optional)
pip install torch torch-geometric

# Start the service
python main.py
```

## Configuration

Edit `config.py` to customize:

```python
class GATSettings(BaseSettings):
    # Service settings
    host: str = "0.0.0.0"
    port: int = 8001
    
    # Model parameters
  input_dim: int = 60          # Node feature dimension (48 event + 8 type + 4 device)
    hidden_dim: int = 64         # Hidden layer size
    output_dim: int = 64         # Output embedding dimension
    num_attention_heads: int = 8 # Multi-head attention
    
    # Authentication thresholds
    similarity_threshold: float = 0.85
    confidence_threshold: float = 0.7
```

## Behavioral Features (60 dimensions)

### 1. Touch Dynamics (12 dims)
- Pressure, contact size, major/minor axis
- Touch position and velocity
- Touch duration and acceleration

### 2. Motion Patterns (12 dims)
- Accelerometer (x, y, z)
- Gyroscope (x, y, z)
- Magnetometer (x, y, z)
- Linear acceleration (x, y, z)

### 3. Typing Patterns (12 dims)
- Dwell time, flight time, typing speed
- Pressure variance, rhythm score
- Key hold patterns, error rates

### 4. App Usage Patterns (12 dims)

### 5. Event Type Embedding (8 dims)
- Deterministic 8‑D embedding derived from event type string

### 6. Device Context (4 dims)
- Battery level, CPU usage, memory usage, network strength (normalized)
- App transition times
- Interaction frequency and intensity
- Navigation patterns, gesture complexity

## Model Architecture

### Graph Attention Network
- **Input**: 60-dim node features (48 event + 8 type + 4 device) + temporal features
- **Processing**: Multi-head GAT layers with temporal encoding
- **Output**: 64-dim session embeddings

### Training Process
- **Siamese network**: Learns similarity metrics
- **Triplet loss**: Anchor-positive-negative training
- **Metric learning**: Distinguishes legitimate vs. anomalous behavior

## Operating Modes

### 1. PyTorch Mode (Full Functionality)
- Real neural network processing
- Model training capability
- Attention visualization

### 2. Simulation Mode (Fallback)
- Runs without PyTorch dependencies
- Simulated authentication decisions
- Useful for testing and development

## Usage Examples

### Basic Authentication
```python
import requests

# Raw behavioral data
data = {
    "events": [...],  # Behavioral events
    "user_id": "user123",
    "session_id": "session456"
}

# Convert to graph format
graph_response = requests.post(
    "http://localhost:8001/convert/behavioral-data",
    json=data
)
temporal_graph = graph_response.json()["temporal_graph"]

# Authenticate
auth_request = {
    "graph": temporal_graph,
    "user_profile_vector": [...],  # 64-dim user profile
    "processing_mode": "inference",
    "similarity_threshold": 0.85
}

auth_response = requests.post(
    "http://localhost:8001/process",
    json=auth_request
)

result = auth_response.json()
print(f"Decision: {result['auth_decision']}")
print(f"Confidence: {result['confidence']}")
```

### User Enrollment
```python
# Enroll new user
enrollment_request = {
    "graph": temporal_graph,
    "processing_mode": "enrollment"
}

enrollment_response = requests.post(
    "http://localhost:8001/enroll",
    json=enrollment_request
)

user_profile = enrollment_response.json()["profile_vector"]
# Save profile_vector for future authentication
```

## Testing

Run the comprehensive test suite:

```bash
python test_gat_service.py
```

Tests include:
- Health check validation
- Data conversion accuracy
- GAT processing functionality
- User enrollment workflow
- Performance benchmarking

## Performance

Expected performance (on modern hardware):
- **Processing time**: 20-100ms per authentication
- **Throughput**: 100+ requests/second
- **Memory usage**: <500MB (simulation), <1GB (PyTorch)

## Integration with CBSA Backend

The GAT service integrates with the main CBSA backend:

1. **Replace simulation**: Update `cbsa-backend/app/main.py` to call real GAT service
2. **Configure endpoint**: Set GAT service URL in backend configuration
3. **Error handling**: Implement fallback to simulation if GAT service unavailable

Example integration:
```python
# In cbsa-backend/app/main.py
async def call_gat_service(temporal_graph_data):
    try:
        response = await http_client.post(
            "http://localhost:8001/process",
            json=temporal_graph_data
        )
        return response.json()
    except Exception:
        # Fallback to simulation
        return simulate_layer3_processing(temporal_graph_data)
```

## Deployment

### Development
```bash
python start_service.py
```

### Production
```bash
# Using gunicorn for production
pip install gunicorn
gunicorn main:app -w 4 -k uvicorn.workers.UvicornWorker --bind 0.0.0.0:8001
```

### Docker (Optional)
```dockerfile
FROM python:3.9-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
EXPOSE 8001
CMD ["python", "main.py"]
```

## Monitoring

### Health Checks
- `GET /health` - Service status and model information
- `GET /models/info` - Detailed model configuration

### Logging
- Service logs written to console
- Configurable log levels in `main.py`
- Processing metrics included in responses

## Troubleshooting

### Common Issues

1. **PyTorch installation errors**
   - Solution: Service runs in simulation mode without PyTorch
   - Install PyTorch manually if needed

2. **Memory issues with large graphs**
   - Solution: Adjust `max_events_per_window` in config
   - Implement graph sampling for very large sessions

3. **Slow processing**
   - Check device configuration (CPU vs GPU)
   - Reduce attention heads or embedding dimensions
   - Implement batch processing for multiple requests

### Error Codes
- `500`: Internal processing error
- `400`: Invalid request format
- `503`: Model not available (training mode)

## Security Considerations

- **Data privacy**: Raw behavioral data never stored permanently
- **Model protection**: Trained models should be secured
- **API security**: Implement authentication for production use
- **Rate limiting**: Prevent abuse with request limiting

## Future Enhancements

1. **Advanced Features**
   - Dynamic attention mechanisms
   - Federated learning support
   - Multi-user session handling

2. **Performance Optimizations**
   - Model quantization
   - TensorRT optimization
   - Distributed processing

3. **Monitoring & Analytics**
   - Authentication analytics dashboard
   - Model drift detection
   - Performance monitoring

## Support

For issues or questions:
1. Check the test suite output
2. Review service logs
3. Validate input data format
4. Ensure proper Python environment

---

**Note**: This is a research/demonstration implementation. For production use, additional security hardening, testing, and optimization would be required.