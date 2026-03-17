# CBSA Backend — Continuous Behavioral Streaming Authentication

A production-grade FastAPI backend for real-time behavioral authentication from mobile applications. The system continuously authenticates users by analyzing behavioral event streams via a four-layer processing pipeline: secure ingestion, prototype-based behavioral modeling, Graph Attention Network deep analysis, and EMA trust scoring.

---

## Features

- **WebSocket-based streaming**: Real-time continuous event ingestion from mobile clients
- **Leakage-free preprocessing**: Drift, stability, and similarity metrics computed against pre-update statistics
- **Prototype quarantine**: New behavioral archetypes require 3 observations, 30s temporal spread, and consistency >= 0.72 before promotion
- **GAT deep analysis**: In-process PyTorch Graph Attention Network on temporal behavioral graphs
- **Adaptive trust model**: EMA trust score with alpha tied to short-term drift
- **Azure Cosmos DB**: Computation logs, user profiles, and enrollment state
- **Azure Blob Storage**: Model checkpoint persistence and retrieval
- **Admin endpoints**: Per-user data wipe and full system truncate

---

## Project Structure

```
cbsa-backend/
├── app/
│   ├── main.py               # FastAPI application, WebSocket + admin endpoints
│   ├── config.py             # Application configuration and environment variables
│   ├── api/
│   │   └── websocket_manager.py   # Multi-client WebSocket connection manager
│   ├── ingestion/
│   │   └── ingestion.py      # Layer 1: event validation and extraction
│   ├── preprocessing/
│   │   ├── preprocessing.py  # Layer 2a: drift, stability, window vector
│   │   ├── buffer_manager.py # Leakage-free session buffer (Welford's algorithm)
│   │   └── drift_engine.py   # Exp-normalized drift and stability computation
│   ├── prototype/
│   │   ├── prototype_engine.py    # Layer 2b: matching, adaptive update, lifecycle
│   │   ├── similarity_engine.py   # Composite similarity (cosine + Mahalanobis + stability)
│   │   └── quarantine_manager.py  # Quarantine-gated prototype promotion
│   ├── gat/
│   │   ├── gat_network.py    # SiameseGATNetwork (56-D input, 64-D output)
│   │   ├── data_processor.py # Behavioral graph construction for GAT
│   │   ├── config.py         # GAT hyperparameters
│   │   ├── models.py         # GAT internal data models
│   │   ├── engine.py         # In-process GAT inference engine
│   │   └── trainer.py        # Triplet loss training across all users
│   ├── layer3/
│   │   ├── layer3_manager.py    # Layer 3 orchestration and escalation
│   │   ├── layer3_processor.py  # Graph construction from session window
│   │   ├── layer3_cloud.py      # Profile store integration
│   │   └── layer3_models.py     # Layer 3 Pydantic models
│   ├── trust/
│   │   └── trust_engine.py   # Layer 4: EMA trust model, zone decisions
│   ├── azure/
│   │   ├── behavioral_logger.py    # Per-user JSONL event logger
│   │   ├── blob_model_store.py     # Azure Blob Storage for .pth checkpoints
│   │   ├── cosmos_logger.py        # Cosmos DB computation log writer
│   │   ├── cosmos_profile_store.py # Cosmos DB 64-D user profile store
│   │   └── enrollment_store.py     # Enrollment state tracker
│   ├── core/
│   │   ├── constants.py      # All numeric constants in one place
│   │   └── invariants.py     # Runtime bounds checks (InvariantError)
│   ├── logging/
│   │   └── structured_logger.py  # Structured per-event audit logging
│   ├── models/
│   │   ├── behaviour_event.py        # BehaviourEvent Pydantic model
│   │   └── preprocessed_behaviour.py # PreprocessedBehaviour model
│   └── storage/
│       ├── cosmos_unified_store.py   # Unified Cosmos DB + SQLite store
│       ├── memory_store.py           # In-memory session state
│       └── repository.py             # BehaviourRepository (unified access)
│
├── tests/
│   ├── runner.py             # End-to-end test scenarios (standard, attack, cold_start, failure)
│   └── evaluation.py         # FAR / FRR / EER evaluation engine
│
├── docs/
│   ├── gat/
│   │   ├── GAT_COMPLETE_GUIDE.md    # GAT architecture and training reference
│   │   └── GAT_EDGE_WINDOWING.md   # Edge construction and windowing rules
│   ├── api/
│   │   └── API_SPEC.md              # WebSocket and REST API specification
│   └── SETUP.md                     # Installation and run guide
│
├── data/
│   ├── profiles/             # Local user profile vectors (development fallback)
│   ├── behavioral_logs/      # Per-user JSONL behavioral event logs
│   ├── checkpoints/          # Local model checkpoints (development fallback)
│   └── samples/              # Behavioral sample data for testing
│
├── CBSA_ARCHITECTURE_DESIGN.md   # Full system architecture and design rationale
├── requirements.txt
├── Dockerfile
└── docker-compose.yml
```

---

## Quick Start

See [docs/SETUP.md](docs/SETUP.md) for full setup instructions.

```bash
# Create and activate virtual environment
python -m venv .venv
.\.venv\Scripts\Activate.ps1    # Windows
source .venv/bin/activate        # macOS/Linux

# Install dependencies
pip install -r requirements.txt

# Start the backend
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

---

## API Endpoints

### HTTP

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Server info and active connection count |
| `/health` | GET | Health check |
| `/event-flow-map` | GET | Event flow map JSON for monitoring UI |
| `/admin/user/{user_id}` | DELETE | Delete all data for a user |
| `/admin/truncate` | DELETE | Delete all data for all users |

### WebSocket

| Endpoint | Description |
|----------|-------------|
| `ws://.../ws/behaviour` | Primary behavioral event stream |
| `ws://.../ws/monitor` | Live event monitoring feed |

Full API specification: [docs/api/API_SPEC.md](docs/api/API_SPEC.md)

---

## Behavioral Event Format

```json
{
  "username":   "alice",
  "session_id": "session_abc_123",
  "timestamp":  1708012800.123,
  "event_type": "scroll",
  "event_data": {
    "nonce":  "abc123unique",
    "vector": [0.12, 0.45, ..., 0.78]
  }
}
```

Requirements:
- `vector` must have exactly **48 values**, all in `[0.0, 1.0]`
- `nonce` must be unique per session
- `timestamp` must be monotonically increasing per session

---

## Configuration

**`app/config.py`** — server settings and Azure connection strings.

**`DEBUG_MODE`**:

| Setting | `True` (development) | `False` (production) |
|---------|---------------------|---------------------|
| User profiles | Cosmos DB + local `data/profiles/` | Cosmos DB only |
| Model checkpoints | Blob Storage + local `data/checkpoints/` | Blob Storage only |

### Environment Variables

| Variable | Purpose |
|----------|---------|
| `COSMOS_ENDPOINT` | Azure Cosmos DB account URI |
| `COSMOS_KEY` | Cosmos DB primary key |
| `COSMOS_DATABASE` | Database name (default: `cbsa-logs`) |
| `COSMOS_CONTAINER` | Computation logs container (default: `computation-logs`) |
| `COSMOS_PROFILES_CONTAINER` | User profiles container (default: `user-profiles`) |
| `AZURE_STORAGE_CONNECTION_STRING` | Full Blob Storage connection string |
| `AZURE_STORAGE_CONTAINER` | Model checkpoint container (default: `cbsa-models`) |

For local development, create a `.env` file at the project root.

---

## Docker

```bash
# Build and start both services
docker compose up --build

# Main backend:  http://localhost:8000/health
# Tear down:     docker compose down
```

---

## Architecture

See [CBSA_ARCHITECTURE_DESIGN.md](CBSA_ARCHITECTURE_DESIGN.md) for the complete architecture document covering all design decisions with mathematical justifications and alternatives considered.

---

## License

MIT
