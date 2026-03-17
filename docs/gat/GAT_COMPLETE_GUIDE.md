# GAT Complete Guide — CBSA Backend

> **Last updated to reflect current implementation.**
> The `gat-service` microservice no longer exists. All GAT logic now runs **in-process** inside the `app/` package. This document is the single authoritative reference for the full GAT workflow: feature engineering, enrollment, training, inference, persistence, and API.

---

## Table of Contents

1. [Architecture Overview](#1-architecture-overview)
2. [Feature Engineering — Node Vector (56-D)](#2-feature-engineering--node-vector-56-d)
3. [Temporal Graph Construction](#3-temporal-graph-construction)
4. [Windowing and Edge Rules](#4-windowing-and-edge-rules)
5. [User Enrollment Flow](#5-user-enrollment-flow)
6. [Training — Triplet Loss on Behavioral Logs](#6-training--triplet-loss-on-behavioral-logs)
7. [Inference Flow (In-Process GAT Engine)](#7-inference-flow-in-process-gat-engine)
8. [Model and Profile Persistence](#8-model-and-profile-persistence)
9. [REST API Reference](#9-rest-api-reference)
10. [Config Reference](#10-config-reference)
11. [Where to Tweak Embedding Dimensions](#11-where-to-tweak-embedding-dimensions)
12. [Quality Gates and Validation](#12-quality-gates-and-validation)
13. [Troubleshooting](#13-troubleshooting)
14. [Full File Index](#14-full-file-index)


---

## 1) Architecture Overview

```
Mobile Device
    │  JSON payload every ~1s (event_type, vector[48], nonce, …)
    ▼
WebSocket /ws/behaviour
    │
    ├─► [Layer 1 / 2] Validate + preprocess + prototype match → metrics response
    │
    └─► [Layer 3 GAT — in-process]
            │
            ├─ Build 20-second sliding event window per session
            ├─ Construct temporal graph (nodes = events, edges = attention spans)
            ├─ Run SiameseGATNetwork → 64-D session embedding
            └─ Cosine-similarity vs user profile → similarity_score
```

**Key architecture changes (vs older docs):**

| Old | Current |
|---|---|
| Separate `gat-service/` HTTP microservice | All GAT runs inside `app/gat/` package, in-process |
| 60-D node features (48 + 8 + 4 device) | **56-D node features** (48 + 8; device context removed) |
| Training via `gat-service/train_from_dataset.py` | Training via `POST /train` REST endpoint |
| Profiles in `profiles/` JSON files | Profiles persisted in **Azure Cosmos DB** (`user-profiles` container) |
| Model in `gat-service/checkpoints/` | Model persisted in **Azure Blob Storage** (`gat_checkpoint.pt`) |

---

## 2) Feature Engineering — Node Vector (56-D)

Each event becomes one node with a **56-dimensional feature vector**:

```
[ behavioral_vector (48-D) | event_type_embedding (8-D) ]
```

### Behavioral vector — 48-D
- Comes directly from `event_data.vector` in the WebSocket payload.
- Must be numeric, in `[0, 1]`, length exactly 48 (padded / truncated if needed).

### Event-type embedding — 8-D
- **Deterministic**, computed from `event_type` string:
  ```python
  digest = hashlib.sha256(event_type.encode("utf-8")).digest()
  embedding = [b / 255.0 for b in digest[:8]]
  ```
- Returns 8 floats in `[0, 1]`.
- Implemented in `app/layer3_processor.py → _event_type_embedding()` and `app/gat/data_processor.py → _event_type_embedding()`.

> **Note:** The 4-D device context vector (battery, CPU, memory, network) that appeared in earlier versions has been removed. The total is now **56**, not 60.

### Output of GAT
- **Session embedding:** 64-D float vector
- **Similarity score:** cosine similarity of session embedding vs user profile vector (float in `[0, 1]`)

---

## 3) Temporal Graph Construction

**Files:**
- `app/layer3_processor.py` — `GATDataProcessor.create_temporal_graph()` (used during live inference)
- `app/gat/data_processor.py` — `BehavioralDataProcessor.process_behavioral_data()` (used during training)

**Steps:**

1. Sort events by timestamp.
2. Apply 20-second time window (drop older events).
3. For each event, create a `GATEventNode` with `node_id`, `timestamp`, `event_type`, and 56-D `behavioral_vector`.
4. Build `GATTemporalEdge` objects using the edge rule (see Section 4).
5. Compute graph-level metadata: `session_duration`, `event_diversity`, `avg_time_between_events`.
6. Return a `GATGraph` / `TemporalGraph` Pydantic model.

**Graph model types:**

| Usage | Node model | Edge model | Graph model |
|---|---|---|---|
| Layer 3 inference | `GATEventNode` | `GATTemporalEdge` | `GATGraph` |
| Internal GAT package | `EventNode` | `TemporalEdge` | `TemporalGraph` |

Both sets are defined in `app/layer3_models.py` and `app/gat/models.py` respectively.

---

## 4) Windowing and Edge Rules

### Time-based window
- Keep **all events within the last 20 seconds** relative to the most recent event.
- Repeated event types are **preserved** (no de-duplication).
- Configured via `GAT_WINDOW_SECONDS = 20` in `app/config.py`.

### Edge construction rule (per node)
For each event node `i`:

```
seen_types = { event[i].type }

for j in i+1 … N-1:
    always connect(i → j)     # even if duplicate type

    if event[j].type not in seen_types:
        seen_types.add(event[j].type)
        if len(seen_types) >= 4:   # GAT_EDGE_DISTINCT_TARGET
            break                  # stop adding edges from node i
```

Key rules:
1. **Always** connect `i` to `j`, even for repeat event types.
2. Track **distinct** event types encountered from node `i`.
3. Stop only when **4 distinct types** have been seen (repeats do not count toward this).
4. Total edges from a node can exceed 4 (because repeats are kept).

**Example:**
Sequence: A, B, C, scroll×4, D, E, F, G

Edges from A:
- A→B (seen: {A, B})
- A→C (seen: {A, B, C})
- A→scroll ×4 (all connected, scroll is repeat, count stays at 3)
- A→D (seen: {A, B, C, D}) → 4th distinct → **stop**

Result: 7 edges from A, even though only 4 distinct types reached.

---

## 5) User Enrollment Flow

Enrollment collects **5 minutes of active behavioral data** (accumulated across sessions) before training is allowed.

### Step-by-step

```
1. Mobile app  →  POST /login { "username": "alice" }
   ← { "status": "enrolling", "time_remaining": 300 }
   (or "enrolled" if profile already exists)

2. Mobile app  →  WebSocket /ws/behaviour
   (send events continuously; each event is logged to Cosmos DB)

3. POST /logout { "username": "alice" }
   (saves accumulated enrollment seconds for the session)

4. Repeat steps 2–3 across multiple sessions until 5 minutes total active time.

5. POST /login { "username": "alice" }
   ← { "status": "enrolling", "message": "Enrollment data collected. Run /train to build your profile." }
   (timer has elapsed — enrollment complete, awaiting training)

6. POST /train { "user_id": "alice" }
   Authorization: Bearer <ADMIN_TOKEN>
   ← { "status": "success", "profile_saved": true, "sessions_used": N }
```

### Enrollment state storage
- **Primary:** Azure Cosmos DB (`enrollment-state` container) via `app/enrollment_store.py`
- **Fallback (DEBUG_MODE):** `data/enrollment_store.json`

### Behavioral log storage (used by training)
- **Primary:** Azure Cosmos DB (`behaviour-logs` container) via `app/behavioral_logger.py`
- **Fallback (DEBUG_MODE):** `data/behavioral_logs/<user_id>.jsonl`

---

## 6) Training — Triplet Loss on Behavioral Logs

### Overview

Training is triggered via the `POST /train` endpoint and runs the `TripletTrainer` in `app/triplet_trainer.py`. It:

1. Loads all behavioral logs from Cosmos DB (or local JSONL fallback).
2. Splits each user's events into **sliding 20-second windows** (2-second stride, ~90% overlap).
3. Converts each window into a `TemporalGraph` → PyTorch Geometric `Data` object.
4. Trains a shared `SiameseGATNetwork` on anchor-positive-negative triplets across **all users**.
5. Saves the model checkpoint.
6. Generates a **64-D profile vector** per user (mean-pool of session embeddings).
7. Saves profiles to Cosmos DB (`user-profiles` container).

### Triplet construction
- **Anchor:** graph from user A, session i
- **Positive:** graph from user A, session j (j ≠ i)
- **Negative:** graph from a different user B (random), or synthetic noisy graph if only one user exists

### Training hyperparameters (defined in `app/triplet_trainer.py`)

| Parameter | Value |
|---|---|
| `INPUT_DIM` | 56 |
| `HIDDEN_DIM` | 64 |
| `OUTPUT_DIM` | 64 |
| `NUM_HEADS` | 4 |
| `DROPOUT` | 0.1 |
| `TEMPORAL_DIM` | 16 |
| `TRIPLET_MARGIN` | 0.5 |
| `LEARNING_RATE` | 0.001 |
| `WINDOW_SECONDS` | 20 |
| `WINDOW_STRIDE_SECONDS` | 2 |
| `MIN_EVENTS_FOR_SESSION` | 5 |
| epochs | 30 |

> **CUDA required for full GAT triplet training.** `train_all()` — which is called by `POST /train` when training all users, or automatically by `train_user()` when no model exists — explicitly requires `torch.device("cuda")` and returns an error if CUDA is unavailable.
> 
> **Numpy fallback (no CUDA / no PyTorch):** If PyTorch is not installed at all, `train_user()` (single-user path) bypasses the GAT entirely and builds a mean-pool profile from raw 56-D vectors. This produces a valid profile but without triplet-loss quality. To use this path, call `POST /train` for a single user on a machine without CUDA.

### Training API call

```http
POST /train
Authorization: Bearer <ADMIN_TOKEN>
Content-Type: application/json

{
  "user_id": "alice",   // omit to train all users
  "force": false        // set true to re-train even if profile exists
}
```

Response (success):
```json
{
  "user_id": "alice",
  "status": "success",
  "message": "GAT triplet training complete in 42.3s",
  "profile_saved": true,
  "sessions_used": 18,
  "training_time_seconds": 42.3
}
```

Response (numpy fallback, no PyTorch):
```json
{
  "user_id": "alice",
  "status": "success",
  "message": "Profile created with numpy mean-pool (no PyTorch)",
  "profile_saved": true,
  "sessions_used": 18
}
```

### User discovery during training
`train_all()` discovers users from two sources (merged, deduplicated):
1. **Cosmos DB** — `behavioral_logger.list_users()` queries the `behaviour-logs` container.
2. **Local JSONL** — `data/behavioral_logs/*.jsonl` (fallback / DEBUG_MODE).

---

## 7) Inference Flow (In-Process GAT Engine)

### Overview

GAT inference runs **in-process** on every enrolled user's WebSocket session. No external HTTP calls are made.

```
WebSocket event arrives
    │
    ├─ gat_manager.add_event_to_session(session_id, event)
    │   └─ Maintains sliding 20s window per session in memory
    │
    └─ Every GAT_INFERENCE_INTERVAL_SECONDS (5.0s), if session has ≥ 5 events:
            │
            ├─ layer3_manager.process_escalated_session()
            │      ├─ GATDataProcessor.create_temporal_graph()  → GATGraph
            │      ├─ Load user profile (in-memory → triplet_trainer → Cosmos DB)
            │      ├─ Prepare GATProcessingRequest
            │      └─ GATCloudInterface.process_temporal_graph()
            │              └─ InternalGATEngine.process_request()
            │                      ├─ Real PyTorch: SiameseGATNetwork → 64-D embedding
            │                      │   + cosine similarity vs profile
            │                      └─ Simulation fallback (if no PyTorch/no model)
            │
            └─ gat_similarity included in response + logged to Cosmos DB
```

### User profile lookup order (in `layer3_manager.py`)
1. In-memory `UserProfileManager` (set via `/gat/enroll`)
2. Triplet-trained profile from disk/Cosmos via `triplet_trainer.load_profile(user_id)`

### GAT is only triggered for enrolled users
- Enrollment check: `enrollment_store.get_enrollment_status(user_id).status == "enrolled"`
- During warm-up or enrollment phase, GAT inference is **skipped**.

### InternalGATEngine (`app/gat_engine.py`)

The engine initializes lazily on first use:
1. Tries to download `gat_checkpoint.pt` from **Azure Blob Storage**.
2. Falls back to `data/checkpoints/gat_checkpoint.pt` on disk (DEBUG_MODE only).
3. If no weights are found, runs the model with **random initialisation**.
4. If PyTorch/torch-geometric are unavailable, runs a **simulation** (random vector).

GAT network config used by the engine:

| Parameter | Value |
|---|---|
| `input_dim` | 56 |
| `hidden_dim` | 128 |
| `output_dim` | 64 |
| `num_heads` | 8 |
| `dropout` | 0.1 |
| `temporal_dim` | 8 |

---

## 8) Model and Profile Persistence

### Model checkpoint — `gat_checkpoint.pt`

| Location | Used when |
|---|---|
| Azure Blob Storage (`cbsa-models` container) | Production (requires `AZURE_STORAGE_CONNECTION_STRING`) |
| `data/checkpoints/gat_checkpoint.pt` | DEBUG_MODE local fallback |

- **Training** saves to both locations (blob first, then disk).
- **Inference engine** downloads from blob on startup; disk fallback in DEBUG_MODE.

### User profiles — 64-D float vector

| Location | Used when |
|---|---|
| Azure Cosmos DB (`user-profiles` container) | Production (requires `COSMOS_ENDPOINT` + `COSMOS_KEY`) |
| Saved locally only in DEBUG_MODE | Debug / development |

- Profile is **mean-pooled** from all session embeddings of a user.
- Managed by `app/cosmos_profile_store.py`.
- Loaded by `triplet_trainer.load_profile(user_id)` → `cosmos_profile_store.load_profile()`.

### Behavioral logs

| Location | Used when |
|---|---|
| Azure Cosmos DB (`behaviour-logs` container) | Production |
| `data/behavioral_logs/<user_id>.jsonl` | DEBUG_MODE fallback |

- Every WebSocket event is logged before GAT processing.
- Used as the training data source.

---

## 9) REST API Reference

### Health and status

| Method | Path | Auth | Description |
|---|---|---|---|
| GET | `/health` | — | Returns Cosmos DB connection status and active WebSocket counts |
| GET | `/` | — | Serves the monitor UI (HTML) |
| GET | `/event-flow-map` | — | Returns `EVENT_FLOW_MAP.json` |

### Authentication / enrollment

| Method | Path | Auth | Description |
|---|---|---|---|
| POST | `/login` | — | Start / check enrollment for a user. Returns `enrolled` or `enrolling` |
| POST | `/logout` | — | End session; saves accumulated enrollment seconds |

**Login request:**
```json
{ "username": "alice" }
```

**Login response (not yet enrolled):**
```json
{
  "status": "enrolling",
  "message": "Enrollment in progress. 180s remaining.",
  "time_remaining": 180.0,
  "total_seconds": 300
}
```

**Login response (enrollment data collected, awaiting training):**
```json
{
  "status": "enrolling",
  "message": "Enrollment data collected. Run /train to build your profile.",
  "time_remaining": 0.0,
  "total_seconds": 300
}
```

**Login response (enrolled):**
```json
{
  "status": "enrolled",
  "message": "User is enrolled."
}
```

### Training

| Method | Path | Auth | Description |
|---|---|---|---|
| POST | `/train` | Bearer token | Train GAT + generate profile for one or all users |

**Train request:**
```json
{
  "user_id": "alice",   // optional; omit to train all users
  "force": false
}
```

### WebSocket streams

| Method | Path | Description |
|---|---|---|
| WS | `/ws/behaviour` | Primary behavioral event stream (send events, receive metrics) |
| WS | `/ws/monitor` | Read-only stream for monitor UI |

**Input event format:**
```json
{
  "username": "alice",
  "session_id": "sess-001",
  "timestamp": 1710000000.123,
  "event_type": "TOUCH_BALANCE_TOGGLE",
  "event_data": {
    "nonce": "abc123",
    "vector": [0.1, 0.2, ...],   // 48 floats in [0,1]
    "deviceInfo": {}             // optional, forwarded to monitor
  },
  "signature": "..."             // optional, verification stubbed
}
```

**Response (warm-up phase):**
```json
{ "status": "WARMUP", "collected_windows": 5 }
```

**Response (after warm-up):**
```json
{
  "similarity_score": 0.87,
  "short_drift": 0.12,
  "long_drift": 0.05,
  "stability_score": 0.93,
  "matched_prototype_id": 3
}
```

### GAT management endpoints

| Method | Path | Auth | Description |
|---|---|---|---|
| POST | `/gat/process` | — | Manually trigger GAT for a buffered session |
| POST | `/gat/enroll` | — | Enroll a user profile from verified session data |
| GET | `/gat/profile/{user_id}` | — | Get user profile metadata |
| GET | `/gat/sessions/{session_id}` | — | Get buffered session window |
| DELETE | `/gat/sessions/{session_id}` | — | Clear session window |
| GET | `/gat/stats` | — | Active sessions and buffered event counts |

**POST /gat/process body:**
```json
{ "session_id": "sess-001" }
```

**POST /gat/enroll body:**
```json
{
  "user_id": "alice",
  "verified_sessions": [
    [ { /* event */ }, { /* event */ }, ... ],
    [ ... ]
  ]
}
```

### Admin endpoints (require `Authorization: Bearer <ADMIN_TOKEN>`)

| Method | Path | Description |
|---|---|---|
| DELETE | `/admin/user/{user_id}` | Delete all data for a user (Cosmos + local) |
| DELETE | `/admin/truncate` | Truncate ALL user data (all users, all containers) |
| POST | `/admin/upload-legacy` | Upload legacy behavioral data files |
| POST | `/admin/cosmos-dump` | Dump all Cosmos containers to disk |
| POST | `/admin/cosmos-dump/download` | Download a Cosmos dump as a ZIP |

---

## 10) Config Reference

### `app/config.py` — Backend settings

| Setting | Default | Description |
|---|---|---|
| `GAT_WINDOW_SECONDS` | `20` | Sliding time window (seconds) for temporal graph |
| `GAT_NODE_FEATURE_DIM` | `56` | Node feature dimension (48 behavioral + 8 event-type) |
| `GAT_EDGE_DISTINCT_TARGET` | `4` | Stop adding edges after 4 distinct event types seen |
| `GAT_INFERENCE_INTERVAL_SECONDS` | `5.0` | Minimum seconds between GAT calls per session |
| `GAT_ESCALATION_THRESHOLD` | `0.5` | Assumed Layer 2 escalation threshold |
| `GAT_CLOUD_ENDPOINT` | `""` | Kept for backwards-compat; unused (GAT is in-process) |
| `GAT_WINDOW_SIZE` | `32` | Deprecated event-count window; kept for compat |
| `DEBUG_MODE` | `False` | Enable local file fallbacks |
| `ADMIN_TOKEN` | `""` (from env) | Bearer token for admin / train endpoints |
| `COSMOS_ENDPOINT` | env | Azure Cosmos DB endpoint |
| `COSMOS_KEY` | env | Azure Cosmos DB key |
| `COSMOS_DATABASE` | `"cbsa-logs"` | Cosmos database name |
| `COSMOS_CONTAINER` | `"computation-logs"` | Computation log container |
| `COSMOS_PROFILES_CONTAINER` | `"user-profiles"` | User profile container |
| `COSMOS_ENROLLMENT_CONTAINER` | `"enrollment-state"` | Enrollment state container |
| `COSMOS_PROTOTYPE_CONTAINER` | `"prototype-store"` | Prototype container |
| `COSMOS_BEHAVIOUR_LOGS_CONTAINER` | `"behaviour-logs"` | Behavioral log container |
| `AZURE_STORAGE_CONNECTION_STRING` | env | Azure Blob Storage connection |
| `AZURE_STORAGE_CONTAINER` | `"cbsa-models"` | Blob container for model checkpoints |

### `app/gat/config.py` — Internal GAT package settings

| Setting | Default | Description |
|---|---|---|
| `input_dim` | `56` | Node feature dimension |
| `hidden_dim` | `128` | Hidden layer size |
| `output_dim` | `64` | Final embedding dimension |
| `num_attention_heads` | `8` | Multi-head attention count |
| `num_layers` | `3` | Number of GAT layers |
| `dropout_rate` | `0.1` | Dropout rate |
| `temporal_dim` | `8` | Temporal encoding dimension |
| `learning_rate` | `0.001` | Training learning rate |
| `triplet_margin` | `0.5` | Triplet loss margin |
| `similarity_threshold` | `0.85` | Authentication threshold |
| `time_window_seconds` | `20` | Sliding window (seconds) |
| `distinct_event_connections` | `4` | Edge distinct target |
| `min_events_per_window` | `5` | Minimum events to form a window |
| `max_events_per_window` | `100` | Maximum events per window |

### `app/triplet_trainer.py` — Training constants

| Constant | Value | Description |
|---|---|---|
| `INPUT_DIM` | `56` | Node feature dimension |
| `HIDDEN_DIM` | `64` | Hidden layer |
| `OUTPUT_DIM` | `64` | Profile vector dimension |
| `NUM_HEADS` | `4` | Attention heads |
| `DROPOUT` | `0.1` | Dropout |
| `TEMPORAL_DIM` | `16` | Temporal encoding size |
| `TRIPLET_MARGIN` | `0.5` | Triplet loss margin |
| `LEARNING_RATE` | `0.001` | Adam learning rate |
| `WINDOW_SECONDS` | `20` | Window size for training |
| `WINDOW_STRIDE_SECONDS` | `2` | Stride for sliding window |
| `MIN_EVENTS_FOR_SESSION` | `5` | Minimum events per window |

---

## 11) Where to Tweak Embedding Dimensions

### Event-type embedding size (currently 8-D)

- `app/layer3_processor.py` → `_event_type_embedding()` (returns `digest[:8]`)
- `app/gat/data_processor.py` → `_event_type_embedding()` (returns `digest[:8]`)

Change the slice `[:8]` to adjust. After changing:

### Device context (currently disabled — 0-D)
Device context was previously 4-D but has been removed. To re-add it:
- Implement `_extract_device_context_vector()` in both processors above.
- Concatenate to the node vector after the event-type embedding.
- Update all dimension settings below.

### After changing any dimension, update all of these

| File | Setting |
|---|---|
| `app/config.py` | `GAT_NODE_FEATURE_DIM` |
| `app/gat/config.py` | `input_dim` |
| `app/gat/gat_network.py` | `input_dim` in `SiameseGATNetwork` |
| `app/gat/models.py` | `behavioral_vector: List[float] = Field(..., min_length=56, max_length=56)` |
| `app/layer3_models.py` | `behavioral_vector: List[float]` |
| `app/triplet_trainer.py` | `INPUT_DIM` constant |

---

## 12) Quality Gates and Validation

### Input validation (every WebSocket event)
- JSON object shape check
- Required fields: `username`, `session_id`, `timestamp`, `event_type`, `event_data`, `nonce`, `vector`
- Vector: exists, length == 48, all numeric, all in `[0, 1]`
- Nonce uniqueness per session
- Monotonic timestamps per session
- Burst-rate guard: reject if delta < 40ms for more than 5 consecutive events

### Training validation
- Minimum 2 session windows required per user
- Minimum 5 events per window (`MIN_EVENTS_FOR_SESSION`)
- CUDA availability checked before GAT training starts
- PyTorch availability checked; falls back to numpy if unavailable

### Cosmos DB health check
- `GET /health` returns per-container connectivity status
- Returns HTTP 503 if any expected Cosmos connection is not available

---

## 13) Troubleshooting

| Problem | Cause | Fix |
|---|---|---|
| `POST /train` returns `"CUDA is not available"` | No GPU in environment | Full GAT triplet training (`train_all`) requires CUDA. To work around: call `POST /train` with a specific `user_id` on a machine without PyTorch installed — this triggers the numpy mean-pool fallback which does not require CUDA or GPU. |
| `POST /train` returns `"No behavioral data found"` | No logs in Cosmos or local files | Send events over WebSocket first; check `DEBUG_MODE=True` for local file fallback. |
| `status: "enrolling"` never completes | Enrollment timer not reaching 5 minutes | Send events and call `/logout` after each session; timer accumulates across sessions. |
| GAT similarity is `null` in response | User not enrolled, or session has < 5 events | Enroll the user first (`/login` → 5 min data → `/train`). |
| GAT returns simulated values | PyTorch not installed or no model checkpoint | Install `torch` and `torch-geometric`, then trigger `/train` to create a checkpoint. |
| Missing azure-cosmos | SDK not installed | `pip install azure-cosmos` |
| Missing azure-storage-blob | SDK not installed | `pip install azure-storage-blob` |
| Profile not found after training | Cosmos not configured | Set `COSMOS_ENDPOINT` and `COSMOS_KEY`, or enable `DEBUG_MODE=True` for local storage. |
| `Admin endpoints are disabled` | `ADMIN_TOKEN` not set | Set `ADMIN_TOKEN` environment variable. |

---

## 14) Full File Index

### Core application
| File | Purpose |
|---|---|
| `app/main.py` | FastAPI app, all REST + WebSocket endpoints |
| `app/config.py` | All backend settings (`Settings` class) |
| `app/models/` | Pydantic models: `BehaviourEvent`, `BehaviourMessage`, `Prototype` |

### GAT engine (in-process)
| File | Purpose |
|---|---|
| `app/gat_engine.py` | `InternalGATEngine` — lazy-init, real inference or simulation fallback |
| `app/gat/gat_network.py` | `SiameseGATNetwork`, `GATTrainer`, `GATInferenceEngine` (PyTorch) |
| `app/gat/data_processor.py` | `BehavioralDataProcessor`, `PyTorchDataConverter` |
| `app/gat/models.py` | Pydantic models: `EventNode`, `TemporalEdge`, `TemporalGraph`, request/response |
| `app/gat/config.py` | `GATSettings` (internal GAT package config) |

### Layer 3 integration
| File | Purpose |
|---|---|
| `app/layer3_manager.py` | `Layer3GATManager` — session window + orchestrates inference + enrollment |
| `app/layer3_processor.py` | `GATDataProcessor` — builds `GATGraph` from `BehaviourMessage` stream |
| `app/layer3_cloud.py` | `GATCloudInterface` — dispatches to `InternalGATEngine` (no HTTP) |
| `app/layer3_models.py` | `GATEventNode`, `GATTemporalEdge`, `GATGraph`, `GATProcessingRequest/Response`, `UserProfile` |

### Training and profiling
| File | Purpose |
|---|---|
| `app/triplet_trainer.py` | `TripletTrainer` — loads logs, windows, builds triplets, trains GAT, saves profiles |
| `app/cosmos_profile_store.py` | Profile CRUD in Cosmos DB (`user-profiles` container) |
| `app/blob_model_store.py` | Model checkpoint upload/download to/from Azure Blob Storage |

### Enrollment and logging
| File | Purpose |
|---|---|
| `app/enrollment_store.py` | `EnrollmentStore` — tracks per-user enrollment time (Cosmos or local file) |
| `app/behavioral_logger.py` | `BehavioralFileLogger` — logs raw events to Cosmos + local JSONL |

### Engine layers (Layer 1/2)
| File | Purpose |
|---|---|
| `app/engine/ingestion.py` | Input validation (`validate_and_extract`) |
| `app/engine/preprocessing.py` | `process_event` — short/long drift, stability |
| `app/engine/prototype_engine.py` | `compute_prototype_metrics` — cosine + Mahalanobis vs prototypes |
| `app/engine/buffer_manager.py` | Per-session in-memory buffers |
| `app/engine/similarity_engine.py` | Similarity helpers |
| `app/engine/drift_engine.py` | Drift calculation |

### Storage
| File | Purpose |
|---|---|
| `app/storage/sqlite_store.py` | SQLite-based prototype + user store (`cbsa.db`) |
| `app/storage/cosmos_prototype_store.py` | Cosmos-based prototype + behaviour log store |
| `app/storage/memory_store.py` | In-memory session state |
| `app/storage/merge_utils.py` | Import/export merge helpers |

### Scripts (legacy / offline)
| File | Purpose |
|---|---|
| `scripts/build_dataset_from_fast.py` | Parses `fast.txt` logs to build `datasets/fast_dataset.json` and `profiles/fast_profiles.json` (offline utility, not part of live pipeline) |

### Documentation
| File | Purpose |
|---|---|
| `GAT_COMPLETE_GUIDE.md` | This document |
| `GAT_EDGE_WINDOWING.md` | Earlier windowing/edge reference (partially superseded by this guide) |
| `CURRENT_IMPLEMENTATION_OVERVIEW.md` | Plain-language description of the full backend pipeline |
| `API_SPEC.md` | Formal API specification |
