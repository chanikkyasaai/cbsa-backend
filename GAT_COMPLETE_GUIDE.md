# GAT End‑to‑End Guide (Dataset → Profiles → Training → Inference)

This document consolidates the full GAT workflow used in this repo, including dataset creation, profile building, training, and tuning knobs. It references the architecture in `image.png` and the current implementation in `/app` and `/gat-service`.

---

## 1) Architecture Summary (from `image.png`)

**Pipeline:**
1. **Mobile device** sends JSON payloads every ~1s or on user actions.
2. **Preprocessing** builds a **20‑second window** of events.
3. **GAT** consumes node features and attention edges to produce a **64‑D session embedding**.
4. **Training** uses Siamese Triplet Loss (anchor / positive / negative).
5. **Inference** compares session vector with user profile via cosine similarity.

---

## 2) Data and Feature Shapes

### Node features (60‑D)
Built as:
- **48** event vector values (from `event_data.vector`)
- **8** event‑type embedding values
- **4** device context values

### Event type embedding (8‑D)
Deterministic embedding from event type string:
- Implemented via SHA‑256 hash → first 8 bytes → normalized to $[0,1]$

### Device context vector (4‑D)
Normalized scalar fields:
- Battery
- CPU usage
- Memory usage
- Network strength

### Output
- **Session embedding:** 64‑D
- **Similarity score:** cosine similarity to profile

---

## 3) Windowing and Edge Rules

### Windowing
- **Time‑based only**: last **20 seconds** of events
- Repeats are preserved

### Edge construction (per event node)
For each event $i$:
1. Connect to **every following event in order**.
2. **Keep repeats** (duplicates are connected).
3. Track **distinct event types** encountered.
4. Stop only after **4 distinct types** are reached.
5. Repeats **do not** decrement the distinct count.

---

## 4) Dataset Build (from `fast.txt`)

### Source file
- `fast.txt` (log lines from websocket stream)

### Build script
- `scripts/build_dataset_from_fast.py`

### Output files
- `datasets/fast_dataset.json`
- `profiles/fast_profiles.json`

### What it does
- Parses `Data received:` payloads
- Applies **20s windowing** per session
- Builds **60‑D node vectors** per event
- Computes **per‑user profile vectors** (mean of node vectors)

### Run it
```powershell
D:/devvenky/cbsa-backend/.venv/Scripts/python.exe d:/devvenky/cbsa-backend/scripts/build_dataset_from_fast.py --input d:/devvenky/cbsa-backend/fast.txt --dataset d:/devvenky/cbsa-backend/datasets/fast_dataset.json --profiles d:/devvenky/cbsa-backend/profiles/fast_profiles.json
```

---

## 5) Training (GAT Service)

### Training script
- `gat-service/train_from_dataset.py`

### Features
- **Progress bars** via `tqdm`
- **Checkpointing** after each epoch
- **Resume** support
- **Ctrl+C** saves checkpoint

### Train
```powershell
D:/devvenky/cbsa-backend/.venv/Scripts/python.exe d:/devvenky/cbsa-backend/gat-service/train_from_dataset.py --dataset d:/devvenky/cbsa-backend/datasets/fast_dataset.json --epochs 10
```

### Resume after Ctrl+C
```powershell
D:/devvenky/cbsa-backend/.venv/Scripts/python.exe d:/devvenky/cbsa-backend/gat-service/train_from_dataset.py --dataset d:/devvenky/cbsa-backend/datasets/fast_dataset.json --epochs 10 --resume
```

### Checkpoint file
- Default: `gat-service/checkpoints/gat_checkpoint.pt`

---

## 6) Inference Flow

### Backend Layer 3
- `app/layer3_manager.py` orchestrates windowing and calls GAT cloud
- `app/layer3_processor.py` builds graph (60‑D features + edges)
- `app/layer3_cloud.py` calls GAT service or simulates locally

### GAT Service
- `gat-service/main.py` processes `/process` requests
- `gat-service/data_processor.py` builds graph
- `gat-service/gat_network.py` runs PyTorch GAT (if available)

---

## 7) Where to Tweak Embedding Dimensions

### Event type embedding size
- **Backend:** `app/layer3_processor.py` → `_event_type_embedding(...)`
- **Service:** `gat-service/data_processor.py` → `_event_type_embedding(...)`
- **Runner:** `gat-service/simple_main.py` → `event_type_embedding(...)`

### Device context vector size
- **Backend:** `app/layer3_processor.py` → `_extract_device_context_vector(...)`
- **Service:** `gat-service/data_processor.py` → `_extract_device_context_vector(...)`
- **Runner:** `gat-service/simple_main.py` → `device_context_vector(...)`

### Update totals if you change sizes
- `app/config.py` → `GAT_NODE_FEATURE_DIM`
- `gat-service/config.py` → `INPUT_DIM`
- `gat-service/gat_network.py` → `input_dim`
- `gat-service/models.py` → `behavioral_vector` length
- `gat-service/simple_main.py` → `behavioral_vector` length

---

## 8) Config Summary

### Backend
- `app/config.py`
  - `GAT_WINDOW_SECONDS = 20`
  - `GAT_EDGE_DISTINCT_TARGET = 4`
  - `GAT_NODE_FEATURE_DIM = 60`

### GAT Service
- `gat-service/config.py`
  - `TIME_WINDOW_SECONDS = 20`
  - `DISTINCT_EVENT_CONNECTIONS = 4`
  - `INPUT_DIM = 60`

---

## 9) Quality Gates / Validation

- Dataset builder prints session/user counts
- Training script supports `--dry-run` to validate loading
- Syntax checks pass for all modified modules

---

## 10) Quick Troubleshooting

- **Missing torch**: training will fail — install with `pip install torch torch-geometric`
- **Missing tqdm**: install with `pip install tqdm`
- **No sessions**: check `fast.txt` contains `Data received:` lines

---

## 11) File Index (Key Files)

- `scripts/build_dataset_from_fast.py`
- `datasets/fast_dataset.json`
- `profiles/fast_profiles.json`
- `gat-service/train_from_dataset.py`
- `app/layer3_processor.py`
- `gat-service/data_processor.py`
- `gat-service/gat_network.py`
- `GAT_EDGE_WINDOWING.md`

---

If you want a diagram‑specific appendix (matching all labels in `image.png`), I can add that too.
