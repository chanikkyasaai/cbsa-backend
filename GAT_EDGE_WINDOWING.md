# GAT Windowing & Edge Construction (Reference Implementation)

This document explains how the CBSA backend builds Layer 3 GAT graphs and how those graphs align with the **reference architecture** shown in `image.png`.

## Overview (Image Reference)

The diagram depicts a **mobile → cloud** pipeline:

1. **Mobile Device** emits JSON payloads every ~1s or on user action.
2. **Preprocessing** builds a temporal window (**20s in our implementation**).
3. **GAT** consumes node features, builds attention edges, and produces a **64‑D embedding**.
4. **Training** uses Siamese Triplet Loss (anchor/positive/negative).
5. **Inference** compares the session vector to the user profile via cosine similarity.

Our implementation mirrors the same stages, with updated rules for **time‑based windowing** and **edge creation** per your requirements.

---

## ✅ Current Windowing Rule

**Time‑based only** (no event‑count truncation):

- We keep **all events within the last 20 seconds**.
- Repeated event types are preserved (no de‑duplication).

### Where this is applied
- `app/layer3_manager.py` → session window pruning (20s)
- `app/layer3_processor.py` → graph creation window (20s)
- `gat-service/data_processor.py` → temporal graph preprocessing (20s)

---

## ✅ Edge Construction Rule (Updated)

For each event node $i$:

1. **Connect to every following event in order**.
2. **Keep repeats** (duplicates are connected).
3. Maintain a set of **distinct event types** encountered.
4. Stop only after **4 distinct types** are reached.
5. **Repeats do not decrement the remaining distinct count**.

### Example (your case)
Sequence:

- 3 distinct events (A, B, C)
- 4 repeats of `scroll_dashboard`
- 4 more distinct events (D, E, F, G)

Edges from A:
- A → B, A → C
- A → scroll_dashboard (x4, all connected)
- A → D, A → E, A → F, A → G (until 4 distinct types reached)

> Result: total edges **can exceed 4**, because repeats are kept and do **not** reduce distinct quota.

---

## Implementation Notes

### Layer 3 (Backend)
**File:** `app/layer3_processor.py`

- Uses a **20‑second window**.
- Builds edges until **4 distinct event types** are reached.
- Adds **every repeat** encountered along the way.

### GAT Service
**File:** `gat-service/data_processor.py`

- Same rule as above for cloud service graphs.
- Distinct target configured as **4**.

### Simple Runner
**File:** `gat-service/simple_main.py`

- Mirrors the same edge rule for local testing.

---

## 🔧 Tuning Knobs (Where to Change Sizes)

### Event‑Type Embedding Dimension
- **Backend:** `app/layer3_processor.py` → `_event_type_embedding(...)`
- **GAT Service:** `gat-service/data_processor.py` → `_event_type_embedding(...)`
- **Simple Runner:** `gat-service/simple_main.py` → `event_type_embedding(...)`

Change the number of bytes returned (currently **8**) to adjust embedding size.

### Device Context Vector Dimension
- **Backend:** `app/layer3_processor.py` → `_extract_device_context_vector(...)`
- **GAT Service:** `gat-service/data_processor.py` → `_extract_device_context_vector(...)`
- **Simple Runner:** `gat-service/simple_main.py` → `device_context_vector(...)`

Update the 4 fields or add/remove fields to change the dimension.

### Total Node Feature Dimension
- `app/config.py` → `GAT_NODE_FEATURE_DIM`
- `gat-service/config.py` → `INPUT_DIM`
- `gat-service/gat_network.py` → `input_dim`
- `gat-service/models.py` → `behavioral_vector` length constraints
- `gat-service/simple_main.py` → `behavioral_vector` length constraints

If you change any embedding sizes, update the totals above accordingly.

---

## Pseudocode (Exact Logic)

```
for each event i in events:
    seen = { event[i].type }

    for j in range(i+1, end):
        connect(i, j)   # ALWAYS connect, even if duplicate

        if event[j].type not in seen:
            seen.add(event[j].type)

            if len(seen) >= 4:
                break
```

---

## Files Updated

- `app/config.py`
  - `GAT_WINDOW_SECONDS = 20`
  - `GAT_EDGE_DISTINCT_TARGET = 4`
- `app/layer3_manager.py`
  - time‑based pruning
- `app/layer3_processor.py`
  - new edge rule
- `gat-service/config.py`
  - distinct target set to 4
- `gat-service/data_processor.py`
  - new edge rule
- `gat-service/simple_main.py`
  - new edge rule

---

## Notes vs Diagram

The diagram shows **60‑D node features** (48 event + 8 event-type + 4 device context), and the backend now matches this shape in both Layer 3 and the GAT service.

---

## Quick Validation

- Python syntax checks run on modified files.
- Time window = 20s, edge rule = 4 distinct target.

If you want an explicit test case for the “3 distinct + 4 repeats + 4 distinct” pattern, I can add a small unit test fixture.