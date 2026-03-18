# Continuous Behavioral Streaming Authentication (CBSA)
## Architecture Design — v1

---

## Abstract

Continuous Behavioral Streaming Authentication (CBSA) is a real-time identity verification system that authenticates users by analyzing their behavioral patterns across a continuous stream of interaction events. Unlike challenge-based authentication (passwords, PINs, OTPs), CBSA operates passively in the background: every touch gesture, scroll event, typing pattern, and navigation interaction contributes to an ongoing authentication signal that adapts to the user over time without any deliberate user action.

The system is structured as a four-layer processing pipeline. Layer 1 validates and extracts behavioral events from a mobile client over WebSocket. Layer 2 extracts leakage-free three-scale temporal drift (micro-behavioral, episodic, and identity-baseline), stability, prototype topology cohesion, prototype-matching signals, and a Behavioral Session Fingerprint (EMA-updated Markov transition surprise) using a geometric, statistical, and sequential model of behavioral identity. Layer 3 runs a Graph Attention Network (GAT) over a temporal behavioral graph to produce a deep session embedding, escalated only when Layer 2 raises uncertainty. Layer 4 combines all signals into a continuous trust score through an Exponential Moving Average model with cohesion-modulated adaptive coefficient, producing zone-based authentication decisions.

This document describes every component of the system, the mathematical foundations of each design decision, the alternatives considered, and the explicit rationale for each choice.

---

## Table of Contents

1. [System Overview](#1-system-overview)
2. [Layer 1 — Secure Ingestion](#2-layer-1--secure-ingestion)
3. [Layer 2 — Behavioral Preprocessing and Prototype Engine](#3-layer-2--behavioral-preprocessing-and-prototype-engine)
   - 3.1 Session Buffer and Leakage-Free Statistics
   - 3.2 Drift Computation
   - 3.3 Stability Score
   - 3.4 Composite Similarity
   - 3.5 Prototype Quarantine
   - 3.6 Adaptive Learning Rate
   - 3.7 Quality-Based Prototype Lifecycle
   - 3.8 Behavioral Session Fingerprint Engine (Layer-2c)
4. [Layer 3 — Graph Attention Network (GAT)](#4-layer-3--graph-attention-network-gat)
   - 4.1 Behavioral Graph Construction
   - 4.2 Node Feature Engineering
   - 4.3 Edge Construction and Windowing
   - 4.4 SiameseGATNetwork Architecture
   - 4.5 Triplet Loss Training
   - 4.6 Inference and Profile Management
5. [Layer 4 — Trust Engine](#5-layer-4--trust-engine)
   - 5.1 Raw Trust Signal
   - 5.2 Adaptive EMA Coefficient
   - 5.3 Trust Zones and Decisions
   - 5.4 GAT Escalation Logic
6. [Storage Architecture](#6-storage-architecture)
7. [Pipeline Execution Order and Invariants](#7-pipeline-execution-order-and-invariants)
8. [Evaluation Methodology](#8-evaluation-methodology)
9. [System Constants Reference](#9-system-constants-reference)

---

## 1. System Overview

```
Mobile Client
     |  JSON event stream (WebSocket, ~1 event/s)
     v
+------------------------------------------------------------------+
|  Layer 1: Secure Ingestion                                       |
|  - Field validation  - Vector bounds  - Nonce uniqueness         |
|  - Monotonic timestamps  - Burst-rate guard                      |
+---------------------------+--------------------------------------+
                            |  BehaviourEvent
                            v
+------------------------------------------------------------------+
|  Layer 2: Preprocessing + Prototype Engine + Fingerprint         |
|  - Leakage-free drift & stability   - Composite similarity       |
|  - Quarantine-gated prototype creation                           |
|  - Adaptive learning rate           - Quality lifecycle          |
|  - Behavioral Session Fingerprint (Markov transition surprise)   |
+---------------------------+--------------------------------------+
                            |  PreprocessedBehaviour + PrototypeMetrics
                            |  (incl. transition_surprise)
                            v
             +--------------+------------------+
             |  Escalate? (anomaly score)       |
             |  trust < theta_risk  OR          |
             |  n_uncertain > threshold         |
             +----------+----------------------+
                        |  (conditional)
                        v
+------------------------------------------------------------------+
|  Layer 3: GAT Deep Analysis (in-process PyTorch engine)          |
|  - Temporal behavioral graph  - 56-D node features               |
|  - Siamese GAT embeddings     - Profile cosine similarity        |
+---------------------------+--------------------------------------+
                            |  gat_similarity in [0,1]
                            v
+------------------------------------------------------------------+
|  Layer 4: Trust Engine                                           |
|  - EMA trust model with adaptive alpha                           |
|  - Zone decisions: ACCEPT / MONITOR / CHALLENGE / REJECT         |
|  - Drift-vs-trust debug logging                                  |
+------------------------------------------------------------------+
```

**Technology stack:** Python 3.10+, FastAPI, WebSocket, PyTorch, torch_geometric, Azure Cosmos DB (profiles, computation logs, enrollment), Azure Blob Storage (model checkpoints).

**Project layout:**

```
cbsa-backend/
├── app/
│   ├── ingestion/        # Layer 1: validate_and_extract
│   ├── preprocessing/    # Layer 2a: buffer, drift, stability
│   ├── prototype/        # Layer 2b: similarity, quarantine, lifecycle
│   ├── gat/              # Layer 3: network, trainer, engine
│   ├── layer3/           # Layer 3 integration: graph builder, manager
│   ├── trust/            # Layer 4: EMA trust model, decisions
│   ├── azure/            # Cosmos DB and Blob Storage wrappers
│   ├── api/              # FastAPI WebSocket handler
│   ├── core/             # Constants, invariant checks
│   ├── logging/          # Structured event logging
│   ├── models/           # Pydantic data models
│   └── storage/          # SQLite store, memory store, repository
├── tests/
│   ├── runner.py          # End-to-end test scenarios
│   └── evaluation.py      # FAR / FRR / EER evaluation
├── docs/
│   ├── gat/               # GAT architecture documentation
│   ├── api/               # API specification
│   └── SETUP.md           # Installation and run guide
└── data/
    ├── profiles/           # Local user profile vectors (fallback)
    ├── behavioral_logs/    # Per-user JSONL event logs
    ├── checkpoints/        # Local model checkpoints (fallback)
    └── samples/            # Behavioral sample data for testing
```

---

## 2. Layer 1 — Secure Ingestion

**File:** `app/ingestion/ingestion.py`

### Purpose

Layer 1 is the sole entry point for behavioral data. Every event arriving over WebSocket passes through a sequence of deterministic validation checks before any downstream processing begins. Rejection at this layer returns an error response immediately; no state is modified.

### Validation Checks

The following checks are performed in strict order:

| # | Check | Rejection Condition |
|---|-------|---------------------|
| 1 | Schema validation | Missing required fields: `username`, `session_id`, `timestamp`, `event_type`, `event_data`, `nonce`, `vector` |
| 2 | Vector dimension | `len(vector) != 48` |
| 3 | Vector range | Any element outside `[0, 1]` |
| 4 | Nonce uniqueness | Nonce already seen in this session |
| 5 | Monotonic timestamp | `timestamp <= last_timestamp` for this session |
| 6 | Burst-rate guard | `delta_t < 40 ms` for more than 5 consecutive events |

### Design Decisions

**Why validate nonce uniqueness?**
Without nonce checking, a replay attack can re-inject a previously captured authentic behavioral event. Since each event produces updates to the prototype engine, repeated authentic events could artificially inflate similarity scores, potentially enabling impersonation by replaying a captured session. The nonce (per-event unique identifier generated by the mobile SDK) makes every event cryptographically distinct.

**Why 48-dimensional vectors?**
The 48-dimensional behavioral feature vector is the agreed interface between the mobile SDK and the backend. Each dimension encodes a normalized behavioral signal (touch pressure, scroll velocity, typing cadence components, etc.). The fixed dimensionality is required by all downstream components: drift computation (L2 distance), Mahalanobis distance (diagonal covariance), and GAT node features (56-D = 48 + 8 event-type embedding).

**Why `[0, 1]` range enforcement?**
All similarity, drift, and stability metrics downstream operate under the assumption that the input space is bounded in `[0,1]^D`. Admitting out-of-range values would break the mathematical bounds guarantees of the composite similarity function and potentially cause numerical overflow in the Mahalanobis distance computation (division by a variance that approaches zero creates amplified sensitivity to out-of-range inputs).

**Why burst-rate guarding?**
Extremely rapid event injection (sub-40ms inter-event intervals) is inconsistent with human interaction patterns. It indicates either automated replay or fuzzing. The threshold of 40ms reflects the practical lower bound of human touch interaction latency. Sustained bursts (5+ consecutive) are rejected; isolated fast events are tolerated.

---

## 3. Layer 2 — Behavioral Preprocessing and Prototype Engine

**Files:** `app/preprocessing/`, `app/prototype/`

Layer 2 is the statistical core of CBSA. It extracts three families of behavioral signals from the event stream: *drift* (deviation from the user's established behavioral baseline), *stability* (coherence of the current interaction window), and *prototype similarity* (proximity to the user's behavioral archetypes). These three signals are combined by Layer 4 into the trust score.

### 3.1 Session Buffer and Leakage-Free Statistics

**File:** `app/preprocessing/buffer_manager.py`

Each user session maintains an in-memory state structure:

- **Short window** (`deque`, maxlen=5): the 5 most recent behavioral vectors (micro-behavioral scale)
- **Medium window** (`deque`, maxlen=20): the 20 most recent behavioral vectors (episodic scale)
- **Running mean**: online estimate of the session long-term mean `mu_long`, updated via Welford's algorithm
- **Running M2**: accumulated sum-of-squared deviations
- **Running variance**: `sigma^2 = M2 / (n-1)` (unbiased estimator)
- **Sample count** `n`

#### Welford's Online Algorithm

For each incoming sample `x` at count `n`:

```
delta1 = x - mu_{n-1}
mu_n   = mu_{n-1} + delta1 / n
delta2 = x - mu_n
M2_n   = M2_{n-1} + delta1 * delta2
sigma^2_n = M2_n / (n - 1)
```

**Why Welford's over a simple running sum?**
Computing variance as `sum(x^2)/n - (sum(x)/n)^2` suffers catastrophic cancellation: the two terms can be nearly equal for large n, causing the difference to lose precision. For sessions of up to `MAX_SESSION_EVENTS = 10,000` events with 48-dimensional vectors, numerical accuracy is a practical requirement. Welford's algorithm is equivalent but numerically stable to machine precision regardless of session length.

Reference: Welford, B.P. (1962). "Note on a method for calculating corrected sums of squares and products." *Technometrics*, 4(3), 419-420.

#### The Pre-Update Snapshot (Leakage Prevention)

A fundamental correctness requirement is that `v_t` must not appear in its own reference distribution when drift is computed:

```
Leakage-free invariant:
    snapshot = {mu_long^{t-1}, sigma^2^{t-1}, window^{t-1}}  <- BEFORE update
    update buffer with v_t
    d_long = ||mu_window^{t-1} - mu_long^{t-1}||             <- no leakage
```

If the buffer were updated first and then drift computed against the updated statistics, `v_t` would appear on both sides of the distance computation. In the extreme case (first event, n=1), the running mean equals `v_t`, so short_drift would always be exactly zero — completely uninformative.

The `update_session_buffer()` function **atomically** captures the pre-update snapshot and then performs the buffer update, returning both. All metric computations downstream consume only snapshot values. This is enforced architecturally: the `PreUpdateSnapshot` dataclass carries only pre-update fields.

**Window vector asymmetry:** The `window_vector` passed to prototype matching is the *post-update* short window mean. This asymmetry is deliberate: prototype matching asks "what is the user's current behavioral state?" and should include `v_t`, while drift asks "how much has `v_t` deviated from prior context?" and must exclude `v_t` from the reference.

### 3.2 Drift Computation

**File:** `app/preprocessing/drift_engine.py`

Three drift signals are computed from the pre-update snapshot, forming a temporal hierarchy at three distinct behavioral scales:

**Short drift** — deviation of `v_t` from the 5-event micro-behavioral window:

```
d_raw_short = ||v_t - mu_short^{t-1}||_2 / sqrt(D)
d_short = 1 - exp(-d_raw_short / sigma)
```

Captures sudden within-session behavioral changes at the finest granularity. This is the primary anomaly signal: an impostor who cannot replicate the user's micro-behavioral patterns will immediately register high short drift.

**Medium drift** — deviation of `v_t` from the 20-event episodic window:

```
d_raw_medium = ||v_t - mu_medium^{t-1}||_2 / sqrt(D)
d_medium = 1 - exp(-d_raw_medium / sigma)
```

Captures behavioral mode transitions at the episodic interaction scale. Slower than short drift (responds over 20 events rather than 5), faster than long drift (responds to within-session shifts, not just session-to-session changes). Detects transitions between interaction contexts — e.g., switching from browsing to form-filling — that short drift misses because they occur gradually.

**Long drift** — deviation of the short window from the Welford running mean:

```
d_raw_long = ||mu_short^{t-1} - mu_long^{t-1}||_2 / sqrt(D)
d_long = 1 - exp(-d_raw_long / sigma)
```

Captures gradual behavioral drift across the full session identity baseline. A slow attacker who incrementally shifts behavioral vectors would evade short and medium drift detectors but register on the long-term baseline.

All three outputs are in `[0, 1)`.

#### Three-Scale Temporal Hierarchy

The three signals operate at qualitatively different behavioral timescales:

| Scale | Window | Timescale | Detects |
| :--- | :------ | :---------- | :--------- |
| Short | 5 events | Micro-behavioral | Sudden impostor switches, replay attacks |
| Medium | 20 events | Episodic | Mode transitions, context switches, gradual impersonation |
| Long | Full session (Welford) | Identity baseline | Slow drift, session-level behavioral evolution |

No single-scale system can detect all three threat profiles. A short-window-only system is blind to gradual episodic drift; a long-window-only system is slow to respond to sudden takeover. The three-scale hierarchy provides simultaneous sensitivity at all three behavioral timescales with zero additional computational cost (all windows are already maintained in session state).

#### Design Decision: `sqrt(D)` Normalization

For a D-dimensional vector in `[0,1]^D`, the maximum possible L2 distance between two points is `sqrt(D)` (the diagonal of the unit hypercube). Dividing by `sqrt(D)` maps the raw distance to `[0, 1]` and makes the metric **dimension-invariant**: a "moderate" drift has the same numerical magnitude regardless of whether D=10 or D=100. Without this normalization, thresholds would need to be re-calibrated for every change in feature dimensionality.

#### Design Decision: Exponential Normalization

Three candidate normalizations were evaluated:

| Approach | Formula | Analysis |
| :---------- | :---------- | :---------- |
| Linear clipping | `min(d / d_max, 1)` | Requires choosing `d_max` without principled basis; clips outliers instead of smoothly mapping them |
| Platt / logistic | `d / (1 + d)` | No natural scale parameter; insensitive at high d (approaches 1 slowly) |
| Exponential | `1 - exp(-d / sigma)` | Natural scale parameter sigma; smooth; derivative strictly decreasing |

The exponential form was chosen because:
1. **Interpretable sigma**: Setting `sigma = 0.15 * sqrt(D)` is grounded in the observation that a "moderate" behavioral change shifts features by 0.15 on average across all D dimensions. The L2 of this uniform shift is `0.15 * sqrt(D)`. At `d = sigma`, drift scores `1 - exp(-1) ≈ 0.632` — the one-sigma point. Thresholds have a principled unit interpretation.
2. **Smooth penalization**: The derivative `exp(-d/sigma)/sigma` is strictly positive and monotonically decreasing. Small deviations produce proportionally smaller scores; extreme deviations approach 1 asymptotically without cliff effects.
3. **Dimension-awareness**: With `sqrt(D)` pre-normalization, sigma is expressed in normalized units and is thus dimension-invariant across systems.

### 3.3 Stability Score

```
S(t) = exp(-(1/D) * sum_i [ Var_short_i / max(Var_global_i, eps) ])
```

Where:
- `Var_short_i`: per-dimension variance of the short window
- `Var_global_i`: per-dimension Welford running variance
- `eps = 1e-8`: numerical floor
- Division by `D` provides dimension-invariant normalization

**Interpretation:** S measures the coherence of the current behavioral window relative to the user's established variability:
- Window variance matches long-term variance (ratio = 1 per dim): `S = exp(-1) ≈ 0.37`
- Window is highly consistent (ratio << 1): `S -> 1.0`
- Window is erratic (ratio >> 1): `S -> 0`

**Design decision:**
Two alternatives were considered:

| Approach | Problem |
| :---------- | :---------- |
| Mean consecutive L2 | Unbounded: the mean L2 between consecutive events can exceed 1.0, breaking the [0,1] guarantee of the composite similarity score |
| Linear variance ratio | Unbounded when short variance exceeds global variance |

The `exp(-)` transform ensures `S in (0, 1]` by construction, regardless of variance ratios. The global variance denominator personalizes the stability threshold: a naturally high-variance user is not penalized for a moderately variable window.

### 3.4 Composite Similarity

**File:** `app/prototype/similarity_engine.py`

Given a window vector `v` and a stored prototype `(mu_k, sigma^2_k)`:

```
sim = 0.50 * cos(v, mu_k) + 0.40 * k_M(v, mu_k) + 0.10 * S
```

**Cosine similarity:**
```
cos(v, mu) = (v . mu) / (||v||_2 * ||mu||_2)    in [-1, 1] -> clipped to [0, 1]
```

**Exp-Mahalanobis kernel:**
```
d_M(v, mu, sigma^2) = sqrt( sum_i (v_i - mu_i)^2 / max(sigma^2_i, eps) )
k_M = exp(-d_M / sqrt(D))                         in (0, 1]
```

**Stability modifier:** `S in (0, 1]` from Section 3.3

**Bounds verification:**
- Maximum (cos=1, k_M=1, S=1): 0.50 + 0.40 + 0.10 = 1.0
- Minimum (cos=0, k_M≈0, S≈0): ≈ 0.0

All three components are independently bounded in `[0,1]` before combination; the composite output is `[0,1]` without clipping.

#### Design Decision: Three-Component Composite

Cosine and Mahalanobis are geometrically orthogonal measures:
- **Cosine** captures directional agreement — the behavioral "shape" or pattern. Scale-invariant: two vectors of different magnitude but same direction have cosine similarity = 1.
- **Mahalanobis** captures magnitude deviation scaled by user-specific variance. The same absolute deviation is penalized more for a low-variance user than a high-variance user — matching the intuition that deviations should be judged relative to individual expectations.
- **Stability** is a quality modifier: a behaviorally coherent window should contribute slightly more to similarity than an erratic one with the same mean.

**Why 50/40/10 weights?**
The dominant weight on cosine (0.50) reflects that behavioral identity is primarily a directional property — two vectors that "point in the same direction" in 48-D behavioral space are likely from the same user. Mahalanobis (0.40) provides strong secondary discrimination. Stability (0.10) is a minor quality adjustment — overweighting it would confuse behavioral consistency with behavioral identity.

#### Design Decision: Exp-Mahalanobis over d/(1+d)

| Property | `exp(-d/sqrt(D))` | `d/(1+d)` |
| :---------- | :---------- | :---------- |
| Probabilistic interpretation | Unnormalized Gaussian kernel (likelihood ratio under Gaussian prototype model) | Logistic/sigmoid |
| Dimensionality normalization | Explicit sqrt(D) | None |
| Discriminability at large d | Faster decay | Slow saturation |
| Gradient | Smooth, C-infinity | Smooth |

The `exp(-d/sqrt(D))` form has a natural probabilistic interpretation and sharper discrimination at large distances, making it more effective at confidently rejecting dissimilar vectors.

#### Design Decision: Diagonal Mahalanobis

Full covariance estimation for D=48 requires at minimum O(D^2) = 2,304 samples for the covariance matrix to be non-singular and well-conditioned. In streaming authentication with sessions of 20-100 events per prototype, full covariance estimation is impractical. The diagonal approximation:
1. Requires only D samples per prototype for reliable estimation
2. Captures per-feature variance (important: touch pressure and scroll velocity have very different natural variances)
3. Is exact when features are uncorrelated — a reasonable approximation for well-designed behavioral features

### 3.5 Prototype Quarantine

**File:** `app/prototype/quarantine_manager.py`

When composite similarity falls below `THRESHOLD_CREATE = 0.50`, the behavioral vector enters a **CandidatePool** rather than immediately creating a new prototype. A CandidatePrototype is promoted to a full Prototype only when all three conditions are simultaneously satisfied:

| Condition | Parameter | Value | Rationale |
| :---------- | :---------- | :------ | :---------- |
| Observation count | `N_MIN` | 3 | A single deviating event cannot create a prototype |
| Temporal spread | `T_MIN` | 30s | Burst events within a short window cannot satisfy count alone |
| Consistency | `CONSISTENCY_THRESHOLD` | 0.72 | Mean cosine to centroid >= 0.72, requiring directional agreement (angle <= 44 degrees in 48-D space) |

Candidates expire after `T_EXPIRE = 600s` if not promoted.

#### Design Decision: Why N_MIN = 3?

One observation: trivially noise. Two observations: statistically insufficient for a 48-dimensional centroid estimate. Three observations provide a minimal centroid and allow consistency to be computed against a non-trivial average. Three is the minimum count that makes all three conditions jointly non-trivial (count, temporal spread, and consistency each play a distinct role).

#### Design Decision: Why T_MIN = 30 seconds?

Human interaction patterns emerge across minutes of natural device use, not seconds. A 30-second spread means the user must have been naturally interacting for at least half a minute before a new behavioral archetype is accepted. This defeats burst-injection attacks: an attacker who rapidly submits multiple crafted events cannot satisfy the temporal spread requirement without sustaining the attack across 30 seconds of interaction.

#### Design Decision: Why Consistency >= 0.72?

In 48-dimensional space, cosine similarity of 0.72 corresponds to an inter-vector angle of `arccos(0.72) ≈ 44 degrees`. Requiring mean cosine to centroid >= 0.72 means all observations assigned to a candidate must lie within a 44-degree cone in behavioral space. This threshold is:
- **Permissive enough** for natural behavioral variation within a consistent context (same task, similar environment)
- **Restrictive enough** to reject observations from fundamentally different behavioral modes

#### Security Consequence

Without quarantine, an attacker with momentary physical access needs O(1) events to inject a new behavioral archetype. With quarantine, the cost rises to at least 3 events spread over 30+ seconds with consistent behavioral pattern — a substantially higher bar that requires sustained, coherent behavioral mimicry.

### 3.6 Adaptive Learning Rate

**File:** `app/prototype/prototype_engine.py`

Prototype updates use Exponential Moving Average (EMA) with an exponentially decaying learning rate:

```
eta(n) = eta_base * exp(-n / tau) + eta_floor

mu_new        = (1 - eta) * mu_old + eta * v
sigma^2_EMA   = (1 - eta) * sigma^2_old + eta * (v - mu_new)^2
sigma^2_final = 0.7 * sigma^2_EMA + 0.3 * sigma^2_session
```

Parameters: `eta_base = 0.30`, `tau = 50`, `eta_floor = 0.01`

| n | eta |
| :--- | :------ |
| 0 | 0.31 |
| 10 | 0.133 |
| 50 | 0.121 |
| 100 | 0.011 |
| inf | 0.01 |

**Why decaying rate with a floor?**

A constant high rate risks prototype drift from behavioral noise. A constant low rate prevents adaptation to genuine long-term behavioral change. The decaying schedule resolves this naturally: new prototypes need fast early calibration (high eta at small n), while mature prototypes should resist noise (low eta at large n). The floor `eta_floor = 0.01` prevents asymptotic freezing: even a prototype with n=10,000 still adapts at rate 0.01, allowing genuine long-term behavioral evolution to propagate through the model.

**Why variance blending (0.7 EMA + 0.3 session)?**

The EMA variance update can shrink arbitrarily small if the prototype receives repeatedly similar vectors. An extremely small prototype variance makes the Mahalanobis distance hypersensitive: tiny deviations produce very large distances, causing false rejection. Blending 30% of the current session's Welford variance maintains calibration to actual current behavioral variability, preventing the prototype from overfitting to a narrow region of behavioral space.

### 3.7 Quality-Based Prototype Lifecycle

```
Q(k) = log(1 + n_k) * exp(-lambda_age * age_k) * max(sim_k, 0.1)
```

Where:
- `n_k`: support count (events matched to prototype k)
- `age_k`: seconds since last update
- `sim_k`: best composite similarity in the current batch
- `lambda_age = 1/86400` (one-day half-life)

When the maximum prototype count (`MAX_PROTOTYPES = 15`) is reached, the prototype with the lowest Q score is removed.

**Component rationale:**

| Component | Formula | Effect |
| :---------- | :---------- | :---------- |
| Support | `log(1+n)` | Logarithmic returns: establishes that 100 obs is valuable but not 10x more than 10 |
| Recency | `exp(-lambda*age)` | A prototype unmatched for 1 day has Q reduced by exp(-1) ≈ 0.37; 7 days: exp(-7) ≈ 0.001 |
| Relevance | `max(sim, 0.1)` | Prototype not matching current behavior scores lower (min 0.1 prevents zero) |

A freshly created prototype (n=1) with recent age and good relevance scores higher than an old high-support prototype that no longer matches current behavior. This keeps the prototype set tuned to the user's current behavioral state rather than their historical one.

### 3.8 Behavioral Session Fingerprint Engine (Layer-2c)

**File:** `app/preprocessing/transition_engine.py`

Layer-2c captures the **sequential navigation structure** of user behavior — which event transitions the user characteristically makes and in which order — via an EMA-updated Markov transition probability matrix.

#### Signal Definition

```
I(prev→curr)        = -log₂( P_EMA[prev][curr] )    (information content, bits)
transition_surprise = 1 - exp( -I(prev→curr) / TRANS_SIGMA )
```

**Properties:**

| Scenario | P | I | ts | Meaning |
| :---------- | :--- | :--- | :---- | :---------- |
| Always-expected transition | ≈1.0 | ≈0 bits | ≈0.0 | No sequential anomaly |
| Familiar transition | 0.5 | 1 bit | 0.28 | Seen roughly half the time |
| Uncommon transition | 0.1 | 3.3 bits | 0.67 | Atypical but not impossible |
| Never-seen transition | → 0 | floored | 0.92 | Completely novel navigation |

**Constants:**

| Constant | Value | Rationale |
| :---------- | :------- | :---------- |
| `TRANS_EMA_ALPHA` | 0.15 | Learning rate for matrix update. At α=0.15, a transition seen 10 times reaches ≈80% of long-run probability. |
| `TRANS_SIGMA` | 3.0 | Normalization for bits→[0,1). At 3 bits, ts≈0.63 — "uncommon but not alarming". |
| `MIN_TRANSITION_PROB` | 1e-4 | Floor to prevent log(0). Never-seen transition produces ts≈0.92. |

#### EMA Update Rule

For each source event type A, the matrix row `P[A]` is updated as follows when transition A→B is observed:

```
For all k in P[A]:  P[A][k] *= (1 - TRANS_EMA_ALPHA)   ← decay all entries
P[A][B] += TRANS_EMA_ALPHA                               ← reinforce observed
```

This produces an unnormalized EMA accumulator. Probability of A→B is `P[A][B] / sum(P[A].values())`.

The EMA decay ensures that transitions performed long ago gradually lose weight, allowing the model to adapt to evolving navigation habits.

#### Leakage-Free Property

Following the same discipline as drift metrics, the transition surprise is computed from the **pre-update** probability (before the current transition is incorporated):

```
Step 1: Read P[prev][curr]        ← pre-update (no leakage)
Step 2: Compute surprise          ← using pre-update probability
Step 3: Update P with A→B         ← post-scoring update
Step 4: Advance prev_event_type   ← chain advance
```

If the matrix were updated before scoring, the current transition would inflate its own probability, systematically under-estimating the surprise of novel transitions.

#### Why Sequential Structure is Orthogonal to Existing Signals

The three existing Layer-2 signals measure:
- **Drift**: how far the current feature vector is from prior means (amplitude/magnitude)
- **Stability**: coherence of the current window (variance within a window)
- **Similarity**: whether the current vector shape matches stored prototype shapes (geometry)

None of these capture **ordering** — whether the user navigates pages in their habitual sequence. An attacker who has observed the genuine user's feature vectors (e.g., via shoulder-surfing) can reproduce the magnitude and shape but is likely to navigate in an unexpected order (e.g., accessing transfer screens before checking balance). The transition engine detects exactly this.

#### Session State

Per-session state in `SessionState`:
- `transition_probs: Dict[str, Dict[str, float]]` — EMA Markov matrix
- `prev_event_type: Optional[str]` — event type of the previous event

For the first event in a session (no prior context), `transition_surprise = 0.0` — a neutral value that makes no sequential claim.

---

## 4. Layer 3 — Graph Attention Network (GAT)

**Files:** `app/gat/`, `app/layer3/`

Layer 3 provides deep session-level behavioral authentication through a Graph Attention Network operating on a temporal behavioral graph. This component was designed and implemented by the GAT team member.

### 4.1 Behavioral Graph Construction

Each behavioral session window is represented as a temporal graph `G = (V, E)` where:
- **Nodes V**: individual behavioral events within the last 20-second sliding window
- **Edges E**: temporal relationships between events, governed by the edge construction rule

The 20-second window captures a meaningful unit of user interaction — sufficient events for the GAT to recognize complex interaction sequences — while remaining responsive to behavioral change.

### 4.2 Node Feature Engineering

Each node (event) is represented as a **56-dimensional feature vector**:

| Dimensions | Source | Description |
| :---------- | :------- | :---------- |
| 0-47 | `event_data.vector` | 48-D behavioral vector from mobile SDK |
| 48-55 | Hash embedding | 8-byte event-type encoding: `sha256(event_type)[:8]` |

**File:** `app/layer3/layer3_processor.py`

**Why hash embedding over one-hot encoding?**
One-hot encoding requires a fixed, known vocabulary of event types. As new event types are introduced, the embedding dimension grows and the model must be retrained. Hash embeddings are fixed-size regardless of vocabulary size. Additionally, semantically related event type names (e.g., `scroll_up` / `scroll_down`) receive nearby hash codes, providing an implicit soft-similarity signal.

### 4.3 Edge Construction and Windowing

**Windowing:** All events within the last 20 seconds are included; no event-count truncation.

**Edge rule:** For each source node `i`:

```python
for each event i in events:
    seen = { event[i].type }
    for j in range(i+1, end):
        connect(i, j)           # always connect, including repeats
        if event[j].type not in seen:
            seen.add(event[j].type)
            if len(seen) >= 4:  # stop after 4 distinct new types
                break
```

Key properties:
- **Repeat-inclusive**: duplicate event types are always connected; they do not count toward the distinct-type limit
- **Distinct-4 termination**: fan-out from node `i` stops after 4 distinct *new* event types are encountered
- **Result**: edges from a node can exceed 4, because repeats do not consume the distinct quota

Repeat-inclusive connectivity is intentional: repetitive behavioral patterns (e.g., repeated scrolls) are identity-bearing signals. The distinct-4 limit prevents over-connected graphs in high-variety event streams while preserving enough context for multi-type pattern recognition.

### 4.4 SiameseGATNetwork Architecture

**File:** `app/gat/gat_network.py`

The GAT model uses a **Siamese architecture**: the session graph and the user's reference profile are both processed through a shared-weight encoder, and the outputs are compared via cosine similarity.

```
Input:   G_session (56-D node features)
         G_profile  (stored 64-D embedding per user)

Encoder (shared weights):
  GATConv(56 -> 128, heads=4)   concat  -> 512-D
  GATConv(512 -> 128, heads=4)  mean    -> 128-D
  Global mean pooling                   -> 128-D
  Linear(128 -> 64)                     -> 64-D session embedding

Score:   cosine_similarity(session_emb, profile_emb) in [0, 1]
```

**Why GAT over GCN?**
Graph Convolutional Networks apply uniform weight to all neighbors. Graph Attention Networks learn per-edge attention weights, allowing the model to focus on the most identity-discriminative temporal transitions. In behavioral authentication, not all event transitions carry equal identity signal — transitions between certain event type pairs are more characteristic than others. GAT learns which transitions matter, without requiring manual feature engineering of these distinctions.

**Why Siamese architecture?**
The Siamese network is the standard architecture for similarity learning under the constraint that the two inputs belong to the same class of objects (both are behavioral sessions). Sharing encoder weights ensures that the session embedding and profile embedding live in the same metric space. Cosine similarity in this learned space provides a scale-invariant, bounded similarity score without requiring additional calibration.

### 4.5 Triplet Loss Training

**File:** `app/gat/trainer.py`

The model is trained with triplet loss across users in the behavioral log store:

```
L = max(0, d(anchor, positive) - d(anchor, negative) + margin)

d(a, b) = 1 - cosine_similarity(a, b)
margin = 0.5
```

Where:
- **Anchor**: a session graph from user U
- **Positive**: another session graph from user U
- **Negative**: a session graph from a different user V

Training: 30 epochs across all available users. After training, the checkpoint is persisted to Azure Blob Storage (`cbsa-models` container).

**Why triplet loss?**
Triplet loss directly optimizes the objective of authentication: same-user sessions should be closer than different-user sessions by at least `margin` in the learned embedding space. This is a direct optimization target, unlike cross-entropy classification which requires a fixed user set at training time and generalizes poorly to unseen users.

### 4.6 Inference and Profile Management

**Files:** `app/gat/engine.py`, `app/layer3/layer3_manager.py`

At inference time:
1. The 20-second event window is converted to a graph by `layer3_processor.py`
2. The GAT encoder produces a 64-D session embedding
3. The user's stored 64-D profile vector (Cosmos DB / local disk fallback) is retrieved
4. Score: `gat_score = cosine_similarity(session_emb, profile_emb)` in `[0, 1]`

If no profile exists, the current embedding is stored as the initial profile. Profiles are updated via EMA after each inference call.

**Adaptive kappa — contribution weight:**

```
kappa = clip(0.10 + 0.30 * (1 - trust_score), 0.10, 0.40)
```

When trust is high (0.9), `kappa = 0.13` — GAT contributes minimally, preserving the Layer 2 signal.
When trust is low (0.2), `kappa = 0.34` — GAT is weighted more heavily to provide deep analysis.

This prevents GAT from overriding a well-established trust signal while amplifying its influence precisely when Layer 2 is uncertain.

---

## 5. Layer 4 — Trust Engine

**File:** `app/trust/trust_engine.py`

Layer 4 aggregates the complete behavioral signal into a continuous trust score via an Exponential Moving Average model with adaptive coefficient, and issues zone-based authentication decisions.

### 5.1 Raw Trust Signal

```
R_t = w_sim * sim_t + w_stab * S_t + w_drift * (1 - D_t) + w_trans * (1 - ts_t)

D_t  = 0.50 * d_short_t + 0.30 * d_medium_t + 0.20 * d_long_t   (three-scale composite drift)
ts_t = transition_surprise_t                                       (Markov sequential fingerprint)

Weights:  w_sim = 0.40,  w_stab = 0.20,  w_drift = 0.30,  w_trans = 0.10
          (sum = 1.0)
```

**Bounds verification:**
- `D_t`: all three drift weights sum to 1.0; each drift term is in [0,1); therefore `D_t in [0, 1)`
- `ts_t in [0, 1)` by construction (see Section 3.8)
- Maximum (sim=1, S=1, D=0, ts=0): 0.40 + 0.20 + 0.30 + 0.10×1 = 1.0
- Minimum (sim=0, S=0, D=1, ts=1): 0 + 0 + 0 + 0 = 0.0

`R_t in [0, 1]` without clipping, by construction.

**Weight rationale:**

| Signal | Weight | Rationale |
| :---------- | :------- | :---------- |
| Similarity | 0.40 | Primary identity signal: how well does current behavior match the stored prototype? |
| Drift-complement | 0.30 | Penalizes behavioral deviation across all three temporal scales. |
| Stability | 0.20 | Behavioral coherence quality. Lower weight: stability captures *how consistently* the user behaves, not *who* they are. |
| Transition fingerprint | 0.10 | Sequential navigation structure. Captures whether the user follows their habitual app-navigation sequence at the event-ordering level — orthogonal to all magnitude/shape signals. |

**Why `(1 - ts_t)` rather than `ts_t` directly?**
The transition engine outputs *surprise* — high values indicate unexpected behavior. Using `(1 - ts_t)` inverts this to a *familiarity* contribution, matching the convention of all other terms in the formula: each term is a "trust evidence" value where high = trustworthy. A user making their characteristic navigation transitions receives the full `w_trans` contribution; an attacker making surprising transitions receives zero.

**Why drift-complement `(1 - D_t)` rather than raw drift?**
Using `(1 - D_t)` inverts the drift signal so all three terms contribute positively to trust: zero drift contributes +0.30; maximal drift contributes 0. This keeps the weight interpretation uniform — each term is a "trust evidence" component.

**Three-scale composite drift rationale (50/30/20):**

| Component | Weight | Rationale |
| :---------- | :------- | :---------- |
| `d_short` (5-event) | 0.50 | Dominant weight: sudden micro-behavioral deviations are the strongest impostor signal. An attacker switching sessions immediately registers high short drift. |
| `d_medium` (20-event) | 0.30 | Secondary: episodic-scale drift catches gradual impersonation and context switches missed by the 5-event window. |
| `d_long` (Welford) | 0.20 | Tertiary: long-term drift is a valuable baseline signal but responds slowly by design; it should not dominate the instantaneous trust computation. |

The three-scale decomposition is strictly more expressive than any two-scale system. A two-scale (short + long) system has a temporal blind spot in the 6–19 event range: behavioral shifts that manifest over 10–15 events are invisible to the 5-event window and too transient to register on the Welford baseline. The medium scale closes this gap.

### 5.2 Adaptive EMA Coefficient

```
alpha_eff_max = ALPHA_MAX * (0.90 + 0.10 * cohesion_t)

alpha_t = clip(alpha_eff_max - gamma * d_short_t, alpha_min, alpha_eff_max)
gamma   = ALPHA_MAX - ALPHA_MIN = 0.55

ALPHA_MIN = 0.30,  ALPHA_MAX = 0.85

T_t = alpha_t * T_{t-1} + (1 - alpha_t) * R_t
```

Where `cohesion_t` is the prototype topology cohesion score (see Section 3, Prototype Engine).

**Effective alpha range by cohesion:**

| cohesion | alpha_eff_max | Interpretation |
|----------|--------------|----------------|
| 1.0 (all prototypes aligned) | 0.85 | Full inertia — single behavioral mode, trust should be stable |
| 0.5 (moderate spread) | 0.8075 | Slightly reduced inertia |
| 0.0 (maximally spread) | 0.765 | Reduced inertia — multi-modal user, trust should respond more dynamically |

**Effective alpha range by d_short (at cohesion=1.0):**

| d_short | alpha | EMA half-life |
|---------|-------|---------------|
| 0.0 (stable) | 0.85 | ~4.3 events |
| 0.5 (moderate) | 0.575 | ~1.3 events |
| 1.0 (extreme) | 0.30 | ~0.57 events |

**Design rationale — drift-modulated alpha:**
Alpha encodes resistance to trust change — the EMA half-life. Making alpha a function of `d_short` implements a principled speed-accuracy trade-off:
- Stable behavior (d_short ≈ 0): `alpha ≈ alpha_eff_max`, strong temporal inertia. A single anomalous event does not immediately drop trust for a well-established session.
- Sudden anomaly (d_short ≈ 1): `alpha = 0.30`, fast response. A sustained impostor who produces high short drift is rejected within 1-2 events rather than waiting for the EMA to propagate.

**Design rationale — cohesion-modulated alpha_eff_max:**
The effective alpha ceiling is modulated by the prototype topology cohesion of the user's behavioral model. This encodes a structural insight: **how much inertia is appropriate depends on the geometric structure of the user's behavioral identity**.

- A user with `cohesion = 1.0` has all stored prototypes pointing in the same direction in 48-D behavioral space. This is a **single-mode behavioral identity**: the user behaves consistently across all contexts. For such a user, trust should be highly stable — a single unusual event is probably noise. Full alpha_max inertia is appropriate.
- A user with low cohesion has prototypes spread across behavioral space — they are a **multi-modal user** with genuinely different behavioral patterns depending on context (e.g., different typing styles at home vs. work). For such a user, trust must respond more dynamically to context changes, otherwise trust computed in one behavioral mode will incorrectly penalize legitimate behavior in another mode. Reducing `alpha_eff_max` slightly lowers temporal inertia, making the EMA more responsive.

The 90/10 split (`0.90 + 0.10 * cohesion`) keeps the cohesion modulation subtle: the ceiling varies only between 0.765 and 0.85. This avoids making the EMA unstable for multi-modal users while still encoding the structural difference. A larger cohesion range would make the EMA overly sensitive; a smaller range would make the modulation meaningless.

**Why linear dependency on d_short?**
A linear coupling `alpha = alpha_eff_max - gamma * d_short` is the simplest monotone function spanning the full `[alpha_min, alpha_eff_max]` range. The linear function has a direct interpretation: each unit of short drift reduces alpha by exactly gamma.

### 5.3 Trust Zones and Decisions

| Zone | Trust Range | Decision | Meaning |
| :------ | :---------- | :------- | :---------- |
| SAFE | `[theta_safe, 1.0]` | ACCEPT | Strong behavioral match |
| MONITOR | `[theta_risk, theta_safe)` | LOG | Mild deviation, passive monitoring |
| RISK | `[0, theta_risk)` | REJECT | Sustained behavioral anomaly |
| WARMUP | (any) | WARMUP | Insufficient data, n < WARMUP_SKIP_EVENTS |

`theta_safe = 0.80`, `theta_risk = 0.40`

The MONITOR dead-zone `[0.40, 0.80)` provides hysteresis: trust scores oscillating near a single threshold do not produce alternating ACCEPT/REJECT decisions. Continuous authentication without hysteresis produces unstable, annoying decisions for legitimate sessions with natural behavioral variance.

### 5.4 GAT Escalation Logic

GAT analysis is triggered conditionally, not periodically:

```python
should_escalate = (
    anomaly_indicator > ANOMALY_ESCALATION_THRESHOLD    # = 0.40
    OR n_consecutive_uncertain > N_UNCERTAIN_ESCALATION   # = 3
)
AND (time_since_last_gat > T_RECHECK_SECONDS)           # = 30s
```

**Why event-driven over periodic?**
- **Efficiency**: Stable sessions (high trust, low drift) do not warrant deep GAT analysis. Periodic GAT wastes compute on unnecessary inference.
- **Responsiveness**: An anomaly triggers GAT at the moment the anomaly indicator exceeds the threshold, not at a fixed future interval.
- **Recheck cooldown**: The 30-second minimum interval between GAT calls prevents repeated inference when the anomaly indicator is persistently high, bounding compute cost while maintaining responsiveness.

---

## 6. Storage Architecture

**Files:** `app/azure/`, `app/storage/`

### Azure Cosmos DB

| Container | Partition Key | Content |
| :---------- | :---------- | :---------- |
| `computation-logs` | `/userId` | Per-event engine metrics, trust scores, GAT similarity |
| `user-profiles` | `/userId` | 64-D GAT profile vectors |
| `cbsa-behavioral` | `/username` | Behavioral event logs (JSONL) |

### Azure Blob Storage

Container `cbsa-models`: trained GAT model checkpoint (`.pth` file). At startup, the GAT engine downloads the latest checkpoint from Blob Storage. Falls back to local `data/checkpoints/` (development) or random initialisation (no checkpoint available).

### SQLite (Development / Local)

Database: `cbsa.db`

| Table | Purpose |
|-------|---------|
| `users` | Identity, initialization state, warmup event count |
| `prototypes` | Per-user prototypes: vector, variance, support count, timestamps |
| `behaviour_logs` | Full behavioral event history per user-session |

### BehaviourRepository

**File:** `app/storage/repository.py`

All storage access within the pipeline goes through `BehaviourRepository`, a unified interface over both SQLite and Cosmos DB stores. This decouples pipeline logic from storage backend — the Cosmos DB containers can be replaced without modifying any pipeline code.

### Session TTL and Event Limits

In-memory session state is bounded:
- `MAX_SESSION_EVENTS = 10,000`: maximum events before the session is evicted and restarted
- `SESSION_TTL_SECONDS = 600`: sessions inactive for 10 minutes are evicted

---

## 7. Pipeline Execution Order and Invariants

The four-layer pipeline executes in strict order per event. No layer may be skipped or reordered:

```
1. validate_and_extract(raw_event)              -> BehaviourEvent       (Layer 1)
2. process_event(behaviour_event)               -> PreprocessedBehaviour (Layer 2a + 2c)
   └─ internally:
      a. update_session_buffer(event)           -> snapshot + session_state
      b. compute drift/stability (pre-snapshot)
      c. compute_transition_surprise(session_state, event.event_type)
         └─ uses pre-update Markov matrix, then updates it (leakage-free)
3. match_and_update_prototypes(preprocessed)    -> PrototypeMetrics      (Layer 2b)
   └─ passes through preprocessed.transition_surprise unchanged
4. [if escalation condition AND len(window) >= MIN_EVENTS_FOR_GAT_ESCALATION]
   gat_engine.infer(session_window)             -> gat_similarity        (Layer 3)
5. evaluate(preprocessed, prototype_metrics,    -> TrustResult           (Layer 4)
            gat_similarity, trust_state)
   └─ R_t includes w_trans * (1 - transition_surprise)
```

**Runtime invariants** (checked by `app/core/invariants.py`):

| Invariant | Check |
| :---------- | :---------- |
| Vector dimension | `len(vector) == 48` |
| Vector range | All elements in `[0, 1]` |
| Drift range | `d_short, d_medium, d_long in [0, 1]` |
| Stability | `stability in (0, 1]` |
| Similarity | `similarity in [0, 1]` |
| Transition surprise | `transition_surprise in [0, 1]` |
| Prototype confidence | `prototype_confidence in [0, 1]` |
| Prototype topology cohesion | `prototype_topology_cohesion in [0, 1]` |
| Anomaly indicator | `anomaly_indicator in [0, 1]` |
| Trust score | `trust_score in [0, 1]` |
| GAT score | `gat_score in [0, 1]` |

Any invariant violation raises `InvariantError` and terminates processing for that event. The pipeline never propagates an invalid intermediate state to downstream layers.

**Warmup protocol:** New users require `WARMUP_SKIP_EVENTS = 20` events before prototype matching begins. During warmup, Layer 4 returns a `WARMUP` status. The 20-event warmup gives Welford's algorithm sufficient observations for a meaningful variance estimate in 48-dimensional space.

---

## 8. Evaluation Methodology

**File:** `tests/evaluation.py`

The system is evaluated using standard biometric authentication metrics.

### False Acceptance Rate (FAR)

```
FAR(tau) = |{attack events : trust > tau}| / |{total attack events}|
```

Fraction of impostor events that produced trust above threshold tau.

### False Rejection Rate (FRR)

```
FRR(tau) = |{genuine events : trust <= tau}| / |{total genuine events}|
```

Fraction of genuine events that produced trust at or below threshold tau.

### Equal Error Rate (EER)

The threshold `tau*` at which `FAR(tau*) ≈ FRR(tau*)`. Computed by sweeping tau across 1,001 evenly spaced values in `[0, 1]` and finding the threshold minimizing `|FAR(tau) - FRR(tau)|`.

EER is the standard single-number summary for biometric systems. Lower is better; a random classifier has EER = 50%.

### Test Scenarios

| Scenario | Description |
| :---------- | :---------- |
| `standard` | Normal behavioral event stream for an enrolled user |
| `attack` | Behavioral stream from a different user (impostor) |
| `cold_start` | New user: verifies quarantine enrollment and trust growth |
| `failure_cases` | Layer 1 rejection: wrong vector length, out-of-range values, duplicate nonce, non-monotonic timestamp |
| `prototype_analytics` | Tabular per-user prototype statistics: support, age, quality scores |

---

## 9. System Constants Reference

**File:** `app/core/constants.py`

| Constant | Value | Description |
| :---------- | :------- | :---------- |
| `VECTOR_DIM` | 48 | Behavioral vector dimensionality |
| `WARMUP_SKIP_EVENTS` | 20 | Events before prototype matching begins |
| `MAX_SESSION_EVENTS` | 10,000 | Session event capacity |
| `SESSION_TTL_SECONDS` | 600 | Session inactivity expiry |
| `ACCEPT_THRESHOLD` | 0.65 | Binary accept/reject threshold for evaluation |
| `THETA_SAFE` | 0.80 | Trust zone: SAFE lower bound |
| `THETA_RISK` | 0.40 | Trust zone: RISK upper bound |
| `ALPHA_MAX` | 0.85 | EMA max coefficient (stable behavior) |
| `ALPHA_MIN` | 0.30 | EMA min coefficient (anomalous behavior) |
| `W_SIM` | 0.40 | Trust signal: similarity weight |
| `W_STAB` | 0.20 | Trust signal: stability weight |
| `W_DRIFT` | 0.30 | Trust signal: drift-complement weight |
| `W_TRANSITION` | 0.10 | Trust signal: Markov transition fingerprint weight |
| `SHORT_DRIFT_WEIGHT` | 0.50 | Composite drift: short-term (5-event) fraction |
| `MEDIUM_DRIFT_WEIGHT` | 0.30 | Composite drift: medium-term (20-event) fraction |
| `LONG_DRIFT_WEIGHT` | 0.20 | Composite drift: long-term (Welford) fraction |
| `COHESION_ALPHA_FLOOR` | 0.90 | Cohesion modulation: base fraction of ALPHA_MAX |
| `COHESION_ALPHA_RANGE` | 0.10 | Cohesion modulation: cohesion contribution range |
| `ETA_BASE` | 0.30 | Adaptive learning rate: initial component |
| `ETA_FLOOR` | 0.01 | Adaptive learning rate: minimum |
| `TAU` | 50 | Adaptive learning rate: decay constant |
| `THRESHOLD_UPDATE` | 0.75 | Similarity threshold to update prototype |
| `THRESHOLD_CREATE` | 0.50 | Similarity threshold to route to quarantine |
| `N_MIN_OBSERVATIONS` | 3 | Quarantine: minimum observations for promotion |
| `T_MIN_SPAN_SECONDS` | 30.0 | Quarantine: minimum temporal spread |
| `CONSISTENCY_THRESHOLD` | 0.72 | Quarantine: minimum cosine consistency |
| `T_EXPIRE_SECONDS` | 600.0 | Quarantine: candidate expiry |
| `MAX_PROTOTYPES` | 15 | Maximum prototypes per user |
| `LAMBDA_AGE` | 1/86400 | Prototype quality: age decay rate (1 per day) |
| `GAT_WINDOW_SECONDS` | 20 | GAT: temporal window for graph |
| `GAT_EDGE_DISTINCT_TARGET` | 4 | GAT: distinct event-type limit per edge fan-out |
| `GAT_NODE_FEATURE_DIM` | 56 | GAT: node feature dim (48 + 8 type embedding) |
| `ANOMALY_ESCALATION_THRESHOLD` | 0.40 | GAT escalation: anomaly indicator threshold |
| `N_UNCERTAIN_ESCALATION` | 3 | GAT escalation: consecutive uncertain events |
| `T_RECHECK_SECONDS` | 30.0 | GAT escalation: recheck cooldown |
| `MIN_EVENTS_FOR_GAT_ESCALATION` | 5 | GAT prerequisite: minimum session-window size |
| `TRANS_EMA_ALPHA` | 0.15 | Transition engine: Markov matrix EMA learning rate |
| `TRANS_SIGMA` | 3.0 | Transition engine: surprise normalization scale |
| `MIN_TRANSITION_PROB` | 1e-4 | Transition engine: probability floor (prevents log₂(0)) |
