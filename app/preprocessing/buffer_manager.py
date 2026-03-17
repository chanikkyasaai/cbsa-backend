"""
Buffer Manager — Leakage-Free Session State Tracking

Statistical Leakage Prevention
--------------------------------
All drift and stability metrics are computed against PRE-UPDATE statistics
(i.e., session state captured before incorporating v_t). This ensures that
the current event is never part of its own reference distribution, preventing
look-ahead bias in all downstream metric computations.

The core invariant enforced by this module:
    snapshot = {μ_long^{t-1}, σ²_long^{t-1}, window^{t-1}}  ← BEFORE update
    update buffer with v_t
    d_long = ||μ_short^{t-1} - μ_long^{t-1}||               ← no leakage

update_session_buffer() ATOMICALLY captures the pre-update snapshot and then
performs the buffer update. Callers receive both the updated state and the
snapshot; all metric computations must use only the snapshot values.

Welford's Online Algorithm (for running_mean / running_variance):
  For sample count n, new sample x:
    δ₁ = x - μ_{n-1}
    μ_n = μ_{n-1} + δ₁/n
    δ₂ = x - μ_n
    M2_n = M2_{n-1} + δ₁ * δ₂
    σ²_n = M2_n / (n - 1)    [unbiased estimator]

  Reference: Welford, B.P. (1962). "Note on a method for calculating corrected
  sums of squares and products." Technometrics, 4(3), 419–420.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from app.models.behaviour_event import BehaviourEvent
from app.storage.memory_store import SessionState, memory_store

MAX_HISTORY = 256


@dataclass
class PreUpdateSnapshot:
    """
    Snapshot of session statistics captured BEFORE incorporating the current event.

    All drift and stability computations in preprocessing.py use ONLY these
    pre-update values, ensuring v_t is never part of its own reference distribution.

    Attributes:
        short_window_mean    : Mean of last W events (not including current v_t)
        short_window_vectors : Stacked array of last W vectors (not including v_t)
        long_term_mean       : Running mean before this event (Welford's μ_{n-1})
        long_term_variance   : Running variance before this event (Welford's σ²_{n-1})
        sample_count         : Number of events seen before this event
        has_window_data      : False only on the very first event ever
    """
    short_window_mean: np.ndarray
    short_window_vectors: np.ndarray
    long_term_mean: np.ndarray
    long_term_variance: np.ndarray
    sample_count: int
    has_window_data: bool


def update_session_buffer(event: BehaviourEvent) -> tuple[SessionState, PreUpdateSnapshot]:
    """
    Update session buffer with the incoming event.

    Returns (session_state, snapshot) where:
      - session_state  : fully updated state (POST-update, for downstream use)
      - snapshot       : statistics captured PRE-update (for drift/stability computation)

    The caller (preprocessing.py) MUST use snapshot for all metric computations
    and may use session_state only for the post-update window_vector.
    """
    session_state = memory_store.get_or_create_session(event.session_id)
    vector = event.vector

    # ──────────────────────────────────────────────────────────────────────
    # STEP 1: Capture pre-update statistics (NO v_t included)
    # ──────────────────────────────────────────────────────────────────────
    pre_count = session_state.sample_count

    if len(session_state.short_window) > 0:
        short_window_vecs = np.vstack(list(session_state.short_window))  # (W, D)
        pre_short_mean = np.mean(short_window_vecs, axis=0)              # (D,)
        has_window = True
    else:
        # Cold start: no prior window — use the incoming vector as its own reference.
        # Drift on the first event is defined as 0 (no deviation from self).
        short_window_vecs = vector.reshape(1, -1)
        pre_short_mean = vector.copy()
        has_window = False

    pre_long_mean = session_state.running_mean.copy()
    pre_long_variance = session_state.running_variance.copy()

    snapshot = PreUpdateSnapshot(
        short_window_mean=pre_short_mean,
        short_window_vectors=short_window_vecs,
        long_term_mean=pre_long_mean,
        long_term_variance=pre_long_variance,
        sample_count=pre_count,
        has_window_data=has_window,
    )

    # ──────────────────────────────────────────────────────────────────────
    # STEP 2: Update session state with v_t (POST-update)
    # ──────────────────────────────────────────────────────────────────────
    session_state.short_window.append(vector)
    session_state.event_history.append(vector)
    if len(session_state.event_history) > MAX_HISTORY:
        session_state.event_history.pop(0)

    session_state.sample_count += 1
    n = session_state.sample_count

    if n == 1:
        session_state.running_mean = vector.copy()
        session_state.m2 = np.zeros_like(vector)
        session_state.running_variance = np.zeros_like(vector)
    else:
        # Welford's online algorithm — numerically stable incremental update
        delta1 = vector - session_state.running_mean
        session_state.running_mean = session_state.running_mean + delta1 / n
        delta2 = vector - session_state.running_mean
        session_state.m2 = session_state.m2 + delta1 * delta2
        session_state.running_variance = session_state.m2 / max(n - 1, 1)

    session_state.last_timestamp = event.timestamp

    return session_state, snapshot
