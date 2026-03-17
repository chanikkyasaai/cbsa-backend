"""
Layer-2 Preprocessing — Leakage-Free Behavioural State Extraction

Processing Order (CRITICAL FOR CORRECTNESS)
--------------------------------------------
1. Capture PRE-UPDATE snapshot of session statistics (before v_t is incorporated)
2. Update session buffer with v_t (post-update state)
3. Compute ALL drift/stability metrics using ONLY the pre-update snapshot

This ordering is enforced by update_session_buffer(), which ATOMICALLY captures
the snapshot and then performs the update, returning both. The caller (this
module) uses only the snapshot for all metric computations.

Window Vector Selection
-----------------------
The window_vector used for prototype matching is the POST-UPDATE short window mean.
This is intentional and correct: prototype matching should represent the current
behavioral state (including v_t), while drift computation requires the prior
context (pre-update) to avoid including v_t in its own reference distribution.

  - Drift: compare v_t against prior context (pre-update) — no leakage
  - Prototype matching: represent current state (post-update) — correct

This asymmetry is deliberate and mathematically justified.
"""

from __future__ import annotations

import numpy as np

from app.preprocessing.buffer_manager import update_session_buffer
from app.preprocessing.drift_engine import (
    _DEFAULT_SIGMA,
    compute_behavioural_consistency,
    compute_long_drift,
    compute_short_drift,
    compute_stability_score,
)
from app.models.behaviour_event import BehaviourEvent
from app.models.preprocessed_behaviour import PreprocessedBehaviour


def process_event(event: BehaviourEvent) -> PreprocessedBehaviour:
    """
    Extract the complete behavioural state from an incoming event.

    Returns a PreprocessedBehaviour with ALL fields bounded in [0,1]
    (or (0,1] for stability_score which is bounded by exp construction).
    No decisions are made here — this is a pure measurement function.

    Parameters
    ----------
    event : BehaviourEvent
        Validated incoming behavioural event (48-D vector, timestamp, user metadata)

    Returns
    -------
    PreprocessedBehaviour
        Rich behavioural state vector for Layer-2 prototype matching and
        Layer-4 trust computation.
    """
    # ── STEP 1 + 2: Atomically capture snapshot and update buffer ──────────
    # session_state: post-update (v_t included)
    # snapshot:      pre-update (v_t NOT included in any statistics)
    session_state, snapshot = update_session_buffer(event)

    current_vector = event.vector

    # ── STEP 3: Compute metrics using PRE-UPDATE statistics only ───────────
    # The sigma parameter is the drift scale for exp-normalization.
    # Using _DEFAULT_SIGMA (= 0.15 * sqrt(48)) globally.
    # Production extension: replace with per-user adaptive sigma.
    sigma = _DEFAULT_SIGMA

    short_drift = compute_short_drift(
        current_vector=current_vector,
        pre_short_window_mean=snapshot.short_window_mean,
        sigma=sigma,
    )

    long_drift = compute_long_drift(
        pre_short_window_mean=snapshot.short_window_mean,
        pre_long_term_mean=snapshot.long_term_mean,
        sigma=sigma,
    )

    stability_score = compute_stability_score(
        pre_short_window_vectors=snapshot.short_window_vectors,
        pre_long_term_variance=snapshot.long_term_variance,
    )

    behavioural_consistency = compute_behavioural_consistency(
        pre_short_window_vectors=snapshot.short_window_vectors,
    )

    # ── Window vector: post-update mean (represents current state) ─────────
    # We use the post-update short window (includes v_t) as the prototype
    # matching vector. This correctly represents "current behavior", unlike
    # the drift reference which must exclude v_t to prevent leakage.
    post_window_vecs = np.vstack(list(session_state.short_window))
    window_vector = np.mean(post_window_vecs, axis=0)

    return PreprocessedBehaviour(
        window_vector=window_vector,
        short_drift=short_drift,
        long_drift=long_drift,
        stability_score=stability_score,
        variance_vector=session_state.running_variance.copy(),
        behavioural_consistency=behavioural_consistency,
        sigma_ref=sigma,
    )
