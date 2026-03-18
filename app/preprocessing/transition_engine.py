"""
Behavioral Session Fingerprint Engine  (Layer-2c)
==================================================

Captures the SEQUENTIAL structure of a user's app-navigation behavior by
maintaining a per-session EMA-updated Markov transition probability matrix
over event types.  After each event the engine produces a *transition
surprise* score in [0, 1):

    transition_surprise = 1 - exp( -I(prev → curr) / TRANS_SIGMA )

where

    I(prev → curr) = -log₂( P[prev][curr] )      (information content, bits)

Properties
----------
  surprise = 0.0   P ≈ 1.0 — the user always makes this transition
                   Full trust contribution from this signal.

  surprise → 1.0   P ≈ 0   — the transition is nearly impossible for this user
                   Maximum suspicion; trust contribution approaches 0.

Why this is orthogonal to existing signals
------------------------------------------
  - Drift metrics   measure HOW FAR the current feature vector deviates from
                    prior means — they are magnitude/amplitude signals.
  - Prototype sim   measures whether the SHAPE of the current window matches
                    known behavioral prototypes — a geometric signal.
  - Transition surp measures SEQUENTIAL ORDERING — whether the user navigates
                    the app in their habitual event-transition patterns.

An attacker who mirrors the genuine user's feature magnitudes and shapes can
still produce unusual navigation sequences (e.g., accessing settings directly
after login rather than visiting the dashboard first).  This engine detects
exactly that pattern.

Leakage-Free Design
-------------------
Transition surprise is computed using the PRIOR probability (before the current
transition is incorporated), mirroring the leakage-free design of drift metrics:
  1. Read P[prev][curr] from the EMA matrix (pre-update).
  2. Compute surprise from the pre-update probability.
  3. Then perform the EMA update with the current observation.

This ensures the current transition cannot inflate its own probability before
the surprise is scored.

EMA Update Rule
---------------
For each source event type A, we maintain a per-row probability vector.
When we observe transition A → B:

    For all k in P[A]:  P[A][k] *= (1 - TRANS_EMA_ALPHA)   # decay all
    P[A][B] += TRANS_EMA_ALPHA                               # reinforce B

This is equivalent to an unnormalized EMA accumulator.  The probability of
transition A → B is the normalised value: P[A][B] / sum(P[A].values()).

The EMA decay ensures that stale transitions (performed long ago) gradually
lose influence, so the matrix adapts to changing navigation habits over time.

Constants
---------
TRANS_EMA_ALPHA   : float = 0.15   Learning rate for transition matrix updates.
                    Larger → faster adaptation; smaller → more stable model.
                    At alpha=0.15, a transition reinforced 10 times achieves
                    ≈80% of its long-run probability.

TRANS_SIGMA       : float = 3.0    Normalization for information content → [0,1).
                    surprise_bits = -log₂(prob).
                    At prob=1.0: bits=0, surprise=0.
                    At prob=0.5: bits=1, surprise≈0.28 (expected/familiar).
                    At prob=0.1: bits≈3.3, surprise≈0.67 (uncommon).
                    At prob=0.01: bits≈6.6, surprise≈0.89 (very unusual).

MIN_TRANSITION_PROB : float = 1e-4  Floor probability.  Prevents log(0) on
                    transitions that have never been observed.  Also provides a
                    meaningful surprise value (≈0.92) for completely novel
                    transitions.
"""

from __future__ import annotations

import math
from typing import Optional

# ── Constants ─────────────────────────────────────────────────────────────────

TRANS_EMA_ALPHA: float = 0.15
TRANS_SIGMA: float = 3.0
MIN_TRANSITION_PROB: float = 1e-4


# ── Core Function ─────────────────────────────────────────────────────────────

def compute_transition_surprise(
    session_state,
    curr_event_type: str,
) -> float:
    """
    Compute the behavioral transition surprise for the current event and update
    the session-local Markov probability matrix.

    Parameters
    ----------
    session_state  : SessionState (from memory_store) — carries
                     `transition_probs` (dict-of-dicts) and `prev_event_type`.
    curr_event_type: The event_type string of the current incoming event.

    Returns
    -------
    float in [0, 1)
        0.0  — no previous event, or transition was fully expected.
        →1.0 — transition was highly unusual for this user.

    Side Effects
    ------------
    Mutates session_state.transition_probs  (EMA update after scoring).
    Mutates session_state.prev_event_type   (advances the chain).
    """
    prev = session_state.prev_event_type

    # ── 1. Compute surprise from PRE-UPDATE probability (leakage-free) ────
    if prev is None:
        # No prior event in this session: no sequential structure yet.
        # Return 0.0 (no penalty; full trust contribution).
        surprise = 0.0
    else:
        row = session_state.transition_probs.get(prev)
        if row:
            total = sum(row.values())
            raw_prob = (row.get(curr_event_type, 0.0) / total) if total > 0.0 else 0.0
        else:
            raw_prob = 0.0

        prob = max(raw_prob, MIN_TRANSITION_PROB)
        surprise_bits = -math.log2(prob)
        surprise = 1.0 - math.exp(-surprise_bits / TRANS_SIGMA)

    surprise = max(0.0, min(1.0, surprise))

    # ── 2. EMA update of the transition matrix (post-scoring) ─────────────
    if prev is not None:
        if prev not in session_state.transition_probs:
            session_state.transition_probs[prev] = {}
        row = session_state.transition_probs[prev]
        # Decay all existing entries for this source type
        for k in list(row.keys()):
            row[k] = row[k] * (1.0 - TRANS_EMA_ALPHA)
        # Reinforce the observed transition
        row[curr_event_type] = row.get(curr_event_type, 0.0) + TRANS_EMA_ALPHA

    # ── 3. Advance the Markov chain ────────────────────────────────────────
    session_state.prev_event_type = curr_event_type

    return float(surprise)
