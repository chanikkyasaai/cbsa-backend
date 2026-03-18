"""
PreprocessedBehaviour — Rich Layer-2 Output Model

All fields are bounded and mathematically defined.
No decisions are made here — only behavioural state representation.
"""

from dataclasses import dataclass

import numpy as np


@dataclass(slots=True)
class PreprocessedBehaviour:
    """
    Behavioural state extracted from a single incoming event.

    Fields
    ------
    window_vector : np.ndarray  (D=48)
        Mean of the post-update short window. Used as the representative
        vector for prototype matching. This is the ONLY field that uses
        post-update statistics — it represents the current behavioral state.

    short_drift : float  [0, 1)
        Short-term drift: deviation of current vector from pre-update 5-event window mean.
            d_short(t) = 1 - exp(-||v_t - mu_short^{t-1}||_2 / (sqrt(D)*sigma))
        Captures sudden within-session behavioral changes (micro-behavioral scale).
        Leakage-free.

    medium_drift : float  [0, 1)
        Medium-term drift: deviation of current vector from pre-update 20-event window mean.
            d_medium(t) = 1 - exp(-||v_t - mu_medium^{t-1}||_2 / (sqrt(D)*sigma))
        Captures behavioral mode transitions at the episodic interaction scale —
        slower than short drift, faster than long drift. Leakage-free.

    long_drift : float  [0, 1)
        Long-term drift: deviation of local window mean from global session mean.
            d_long(t) = 1 - exp(-||mu_short^{t-1} - mu_global^{t-1}||_2 / (sqrt(D)*sigma))
        Captures gradual behavioral drift across the full session. Leakage-free.

    stability_score : float  (0, 1]
        Exponential variance-ratio stability measure:
            S(t) = exp(-(1/D) * sum_i [Var_short,i / max(Var_global,i, eps)])
        When short variance near 0: S approaches 1 (stable behavior).
        When short variance >> long variance: S approaches 0 (erratic behavior).
        Bounded by construction via the exponential — no clipping required.

    variance_vector : np.ndarray  (D=48)
        Per-dimension running variance of the current session (Welford's sigma^2).
        Used for Mahalanobis distance computation in prototype matching.

    behavioural_consistency : float  [0, 1]
        Mean cosine similarity of recent window vectors to their centroid.
            C(t) = (1/|W|) * sum_{v in W} cos(v, mean(W))
        Measures directional agreement within the recent window.
        Distinct from stability_score: consistency captures directional alignment,
        stability captures amplitude variance.

    sigma_ref : float
        The drift scale parameter sigma used for exp-normalization.
        Logged for diagnostics. Candidate for per-user adaptation in production.

    transition_surprise : float  [0, 1)
        Behavioral Session Fingerprint score from Layer-2c (transition_engine).
        Captures the sequential structure of the user's navigation pattern via
        an EMA-updated Markov transition probability matrix.

            I(prev→curr) = -log₂( P_EMA[prev][curr] )       (information content)
            transition_surprise = 1 - exp( -I(prev→curr) / TRANS_SIGMA )

        High surprise (→ 1.0): current event transition was unexpected given the
        user's established navigation habits — characteristic of an attacker who
        does not know the user's typical app-navigation flow.

        Low surprise (→ 0.0): transition was exactly as expected — strong signal
        that this is the genuine user following their habitual navigation sequence.

        Returns 0.0 for the first event in a session (no prior context yet).
        Leakage-free: P is read BEFORE incorporating the current transition.
    """
    window_vector: np.ndarray
    short_drift: float
    medium_drift: float
    long_drift: float
    stability_score: float
    variance_vector: np.ndarray
    behavioural_consistency: float
    sigma_ref: float
    transition_surprise: float
