"""
Drift Engine — Leakage-Free, Mathematically Bounded Drift Computation

Design Principles
-----------------
1. LEAKAGE-FREE: All metrics use PRE-UPDATE statistics (snapshot before v_t).
   The caller (preprocessing.py) passes a PreUpdateSnapshot, not the live state.

2. DIMENSION-NORMALIZED L2:
       d_raw / sqrt(D)
   Dividing by sqrt(D) makes the metric scale-invariant across vector dimensions.
   For D-dimensional unit vectors, maximum L2 = 2, so normalized max = 2/sqrt(D),
   independent of D. This allows meaningful comparison across systems.

3. EXPONENTIAL NORMALIZATION:
       f(d) = 1 - exp(-d / sigma)
   Maps [0, inf) -> [0, 1) monotonically.
   - f(0)      = 0       (no drift)
   - f(sigma)  ≈ 0.632   (one-sigma drift, moderate)
   - f(inf)    -> 1      (extreme drift, asymptotic upper bound)
   Preferred over d/(1+d) (Platt-style) because:
     (a) Has a natural scale parameter sigma that can be adapted per-user
     (b) Exponential tail provides better discrimination at high drift values
     (c) Derivative is always positive with decreasing magnitude — smooth response

4. STABILITY (BOUNDED BY CONSTRUCTION):
       S(t) = exp(-(1/D) * sum_i [Var_short,i / max(Var_global,i, eps)])
   The exp(-·) ensures S in (0,1] regardless of variance values.
   The 1/D normalization is dimension-count invariant.

Sigma Reference Value
---------------------
The default sigma = 0.15 * sqrt(48) ≈ 1.04 is grounded as follows:
  - Behavioral vectors are in [0,1]^48 (normalized features)
  - A "moderate" behavioral change might shift features by 0.15 on average
  - The L2 of a uniform 0.15 shift across all 48 dims = 0.15 * sqrt(48) ≈ 1.04
  - At sigma=1.04, a one-sigma deviation maps to d_normalized ≈ 0.632
  - Larger shifts produce d_normalized -> 1 asymptotically
This gives an interpretable threshold: d > 0.6 is meaningfully elevated drift.

Per-User Adaptation (Future)
-----------------------------
sigma should be adapted per-user based on historical drift distribution:
    sigma_u = mean(d_short_history_u) * 1.5
This would replace the global default for enrolled users.
"""

from __future__ import annotations

import numpy as np

VECTOR_DIM: int = 48
_SQRT_D: float = float(np.sqrt(VECTOR_DIM))

# Default scale parameter sigma for exp-normalization during cold start.
# sigma = 0.15 * sqrt(D): see docstring rationale above.
_DEFAULT_SIGMA: float = 0.15 * _SQRT_D


def normalized_l2(vector_a: np.ndarray, vector_b: np.ndarray) -> float:
    """
    Dimension-normalized L2 distance.

        d_norm(a, b) = ||a - b||_2 / sqrt(D)

    Returns float in [0, inf).
    For unit vectors in [0,1]^D, practical range is [0, 2/sqrt(D)].
    """
    return float(np.linalg.norm(vector_a - vector_b)) / _SQRT_D


def exp_normalize(raw: float, sigma: float = _DEFAULT_SIGMA) -> float:
    """
    Map [0, inf) -> [0, 1) via exponential normalization.

        f(d) = 1 - exp(-d / sigma)

    The sigma parameter sets the characteristic scale: at d = sigma,
    f(sigma) = 1 - 1/e ≈ 0.632.
    """
    if sigma <= 0.0:
        sigma = _DEFAULT_SIGMA
    return float(1.0 - np.exp(-raw / sigma))


def compute_short_drift(
    current_vector: np.ndarray,
    pre_short_window_mean: np.ndarray,
    sigma: float = _DEFAULT_SIGMA,
) -> float:
    """
    Short-term drift: deviation of the current event from the recent local context.

        d_short(t) = 1 - exp(-||v_t - mu_window^{t-1}||_2 / (sqrt(D) * sigma))

    Uses PRE-UPDATE window mean — v_t is NOT included in mu_window^{t-1},
    ensuring the current event is never part of its own reference distribution.

    Interpretation:
      d_short near 0: current behavior matches recent context (expected)
      d_short near 1: sudden deviation from recent behavior (alert signal)

    Returns: float in [0, 1)
    """
    raw = normalized_l2(current_vector, pre_short_window_mean)
    return exp_normalize(raw, sigma)


def compute_long_drift(
    pre_short_window_mean: np.ndarray,
    pre_long_term_mean: np.ndarray,
    sigma: float = _DEFAULT_SIGMA,
) -> float:
    """
    Long-term drift: deviation of local window from the global session baseline.

        d_long(t) = 1 - exp(-||mu_window^{t-1} - mu_global^{t-1}||_2 / (sqrt(D) * sigma))

    Both means are pre-update (snapshot), ensuring no leakage.
    Long drift captures gradual behavioral drift over time:
      - Increases slowly as behavior shifts session-by-session
      - A legitimate user changing habits will show elevated d_long before
        the baseline adapts via the Welford running mean

    Returns: float in [0, 1)
    """
    raw = normalized_l2(pre_short_window_mean, pre_long_term_mean)
    return exp_normalize(raw, sigma)


def compute_stability_score(
    pre_short_window_vectors: np.ndarray,
    pre_long_term_variance: np.ndarray,
) -> float:
    """
    Stability score: within-window variance relative to long-term variance.

        S(t) = exp(-(1/D) * sum_i [Var_short,i / max(Var_global,i, eps)])

    The variance ratio per dimension measures how much the current behavioral
    window deviates from the user's established variability pattern.

    Interpretation:
      Short_var << Long_var  ->  ratio near 0  ->  S near 1  (very stable)
      Short_var == Long_var  ->  ratio = 1     ->  S = exp(-1) ≈ 0.37  (neutral)
      Short_var >> Long_var  ->  ratio large   ->  S near 0  (very erratic)

    Mathematical guarantee: S in (0, 1] by construction (exp of negative value).
    No clipping required. The 1/D normalization ensures D-independence.

    Cold start: Returns 1.0 if fewer than 2 window vectors exist (neutral assumption).

    Returns: float in (0, 1]
    """
    if pre_short_window_vectors.shape[0] < 2:
        return 1.0

    short_var = np.var(pre_short_window_vectors, axis=0)        # (D,)
    long_var = np.maximum(pre_long_term_variance, 1e-8)          # (D,) — avoid div/0

    # Per-dimension variance ratio
    ratio = short_var / long_var                                  # (D,)
    mean_ratio = float(np.mean(ratio))

    return float(np.exp(-mean_ratio))


def compute_behavioural_consistency(
    pre_short_window_vectors: np.ndarray,
) -> float:
    """
    Behavioural consistency: mean cosine similarity of window vectors to centroid.

        C(t) = (1/|W|) * sum_{v in W} cos(v, mean(W))

    Captures directional coherence within the current behavioral episode.
    When all vectors point in the same behavioral direction: C approaches 1.
    When the window is scattered: C approaches 0.

    Distinct from stability_score:
      - stability_score measures amplitude (variance magnitude)
      - behavioural_consistency measures direction (cosine alignment)
    An attacker replaying high-amplitude but misdirected events would show
    high stability but low consistency — these metrics together provide
    orthogonal discriminative signals.

    Cold start: Returns 1.0 if fewer than 2 vectors available.

    Returns: float in [0, 1]
    """
    if pre_short_window_vectors.shape[0] < 2:
        return 1.0

    window_mean = np.mean(pre_short_window_vectors, axis=0)
    norm_mean = float(np.linalg.norm(window_mean))
    if norm_mean < 1e-10:
        return 0.0

    sims = []
    for vec in pre_short_window_vectors:
        norm_v = float(np.linalg.norm(vec))
        if norm_v < 1e-10:
            sims.append(0.0)
        else:
            sim = float(np.dot(vec, window_mean) / (norm_v * norm_mean))
            sims.append(max(0.0, min(1.0, sim)))

    return float(np.mean(sims))
