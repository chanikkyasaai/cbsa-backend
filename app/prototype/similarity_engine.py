"""
Similarity Engine — Research-Grade Composite Similarity

Design Principles
-----------------
1. ALL COMPONENTS BOUNDED IN [0, 1]:
   Every input to the composite similarity function is independently bounded
   in [0, 1] before combination. This ensures the output is always in [0, 1]
   by construction, without requiring post-hoc clipping.

2. EXP-MAHALANOBIS KERNEL:
       k_M(d) = exp(-d_M / sqrt(D))
   Why exp(-d/sqrt(D)) over d/(1+d)?
     (a) Probabilistic interpretation: k_M is an unnormalized Gaussian kernel.
         It represents the likelihood ratio under a Gaussian prototype model.
     (b) sqrt(D) normalization: For a D-dimensional standard normal vector,
         E[||x||_2] ≈ sqrt(D). Dividing by sqrt(D) normalizes for dimensionality,
         making the kernel value comparable across systems with different D.
     (c) Better discriminability at large distances: exp(-d) decays faster
         than d/(1+d) for d > 1, providing sharper rejection of dissimilar vectors.
     (d) Smooth gradient: exp is infinitely differentiable, useful if this
         score is ever used in a gradient-based update.

3. COMPOSITE SIMILARITY WEIGHTS:
       sim = 0.50 * cos + 0.40 * k_M + 0.10 * S
   Rationale for weights:
     - Cosine (0.50): Primary discriminator. Captures behavioral direction
       (the "shape" of the behavioral pattern). Scale-invariant: two vectors
       of different magnitudes but same direction are maximally similar.
     - Mahalanobis kernel (0.40): Secondary discriminator. Captures deviation
       accounting for per-user variability. A deviation that is large relative
       to the user's typical variance is penalized more than the same absolute
       deviation for a high-variance user.
     - Stability (0.10): Quality modifier. Reduces score when the current
       behavioral window is erratic. Keeps the primary discrimination in
       cosine + Mahalanobis, with stability as a minor quality adjustment.
   The 50/40/10 split is intentional: cosine and Mahalanobis are orthogonal
   measures (one direction, one magnitude-relative), combined they cover the
   full geometry. Overweighting stability would confuse behavioral quality
   with behavioral identity.

4. DIAGONAL MAHALANOBIS APPROXIMATION:
       d_M = sqrt(sum_i [(v_i - mu_i)^2 / max(sigma^2_i, eps)])
   Full covariance requires O(D^2) samples for stable estimation.
   For D=48, this requires ~2304 samples before the covariance matrix
   is reliably estimated. In streaming behavioral authentication with
   ~10s windows, this is impractical. The diagonal approximation:
     (a) Requires only D samples for reliable estimation
     (b) Captures per-feature variance (important: some features are
         naturally high-variance, others low-variance)
     (c) Is exact when features are uncorrelated (common after PCA/decorrelation)
"""

import numpy as np

VECTOR_DIM: int = 48
_SQRT_D: float = float(np.sqrt(VECTOR_DIM))
_LOG1P_N_MAX: float = float(np.log1p(200))   # log(201), for support_strength normalization

# Composite similarity weights — must sum to 1.0
_W_COSINE: float = 0.50
_W_MAHAL: float = 0.40
_W_STABILITY: float = 0.10


def cosine_similarity(vector_a: np.ndarray, vector_b: np.ndarray) -> float:
    """
    Cosine similarity in [0, 1].

        cos(a, b) = (a · b) / (||a|| * ||b||)

    Clipped to [0, 1]: behavioral vectors are non-negative (normalized to [0,1]^D),
    so negative cosine similarity is not meaningful in this domain.
    Clipping rather than using abs() is preferred: a negative cosine indicates
    near-orthogonality (maximally dissimilar), not mirrored similarity.

    Returns: float in [0, 1]
    """
    denom = float(np.linalg.norm(vector_a) * np.linalg.norm(vector_b))
    if denom < 1e-10:
        return 0.0
    value = float(np.dot(vector_a, vector_b) / denom)
    return max(0.0, min(1.0, value))


def mahalanobis_distance(
    sample: np.ndarray,
    prototype: np.ndarray,
    variance_vector: np.ndarray,
) -> float:
    """
    Diagonal Mahalanobis distance.

        d_M(v, mu, sigma^2) = sqrt(sum_i [(v_i - mu_i)^2 / max(sigma^2_i, eps)])

    The diagonal approximation (treating features as independent) is justified
    in streaming settings where full covariance estimation is infeasible.
    See module docstring for full rationale.

    Returns: float in [0, inf)
    """
    safe_var = np.maximum(variance_vector, 1e-8)
    delta = sample - prototype
    return float(np.sqrt(np.sum((delta * delta) / safe_var)))


def mahalanobis_kernel(distance: float) -> float:
    """
    Gaussian kernel mapping Mahalanobis distance to similarity.

        k_M(d) = exp(-d / sqrt(D))

    The sqrt(D) normalization accounts for dimensionality: for a D-dimensional
    standard normal vector x, E[||x||_2] ≈ sqrt(D). Dividing by sqrt(D)
    normalizes the distance so that one-sigma deviation maps to:
        k_M(sqrt(D)) = exp(-1) ≈ 0.368

    This means a Mahalanobis distance equal to the expected magnitude of a
    random unit normal vector gives a kernel value of ~0.37 — a natural calibration.

    Returns: float in (0, 1]
    """
    return float(np.exp(-distance / _SQRT_D))


def composite_similarity(
    cosine: float,
    mahalanobis_dist: float,
    stability_score: float,
) -> float:
    """
    Composite similarity in [0, 1].

        sim = 0.50 * cos(v, mu) + 0.40 * exp(-d_M/sqrt(D)) + 0.10 * S

    All three components are bounded in [0, 1] by construction:
      - cosine:             clipped to [0, 1]
      - mahalanobis_kernel: exp(-·) in (0, 1]
      - stability_score:    exp(-variance_ratio) in (0, 1]

    Therefore the weighted sum is in [0, 1] without additional clipping.
    The final clip is a numerical safety guard only.

    IMPORTANT: This function takes mahalanobis_dist (raw distance), not the
    kernel value — it calls mahalanobis_kernel() internally. This keeps the
    interface consistent (callers compute distance, not kernel).

    Returns: float in [0, 1]
    """
    mahal_sim = mahalanobis_kernel(mahalanobis_dist)
    score = _W_COSINE * cosine + _W_MAHAL * mahal_sim + _W_STABILITY * stability_score
    return max(0.0, min(1.0, float(score)))


def compute_prototype_confidence(similarity: float, support_count: int) -> float:
    """
    Prototype confidence: similarity adjusted by prototype maturity.

        conf(sim, n) = sim * (1 - exp(-n / n_ref)),    n_ref = 20

    Rationale: A newly created prototype (n=1) has high statistical uncertainty.
    Even if the current vector matches it perfectly (sim=1), the prototype may
    represent a noise artefact or transient behavior. The maturity factor:
        (1 - exp(-n/20))
    grows from 0 toward 1 as n increases:
      n=1:   maturity ≈ 0.049   (very low confidence)
      n=10:  maturity ≈ 0.394   (moderate)
      n=20:  maturity ≈ 0.632   (one n_ref)
      n=60:  maturity ≈ 0.950   (high confidence)
      n=100: maturity ≈ 0.993   (near full)

    This maturity weighting is inspired by the concept of "effective sample size"
    in Bayesian statistics: a prior-dominated estimate for small n, data-dominated
    for large n.

    Returns: float in [0, 1]
    """
    if support_count <= 0:
        return 0.0
    n_ref = 20.0
    maturity = 1.0 - float(np.exp(-support_count / n_ref))
    return max(0.0, min(1.0, float(similarity) * maturity))


def compute_prototype_support_strength(support_count: int) -> float:
    """
    Log-normalised support count in [0, 1].

        strength = log(1 + n) / log(1 + n_max),    n_max = 200

    Log scaling is used because:
      (a) The marginal value of additional observations diminishes (log utility)
      (b) Prevents high-count prototypes from dominating numerically in comparisons
      (c) A prototype with n=200 vs n=100 should not be twice as "strong" —
          both are well-established. Log(201)/log(101) ≈ 1.05 vs 1.0 reflects this.

    n_max = 200 is chosen as the practical ceiling for a single prototype's
    support count in a continuous authentication session.

    Returns: float in [0, 1]
    """
    if support_count <= 0:
        return 0.0
    return min(1.0, float(np.log1p(support_count)) / _LOG1P_N_MAX)


def compute_anomaly_indicator(similarity: float, short_drift: float) -> float:
    """
    Anomaly indicator: joint signal of low similarity AND high drift.

        anomaly(t) = (1 - sim_t) * (0.5 + 0.5 * d_short_t)

    Mathematical properties:
      sim=1, d_short=0  ->  0.0 * 0.5 = 0.0   (no anomaly)
      sim=0, d_short=1  ->  1.0 * 1.0 = 1.0   (maximum anomaly)
      sim=0, d_short=0  ->  1.0 * 0.5 = 0.5   (similarity failure, no drift)
      sim=0.5, d_short=0.5 ->  0.5 * 0.75 = 0.375

    The 0.5 base weight (d_short=0 term) is critical: even without observed
    drift, a similarity failure is a meaningful anomaly signal (e.g., session
    hijacking or replay attacks that don't deviate from normal drift patterns).

    The drift multiplier (0.5 * d_short) amplifies the indicator when behavior
    is both dissimilar AND rapidly changing — the hallmark of session hijacking
    rather than legitimate behavioral change (which typically shows gradual drift
    with maintained similarity).

    Returns: float in [0, 1]
    """
    return max(0.0, min(1.0, float((1.0 - similarity) * (0.5 + 0.5 * short_drift))))


# ── Legacy compatibility shim ───────────────────────────────────────────────
# The old normalize_mahalanobis(d) = d/(1+d) is replaced by mahalanobis_kernel.
# This alias is kept to avoid breaking any external callers that may reference it.
def normalize_mahalanobis(distance: float) -> float:
    """
    DEPRECATED: Use mahalanobis_kernel() instead.
    Retained for backward compatibility. Will be removed in a future version.
    The old d/(1+d) formulation is less discriminative than exp(-d/sqrt(D)).
    """
    return mahalanobis_kernel(distance)
