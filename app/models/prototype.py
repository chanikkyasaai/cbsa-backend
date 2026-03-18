"""
Prototype and PrototypeMetrics — Extended Layer-2 Data Models

PrototypeMetrics now produces a rich behavioural state vector,
not just a similarity score. All fields are in [0,1] and mathematically
bounded. NO decisions are made at this layer — that is Layer-4's responsibility.
"""

from dataclasses import dataclass
from datetime import datetime
from typing import Optional

import numpy as np


@dataclass(slots=True)
class Prototype:
    """
    A stored behavioural prototype for a user.

    The vector (centroid) and variance are updated via adaptive EMA:
        eta(n) = eta_base * exp(-n/tau) + eta_floor
        mu_new = (1 - eta) * mu_old + eta * v
        sigma^2_new = (1 - eta) * sigma^2_old + eta * (v - mu_new)^2

    This adaptive rate allows rapid learning early (large eta) and
    stabilises as the prototype matures (eta -> eta_floor), while never
    completely freezing — accommodating legitimate behavioural drift.
    """
    prototype_id: int
    vector: np.ndarray        # (D=48,) centroid — EMA of supporting observations
    variance: np.ndarray      # (D=48,) per-dimension EMA variance
    support_count: int        # total observations incorporated into this prototype
    created_at: datetime
    last_updated: datetime


@dataclass(slots=True)
class PrototypeMetrics:
    """
    Rich behavioural state output from Layer-2.

    This is the complete Layer-2 -> Layer-4 interface. Every field is
    bounded in [0,1] with a precise mathematical definition. Layer-4
    (TrustEngine) combines these into a continuous trust score and decision.
    This layer produces NO decisions — it is a pure measurement layer.

    Fields
    ------
    similarity_score : float  [0, 1]
        Composite similarity to the best matching prototype:
            sim = 0.50*cos(v, mu) + 0.40*exp(-d_M/sqrt(D)) + 0.10*S
        All three components are bounded in [0,1]; the weighted sum is too.

    short_drift : float  [0, 1)
        Leakage-free, exp-normalized short-term drift (5-event micro-behavioral scale).

    medium_drift : float  [0, 1)
        Leakage-free, exp-normalized medium-term drift (20-event episodic scale).
        Captures behavioral mode transitions between the short micro-behavioral
        scale and the long identity baseline.

    long_drift : float  [0, 1)
        Leakage-free, exp-normalized long-term drift (full session running mean).

    stability_score : float  (0, 1]
        Variance-ratio stability. Bounded by exp construction. See PreprocessedBehaviour.

    matched_prototype_id : Optional[int]
        DB id of the best matching prototype. None during cold start quarantine phase.

    prototype_confidence : float  [0, 1]
        Similarity adjusted by prototype maturity:
            conf = sim * (1 - exp(-n / n_ref)),  n_ref = 20
        A prototype with support_count=1 gives low confidence even at perfect
        similarity. Full confidence (>0.95) is achieved around n=60 observations.

    behavioural_consistency : float  [0, 1]
        Mean cosine similarity of recent window vectors to the window centroid.
        Measures directional coherence of the current behavioral episode.

    prototype_support_strength : float  [0, 1]
        Log-normalised support count:
            strength = log(1 + n) / log(1 + n_max),  n_max = 200
        Log scaling prevents high-count prototypes from dominating numerically.
        Reflects how well-established this prototype is relative to the system maximum.

    anomaly_indicator : float  [0, 1]
        Joint signal of low similarity AND high short-drift:
            anomaly = (1 - sim) * (0.5 + 0.5 * d_short)
        Mathematical properties:
          sim=1, d_short=0  ->  anomaly = 0.0  (no anomaly)
          sim=0, d_short=1  ->  anomaly = 1.0  (maximum anomaly)
          sim=0, d_short=0  ->  anomaly = 0.5  (similarity failure, no drift)
        The 0.5 base ensures even zero-drift anomalies (e.g., replay attacks)
        are captured. The drift multiplier amplifies the score when behavior
        is both dissimilar AND rapidly changing — characteristic of session
        hijacking rather than legitimate use.

    prototype_topology_cohesion : float  [0, 1]
        Mean pairwise cosine similarity between all of the user's stored prototypes.
            cohesion = (2 / (K*(K-1))) * sum_{i<j} cos(mu_i, mu_j)
        Measures the geometric tightness of the user's behavioral model:
          cohesion = 1.0 : all prototypes point in the same behavioral direction
                           (single consistent behavioral mode)
          cohesion < 0.5 : prototypes are spread across behavioral space
                           (multi-modal user: different contexts produce different modes)
        A single prototype always returns cohesion = 1.0.
        Used by Layer-4 to modulate EMA inertia: tight behavioral models deserve
        higher temporal smoothing; spread models should respond more dynamically.

    transition_surprise : float  [0, 1)
        Behavioral Session Fingerprint score passed through from Layer-2c.
        Measures the unexpectedness of the current event transition relative to
        the user's established Markov navigation model.  High surprise indicates
        an atypical navigation sequence; low surprise indicates habitual flow.
        Passed through from PreprocessedBehaviour unchanged; included here so
        Layer-4 receives the full behavioral state vector in one object.
    """
    similarity_score: float
    short_drift: float
    medium_drift: float
    long_drift: float
    stability_score: float
    matched_prototype_id: Optional[int]
    prototype_confidence: float
    behavioural_consistency: float
    prototype_support_strength: float
    anomaly_indicator: float
    prototype_topology_cohesion: float
    transition_surprise: float
