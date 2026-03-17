"""
Prototype Engine — Prototype Matching, Update, and Lifecycle Management

This module is the core of Layer-2's prototype-based behavioral modeling,
implementing the following design principles:

1. QUARANTINE-GATED PROTOTYPE CREATION:
   New behavioral patterns must pass quarantine (N_MIN observations,
   temporal spread T_MIN, consistency threshold) before becoming prototypes.
   This prevents transient or adversarially injected patterns from polluting
   the prototype set. See quarantine_manager.py for the full protocol.

2. ADAPTIVE LEARNING RATE:
   Prototype updates use an exponentially decaying learning rate with a floor:
       eta(n) = eta_base * exp(-n/tau) + eta_floor
   At n=0: eta ≈ 0.31 (fast early learning for new prototypes).
   At n→∞: eta → eta_floor = 0.01 (slow adaptation for mature prototypes).
   The floor ensures prototypes never freeze — users with legitimately
   evolving behavior can still update their profile indefinitely.

3. QUALITY-BASED PROTOTYPE LIFECYCLE:
   Prototype retention uses a composite quality score:
       Q(k) = log(1+n) * exp(-lambda * age) * max(sim, 0.1)
   balancing support (n), recency (age decay), and current relevance (similarity).
   When the prototype limit is reached, the lowest-Q prototype is removed,
   prioritising deletion of old, rarely-matched, low-support prototypes.

4. RICH OUTPUT:
   Full 9-field PrototypeMetrics output including prototype_confidence,
   behavioural_consistency, prototype_support_strength, and anomaly_indicator.
   All fields are bounded in [0,1] for use by the Layer-4 Trust Engine.

Adaptive Threshold Design
--------------------------
  THRESHOLD_UPDATE  = 0.75: similarity >= 0.75 -> update existing prototype
  THRESHOLD_CREATE  = 0.50: similarity < 0.50  -> route to quarantine
  Gap [0.50, 0.75]: no action — behavior is moderately similar but not
                    sufficiently aligned to update. Re-evaluated at next event.

The dead-zone in [0.50, 0.75] prevents proto-drift under similarity-boundary
attacks where an adversary probes events at just below the update threshold.

Future extension: THRESHOLD_UPDATE and THRESHOLD_CREATE should be per-user,
derived from the distribution of the user's historical similarity scores:
  threshold_update(u) = mu_sim(u) - 1.5 * sigma_sim(u)
  threshold_create(u) = mu_sim(u) - 3.0 * sigma_sim(u)
"""

from __future__ import annotations

import time
from datetime import datetime
from typing import Optional, Union

import numpy as np

from app.prototype.quarantine_manager import quarantine_manager
from app.prototype.similarity_engine import (
    composite_similarity,
    compute_anomaly_indicator,
    compute_prototype_confidence,
    compute_prototype_support_strength,
    cosine_similarity,
    mahalanobis_distance,
)
from app.models.preprocessed_behaviour import PreprocessedBehaviour
from app.models.prototype import Prototype, PrototypeMetrics

# ── Configuration ────────────────────────────────────────────────────────────

MAX_PROTOTYPES_PER_USER: int = 15

# Adaptive learning rate parameters
ETA_BASE: float = 0.30    # Initial (maximum) learning rate
ETA_FLOOR: float = 0.01   # Minimum learning rate (prevents prototype freezing)
TAU: float = 50.0         # Time constant: rate halves every ~tau*ln2 ≈ 35 updates

# Global fallback prototype matching thresholds (used when user has < 30 similarity observations)
THRESHOLD_UPDATE: float = 0.75   # Minimum similarity to update an existing prototype
THRESHOLD_CREATE: float = 0.50   # Maximum similarity before routing to quarantine

# Adaptive threshold requires at least this many similarity observations per user
MIN_ADAPTIVE_OBSERVATIONS: int = 30

# Quality score age decay
LAMBDA_AGE: float = 1.0 / (3600 * 24)   # Decay constant: 1/day (prototype ages over ~days)


# ── Adaptive Learning Rate ───────────────────────────────────────────────────

def _adaptive_learning_rate(support_count: int) -> float:
    """
    Compute the adaptive learning rate for prototype updates.

        eta(n) = eta_base * exp(-n / tau) + eta_floor

    Design rationale:
      - At n=0:   eta ≈ 0.30 + 0.01 = 0.31  (fast: new prototype, high uncertainty)
      - At n=tau: eta ≈ 0.30/e + 0.01 ≈ 0.12  (moderate)
      - At n=inf: eta → eta_floor = 0.01      (slow: mature, adapts only to real drift)

    The eta_floor (0.01) is critical: it ensures prototypes never become
    completely frozen. A user whose behavior legitimately evolves will still
    see their prototype adapt, just slowly. Without eta_floor, prototypes
    would asymptotically freeze and legitimate long-term behavioral change
    would be permanently flagged as anomalous.

    Numerical reference:
      - n=0:    eta ≈ 0.31    (rapid initial learning)
      - n=10:   eta ≈ 0.133   (fast learning for young prototypes)
      - n=100:  eta ≈ 0.0099  (slow for mature prototypes, near floor)
      - n=1000: eta = 0.01    (floor — never completely frozen)
    """
    return float(ETA_BASE * np.exp(-support_count / TAU) + ETA_FLOOR)


# ── Prototype Update ─────────────────────────────────────────────────────────

def _update_prototype(
    prototype: Prototype,
    current_vector: np.ndarray,
    current_session_variance: np.ndarray,
) -> Prototype:
    """
    Update prototype via Exponential Moving Average (EMA).

        eta = eta_base * exp(-n/tau) + eta_floor
        mu_new = (1 - eta) * mu_old + eta * v
        sigma^2_new = (1 - eta) * sigma^2_old + eta * (v - mu_new)^2

    The variance update uses mu_new (post-update mean), consistent with the
    standard EMA variance formulation.

    Additionally, we blend the EMA variance with the current session variance
    to maintain calibration to the current session's observed variability:
        sigma^2_blended = 0.7 * sigma^2_EMA + 0.3 * sigma^2_session

    This blending prevents the prototype variance from becoming arbitrarily
    small (which would make the Mahalanobis distance overly sensitive) while
    keeping it responsive to the user's actual behavioral variability.

    Parameters
    ----------
    prototype                 : The prototype being updated
    current_vector            : The new behavioral observation (window_vector)
    current_session_variance  : Per-dimension session variance from Welford's algorithm

    Returns
    -------
    Prototype : A new Prototype instance with updated fields (immutable update pattern)
    """
    eta = _adaptive_learning_rate(prototype.support_count)

    new_vector = (1.0 - eta) * prototype.vector + eta * current_vector
    deviation = current_vector - new_vector              # deviation from POST-update mean
    ema_variance = (1.0 - eta) * prototype.variance + eta * np.square(deviation)
    blended_variance = 0.7 * ema_variance + 0.3 * current_session_variance
    new_variance = np.maximum(blended_variance, 1e-8)

    return Prototype(
        prototype_id=prototype.prototype_id,
        vector=new_vector,
        variance=new_variance,
        support_count=prototype.support_count + 1,
        created_at=prototype.created_at,
        last_updated=datetime.utcnow(),
    )


# ── Quality-Based Prototype Lifecycle ────────────────────────────────────────

def _quality_score(prototype: Prototype, best_similarity: float, current_time: float) -> float:
    """
    Composite quality score for lifecycle management decisions.

        Q(k) = log(1 + n_k) * exp(-lambda_age * age_k) * max(sim_k, 0.1)

    Components:
      log(1 + n_k)  : Logarithmic support — well-established prototypes score higher,
                      but with diminishing returns (100 obs is not 10x better than 10)
      exp(-lambda * age)  : Age decay — prototypes not recently updated lose quality.
                      lambda = 1/day, so a 7-day-old unmatched prototype has quality
                      reduced by exp(-7) ≈ 0.001 relative to a fresh one.
      max(sim_k, 0.1) : Similarity relevance — prototypes that don't match current
                      behavior score lower (but min 0.1 to prevent total zeroing).

    When the prototype limit is reached, the lowest-Q prototype is deleted.
    This prioritises deleting old, rarely-matched, low-support prototypes over
    newly created prototypes that represent current behavior.

    Returns: float >= 0 (not bounded above, used only for comparison)
    """
    age_s = max(0.0, current_time - prototype.last_updated.timestamp())
    age_factor = float(np.exp(-LAMBDA_AGE * age_s))
    support_factor = float(np.log1p(prototype.support_count))
    sim_factor = max(0.1, float(best_similarity))
    return support_factor * age_factor * sim_factor


def _enforce_prototype_limit_quality(
    store,
    username: str,
    prototypes: list,
    best_similarity: float,
    limit: int,
    current_time: float,
) -> None:
    """
    Enforce the prototype limit using quality-based selection.

    If len(prototypes) > limit, delete the prototype with the lowest Q(k).
    Only one prototype is deleted per call (caller invokes after each insertion).
    """
    if len(prototypes) <= limit:
        return

    scored = [
        (p, _quality_score(p, best_similarity, current_time))
        for p in prototypes
    ]
    scored.sort(key=lambda x: x[1])   # ascending: worst quality first
    to_delete = scored[0][0]
    # CosmosUnifiedStore.delete_prototype requires (proto_id, username).
    # SQLiteStore.delete_prototype only requires (proto_id).
    try:
        store.delete_prototype(to_delete.prototype_id, username)
    except TypeError:
        store.delete_prototype(to_delete.prototype_id)


# ── Main Entry Point ─────────────────────────────────────────────────────────

def _get_adaptive_thresholds(store, username: str) -> tuple[float, float]:
    """
    Load per-user adaptive thresholds from the store.

    Returns (threshold_update, threshold_create).

    If the store supports get_user_adaptive_fields() and the user has
    >= MIN_ADAPTIVE_OBSERVATIONS similarity observations, compute:
        theta_update = mu_sim - 1.5 * sigma_sim
        theta_create = mu_sim - 3.0 * sigma_sim

    Otherwise fall back to the global defaults.
    """
    if not hasattr(store, "get_user_adaptive_fields"):
        return THRESHOLD_UPDATE, THRESHOLD_CREATE

    try:
        fields = store.get_user_adaptive_fields(username)
        if fields is None:
            return THRESHOLD_UPDATE, THRESHOLD_CREATE

        n_sim = int(fields.get("sim_count", 0))
        if n_sim < MIN_ADAPTIVE_OBSERVATIONS:
            return THRESHOLD_UPDATE, THRESHOLD_CREATE

        mu_sim = float(fields.get("sim_mean", THRESHOLD_UPDATE))
        # Welford variance → std
        sim_m2 = float(fields.get("sim_m2", 0.0))
        sigma_sim = float(np.sqrt(sim_m2 / n_sim)) if n_sim > 0 else 0.0

        t_update = float(np.clip(mu_sim - 1.5 * sigma_sim, 0.50, 0.95))
        t_create = float(np.clip(mu_sim - 3.0 * sigma_sim, 0.10, t_update - 0.10))
        return t_update, t_create
    except Exception:
        return THRESHOLD_UPDATE, THRESHOLD_CREATE


def compute_prototype_metrics(
    store,
    username: str,
    preprocessed: PreprocessedBehaviour,
    current_time: Optional[float] = None,
) -> PrototypeMetrics:
    """
    Layer-2 prototype matching, update, and rich metric computation.

    Pipeline
    --------
    1. Retrieve all prototypes for the user from the store
    2. Compute composite similarity against each prototype (cosine + Mahalanobis kernel)
    3. Identify the best matching prototype
    4. Apply update / quarantine routing based on adaptive thresholds
    5. Update per-user adaptive fields (similarity + drift distributions)
    6. Return the full 9-field PrototypeMetrics vector (no decisions)

    This function makes NO trust decisions. It produces measurements that
    Layer-4 (TrustEngine) uses for decision making.

    Parameters
    ----------
    store        : CosmosUnifiedStore or compatible
    username     : User identifier
    preprocessed : Output of preprocessing.process_event()
    current_time : Unix timestamp (defaults to time.time()). Pass the event
                   timestamp to ensure quarantine and lifecycle decisions use
                   a consistent clock (important for tests and replay).

    Returns
    -------
    PrototypeMetrics: Full behavioural state vector for Layer-4 consumption
    """
    current_time = current_time if current_time is not None else time.time()
    prototypes = store.get_prototypes(username)

    # Load per-user adaptive thresholds (falls back to globals if insufficient data)
    threshold_update, threshold_create = _get_adaptive_thresholds(store, username)

    # ── COLD START: No prototypes yet ────────────────────────────────────────
    if not prototypes:
        # Route to quarantine instead of immediate prototype creation.
        # The user's first behavioral observations must survive quarantine
        # before becoming a reference prototype.
        if hasattr(store, "submit_quarantine_candidate"):
            promotion = store.submit_quarantine_candidate(
                username=username,
                vector=preprocessed.window_vector,
                current_time=current_time,
            )
        else:
            promotion = quarantine_manager.submit(
                username=username,
                vector=preprocessed.window_vector,
                current_time=current_time,
            )
        prototype_id: Optional[int] = None
        if promotion is not None:
            centroid, variance, support = promotion
            prototype_id = store.insert_prototype(
                username=username,
                vector=centroid,
                variance=variance,
                support_count=support,
            )

        return PrototypeMetrics(
            similarity_score=0.0,
            short_drift=preprocessed.short_drift,
            long_drift=preprocessed.long_drift,
            stability_score=preprocessed.stability_score,
            matched_prototype_id=prototype_id,
            prototype_confidence=0.0,
            behavioural_consistency=preprocessed.behavioural_consistency,
            prototype_support_strength=0.0,
            anomaly_indicator=compute_anomaly_indicator(0.0, preprocessed.short_drift),
        )

    # ── FIND BEST MATCHING PROTOTYPE ─────────────────────────────────────────
    best_prototype: Optional[Prototype] = None
    best_similarity: float = -1.0

    for prototype in prototypes:
        # Effective variance: blend prototype's stored variance with current
        # session variance. This prevents the Mahalanobis distance from being
        # computed against a prototype variance that is too narrow (over-fitted
        # to historical behavior) or too wide (under-fitted to current variability).
        effective_variance = np.maximum(
            0.5 * prototype.variance + 0.5 * preprocessed.variance_vector,
            1e-8,
        )

        cosine = cosine_similarity(preprocessed.window_vector, prototype.vector)
        m_dist = mahalanobis_distance(
            preprocessed.window_vector,
            prototype.vector,
            effective_variance,
        )
        sim = composite_similarity(
            cosine=cosine,
            mahalanobis_dist=m_dist,
            stability_score=preprocessed.stability_score,
        )

        if sim > best_similarity:
            best_similarity = sim
            best_prototype = prototype

    best_similarity = max(0.0, float(best_similarity))
    matched_id: Optional[int] = best_prototype.prototype_id if best_prototype else None

    # ── PROTOTYPE UPDATE / QUARANTINE ROUTING ────────────────────────────────
    if best_prototype is not None and best_similarity >= threshold_update:
        # High similarity: the current behavior matches an established prototype.
        # Update the prototype to incorporate this new observation.
        updated = _update_prototype(
            best_prototype,
            preprocessed.window_vector,
            preprocessed.variance_vector,
        )
        store.update_prototype(updated)

    elif best_similarity < threshold_create:
        # Low similarity: the current behavior is too dissimilar to any prototype.
        # Route to quarantine — only promoted candidates become new prototypes.
        # Use Cosmos-backed quarantine if available, otherwise in-memory fallback.
        if hasattr(store, "submit_quarantine_candidate"):
            promotion = store.submit_quarantine_candidate(
                username=username,
                vector=preprocessed.window_vector,
                current_time=current_time,
            )
        else:
            promotion = quarantine_manager.submit(
                username=username,
                vector=preprocessed.window_vector,
                current_time=current_time,
            )
        if promotion is not None:
            centroid, variance, support = promotion
            new_id = store.insert_prototype(
                username=username,
                vector=centroid,
                variance=variance,
                support_count=support,
            )
            matched_id = new_id
            # Enforce prototype limit using quality-based deletion
            all_protos = store.get_prototypes(username)
            _enforce_prototype_limit_quality(
                store, username, all_protos,
                best_similarity, MAX_PROTOTYPES_PER_USER, current_time,
            )

    # else: threshold_create <= similarity < threshold_update
    #   Behavior is moderately similar but not enough to update.
    #   No action taken — system re-evaluates on next event.
    #   This dead-zone prevents prototype corruption via boundary attacks.

    # ── UPDATE PER-USER ADAPTIVE FIELDS ─────────────────────────────────────
    # Feed the similarity and short_drift observations into the user's running
    # Welford accumulators. These power adaptive sigma and threshold computation.
    if hasattr(store, "update_user_adaptive_fields"):
        try:
            store.update_user_adaptive_fields(
                username=username,
                similarity=best_similarity,
                drift=preprocessed.short_drift,
            )
        except Exception:
            pass  # Adaptive field update is best-effort; never crash the pipeline

    # ── RICH OUTPUT METRICS ──────────────────────────────────────────────────
    support_count = best_prototype.support_count if best_prototype else 0

    prototype_confidence = compute_prototype_confidence(best_similarity, support_count)
    prototype_support_strength = compute_prototype_support_strength(support_count)
    anomaly = compute_anomaly_indicator(best_similarity, preprocessed.short_drift)

    return PrototypeMetrics(
        similarity_score=best_similarity,
        short_drift=preprocessed.short_drift,
        long_drift=preprocessed.long_drift,
        stability_score=preprocessed.stability_score,
        matched_prototype_id=matched_id,
        prototype_confidence=prototype_confidence,
        behavioural_consistency=preprocessed.behavioural_consistency,
        prototype_support_strength=prototype_support_strength,
        anomaly_indicator=anomaly,
    )
