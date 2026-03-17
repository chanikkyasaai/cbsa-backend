"""
Runtime Invariant Checks — Layer-2/4 Pipeline Guard Rails

This module provides lightweight assertion functions that are called at
critical pipeline boundaries to detect and surface data quality violations
early, before they silently corrupt downstream computations.

Design philosophy:
  - Assertions are cheap for valid data (single pass through the array)
  - Violations raise InvariantError (a ValueError subclass) with a clear
    message identifying which check failed and the offending value
  - All checks are pure functions with no side effects — safe to call
    anywhere in the pipeline

Usage:
    from app.core.invariants import check_vector, check_scalar_01, InvariantError

    check_vector(event.vector, "input vector")           # NaN/Inf/shape
    check_scalar_01(similarity, "similarity_score")      # must be in [0,1]

The checks are not exhaustive — they target the boundaries where bad data
is most likely to enter (external input, cross-layer hand-offs).
"""

from __future__ import annotations

import numpy as np

VECTOR_DIM: int = 48


class InvariantError(ValueError):
    """Raised when a runtime invariant is violated."""


# ── Vector invariants ─────────────────────────────────────────────────────────

def check_vector(v: np.ndarray, name: str = "vector") -> None:
    """
    Assert that v is a valid 48-D behavioral vector.

    Checks:
      1. Shape: must be 1-D with VECTOR_DIM elements
      2. NaN: no NaN values
      3. Inf: no infinite values

    Note: we do NOT assert [0,1] bounds on raw vectors because some
    preprocessing steps operate in unnormalized feature space. The
    caller is responsible for normalisation before calling downstream
    similarity functions.
    """
    if v.ndim != 1 or v.shape[0] != VECTOR_DIM:
        raise InvariantError(
            f"{name}: expected shape ({VECTOR_DIM},), got {v.shape}"
        )
    if np.isnan(v).any():
        n_nan = int(np.isnan(v).sum())
        raise InvariantError(
            f"{name}: contains {n_nan} NaN value(s)"
        )
    if np.isinf(v).any():
        n_inf = int(np.isinf(v).sum())
        raise InvariantError(
            f"{name}: contains {n_inf} Inf value(s)"
        )


def check_variance_vector(v: np.ndarray, name: str = "variance") -> None:
    """
    Assert that v is a valid variance vector: shape (48,), non-negative, finite.
    """
    check_vector(v, name)
    if (v < 0.0).any():
        n_neg = int((v < 0.0).sum())
        raise InvariantError(
            f"{name}: contains {n_neg} negative variance(s)"
        )


# ── Scalar invariants ─────────────────────────────────────────────────────────

def check_scalar_01(x: float, name: str = "value") -> None:
    """
    Assert that x is a finite float in [0, 1].

    Used for: similarity_score, stability_score, drift values, trust_score,
              anomaly_indicator, prototype_confidence, etc.
    """
    if not isinstance(x, (int, float)):
        raise InvariantError(
            f"{name}: expected numeric, got {type(x).__name__}"
        )
    if np.isnan(x):
        raise InvariantError(f"{name}: is NaN")
    if np.isinf(x):
        raise InvariantError(f"{name}: is Inf ({x:+.2e})")
    if not (0.0 <= x <= 1.0):
        raise InvariantError(
            f"{name}: {x:.6f} is outside [0, 1]"
        )


def check_scalar_nonneg(x: float, name: str = "value") -> None:
    """Assert x is a finite non-negative float."""
    if np.isnan(x) or np.isinf(x):
        raise InvariantError(f"{name}: is NaN or Inf ({x})")
    if x < 0.0:
        raise InvariantError(f"{name}: {x:.6f} is negative")


# ── Pipeline boundary checks ──────────────────────────────────────────────────

def check_preprocessed_behaviour(pb) -> None:
    """
    Assert all PreprocessedBehaviour fields are within valid bounds.

    Call immediately after preprocessing.process_event() returns.
    """
    check_vector(pb.window_vector, "window_vector")
    check_scalar_01(pb.short_drift, "short_drift")
    check_scalar_01(pb.long_drift, "long_drift")
    check_scalar_01(pb.stability_score, "stability_score")
    check_scalar_01(pb.behavioural_consistency, "behavioural_consistency")
    check_variance_vector(pb.variance_vector, "variance_vector")
    check_scalar_nonneg(pb.sigma_ref, "sigma_ref")


def check_prototype_metrics(pm) -> None:
    """
    Assert all PrototypeMetrics fields are within valid bounds.

    Call immediately after prototype_engine.compute_prototype_metrics() returns.
    """
    check_scalar_01(pm.similarity_score, "similarity_score")
    check_scalar_01(pm.short_drift, "short_drift")
    check_scalar_01(pm.long_drift, "long_drift")
    check_scalar_01(pm.stability_score, "stability_score")
    check_scalar_01(pm.prototype_confidence, "prototype_confidence")
    check_scalar_01(pm.behavioural_consistency, "behavioural_consistency")
    check_scalar_01(pm.prototype_support_strength, "prototype_support_strength")
    check_scalar_01(pm.anomaly_indicator, "anomaly_indicator")


def check_trust_result(tr) -> None:
    """
    Assert all TrustResult fields are within valid bounds.

    Call immediately after trust_engine.update_trust() returns.
    """
    check_scalar_01(tr.trust_score, "trust_score")
    check_scalar_01(tr.raw_trust_signal, "raw_trust_signal")
    check_scalar_01(tr.alpha_t, "alpha_t")
    check_scalar_01(tr.anomaly_indicator, "anomaly_indicator")
    if tr.decision not in ("SAFE", "UNCERTAIN", "RISK"):
        raise InvariantError(
            f"trust_result.decision: invalid value '{tr.decision}'"
        )
