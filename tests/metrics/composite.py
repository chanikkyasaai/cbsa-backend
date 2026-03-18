"""
tests/metrics/composite.py — Full Composite Metrics for CBSA Evaluation

Metric families
---------------
BiometricMetrics
    Standard biometric authentication metrics:
    FAR, FRR, EER, AUC-ROC, AUC-DET

DetectionMetrics
    Binary classification quality metrics:
    Precision, Recall, F1, AUPRC (Area Under Precision-Recall Curve)
    Confusion matrix: TP, FP, TN, FN

TemporalMetrics
    Temporal behaviour of the authentication stream:
    - Attack detection latency (events to first RISK after attack begins)
    - Trust convergence time (events until genuine user reaches SAFE)
    - Trust score stability (std-dev over stable genuine windows)
    - Trust recovery time (events from RISK back to SAFE after attack stops)

DriftSensitivityMetrics
    How responsive each temporal drift scale is:
    - Mean short/medium/long drift in stable vs. attack windows
    - Drift SNR: signal-to-noise ratio (attack drift / genuine drift per scale)
    - Three-scale separation score

PrototypeHealthMetrics
    Prototype model quality indicators:
    - Mean/std prototype support strength
    - Prototype cohesion distribution statistics
    - Prototype confidence trajectory

SystemHealthMetrics
    Pipeline-level operational metrics:
    - Decision distribution (SAFE / UNCERTAIN / RISK / WARMUP %)
    - Warmup duration (events until first non-WARMUP decision)
    - Mean/std trust score over full run
    - Anomaly indicator distribution

CompositeMetrics
    Combines all families into a single MetricsReport with a unified score
    (Composite Authentication Quality Index, CAQI) and printable report.

CAQI definition
---------------
    CAQI = 0.30*(1-EER) + 0.20*F1 + 0.20*precision
         + 0.15*(1 - attack_detection_latency_norm)
         + 0.15*(trust_separation)

    Where:
      - EER             : lower is better, so (1-EER) rewards good discrimination
      - F1              : harmonic mean of precision and recall
      - precision       : attack rejection quality
      - latency_norm    : attack_detection_latency / 20 (normalised to [0,1], capped)
      - trust_separation: avg_trust_genuine - avg_trust_attack  (clamped [0,1])

    CAQI = 1.0: perfect system
    CAQI > 0.80: excellent
    CAQI > 0.65: good
    CAQI < 0.50: needs tuning

Input format
------------
Each result list is produced by tests/runner.run_pipeline():
    {
        "event":        int,
        "similarity":   float,
        "short_drift":  float,
        "medium_drift": float,     # three-scale drift
        "stability":    float,
        "anomaly":      float,
        "trust":        float,
        "decision":     str,       # "SAFE" | "UNCERTAIN" | "RISK" | "WARMUP"
        "proto_id":     int | None,
        "n_protos":     int,
    }
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np

# Default threshold for binary ACCEPT/REJECT classification
ACCEPT_THRESHOLD: float = 0.65
# Warmup events to skip when computing genuine-user FRR
WARMUP_SKIP: int = 20


# ─────────────────────────────────────────────────────────────────────────────
# Data classes
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class BiometricMetrics:
    """Standard biometric authentication metrics."""
    far: float = 0.0         # False Accept Rate
    frr: float = 0.0         # False Reject Rate
    eer: float = 0.0         # Equal Error Rate
    auc_roc: float = 0.0     # Area under ROC curve
    threshold_used: float = ACCEPT_THRESHOLD
    n_genuine: int = 0
    n_attack: int = 0
    tp: int = 0
    fp: int = 0
    tn: int = 0
    fn: int = 0

    @property
    def accuracy(self) -> float:
        total = self.tp + self.fp + self.tn + self.fn
        return (self.tp + self.tn) / total if total > 0 else 0.0

    @property
    def status(self) -> str:
        sep = self.avg_trust_genuine - self.avg_trust_attack \
            if hasattr(self, 'avg_trust_genuine') else 0.0
        if self.eer < 0.05:
            return "EXCELLENT"
        if self.eer < 0.10:
            return "GOOD"
        if self.eer < 0.15:
            return "STABLE"
        return "NEEDS TUNING"

    avg_trust_genuine: float = 0.0
    avg_trust_attack: float = 0.0


@dataclass
class DetectionMetrics:
    """Binary classification quality metrics."""
    precision: float = 0.0   # TP / (TP + FP)  — attack rejection precision
    recall: float = 0.0      # TP / (TP + FN)  — genuine acceptance recall
    f1: float = 0.0          # Harmonic mean of precision and recall
    auprc: float = 0.0       # Area Under Precision-Recall Curve
    specificity: float = 0.0 # TN / (TN + FP)  — attack rejection rate
    balanced_accuracy: float = 0.0  # (recall + specificity) / 2
    matthews_cc: float = 0.0  # Matthews Correlation Coefficient


@dataclass
class TemporalMetrics:
    """Temporal behaviour of the authentication stream."""
    # How many events after the attack begins until the first RISK decision
    attack_detection_latency: float = 0.0    # events (lower is better)
    # How many events until genuine user first reaches SAFE (after warmup)
    trust_convergence_events: float = 0.0    # events
    # Std-dev of trust score over the stable (post-warmup genuine) window
    trust_stability_std: float = 0.0         # lower is better
    # How many events after attack ends until trust returns to SAFE
    trust_recovery_events: float = 0.0       # events (lower is better)
    # Mean trust score over the last 10 genuine events (convergence target)
    final_genuine_trust: float = 0.0
    # Mean trust score over the first 10 attack events
    early_attack_trust: float = 0.0


@dataclass
class DriftSensitivityMetrics:
    """Responsiveness of each temporal drift scale."""
    # Mean drift in genuine stable window (events 25-60)
    genuine_short_drift: float = 0.0
    genuine_medium_drift: float = 0.0
    genuine_long_drift: float = 0.0
    # Mean drift in attack window
    attack_short_drift: float = 0.0
    attack_medium_drift: float = 0.0
    attack_long_drift: float = 0.0
    # Signal-to-noise ratio: attack_drift / max(genuine_drift, eps)
    snr_short: float = 0.0
    snr_medium: float = 0.0
    snr_long: float = 0.0
    # Composite separation: weighted SNR across three scales
    three_scale_separation: float = 0.0


@dataclass
class PrototypeHealthMetrics:
    """Prototype model quality indicators."""
    # Prototype support strength statistics
    mean_support_strength: float = 0.0
    std_support_strength: float = 0.0
    # Prototype confidence trajectory
    mean_confidence: float = 0.0
    final_confidence: float = 0.0      # last 10 events avg
    # Cohesion (only present if prototype_topology_cohesion logged)
    mean_cohesion: float = 0.0         # 0 if not available
    # How many events until prototype was first created (first non-None proto_id)
    prototype_creation_event: int = 0


@dataclass
class SystemHealthMetrics:
    """Pipeline-level operational metrics."""
    # Decision distribution over the full genuine run
    pct_safe: float = 0.0
    pct_uncertain: float = 0.0
    pct_risk: float = 0.0
    pct_warmup: float = 0.0
    # Number of events with WARMUP decision
    warmup_duration_events: int = 0
    # Mean ± std trust over full genuine run
    mean_trust: float = 0.0
    std_trust: float = 0.0
    # Mean anomaly indicator over genuine run
    mean_anomaly_genuine: float = 0.0
    # Mean anomaly indicator over attack run
    mean_anomaly_attack: float = 0.0
    # Anomaly SNR: attack_anomaly / max(genuine_anomaly, eps)
    anomaly_snr: float = 0.0


@dataclass
class MetricsReport:
    """Unified metrics report combining all families."""
    biometric: BiometricMetrics = field(default_factory=BiometricMetrics)
    detection: DetectionMetrics = field(default_factory=DetectionMetrics)
    temporal: TemporalMetrics = field(default_factory=TemporalMetrics)
    drift_sensitivity: DriftSensitivityMetrics = field(default_factory=DriftSensitivityMetrics)
    prototype_health: PrototypeHealthMetrics = field(default_factory=PrototypeHealthMetrics)
    system_health: SystemHealthMetrics = field(default_factory=SystemHealthMetrics)
    caqi: float = 0.0    # Composite Authentication Quality Index [0,1]

    def print_full(self, title: str = "CBSA COMPOSITE METRICS REPORT") -> None:
        W = 68
        sep = "=" * W
        thin = "-" * W

        def hdr(s: str) -> None:
            print(f"\n  ── {s} {'─' * max(0, W - 6 - len(s))}")

        def row(label: str, value: str) -> None:
            print(f"  {label:<40} {value}")

        print(f"\n{sep}")
        print(f"  {title}")
        print(sep)

        # ── Biometric ─────────────────────────────────────────────────────
        hdr("BIOMETRIC METRICS")
        b = self.biometric
        row("Genuine events evaluated", str(b.n_genuine))
        row("Attack  events evaluated", str(b.n_attack))
        row("Accept threshold", f"{b.threshold_used:.2f}")
        print()
        row("True  Positives (TP)", str(b.tp))
        row("False Negatives (FN)", str(b.fn))
        row("True  Negatives (TN)", str(b.tn))
        row("False Positives (FP)", str(b.fp))
        print()
        row("Accuracy", f"{b.accuracy*100:.2f}%")
        row("FAR  (False Accept Rate)", f"{b.far*100:.2f}%")
        row("FRR  (False Reject Rate)", f"{b.frr*100:.2f}%")
        row("EER  (Equal Error Rate)", f"{b.eer*100:.2f}%")
        row("AUC-ROC", f"{b.auc_roc:.4f}")
        row("Avg trust — genuine", f"{b.avg_trust_genuine:.4f}")
        row("Avg trust — attack",  f"{b.avg_trust_attack:.4f}")
        row("Trust separation", f"{b.avg_trust_genuine - b.avg_trust_attack:.4f}")
        row("Biometric status", b.status)

        # ── Detection ─────────────────────────────────────────────────────
        hdr("DETECTION METRICS")
        d = self.detection
        row("Precision", f"{d.precision:.4f}")
        row("Recall (sensitivity)", f"{d.recall:.4f}")
        row("Specificity", f"{d.specificity:.4f}")
        row("F1 Score", f"{d.f1:.4f}")
        row("AUPRC", f"{d.auprc:.4f}")
        row("Balanced Accuracy", f"{d.balanced_accuracy:.4f}")
        row("Matthews Corr. Coeff.", f"{d.matthews_cc:.4f}")

        # ── Temporal ──────────────────────────────────────────────────────
        hdr("TEMPORAL METRICS")
        t = self.temporal
        row("Attack detection latency (events)", f"{t.attack_detection_latency:.1f}")
        row("Trust convergence (events to SAFE)", f"{t.trust_convergence_events:.1f}")
        row("Trust stability std-dev (genuine)", f"{t.trust_stability_std:.4f}")
        row("Trust recovery after attack (events)", f"{t.trust_recovery_events:.1f}")
        row("Final genuine trust (last 10)", f"{t.final_genuine_trust:.4f}")
        row("Early attack trust (first 10)", f"{t.early_attack_trust:.4f}")

        # ── Drift sensitivity ─────────────────────────────────────────────
        hdr("DRIFT SENSITIVITY (Three-Scale)")
        ds = self.drift_sensitivity
        row("Short drift — genuine", f"{ds.genuine_short_drift:.4f}")
        row("Short drift — attack",  f"{ds.attack_short_drift:.4f}")
        row("SNR short",             f"{ds.snr_short:.2f}x")
        row("Medium drift — genuine", f"{ds.genuine_medium_drift:.4f}")
        row("Medium drift — attack",  f"{ds.attack_medium_drift:.4f}")
        row("SNR medium",             f"{ds.snr_medium:.2f}x")
        row("Long drift — genuine",  f"{ds.genuine_long_drift:.4f}")
        row("Long drift — attack",   f"{ds.attack_long_drift:.4f}")
        row("SNR long",              f"{ds.snr_long:.2f}x")
        row("Three-scale separation score", f"{ds.three_scale_separation:.4f}")

        # ── Prototype health ──────────────────────────────────────────────
        hdr("PROTOTYPE HEALTH")
        ph = self.prototype_health
        row("Prototype created at event", str(ph.prototype_creation_event))
        row("Mean support strength", f"{ph.mean_support_strength:.4f}")
        row("Std  support strength", f"{ph.std_support_strength:.4f}")
        row("Mean prototype confidence", f"{ph.mean_confidence:.4f}")
        row("Final prototype confidence", f"{ph.final_confidence:.4f}")
        if ph.mean_cohesion > 0:
            row("Mean topology cohesion", f"{ph.mean_cohesion:.4f}")

        # ── System health ─────────────────────────────────────────────────
        hdr("SYSTEM HEALTH")
        sh = self.system_health
        row("Decision distribution — SAFE",      f"{sh.pct_safe*100:.1f}%")
        row("Decision distribution — UNCERTAIN", f"{sh.pct_uncertain*100:.1f}%")
        row("Decision distribution — RISK",      f"{sh.pct_risk*100:.1f}%")
        row("Decision distribution — WARMUP",    f"{sh.pct_warmup*100:.1f}%")
        row("Warmup duration (events)", str(sh.warmup_duration_events))
        row("Mean trust (full genuine run)", f"{sh.mean_trust:.4f}")
        row("Std  trust (full genuine run)", f"{sh.std_trust:.4f}")
        row("Mean anomaly — genuine", f"{sh.mean_anomaly_genuine:.4f}")
        row("Mean anomaly — attack",  f"{sh.mean_anomaly_attack:.4f}")
        row("Anomaly SNR",            f"{sh.anomaly_snr:.2f}x")

        # ── CAQI ──────────────────────────────────────────────────────────
        hdr("COMPOSITE AUTHENTICATION QUALITY INDEX (CAQI)")
        bar_len = 40
        filled = round(self.caqi * bar_len)
        bar = "█" * filled + "░" * (bar_len - filled)
        grade = _caqi_grade(self.caqi)
        row("CAQI", f"{self.caqi:.4f}  [{bar}]  {grade}")

        print(f"\n{sep}\n")


# ─────────────────────────────────────────────────────────────────────────────
# Core computation
# ─────────────────────────────────────────────────────────────────────────────

class CompositeMetrics:
    """
    Compute all metric families from genuine and attack result lists.

    Parameters
    ----------
    genuine_results : list of dicts from run_pipeline() for the genuine user
    attack_results  : list of dicts from run_pipeline() for an impostor
    threshold       : binary ACCEPT threshold (default 0.65)
    warmup_skip     : genuine events to exclude from FRR computation

    Returns
    -------
    MetricsReport
    """

    @staticmethod
    def compute(
        genuine_results: List[dict],
        attack_results: List[dict],
        threshold: float = ACCEPT_THRESHOLD,
        warmup_skip: int = WARMUP_SKIP,
    ) -> MetricsReport:
        report = MetricsReport()

        g_eval = genuine_results[warmup_skip:]   # post-warmup genuine
        g_trust = np.array([r["trust"] for r in g_eval], dtype=float)
        a_trust = np.array([r["trust"] for r in attack_results], dtype=float)

        report.biometric = _compute_biometric(g_trust, a_trust, genuine_results, attack_results, threshold, warmup_skip)
        report.detection = _compute_detection(report.biometric)
        report.temporal = _compute_temporal(genuine_results, attack_results, threshold, warmup_skip)
        report.drift_sensitivity = _compute_drift_sensitivity(genuine_results, attack_results, warmup_skip)
        report.prototype_health = _compute_prototype_health(genuine_results, warmup_skip)
        report.system_health = _compute_system_health(genuine_results, attack_results, warmup_skip)
        report.caqi = _compute_caqi(report)

        return report


# ─────────────────────────────────────────────────────────────────────────────
# Private helpers
# ─────────────────────────────────────────────────────────────────────────────

def _compute_biometric(
    g_trust: np.ndarray,
    a_trust: np.ndarray,
    genuine_results: List[dict],
    attack_results: List[dict],
    threshold: float,
    warmup_skip: int,
) -> BiometricMetrics:
    b = BiometricMetrics(
        threshold_used=threshold,
        n_genuine=len(g_trust),
        n_attack=len(a_trust),
    )

    b.tp = int(np.sum(g_trust > threshold))
    b.fn = int(np.sum(g_trust <= threshold))
    b.tn = int(np.sum(a_trust <= threshold))
    b.fp = int(np.sum(a_trust > threshold))

    b.far = b.fp / b.n_attack if b.n_attack > 0 else 0.0
    b.frr = b.fn / b.n_genuine if b.n_genuine > 0 else 0.0
    b.eer = _eer(g_trust, a_trust)
    b.auc_roc = _auc_roc(g_trust, a_trust)
    b.avg_trust_genuine = float(np.mean(g_trust)) if len(g_trust) > 0 else 0.0
    b.avg_trust_attack = float(np.mean(a_trust)) if len(a_trust) > 0 else 1.0
    return b


def _compute_detection(b: BiometricMetrics) -> DetectionMetrics:
    d = DetectionMetrics()
    tp, fp, tn, fn = b.tp, b.fp, b.tn, b.fn

    d.precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    d.recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    d.specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    d.f1 = 2 * d.precision * d.recall / (d.precision + d.recall) \
        if (d.precision + d.recall) > 0 else 0.0
    d.balanced_accuracy = (d.recall + d.specificity) / 2.0
    d.auprc = _auprc_approx(b)

    # Matthews Correlation Coefficient
    denom = ((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)) ** 0.5
    d.matthews_cc = (tp * tn - fp * fn) / denom if denom > 0 else 0.0
    return d


def _compute_temporal(
    genuine_results: List[dict],
    attack_results: List[dict],
    threshold: float,
    warmup_skip: int,
) -> TemporalMetrics:
    t = TemporalMetrics()

    # ── Convergence: genuine events to first SAFE decision ─────────────────
    for i, r in enumerate(genuine_results[warmup_skip:], start=1):
        if r["decision"] == "SAFE":
            t.trust_convergence_events = float(i)
            break
    else:
        t.trust_convergence_events = float(len(genuine_results))

    # ── Trust stability over stable genuine window ─────────────────────────
    stable = genuine_results[warmup_skip:]
    if stable:
        trusts = np.array([r["trust"] for r in stable], dtype=float)
        t.trust_stability_std = float(np.std(trusts))

    # ── Attack detection latency ───────────────────────────────────────────
    # How many events from start of attack_results until first RISK/UNCERTAIN
    t.attack_detection_latency = float(len(attack_results))  # worst case
    for i, r in enumerate(attack_results, start=1):
        if r["decision"] in ("RISK", "UNCERTAIN") or r["trust"] < threshold:
            t.attack_detection_latency = float(i)
            break

    # ── Final genuine trust ────────────────────────────────────────────────
    tail = genuine_results[-10:] if len(genuine_results) >= 10 else genuine_results
    t.final_genuine_trust = float(np.mean([r["trust"] for r in tail])) if tail else 0.0

    # ── Early attack trust ─────────────────────────────────────────────────
    head = attack_results[:10] if len(attack_results) >= 10 else attack_results
    t.early_attack_trust = float(np.mean([r["trust"] for r in head])) if head else 1.0

    # ── Trust recovery (not computable without post-attack genuine data) ───
    t.trust_recovery_events = 0.0   # placeholder — requires mixed scenario

    return t


def _compute_drift_sensitivity(
    genuine_results: List[dict],
    attack_results: List[dict],
    warmup_skip: int,
) -> DriftSensitivityMetrics:
    ds = DriftSensitivityMetrics()
    eps = 1e-9

    g_stable = genuine_results[warmup_skip:]
    a_results = attack_results

    def _mean(results: List[dict], key: str) -> float:
        vals = [r.get(key, 0.0) for r in results if key in r]
        return float(np.mean(vals)) if vals else 0.0

    ds.genuine_short_drift = _mean(g_stable, "short_drift")
    ds.genuine_medium_drift = _mean(g_stable, "medium_drift")
    ds.genuine_long_drift = _mean(g_stable, "long_drift") if g_stable and "long_drift" in g_stable[0] else 0.0

    ds.attack_short_drift = _mean(a_results, "short_drift")
    ds.attack_medium_drift = _mean(a_results, "medium_drift")
    ds.attack_long_drift = _mean(a_results, "long_drift") if a_results and "long_drift" in a_results[0] else 0.0

    ds.snr_short = ds.attack_short_drift / max(ds.genuine_short_drift, eps)
    ds.snr_medium = ds.attack_medium_drift / max(ds.genuine_medium_drift, eps)
    ds.snr_long = ds.attack_long_drift / max(ds.genuine_long_drift, eps)

    # Weighted three-scale separation (mirrors trust engine weights 50/30/20)
    ds.three_scale_separation = (
        0.50 * (ds.attack_short_drift - ds.genuine_short_drift) +
        0.30 * (ds.attack_medium_drift - ds.genuine_medium_drift) +
        0.20 * (ds.attack_long_drift - ds.genuine_long_drift)
    )
    ds.three_scale_separation = max(0.0, float(ds.three_scale_separation))

    return ds


def _compute_prototype_health(
    genuine_results: List[dict],
    warmup_skip: int,
) -> PrototypeHealthMetrics:
    ph = PrototypeHealthMetrics()

    # Prototype creation event: first event where proto_id is not None
    for r in genuine_results:
        if r.get("proto_id") is not None:
            ph.prototype_creation_event = r["event"]
            break

    g_stable = genuine_results[warmup_skip:]
    if not g_stable:
        return ph

    support_vals = [r.get("prototype_support_strength", 0.0) for r in g_stable
                    if "prototype_support_strength" in r]
    if support_vals:
        ph.mean_support_strength = float(np.mean(support_vals))
        ph.std_support_strength = float(np.std(support_vals))

    conf_vals = [r.get("prototype_confidence", 0.0) for r in g_stable
                 if "prototype_confidence" in r]
    if conf_vals:
        ph.mean_confidence = float(np.mean(conf_vals))
        ph.final_confidence = float(np.mean(conf_vals[-10:])) if len(conf_vals) >= 10 else float(np.mean(conf_vals))

    cohesion_vals = [r.get("prototype_topology_cohesion", 0.0) for r in g_stable
                     if "prototype_topology_cohesion" in r]
    if cohesion_vals:
        ph.mean_cohesion = float(np.mean(cohesion_vals))

    return ph


def _compute_system_health(
    genuine_results: List[dict],
    attack_results: List[dict],
    warmup_skip: int,
) -> SystemHealthMetrics:
    sh = SystemHealthMetrics()
    eps = 1e-9

    all_genuine = genuine_results
    n = len(all_genuine)
    if n == 0:
        return sh

    decisions = [r["decision"] for r in all_genuine]
    sh.pct_safe     = decisions.count("SAFE") / n
    sh.pct_uncertain = decisions.count("UNCERTAIN") / n
    sh.pct_risk     = decisions.count("RISK") / n
    sh.pct_warmup   = decisions.count("WARMUP") / n
    sh.warmup_duration_events = decisions.count("WARMUP")

    trusts = np.array([r["trust"] for r in all_genuine], dtype=float)
    sh.mean_trust = float(np.mean(trusts))
    sh.std_trust = float(np.std(trusts))

    g_anomaly = [r.get("anomaly", 0.0) for r in genuine_results[warmup_skip:]]
    a_anomaly = [r.get("anomaly", 0.0) for r in attack_results]
    sh.mean_anomaly_genuine = float(np.mean(g_anomaly)) if g_anomaly else 0.0
    sh.mean_anomaly_attack = float(np.mean(a_anomaly)) if a_anomaly else 0.0
    sh.anomaly_snr = sh.mean_anomaly_attack / max(sh.mean_anomaly_genuine, eps)

    return sh


def _compute_caqi(report: MetricsReport) -> float:
    """
    Composite Authentication Quality Index.

    CAQI = 0.30*(1-EER) + 0.20*F1 + 0.20*precision
         + 0.15*(1 - attack_latency_norm)
         + 0.15*(trust_separation_clamped)
    """
    eer_term = 1.0 - report.biometric.eer
    f1_term = report.detection.f1
    prec_term = report.detection.precision
    latency_norm = min(report.temporal.attack_detection_latency / 20.0, 1.0)
    latency_term = 1.0 - latency_norm
    sep = report.biometric.avg_trust_genuine - report.biometric.avg_trust_attack
    sep_term = max(0.0, min(1.0, sep))

    caqi = (
        0.30 * eer_term +
        0.20 * f1_term +
        0.20 * prec_term +
        0.15 * latency_term +
        0.15 * sep_term
    )
    return round(float(np.clip(caqi, 0.0, 1.0)), 4)


def _caqi_grade(caqi: float) -> str:
    if caqi >= 0.80:
        return "EXCELLENT"
    if caqi >= 0.65:
        return "GOOD"
    if caqi >= 0.50:
        return "ADEQUATE"
    return "NEEDS TUNING"


# ─────────────────────────────────────────────────────────────────────────────
# Statistical helpers
# ─────────────────────────────────────────────────────────────────────────────

def _eer(genuine: np.ndarray, attack: np.ndarray) -> float:
    if len(genuine) == 0 or len(attack) == 0:
        return 0.0
    thresholds = np.linspace(0.0, 1.0, 1001)
    best_eer = 1.0
    best_diff = float("inf")
    for t in thresholds:
        far = float(np.mean(attack > t))
        frr = float(np.mean(genuine <= t))
        diff = abs(far - frr)
        if diff < best_diff:
            best_diff = diff
            best_eer = (far + frr) / 2.0
    return float(best_eer)


def _auc_roc(genuine: np.ndarray, attack: np.ndarray) -> float:
    """
    AUC-ROC via trapezoidal rule over 1001 thresholds.
    genuine labels = 1 (should be accepted), attack labels = 0.
    """
    if len(genuine) == 0 or len(attack) == 0:
        return 0.5
    thresholds = np.linspace(0.0, 1.0, 1001)
    tpr_list = []
    fpr_list = []
    for t in thresholds:
        tpr = float(np.mean(genuine > t))   # sensitivity
        fpr = float(np.mean(attack > t))    # 1 - specificity
        tpr_list.append(tpr)
        fpr_list.append(fpr)
    # Sort by FPR ascending for proper trapezoidal integration
    pairs = sorted(zip(fpr_list, tpr_list))
    fprs = np.array([p[0] for p in pairs])
    tprs = np.array([p[1] for p in pairs])
    return float(np.trapz(tprs, fprs))


def _auprc_approx(b: BiometricMetrics) -> float:
    """
    Approximate AUPRC from the single operating point.
    A full AUPRC requires per-event scores — this gives a conservative estimate.
    """
    p = b.tp / (b.tp + b.fp) if (b.tp + b.fp) > 0 else 0.0
    r = b.tp / (b.tp + b.fn) if (b.tp + b.fn) > 0 else 0.0
    # Interpolated estimate: triangle under the precision-recall curve
    return float(p * r)
