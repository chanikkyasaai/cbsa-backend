"""
tests/metrics — Composite Behavioral Authentication Metrics Package

Exports:
    CompositeMetrics     — full evaluation over genuine + attack result lists
    BiometricMetrics     — FAR, FRR, EER, AUC-ROC
    DetectionMetrics     — Precision, Recall, F1, AUPRC
    TemporalMetrics      — attack detection latency, trust convergence, stability
    DriftSensitivityMetrics — three-scale drift response analysis
    PrototypeHealthMetrics  — cohesion distribution, support strength, lifecycle
    SystemHealthMetrics  — decision distribution, warmup characterisation
    MetricsReport        — unified printable report

Usage:
    from tests.metrics import CompositeMetrics
    report = CompositeMetrics.compute(genuine_results, attack_results)
    report.print_full()
"""

from tests.metrics.composite import (
    CompositeMetrics,
    BiometricMetrics,
    DetectionMetrics,
    TemporalMetrics,
    DriftSensitivityMetrics,
    PrototypeHealthMetrics,
    SystemHealthMetrics,
    MetricsReport,
)

__all__ = [
    "CompositeMetrics",
    "BiometricMetrics",
    "DetectionMetrics",
    "TemporalMetrics",
    "DriftSensitivityMetrics",
    "PrototypeHealthMetrics",
    "SystemHealthMetrics",
    "MetricsReport",
]
