"""
Structured Centralized Event Logger

Writes one document per processed behavioural event to the Cosmos DB
``behaviour-logs`` container via the unified store's log_behaviour_event()
method. Also emits a structured JSON line to the Python logger for local
development / log aggregation.

All Layer-2 and Layer-4 outputs are captured in a single log record:

    {
        "username":                 str,
        "session_id":               str,
        "event_timestamp":          float,
        "event_type":               str,
        # Layer-2 outputs
        "similarity_score":         float,
        "short_drift":              float,
        "long_drift":               float,
        "stability_score":          float,
        "behavioural_consistency":  float,
        "prototype_confidence":     float,
        "prototype_support_strength": float,
        "anomaly_indicator":        float,
        "matched_prototype_id":     int | None,
        # Layer-4 outputs
        "trust_score":              float,
        "raw_trust_signal":         float,
        "alpha_t":                  float,
        "decision":                 str,
        "consecutive_risk":         int,
        "consecutive_uncertain":    int,
        "layer3_used":              bool,
        "gat_augmented":            bool,
    }

Usage:
    from app.logging.structured_logger import structured_logger
    structured_logger.log(
        username=username,
        session_id=session_id,
        event_timestamp=event.timestamp,
        event_type=event.event_type,
        proto_metrics=proto_result,
        trust_result=trust_result,
    )
"""

from __future__ import annotations

import json
import logging
from typing import Optional

from app.models.prototype import PrototypeMetrics
from app.trust.trust_engine import TrustResult

logger = logging.getLogger(__name__)


class StructuredLogger:
    """
    Thin façade over the CosmosUnifiedStore's log_behaviour_event().

    Import is deferred to avoid circular imports at module load time
    (cosmos_unified_store → prototype_engine → structured_logger).
    """

    def __init__(self) -> None:
        self._store = None

    def _get_store(self):
        if self._store is None:
            from app.storage.cosmos_unified_store import cosmos_prototype_store
            self._store = cosmos_prototype_store
        return self._store

    def log(
        self,
        username: str,
        session_id: str,
        event_timestamp: float,
        event_type: str,
        proto_metrics: PrototypeMetrics,
        trust_result: TrustResult,
    ) -> None:
        """
        Write a complete pipeline event record.

        Errors are caught and logged — logging must never crash the pipeline.
        """
        record = {
            "username": username,
            "session_id": session_id,
            "event_timestamp": event_timestamp,
            "event_type": event_type,
            # Layer-2
            "similarity_score": round(proto_metrics.similarity_score, 6),
            "short_drift": round(proto_metrics.short_drift, 6),
            "long_drift": round(proto_metrics.long_drift, 6),
            "stability_score": round(proto_metrics.stability_score, 6),
            "behavioural_consistency": round(proto_metrics.behavioural_consistency, 6),
            "prototype_confidence": round(proto_metrics.prototype_confidence, 6),
            "prototype_support_strength": round(proto_metrics.prototype_support_strength, 6),
            "anomaly_indicator": round(proto_metrics.anomaly_indicator, 6),
            "matched_prototype_id": proto_metrics.matched_prototype_id,
            # Layer-4
            "trust_score": round(trust_result.trust_score, 6),
            "raw_trust_signal": round(trust_result.raw_trust_signal, 6),
            "alpha_t": round(trust_result.alpha_t, 6),
            "decision": trust_result.decision,
            "consecutive_risk": trust_result.consecutive_risk,
            "consecutive_uncertain": trust_result.consecutive_uncertain,
            "layer3_used": trust_result.escalate_to_layer3,
            "gat_augmented": trust_result.gat_augmented,
        }

        # Emit structured line to Python logger (visible in console / log aggregator)
        logger.info("CBSA_EVENT %s", json.dumps(record))

        # Persist to Cosmos DB
        try:
            store = self._get_store()
            store.log_behaviour_event(
                username=username,
                session_id=session_id,
                event_timestamp=event_timestamp,
                event_type=event_type,
                similarity_score=proto_metrics.similarity_score,
                short_drift=proto_metrics.short_drift,
                long_drift=proto_metrics.long_drift,
                stability_score=proto_metrics.stability_score,
                behavioural_consistency=proto_metrics.behavioural_consistency,
                prototype_confidence=proto_metrics.prototype_confidence,
                anomaly_indicator=proto_metrics.anomaly_indicator,
                trust_score=trust_result.trust_score,
                decision=trust_result.decision,
                matched_prototype_id=proto_metrics.matched_prototype_id,
                layer3_used=trust_result.escalate_to_layer3,
            )
        except Exception as exc:
            logger.error("structured_logger: failed to persist event log: %s", exc)


# Module-level singleton
structured_logger: StructuredLogger = StructuredLogger()
