from __future__ import annotations

from typing import Any, Dict, Iterable

import numpy as np

from app.models.behaviour_event import BehaviourEvent
from app.storage.memory_store import memory_store


VECTOR_SIZE = 48


def verify_signature(_: Dict[str, Any]) -> bool:
    return True


def _is_numeric_list(values: Iterable[Any]) -> bool:
    return all(isinstance(value, (int, float)) for value in values)


def validate_and_extract(raw_json: dict) -> BehaviourEvent:
    if not isinstance(raw_json, dict):
        raise ValueError("Invalid payload: expected JSON object")

    if not verify_signature(raw_json):
        raise ValueError("Invalid signature")

    user_id = raw_json.get("user_id")
    if not isinstance(user_id, str) or not user_id.strip():
        raise ValueError("Invalid user_id")
    user_id = user_id.strip()

    session_id = raw_json.get("session_id")
    if not isinstance(session_id, str) or not session_id.strip():
        raise ValueError("Invalid session_id")

    timestamp = raw_json.get("timestamp")
    if not isinstance(timestamp, (int, float)):
        raise ValueError("Invalid timestamp")
    timestamp = float(timestamp)

    event_type = raw_json.get("event_type")
    if not isinstance(event_type, str) or not event_type.strip():
        raise ValueError("Invalid event_type")

    event_data = raw_json.get("event_data")
    if not isinstance(event_data, dict):
        raise ValueError("Invalid event_data")

    nonce = event_data.get("nonce")
    if not isinstance(nonce, str) or not nonce.strip():
        raise ValueError("Invalid nonce")

    vector = event_data.get("vector")
    if vector is None:
        raise ValueError("Missing vector")
    if not isinstance(vector, list):
        raise ValueError("Invalid vector: expected list")
    if len(vector) != VECTOR_SIZE:
        raise ValueError("Invalid vector length: expected 48")
    if not _is_numeric_list(vector):
        raise ValueError("Invalid vector values: all elements must be numeric")

    vector_array = np.asarray(vector, dtype=np.float64)
    if np.any(vector_array < 0.0) or np.any(vector_array > 1.0):
        raise ValueError("Invalid vector values: must be within [0, 1]")

    session_state = memory_store.get_or_create_session(session_id)

    if nonce in session_state.seen_nonces:
        raise ValueError("Duplicate nonce for session")

    if session_state.last_timestamp is not None and timestamp <= session_state.last_timestamp:
        raise ValueError("Non-monotonic timestamp for session")

    if session_state.last_timestamp is not None:
        delta = timestamp - session_state.last_timestamp
        if delta < 0.040:
            session_state.fast_delta_count += 1
        else:
            session_state.fast_delta_count = 0

        if session_state.fast_delta_count > 5:
            raise ValueError("Rejected: too many consecutive events with delta < 40ms")

    session_state.seen_nonces.add(nonce)
    session_state.last_timestamp = timestamp

    return BehaviourEvent(
        user_id=user_id,
        session_id=session_id,
        vector=vector_array,
        timestamp=timestamp,
        nonce=nonce,
        event_type=event_type,
    )
