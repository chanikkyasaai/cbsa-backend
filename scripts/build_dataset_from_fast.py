"""Build dataset and user profiles from fast.txt log."""
from __future__ import annotations

import argparse
import ast
import json
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Any, Dict, List


WINDOW_SECONDS = 20
EVENT_TYPE_EMBED_DIM = 8
DEVICE_CONTEXT_DIM = 4
BASE_VECTOR_DIM = 48
NODE_VECTOR_DIM = BASE_VECTOR_DIM + EVENT_TYPE_EMBED_DIM + DEVICE_CONTEXT_DIM


@dataclass
class ParsedEvent:
    user_id: str
    session_id: str
    timestamp: float
    event_type: str
    event_data: Dict[str, Any]


def event_type_embedding(event_type: str) -> List[float]:
    digest = sha256(event_type.encode("utf-8")).digest()
    return [b / 255.0 for b in digest[:EVENT_TYPE_EMBED_DIM]]


def device_context_vector(device_info: Dict[str, Any]) -> List[float]:
    battery = float(device_info.get("battery", 0.0))
    cpu = float(device_info.get("cpu", 0.0))
    memory = float(device_info.get("memory", 0.0))
    network = float(device_info.get("signal", 0.0))

    return [
        max(min(battery, 1.0), 0.0),
        max(min(cpu, 1.0), 0.0),
        max(min(memory, 1.0), 0.0),
        max(min(network, 1.0), 0.0),
    ]


def build_node_vector(event: ParsedEvent) -> List[float]:
    print(event)
    vector = event.event_data.get("vector", [0.0] * BASE_VECTOR_DIM)
    if len(vector) != BASE_VECTOR_DIM:
        vector = (vector + [0.0] * BASE_VECTOR_DIM)[:BASE_VECTOR_DIM]

    vector = list(vector)
    vector.extend(event_type_embedding(event.event_type))
    vector.extend(device_context_vector(event.event_data.get("deviceInfo", {})))

    if len(vector) < NODE_VECTOR_DIM:
        vector.extend([0.0] * (NODE_VECTOR_DIM - len(vector)))

    return vector[:NODE_VECTOR_DIM]


def parse_fast_file(path: Path) -> List[ParsedEvent]:
    events: List[ParsedEvent] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if "Data received:" not in line:
            continue
        _, payload = line.split("Data received:", 1)
        payload = payload.strip()
        try:
            data = ast.literal_eval(payload)
        except Exception:
            continue

        user_id = data.get("user_id") or "unknown"
        session_id = data.get("session_id") or "unknown_session"
        timestamp = float(data.get("timestamp") or 0.0)
        event_type = data.get("event_type") or "unknown"
        event_data = data.get("event_data") or {}

        events.append(
            ParsedEvent(
                user_id=user_id,
                session_id=session_id,
                timestamp=timestamp,
                event_type=event_type,
                event_data=event_data,
            )
        )
    return events


def build_dataset(events: List[ParsedEvent]) -> Dict[str, Any]:
    sessions: Dict[str, List[ParsedEvent]] = {}
    for event in events:
        sessions.setdefault(event.session_id, []).append(event)

    dataset = {
        "window_seconds": WINDOW_SECONDS,
        "node_vector_dim": NODE_VECTOR_DIM,
        "sessions": [],
    }

    for session_id, session_events in sessions.items():
        session_events.sort(key=lambda e: e.timestamp)
        if not session_events:
            continue

        latest_ts = session_events[-1].timestamp
        window_start = latest_ts - WINDOW_SECONDS
        window_events = [e for e in session_events if e.timestamp >= window_start]

        session_payload = {
            "session_id": session_id,
            "user_id": window_events[0].user_id,
            "window_start": window_start,
            "window_end": latest_ts,
            "events": [],
        }

        for event in window_events:
            session_payload["events"].append(
                {
                    "timestamp": event.timestamp,
                    "event_type": event.event_type,
                    "event_data": event.event_data,
                    "node_vector": build_node_vector(event),
                }
            )

        dataset["sessions"].append(session_payload)

    return dataset

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="fast.txt")
    parser.add_argument("--dataset", default="datasets/fast_dataset.json")
    parser.add_argument("--profiles", default="profiles/fast_profiles.json")
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        raise SystemExit(f"Input file not found: {input_path}")

    events = parse_fast_file(input_path)
    dataset = build_dataset(events)

    Path(args.dataset).write_text(json.dumps(dataset, indent=2), encoding="utf-8")

    print(f"Wrote dataset to {args.dataset} (sessions: {len(dataset['sessions'])})")


if __name__ == "__main__":
    main()
