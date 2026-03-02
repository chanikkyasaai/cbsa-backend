"""
Behavioral Data File Logger
Appends incoming behavioral events to per-user JSONL log files.
"""

import json
import logging
import time
from pathlib import Path
from typing import Any, Dict

logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
BEHAVIORAL_LOG_DIR = DATA_DIR / "behavioral_logs"


class BehavioralFileLogger:
    """Logs raw behavioral events to JSONL files per user."""

    def __init__(self):
        BEHAVIORAL_LOG_DIR.mkdir(parents=True, exist_ok=True)
        logger.info(f"BehavioralFileLogger initialized. Logs at: {BEHAVIORAL_LOG_DIR}")

    def _log_path(self, user_id: str) -> Path:
        safe_uid = user_id.replace("/", "_").replace("\\", "_")
        return BEHAVIORAL_LOG_DIR / f"{safe_uid}.jsonl"

    def log_event(self, user_id: str, session_id: str, event_data: Dict[str, Any]):
        """Append a single event to the user's log file."""
        record = {
            "logged_at": time.time(),
            "user_id": user_id,
            "session_id": session_id,
            **event_data,
        }
        path = self._log_path(user_id)
        try:
            with path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(record) + "\n")
        except Exception as e:
            logger.error(f"Failed to log event for user {user_id}: {e}")

    def load_user_events(self, user_id: str) -> list:
        """Load all logged events for a user. Returns list of dicts."""
        path = self._log_path(user_id)
        if not path.exists():
            return []
        events = []
        try:
            with path.open("r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        try:
                            events.append(json.loads(line))
                        except json.JSONDecodeError:
                            pass
        except Exception as e:
            logger.error(f"Failed to load events for user {user_id}: {e}")
        return events

    def list_users(self) -> list:
        """Return list of user IDs that have logged data."""
        return [p.stem for p in BEHAVIORAL_LOG_DIR.glob("*.jsonl")]


# Singleton
behavioral_logger = BehavioralFileLogger()
