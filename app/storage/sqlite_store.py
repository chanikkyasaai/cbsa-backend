from __future__ import annotations

import json
import os
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from app.models.prototype import Prototype
from app.storage.memory_store import memory_store


DB_PATH = Path(os.environ["DB_PATH"]) if os.environ.get("DB_PATH") else Path(__file__).resolve().parents[2] / "cbsa.db"
WARMUP_WINDOW_COUNT = 20


def _utc_now_iso() -> str:
    return datetime.utcnow().isoformat()


def _to_json_array(array: np.ndarray) -> str:
    return json.dumps(array.astype(float).tolist(), separators=(",", ":"))


def _from_json_array(value: str) -> np.ndarray:
    parsed = json.loads(value)
    return np.asarray(parsed, dtype=np.float64)


class SQLiteStore:
    def __init__(self, db_path: Path) -> None:
        self.db_path = db_path
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.db_path)
        connection.row_factory = sqlite3.Row
        return connection

    def _initialize(self) -> None:
        with self._connect() as connection:
            cursor = connection.cursor()
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS users (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    username TEXT UNIQUE NOT NULL,
                    initialized INTEGER DEFAULT 0,
                    created_at TEXT
                )
                """
            )
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS prototypes (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    username TEXT NOT NULL,
                    vector_json TEXT NOT NULL,
                    variance_json TEXT NOT NULL,
                    support_count INTEGER,
                    created_at TEXT,
                    updated_at TEXT
                )
                """
            )
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS behaviour_logs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    username TEXT NOT NULL,
                    session_id TEXT NOT NULL,
                    event_timestamp REAL,
                    event_type TEXT,
                    vector_json TEXT,
                    short_drift REAL,
                    long_drift REAL,
                    stability_score REAL,
                    created_at TEXT
                )
                """
            )
            connection.commit()

    def ensure_user(self, username: str) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                INSERT OR IGNORE INTO users (username, initialized, created_at)
                VALUES (?, 0, ?)
                """,
                (username, _utc_now_iso()),
            )
            connection.commit()

    def get_user_initialized(self, username: str) -> bool:
        self.ensure_user(username)
        with self._connect() as connection:
            row = connection.execute(
                "SELECT initialized FROM users WHERE username = ?",
                (username,),
            ).fetchone()
            return bool(row["initialized"]) if row else False

    def set_user_initialized(self, username: str, initialized: bool) -> None:
        with self._connect() as connection:
            connection.execute(
                "UPDATE users SET initialized = ? WHERE username = ?",
                (1 if initialized else 0, username),
            )
            connection.commit()

    def insert_behaviour_log(
        self,
        username: str,
        session_id: str,
        event_timestamp: float,
        event_type: str,
        vector: np.ndarray,
        short_drift: float,
        long_drift: float,
        stability_score: float,
    ) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO behaviour_logs (
                    username,
                    session_id,
                    event_timestamp,
                    event_type,
                    vector_json,
                    short_drift,
                    long_drift,
                    stability_score,
                    created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    username,
                    session_id,
                    event_timestamp,
                    event_type,
                    _to_json_array(vector),
                    float(short_drift),
                    float(long_drift),
                    float(stability_score),
                    _utc_now_iso(),
                ),
            )
            connection.commit()

    def get_prototypes(self, username: str) -> List[Prototype]:
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT id, vector_json, variance_json, support_count, created_at, updated_at
                FROM prototypes
                WHERE username = ?
                ORDER BY id ASC
                """,
                (username,),
            ).fetchall()

        prototypes: List[Prototype] = []
        for row in rows:
            created_at = datetime.fromisoformat(row["created_at"]) if row["created_at"] else datetime.utcnow()
            updated_at = datetime.fromisoformat(row["updated_at"]) if row["updated_at"] else created_at
            prototypes.append(
                Prototype(
                    prototype_id=int(row["id"]),
                    vector=_from_json_array(row["vector_json"]),
                    variance=np.maximum(_from_json_array(row["variance_json"]), 1e-6),
                    support_count=int(row["support_count"] or 0),
                    created_at=created_at,
                    last_updated=updated_at,
                )
            )
        return prototypes

    def insert_prototype(self, username: str, vector: np.ndarray, variance: np.ndarray, support_count: int) -> int:
        now = _utc_now_iso()
        with self._connect() as connection:
            cursor = connection.execute(
                """
                INSERT INTO prototypes (
                    username,
                    vector_json,
                    variance_json,
                    support_count,
                    created_at,
                    updated_at
                ) VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    username,
                    _to_json_array(vector),
                    _to_json_array(np.maximum(variance, 1e-6)),
                    int(support_count),
                    now,
                    now,
                ),
            )
            connection.commit()
            return int(cursor.lastrowid)

    def update_prototype(self, prototype: Prototype) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                UPDATE prototypes
                SET vector_json = ?, variance_json = ?, support_count = ?, updated_at = ?
                WHERE id = ?
                """,
                (
                    _to_json_array(prototype.vector),
                    _to_json_array(np.maximum(prototype.variance, 1e-6)),
                    int(prototype.support_count),
                    _utc_now_iso(),
                    int(prototype.prototype_id),
                ),
            )
            connection.commit()

    def enforce_prototype_limit(self, username: str, limit: int) -> None:
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT id
                FROM prototypes
                WHERE username = ?
                ORDER BY support_count ASC, id ASC
                """,
                (username,),
            ).fetchall()
            if len(rows) <= limit:
                return

            delete_count = len(rows) - limit
            ids_to_delete = [int(row["id"]) for row in rows[:delete_count]]
            connection.executemany(
                "DELETE FROM prototypes WHERE id = ?",
                [(prototype_id,) for prototype_id in ids_to_delete],
            )
            connection.commit()

    def collect_warmup_window(self, username: str, window_vector: np.ndarray) -> Dict[str, int | bool]:
        initialized = self.get_user_initialized(username)
        if initialized:
            return {"warmup": False, "collected_windows": WARMUP_WINDOW_COUNT}

        warmup_buffer = memory_store.get_or_create_warmup_buffer(username)
        warmup_buffer.append(window_vector.copy())
        collected = len(warmup_buffer)

        if collected < WARMUP_WINDOW_COUNT:
            return {"warmup": True, "collected_windows": collected}

        buffer_matrix = np.vstack(warmup_buffer)
        mean_vector = np.mean(buffer_matrix, axis=0)
        variance_vector = np.var(buffer_matrix, axis=0)
        self.insert_prototype(username, mean_vector, np.maximum(variance_vector, 1e-6), support_count=collected)
        self.set_user_initialized(username, True)
        memory_store.clear_warmup_buffer(username)

        return {"warmup": True, "collected_windows": collected}

    def export_user(self, username: str) -> Dict[str, object]:
        self.ensure_user(username)
        with self._connect() as connection:
            user_row = connection.execute(
                "SELECT username, initialized, created_at FROM users WHERE username = ?",
                (username,),
            ).fetchone()
            prototype_rows = connection.execute(
                """
                SELECT id, username, vector_json, variance_json, support_count, created_at, updated_at
                FROM prototypes WHERE username = ? ORDER BY id ASC
                """,
                (username,),
            ).fetchall()
            log_rows = connection.execute(
                """
                SELECT id, username, session_id, event_timestamp, event_type, vector_json,
                       short_drift, long_drift, stability_score, created_at
                FROM behaviour_logs WHERE username = ? ORDER BY id ASC
                """,
                (username,),
            ).fetchall()

        return {
            "username": username,
            "user": dict(user_row) if user_row else {"username": username, "initialized": 0, "created_at": None},
            "prototypes": [dict(row) for row in prototype_rows],
            "behaviour_logs": [dict(row) for row in log_rows],
        }

    def import_user(self, data: Dict[str, object]) -> None:
        username = str(data.get("username", "")).strip()
        if not username:
            raise ValueError("Invalid import payload: username is required")

        self.ensure_user(username)

        incoming_user = data.get("user") if isinstance(data.get("user"), dict) else {}
        incoming_created_at = incoming_user.get("created_at") if isinstance(incoming_user, dict) else None
        incoming_initialized = int(incoming_user.get("initialized", 0)) if isinstance(incoming_user, dict) else 0

        with self._connect() as connection:
            existing_user = connection.execute(
                "SELECT created_at, initialized FROM users WHERE username = ?",
                (username,),
            ).fetchone()

            earliest_created_at = existing_user["created_at"] if existing_user and existing_user["created_at"] else incoming_created_at
            if existing_user and existing_user["created_at"] and incoming_created_at:
                earliest_created_at = min(existing_user["created_at"], incoming_created_at)

            merged_initialized = max(int(existing_user["initialized"] if existing_user else 0), incoming_initialized)

            connection.execute(
                "UPDATE users SET created_at = ?, initialized = ? WHERE username = ?",
                (earliest_created_at or _utc_now_iso(), merged_initialized, username),
            )

            existing_proto_vectors = {
                row["vector_json"]
                for row in connection.execute(
                    "SELECT vector_json FROM prototypes WHERE username = ?",
                    (username,),
                ).fetchall()
            }

            for prototype in data.get("prototypes", []):
                if not isinstance(prototype, dict):
                    continue
                vector_json = prototype.get("vector_json")
                variance_json = prototype.get("variance_json")
                if not isinstance(vector_json, str) or not isinstance(variance_json, str):
                    continue
                if vector_json in existing_proto_vectors:
                    continue

                connection.execute(
                    """
                    INSERT INTO prototypes (username, vector_json, variance_json, support_count, created_at, updated_at)
                    VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (
                        username,
                        vector_json,
                        variance_json,
                        int(prototype.get("support_count", 1)),
                        prototype.get("created_at") or _utc_now_iso(),
                        prototype.get("updated_at") or prototype.get("created_at") or _utc_now_iso(),
                    ),
                )
                existing_proto_vectors.add(vector_json)

            for log_item in data.get("behaviour_logs", []):
                if not isinstance(log_item, dict):
                    continue
                connection.execute(
                    """
                    INSERT INTO behaviour_logs (
                        username, session_id, event_timestamp, event_type, vector_json,
                        short_drift, long_drift, stability_score, created_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        username,
                        str(log_item.get("session_id", "")),
                        float(log_item.get("event_timestamp", 0.0)),
                        str(log_item.get("event_type", "")),
                        str(log_item.get("vector_json", "[]")),
                        float(log_item.get("short_drift", 0.0)),
                        float(log_item.get("long_drift", 0.0)),
                        float(log_item.get("stability_score", 0.0)),
                        str(log_item.get("created_at") or _utc_now_iso()),
                    ),
                )

            connection.commit()

