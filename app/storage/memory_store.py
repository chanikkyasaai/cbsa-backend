from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set

import numpy as np


VECTOR_SIZE = 48
SESSION_TTL_SECONDS: float = 600.0    # 10 minutes of inactivity → evict session
MAX_SESSION_EVENTS: int = 10_000      # cap on event_history per session (memory guard)


@dataclass
class TrustState:
    """
    Per-session trust tracking state for Layer-4 (TrustEngine).

    Uses a plain dataclass (not slots=True) so that trust_engine.update_trust()
    can modify fields in-place. Embedded in SessionState for co-location.

    Attributes
    ----------
    trust_score           : Current EMA trust T_t in [0,1], initial 0.5 (neutral)
    consecutive_risk      : Consecutive RISK decisions; resets on SAFE/UNCERTAIN
    consecutive_uncertain : Consecutive UNCERTAIN decisions; resets on SAFE/RISK
    consecutive_safe      : Consecutive SAFE decisions; resets on RISK/UNCERTAIN
    last_gat_time         : Unix timestamp of last Layer-3 escalation (0 = never)
    event_count           : Total events processed in this session
    """
    trust_score: float = 0.5
    consecutive_risk: int = 0
    consecutive_uncertain: int = 0
    consecutive_safe: int = 0
    last_gat_time: float = 0.0
    event_count: int = 0


@dataclass(slots=True)
class SessionState:
    short_window: deque = field(default_factory=lambda: deque(maxlen=5))
    medium_window: deque = field(default_factory=lambda: deque(maxlen=20))
    running_mean: np.ndarray = field(default_factory=lambda: np.zeros(VECTOR_SIZE, dtype=np.float64))
    running_variance: np.ndarray = field(default_factory=lambda: np.zeros(VECTOR_SIZE, dtype=np.float64))
    m2: np.ndarray = field(default_factory=lambda: np.zeros(VECTOR_SIZE, dtype=np.float64))
    sample_count: int = 0
    event_history: List[np.ndarray] = field(default_factory=list)
    last_timestamp: Optional[float] = None
    seen_nonces: Set[str] = field(default_factory=set)
    fast_delta_count: int = 0
    trust_state: TrustState = field(default_factory=TrustState)
    last_activity: float = field(default_factory=time.time)   # Unix timestamp of last event

    # ── Behavioral Session Fingerprint (Layer-2c) ─────────────────────────
    # EMA-updated Markov transition probability matrix.
    # transition_probs[prev_type][curr_type] = EMA-probability of this transition.
    # Used by transition_engine.py to compute per-event transition_surprise.
    transition_probs: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    # The event_type of the previous event — None until the first event is processed.
    prev_event_type: Optional[str] = None


class MemoryStore:
    def __init__(self) -> None:
        self.sessions: Dict[str, SessionState] = {}
        self.warmup_buffers: Dict[str, List[np.ndarray]] = {}

    def get_or_create_session(self, session_id: str) -> SessionState:
        if session_id not in self.sessions:
            self.sessions[session_id] = SessionState()
        state = self.sessions[session_id]
        state.last_activity = time.time()
        # Task 9: enforce max session size cap to prevent unbounded memory growth
        if len(state.event_history) > MAX_SESSION_EVENTS:
            # Keep the most recent half — preserve temporal context
            state.event_history = state.event_history[MAX_SESSION_EVENTS // 2:]
        return state

    def touch_session(self, session_id: str) -> None:
        """Update last_activity for an active session."""
        if session_id in self.sessions:
            self.sessions[session_id].last_activity = time.time()

    def evict_expired_sessions(self) -> int:
        """
        Remove sessions that have been inactive for SESSION_TTL_SECONDS.
        Returns the number of evicted sessions.
        """
        now = time.time()
        expired = [
            sid for sid, state in self.sessions.items()
            if (now - state.last_activity) > SESSION_TTL_SECONDS
        ]
        for sid in expired:
            del self.sessions[sid]
        return len(expired)

    def get_or_create_warmup_buffer(self, username: str) -> List[np.ndarray]:
        if username not in self.warmup_buffers:
            self.warmup_buffers[username] = []
        return self.warmup_buffers[username]

    def clear_warmup_buffer(self, username: str) -> None:
        self.warmup_buffers.pop(username, None)


memory_store = MemoryStore()
