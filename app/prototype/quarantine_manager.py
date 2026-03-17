"""
Prototype Quarantine Manager

Design Rationale
-----------------
Immediate prototype creation upon any low-similarity event creates two problems:

  SECURITY: An attacker who gains momentary access can inject a single
  deviating behavioral event that permanently alters the user's prototype
  set. Subsequent attacks matching this injected prototype would pass
  authentication.

  CORRECTNESS: Transient behavioral patterns (e.g., a user briefly distracted,
  a network glitch causing anomalous sensor readings, or a one-off interaction
  with an unfamiliar UI element) should NOT become permanent behavioral
  reference points.

Quarantine Protocol
-------------------
New behavioral observations enter a per-user CandidatePool in memory.
A CandidatePrototype is promoted to a full Prototype only when it satisfies
ALL of the following criteria simultaneously:

  1. OBSERVATION COUNT:  count >= N_MIN_OBSERVATIONS (default: 3)
     The pattern must be observed at least 3 times. A single anomalous event
     cannot trigger promotion. This prevents noise and single-event injection.

  2. TEMPORAL SPREAD:  time_span >= T_MIN_SPAN_SECONDS (default: 30.0)
     The observations must span at least 30 seconds. This prevents a burst
     of events within a single second from satisfying the count requirement.
     Legitimate behavioral patterns emerge across natural interaction time.

  3. CONSISTENCY:  mean_cosine_to_centroid >= CONSISTENCY_THRESHOLD (default: 0.72)
     All observations assigned to this candidate must directionally agree.
     Consistency = mean cosine similarity of observations to the candidate centroid.
     0.72 corresponds to a mean inter-vector angle of ≤ 44°, requiring strong
     directional alignment in 48-dimensional behavioral space.

Candidate Matching
------------------
A new vector v is assigned to existing candidate C_j if:
    cos(v, centroid_j) >= CANDIDATE_MATCH_THRESHOLD  (default: 0.75)

If no candidate matches, a new candidate is created (up to MAX_CANDIDATES_PER_USER).

Candidate Expiry
----------------
Candidates older than T_EXPIRE_SECONDS (default: 600s = 10 minutes) without
promotion are silently deleted. This prevents memory growth from abandoned
behavioral patterns.

Why These Specific Values?
--------------------------
  N_MIN = 3:    Minimum for statistical validity (mean of 3 samples is more
                stable than mean of 1-2). Balances security vs. enrollment speed.

  T_MIN = 30s:  A typical user interaction episode is 30-60s. Requiring 30s
                ensures the pattern spans at least one natural behavioral segment.

  CONSISTENCY = 0.72:  Empirically chosen as a balance between strictness
                (rejecting noisy patterns) and permissiveness (accepting
                legitimate but slightly variable behavior). In 48-D space,
                0.72 cosine similarity is a strong directional agreement.

  MATCH = 0.75:  Higher than CONSISTENCY_THRESHOLD because assignment to a
                 candidate must be confident. If a new vector weakly matches
                 an existing candidate, creating a new candidate is preferable.

  T_EXPIRE = 600s:  10 minutes without promotion suggests the behavioral
                    pattern is not recurring — it was likely transient.
"""

from __future__ import annotations

import time
from typing import Dict, List, Optional, Tuple

import numpy as np

# ── Quarantine parameters ───────────────────────────────────────────────────

N_MIN_OBSERVATIONS: int = 3           # Minimum observations before promotion
CONSISTENCY_THRESHOLD: float = 0.72   # Minimum mean cosine similarity to centroid
T_MIN_SPAN_SECONDS: float = 30.0      # Minimum time between first and last observation
T_EXPIRE_SECONDS: float = 600.0       # Candidate expiry (10 minutes of inactivity)
CANDIDATE_MATCH_THRESHOLD: float = 0.75   # Cosine threshold for assigning to a candidate
MAX_CANDIDATES_PER_USER: int = 20         # Per-user pool size cap
MAX_OBSERVATIONS_PER_CANDIDATE: int = 30  # Cap to prevent unbounded memory growth


class CandidatePrototype:
    """
    A behavioral pattern candidate awaiting promotion to a full Prototype.

    The centroid is maintained as an online mean (no need to store all vectors
    for centroid computation, but observations are kept for consistency evaluation
    up to MAX_OBSERVATIONS_PER_CANDIDATE).
    """

    __slots__ = ("centroid", "observations", "first_seen", "last_seen")

    def __init__(self, initial_vector: np.ndarray, timestamp: float) -> None:
        self.centroid: np.ndarray = initial_vector.copy()
        self.observations: List[np.ndarray] = [initial_vector.copy()]
        self.first_seen: float = timestamp
        self.last_seen: float = timestamp

    # ── Properties ──────────────────────────────────────────────────────────

    def observation_count(self) -> int:
        return len(self.observations)

    def time_span(self) -> float:
        """Seconds between first and most recent observation."""
        return self.last_seen - self.first_seen

    def consistency(self) -> float:
        """
        Mean cosine similarity of all stored observations to the centroid.

            C = (1/|obs|) * sum_{v in obs} cos(v, centroid)

        Returns 1.0 when only one observation exists (trivially consistent).
        Returns float in [0, 1].
        """
        if len(self.observations) < 2:
            return 1.0
        norm_c = float(np.linalg.norm(self.centroid))
        if norm_c < 1e-10:
            return 0.0
        sims: List[float] = []
        for obs in self.observations:
            norm_o = float(np.linalg.norm(obs))
            if norm_o < 1e-10:
                sims.append(0.0)
            else:
                sim = float(np.dot(obs, self.centroid) / (norm_o * norm_c))
                sims.append(max(0.0, min(1.0, sim)))
        return float(np.mean(sims))

    def is_expired(self, current_time: float) -> bool:
        return (current_time - self.last_seen) > T_EXPIRE_SECONDS

    def is_ready_for_promotion(self) -> bool:
        """
        Check all three promotion criteria simultaneously.
        All must be satisfied for promotion.
        """
        return (
            self.observation_count() >= N_MIN_OBSERVATIONS
            and self.time_span() >= T_MIN_SPAN_SECONDS
            and self.consistency() >= CONSISTENCY_THRESHOLD
        )

    # ── Mutation ─────────────────────────────────────────────────────────────

    def update(self, new_vector: np.ndarray, timestamp: float) -> None:
        """
        Incorporate a new observation into this candidate.

        Online mean update (avoids recomputing from scratch):
            centroid_new = centroid_old + (v - centroid_old) / (n + 1)

        This is numerically equivalent to the arithmetic mean but O(D)
        instead of O(n*D), making it suitable for streaming updates.
        """
        n = len(self.observations)
        self.centroid = self.centroid + (new_vector - self.centroid) / (n + 1)
        if len(self.observations) < MAX_OBSERVATIONS_PER_CANDIDATE:
            self.observations.append(new_vector.copy())
        self.last_seen = timestamp


class QuarantineManager:
    """
    Manages the per-user candidate prototype pool.

    This manager is in-memory only. Candidates are not persisted to disk —
    only promoted prototypes are persisted via the store. Loss of in-memory
    state (server restart) simply resets candidate pools; this is acceptable
    because legitimate behavioral patterns will re-accumulate.
    """

    def __init__(self) -> None:
        # username -> list of CandidatePrototype
        self._pools: Dict[str, List[CandidatePrototype]] = {}

    # ── Internal helpers ─────────────────────────────────────────────────────

    def _get_pool(self, username: str) -> List[CandidatePrototype]:
        if username not in self._pools:
            self._pools[username] = []
        return self._pools[username]

    def _purge_expired(self, username: str, current_time: float) -> None:
        """Remove expired candidates to prevent memory growth."""
        pool = self._get_pool(username)
        self._pools[username] = [c for c in pool if not c.is_expired(current_time)]

    # ── Public interface ──────────────────────────────────────────────────────

    def submit(
        self,
        username: str,
        vector: np.ndarray,
        current_time: Optional[float] = None,
    ) -> Optional[Tuple[np.ndarray, np.ndarray, int]]:
        """
        Submit a behavioral vector to the candidate pool.

        Matching algorithm:
          1. Find the candidate C_j whose centroid has cosine similarity
             >= CANDIDATE_MATCH_THRESHOLD with the new vector.
             Among multiple matches, the highest-similarity candidate wins.
          2. If matched: update the candidate's centroid and observations.
             If the updated candidate satisfies all promotion criteria,
             remove it from the pool and return promotion payload.
          3. If no match: create a new candidate (if pool not full).

        Args:
            username     : User identifier (for per-user pool isolation)
            vector       : 48-D behavioral vector (must be normalized to [0,1])
            current_time : Unix timestamp (defaults to time.time())

        Returns:
            (centroid, variance, support_count) if a candidate is ready for
            promotion to a full prototype; None otherwise.
        """
        t = current_time if current_time is not None else time.time()
        self._purge_expired(username, t)
        pool = self._get_pool(username)

        # ── Step 1: Find best matching candidate ──────────────────────────
        best_idx: int = -1
        best_sim: float = CANDIDATE_MATCH_THRESHOLD - 1e-9  # strict threshold

        for idx, candidate in enumerate(pool):
            norm_c = float(np.linalg.norm(candidate.centroid))
            norm_v = float(np.linalg.norm(vector))
            if norm_c < 1e-10 or norm_v < 1e-10:
                continue
            sim = float(np.dot(vector, candidate.centroid) / (norm_c * norm_v))
            if sim > best_sim:
                best_sim = sim
                best_idx = idx

        # ── Step 2: Update matched candidate or create new one ────────────
        if best_idx >= 0:
            pool[best_idx].update(vector, t)
            if pool[best_idx].is_ready_for_promotion():
                promoted = pool.pop(best_idx)
                return self._build_promotion_payload(promoted)
        elif len(pool) < MAX_CANDIDATES_PER_USER:
            pool.append(CandidatePrototype(vector, t))

        return None

    def _build_promotion_payload(
        self,
        candidate: CandidatePrototype,
    ) -> Tuple[np.ndarray, np.ndarray, int]:
        """
        Build (centroid, variance, support_count) from a promoted candidate.

        Variance is computed across all stored observations. For candidates
        with only MAX_OBSERVATIONS_PER_CANDIDATE stored (cap enforced), the
        variance represents that sample — a good approximation of the true
        variance when observations are drawn consistently.
        """
        obs = candidate.observations
        if len(obs) > 1:
            obs_matrix = np.vstack(obs)
            variance = np.var(obs_matrix, axis=0)         # unbiased via numpy default
        else:
            variance = np.zeros_like(candidate.centroid)

        return (
            candidate.centroid.copy(),
            np.maximum(variance, 1e-8),
            len(obs),
        )

    def get_pool_size(self, username: str) -> int:
        """Return current number of candidates in the user's pool."""
        return len(self._get_pool(username))

    def clear_user(self, username: str) -> None:
        """Remove all candidates for a user (e.g., after re-enrollment)."""
        self._pools.pop(username, None)

    def get_pool_status(self, username: str) -> List[dict]:
        """
        Return diagnostic information about the candidate pool.
        Used for monitoring and debugging only.
        """
        pool = self._get_pool(username)
        return [
            {
                "observations": c.observation_count(),
                "time_span_s": round(c.time_span(), 1),
                "consistency": round(c.consistency(), 3),
                "is_expired": c.is_expired(time.time()),
                "ready": c.is_ready_for_promotion(),
            }
            for c in pool
        ]


# Module-level singleton — shared across all requests (in-process)
quarantine_manager: QuarantineManager = QuarantineManager()
