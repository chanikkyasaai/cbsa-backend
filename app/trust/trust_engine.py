"""
Layer-4: Continuous Trust Engine — Decision and Escalation

Overview
--------
Layer-4 aggregates the rich behavioural state vector from Layer-2 (and
optionally Layer-3's GAT similarity) into a single continuous trust score,
applies temporal smoothing, makes zone-based decisions, and determines
when to escalate to the expensive Layer-3 (GAT) analysis.

This is a stateless engine — all per-session state is maintained in TrustState
objects managed by the caller (main.py WebSocket handler). This design keeps
the engine reentrant and testable.

Continuous Trust Model
----------------------
The trust score follows an Exponential Moving Average (EMA):

    T_t = alpha_t * T_{t-1} + (1 - alpha_t) * R_t

Where:
  T_t:    Trust score at time t, in [0, 1]
  T_{t-1}: Previous trust score (temporal memory)
  R_t:    Raw trust signal from Layer-2 metrics, in [0, 1]
  alpha_t: Adaptive EMA coefficient, in [alpha_min, alpha_max]

Raw Trust Signal
----------------
    R_t = w_sim * sim_t + w_stab * stab_t + w_drift * (1 - D_t)
          + w_trans * (1 - ts_t)

    D_t  = 0.50 * d_short_t + 0.30 * d_medium_t + 0.20 * d_long_t
           (three-scale composite drift)
    ts_t = transition_surprise_t  (Markov sequential fingerprint)

    Weights: w_sim=0.40, w_stab=0.20, w_drift=0.30, w_trans=0.10
             (sum = 1.0)

Boundary verification:
  Maximum R_t (sim=1, stab=1, D_t=0, ts=0): 0.40+0.20+0.30+0.10*1 = 1.0
  Minimum R_t (sim=0, stab=0, D_t=1, ts=1): 0 + 0 + 0 + 0.10*0   = 0.0
  R_t is naturally bounded in [0, 1] without clipping.

Why these weights?
  - Similarity (0.40): Primary authentication signal — how well does current
    behavior match the established prototype? Highest weight.
  - Drift-complement (0.30): Penalizes behavioral deviation. Three-scale
    composite: short (0.50) > medium (0.30) > long (0.20). Short drift
    detects single-event anomalies; medium drift detects interaction-mode
    transitions; long drift detects identity-baseline evolution.
  - Stability (0.20): Behavioral consistency quality. Lower weight because
    stability measures intra-window coherence, not inter-prototype agreement.
  - Transition fingerprint (0.10): Sequential navigation structure. Captures
    whether the user follows their habitual app-navigation patterns at the
    event-sequence level — orthogonal to all magnitude/shape signals above.
    An attacker who mirrors feature vectors but uses atypical navigation
    sequences is penalized here while the other signals remain unaffected.

Prototype Topology Cohesion — EMA Alpha Modulation
----------------------------------------------------
The adaptive EMA coefficient alpha_max is further modulated by prototype
topology cohesion (the mean pairwise cosine similarity between all prototypes):

    alpha_eff_max = alpha_max * (COHESION_ALPHA_FLOOR + COHESION_ALPHA_RANGE * cohesion)

    COHESION_ALPHA_FLOOR = 0.90,  COHESION_ALPHA_RANGE = 0.10

  cohesion = 1.0 (tight model): alpha_eff_max = alpha_max * 1.0 = 0.85 (full inertia)
  cohesion = 0.0 (spread model): alpha_eff_max = alpha_max * 0.90 = 0.765 (slightly responsive)

Rationale: a user with a tight, consistent behavioral model (high cohesion) has a
well-defined identity baseline — the trust score should be stable (high inertia).
A user with a spread prototype set (multiple behavioral modes) has a more ambiguous
baseline — the trust score should adapt slightly more dynamically to behavioral signals.

Adaptive EMA Coefficient
------------------------
    alpha_t = clip(alpha_max - gamma * d_short_t, alpha_min, alpha_max)
    gamma = alpha_max - alpha_min

When short drift is zero (stable): alpha_t = alpha_max = 0.85
  -> Strong temporal smoothing, resistant to single-event noise
When short drift is maximal (d_short=1): alpha_t = alpha_min = 0.30
  -> Fast response, system reacts quickly to behavioral anomaly

Mathematical justification: alpha_t modulates the EMA half-life.
  EMA half-life = -log(2) / log(alpha)
  alpha=0.85: half-life ≈ 4.3 events (slow, smooth)
  alpha=0.30: half-life ≈ 0.76 events (fast, responsive)

This adaptive mechanism prevents two failure modes:
  1. Noisy decisions when behavior is stable (high alpha smooths noise)
  2. Slow response to genuine threats (low alpha reacts quickly to drift)

GAT Augmentation (Layer-3 Integration)
---------------------------------------
When Layer-3 provides a similarity score GAT_t:

    R_t^aug = (1 - kappa) * R_t + kappa * GAT_t,    kappa = 0.25

Rationale:
  - GAT operates on temporal graph structure of the session, providing a
    higher-level behavioural signal than per-event prototype matching
  - kappa = 0.25 keeps GAT as a supplementary signal, not primary
  - If GAT is unavailable (not triggered, network failure): R_t is used as-is
  - This ensures Layer-2 remains the primary decision authority, with GAT
    providing refinement rather than override

Decision Zones
--------------
    SAFE      : T_t > theta_safe  (default: 0.80)
    UNCERTAIN : theta_risk <= T_t <= theta_safe
    RISK      : T_t < theta_risk  (default: 0.40)

The gap between 0.80 and 0.40 is the uncertainty zone — not normal, not
definitively risky. This zone triggers Layer-3 escalation (see below).

Threshold Justification:
  - theta_safe = 0.80: Requires sustained strong behavioral match before
    granting SAFE status. At R_t=0.85 consistently, EMA reaches 0.80 in
    ~5-6 events from neutral (0.5). Impostors with R_t in 0.55-0.70 range
    remain in UNCERTAIN throughout, keeping consecutive_uncertain climbing
    and ensuring repeated GAT escalation every T_RECHECK seconds.
  - theta_risk = 0.40: At this level, trust has been consistently low.
    Multiple consecutive low R_t values drag T below 0.40.
  - These defaults can be adapted per-user based on historical trust distributions.

Escalation Logic — Event-Driven GAT
--------------------------------------
GAT escalation is event-driven and uncertainty-based rather than periodic.
Periodic GAT is suboptimal for two reasons:
  1. WASTE: Stable behavior (high trust, low drift) does not warrant deep
     analysis every N seconds — compute is wasted on unnecessary inference.
  2. LATENCY: Anomalous behavior at second 1 of a fixed interval still
     waits until second N before GAT analysis is triggered.

The event-driven escalation is:

Escalation triggers (ANY of):
  1. decision == "RISK"
     -> Trust has collapsed: immediate deep analysis required
  2. decision == "UNCERTAIN" AND anomaly_indicator > 0.25
     -> Uncertain zone with elevated anomaly: potential threat
     -> Threshold lowered from 0.40: careful impostors reach 0.22-0.28
  3. consecutive_uncertain >= 3
     -> Sustained uncertainty: system cannot resolve without deep analysis

Escalation suppression:
  - Re-check interval T_RECHECK = 15s: prevent GAT from running on every event
    when trust fluctuates near the boundary. Reduced from 30s to improve
    responsiveness to sustained impostor sessions.

This event-driven approach ensures:
  - Low compute when trust is stable (SAFE)
  - Immediate deep analysis when genuinely alarming
  - No unnecessary overhead during normal operation
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

# ── Trust EMA parameters ─────────────────────────────────────────────────────

ALPHA_MAX: float = 0.85      # EMA coefficient at zero drift (slow, smooth)
ALPHA_MIN: float = 0.30      # EMA coefficient at maximum drift (fast, responsive)
_ALPHA_GAMMA: float = ALPHA_MAX - ALPHA_MIN   # Modulation range

# ── Raw signal weights (must satisfy: w_sim + w_stab + w_drift + w_trans = 1.0) ──

W_SIMILARITY: float = 0.40
W_STABILITY: float = 0.20
W_DRIFT: float = 0.30
W_TRANSITION: float = 0.10   # Behavioral Session Fingerprint (Markov transition surprise)

# ── Three-scale composite drift weights (must sum to 1.0) ──────────────────────

W_SHORT_DRIFT: float = 0.50    # Micro-behavioral: single-event anomaly detection
W_MEDIUM_DRIFT: float = 0.30   # Episodic: interaction-mode transition detection
W_LONG_DRIFT: float = 0.20     # Identity baseline: long-term behavioral drift

# ── Cohesion-modulated alpha parameters ───────────────────────────────────────

# alpha_effective_max = ALPHA_MAX * (COHESION_ALPHA_FLOOR + COHESION_ALPHA_RANGE * cohesion)
# cohesion=1.0 -> multiplier=1.00 -> alpha_eff_max = 0.850 (full inertia)
# cohesion=0.0 -> multiplier=0.90 -> alpha_eff_max = 0.765 (slightly more responsive)
COHESION_ALPHA_FLOOR: float = 0.90
COHESION_ALPHA_RANGE: float = 0.10

# ── GAT augmentation ─────────────────────────────────────────────────────────

# Adaptive kappa: κ = KAPPA_BASE + KAPPA_RANGE * (1 - trust_score)
# At trust=1.0 (highly trusted): κ = 0.10  (GAT has minimal influence)
# At trust=0.0 (zero trust):     κ = 0.40  (GAT has maximum influence)
# Rationale: when trust is low, the system is uncertain — GAT's deeper
# structural analysis should carry more weight. When trust is high, the
# lighter Layer-2 signal is already reliable.
KAPPA_BASE: float = 0.10
KAPPA_RANGE: float = 0.30


def _adaptive_kappa(trust_score: float) -> float:
    """Compute adaptive GAT blending coefficient κ ∈ [0.10, 0.40]."""
    return float(np.clip(KAPPA_BASE + KAPPA_RANGE * (1.0 - trust_score), KAPPA_BASE, KAPPA_BASE + KAPPA_RANGE))

# ── Decision zone thresholds ──────────────────────────────────────────────────

THETA_SAFE_DEFAULT: float = 0.80    # Trust > this: SAFE (aligned with architecture doc)
THETA_RISK_DEFAULT: float = 0.40    # Trust < this: RISK

# ── Escalation parameters ─────────────────────────────────────────────────────

ANOMALY_ESCALATION_THRESHOLD: float = 0.25   # anomaly_indicator threshold for escalation
                                              # Lowered from 0.40: careful impostors produce
                                              # anomaly in 0.22-0.28 range; 0.40 was unreachable
N_UNCERTAIN_ESCALATION: int = 3              # consecutive UNCERTAIN events before escalation
T_RECHECK_SECONDS: float = 15.0             # Minimum seconds between GAT invocations
                                             # Reduced from 30s: at ~1 event/s, 30s = 30 events
                                             # of suppression; 15s balances responsiveness vs cost

# Minimum Layer-3 session-window size required before GAT can be invoked.
# Callers (e.g. main.py) enforce this prerequisite before calling GAT.
# Defined here so the threshold is co-located with all other escalation logic
# and readable from tests without importing app.core.constants.
MIN_EVENTS_FOR_GAT_ESCALATION: int = 5


# ── Data classes ──────────────────────────────────────────────────────────────

@dataclass
class TrustState:
    """
    Per-session mutable trust tracking state.

    One instance per active WebSocket session. Mutated in-place by
    TrustEngine.update_trust(). Stored in memory_store.SessionState.
    """
    trust_score: float = 0.5           # Initial neutral trust (no information)
    consecutive_risk: int = 0
    consecutive_uncertain: int = 0
    consecutive_safe: int = 0
    last_gat_time: float = 0.0         # Unix timestamp of last GAT invocation
    event_count: int = 0               # Total events processed in this session


@dataclass
class TrustResult:
    """
    Output of a single trust engine update cycle.

    All numeric fields in [0, 1] except consecutive counts.
    This is the complete Layer-4 output — sent to the mobile client
    alongside Layer-2 metrics.
    """
    trust_score: float                  # T_t in [0,1]: current continuous trust
    raw_trust_signal: float             # R_t in [0,1]: before EMA (diagnostic)
    alpha_t: float                      # Adaptive EMA coefficient used [alpha_min, alpha_max]
    decision: str                       # "SAFE" | "UNCERTAIN" | "RISK"
    escalate_to_layer3: bool            # Whether to invoke GAT this event
    consecutive_risk: int               # Count of consecutive RISK decisions
    consecutive_uncertain: int          # Count of consecutive UNCERTAIN decisions
    anomaly_indicator: float            # From Layer-2, forwarded for logging
    gat_augmented: bool                 # Whether GAT score was incorporated this cycle


# ── Trust Engine ──────────────────────────────────────────────────────────────

class TrustEngine:
    """
    Stateless Layer-4 trust computation engine.

    The engine itself holds no session state — all state is passed via
    TrustState. This design:
      - Allows a single TrustEngine instance to serve all sessions
      - Makes the engine fully testable without session setup
      - Prevents state leakage between sessions

    Usage:
        engine = TrustEngine()
        state = TrustState()
        result = engine.update_trust(state, sim, stab, d_short, d_long, anomaly)
    """

    def __init__(
        self,
        theta_safe: float = THETA_SAFE_DEFAULT,
        theta_risk: float = THETA_RISK_DEFAULT,
    ) -> None:
        if theta_risk >= theta_safe:
            raise ValueError(
                f"theta_risk ({theta_risk}) must be strictly less than "
                f"theta_safe ({theta_safe})"
            )
        self.theta_safe = theta_safe
        self.theta_risk = theta_risk

    def compute_raw_signal(
        self,
        similarity_score: float,
        stability_score: float,
        short_drift: float,
        medium_drift: float,
        long_drift: float,
        transition_surprise: float = 0.0,
    ) -> float:
        """
        Compute raw trust signal R_t from Layer-2 metrics.

            D_t  = 0.50 * d_short + 0.30 * d_medium + 0.20 * d_long
            R_t  = 0.40 * sim + 0.20 * stab + 0.30 * (1 - D_t) + 0.10 * (1 - ts)

        Three-scale composite drift captures micro-behavioral anomalies (short),
        episodic mode transitions (medium), and identity baseline drift (long).
        Transition surprise (ts) captures sequential navigation structure via an
        EMA-updated Markov model.  All inputs expected in [0, 1].
        Output is naturally bounded in [0, 1] (convex combination of [0,1] values).

        Parameters
        ----------
        transition_surprise : float in [0, 1), default 0.0
            When no prior event exists (session start), pass 0.0 — this gives
            the full trust contribution from the fingerprint term, which is
            correct: we cannot assess sequential surprise without prior context.

        Returns: float in [0, 1]
        """
        composite_drift = (
            W_SHORT_DRIFT * short_drift
            + W_MEDIUM_DRIFT * medium_drift
            + W_LONG_DRIFT * long_drift
        )
        raw = (
            W_SIMILARITY * similarity_score
            + W_STABILITY * stability_score
            + W_DRIFT * (1.0 - composite_drift)
            + W_TRANSITION * (1.0 - transition_surprise)
        )
        return max(0.0, min(1.0, float(raw)))

    def compute_adaptive_alpha(
        self,
        short_drift: float,
        prototype_topology_cohesion: float = 1.0,
    ) -> float:
        """
        Compute the adaptive EMA coefficient for this event.

            alpha_eff_max = ALPHA_MAX * (COHESION_ALPHA_FLOOR + COHESION_ALPHA_RANGE * cohesion)
            alpha_t = clip(alpha_eff_max - gamma * d_short, alpha_min, alpha_eff_max)

        Cohesion modulation:
          cohesion=1.0 (tight behavioral model) -> alpha_eff_max = 0.85 (full inertia)
          cohesion=0.0 (spread behavioral model) -> alpha_eff_max = 0.765 (more dynamic)

        Short-drift modulation:
          d_short=0 (stable)  -> alpha = alpha_eff_max (slow EMA, resistant to noise)
          d_short=1 (extreme) -> alpha = ALPHA_MIN=0.30 (fast EMA, rapid response)

        Returns: float in [ALPHA_MIN, alpha_eff_max]
        """
        cohesion = max(0.0, min(1.0, float(prototype_topology_cohesion)))
        alpha_eff_max = ALPHA_MAX * (COHESION_ALPHA_FLOOR + COHESION_ALPHA_RANGE * cohesion)
        alpha = alpha_eff_max - _ALPHA_GAMMA * float(short_drift)
        return max(ALPHA_MIN, min(alpha_eff_max, alpha))

    def update_trust(
        self,
        state: TrustState,
        similarity_score: float,
        stability_score: float,
        short_drift: float,
        medium_drift: float,
        long_drift: float,
        anomaly_indicator: float,
        prototype_topology_cohesion: float = 1.0,
        transition_surprise: float = 0.0,
        gat_similarity: Optional[float] = None,
        current_time: Optional[float] = None,
    ) -> TrustResult:
        """
        Process one event cycle: update trust score, determine decision.

        Args
        ----
        state                      : Mutable per-session trust state (modified in-place)
        similarity_score           : Layer-2 composite similarity [0,1]
        stability_score            : Layer-2 stability score [0,1]
        short_drift                : Layer-2 short-term drift (5-event) [0,1)
        medium_drift               : Layer-2 medium-term drift (20-event) [0,1)
        long_drift                 : Layer-2 long-term drift (running) [0,1)
        anomaly_indicator          : Layer-2 anomaly indicator [0,1]
        prototype_topology_cohesion: Geometric tightness of user's prototype set [0,1]
        transition_surprise        : Layer-2c Markov sequential fingerprint [0,1)
        gat_similarity             : Optional Layer-3 GAT similarity [0,1]
        current_time               : Unix timestamp (defaults to time.time())

        Returns
        -------
        TrustResult: Complete Layer-4 output for this event
        """
        t = current_time if current_time is not None else time.time()
        state.event_count += 1

        # ── 1. Compute raw signal from Layer-2/2c metrics ─────────────────
        raw_signal = self.compute_raw_signal(
            similarity_score, stability_score, short_drift, medium_drift, long_drift,
            transition_surprise,
        )

        # ── 2. Optional GAT augmentation (Layer-3 integration) ────────────
        gat_augmented = False
        if gat_similarity is not None:
            # Adaptive kappa: GAT carries more weight when trust is low.
            # κ = 0.10 + 0.30 * (1 - T_{t-1})
            # High trust → κ≈0.10 (Layer-2 dominant)
            # Low  trust → κ≈0.40 (GAT provides significant refinement)
            kappa = _adaptive_kappa(state.trust_score)
            raw_signal = (1.0 - kappa) * raw_signal + kappa * float(gat_similarity)
            raw_signal = max(0.0, min(1.0, raw_signal))
            gat_augmented = True

        # ── 3. Adaptive EMA coefficient (drift-modulated + cohesion-modulated) ──
        alpha_t = self.compute_adaptive_alpha(short_drift, prototype_topology_cohesion)

        # ── 4. EMA trust update ───────────────────────────────────────────
        # T_t = alpha_t * T_{t-1} + (1 - alpha_t) * R_t
        # Convex combination: T_t stays in [0,1] when both operands are in [0,1]
        new_trust = alpha_t * state.trust_score + (1.0 - alpha_t) * raw_signal
        new_trust = max(0.0, min(1.0, float(new_trust)))
        state.trust_score = new_trust

        # ── 5. Decision zone classification ───────────────────────────────
        if new_trust > self.theta_safe:
            decision = "SAFE"
            state.consecutive_risk = 0
            state.consecutive_uncertain = 0
            state.consecutive_safe += 1
        elif new_trust < self.theta_risk:
            decision = "RISK"
            state.consecutive_risk += 1
            state.consecutive_uncertain = 0
            state.consecutive_safe = 0
        else:
            decision = "UNCERTAIN"
            state.consecutive_uncertain += 1
            state.consecutive_risk = 0
            state.consecutive_safe = 0

        # ── 6. Escalation decision ────────────────────────────────────────
        escalate = self._should_escalate(state, decision, anomaly_indicator, t)
        if escalate:
            state.last_gat_time = t   # Record escalation time to enforce re-check interval

        return TrustResult(
            trust_score=new_trust,
            raw_trust_signal=raw_signal,
            alpha_t=alpha_t,
            decision=decision,
            escalate_to_layer3=escalate,
            consecutive_risk=state.consecutive_risk,
            consecutive_uncertain=state.consecutive_uncertain,
            anomaly_indicator=anomaly_indicator,
            gat_augmented=gat_augmented,
        )

    def _should_escalate(
        self,
        state: TrustState,
        decision: str,
        anomaly_indicator: float,
        current_time: float,
    ) -> bool:
        """
        Determine if Layer-3 (GAT) should be invoked for this event.

        Escalation triggers (any one is sufficient):
          1. RISK zone: trust has collapsed — immediate deep analysis
          2. UNCERTAIN + high anomaly: potentially malicious, not just uncertain
          3. Sustained uncertainty: >= N_UNCERTAIN_ESCALATION consecutive events

        Escalation suppression:
          - Re-check interval: if last GAT call was < T_RECHECK_SECONDS ago,
            suppress escalation. This prevents GAT from being called on every
            uncertain event when trust fluctuates near the boundary.

        Returns: bool — True if Layer-3 should be invoked this cycle
        """
        # Enforce minimum re-check interval
        time_since_last = current_time - state.last_gat_time
        if time_since_last < T_RECHECK_SECONDS:
            return False

        # Trigger 1: RISK zone
        if decision == "RISK":
            return True

        # Trigger 2: UNCERTAIN zone with elevated anomaly signal
        if decision == "UNCERTAIN" and anomaly_indicator > ANOMALY_ESCALATION_THRESHOLD:
            return True

        # Trigger 3: Sustained uncertainty exceeds threshold
        if state.consecutive_uncertain >= N_UNCERTAIN_ESCALATION:
            return True

        return False


# ── Module-level singleton (stateless engine, one instance serves all sessions) ──
trust_engine: TrustEngine = TrustEngine()
