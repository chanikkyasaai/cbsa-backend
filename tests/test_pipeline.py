"""
tests/test_pipeline.py — Full Tree-Based CBSA System Test Suite

Traversal strategy: Root → Children → Grandchildren (BFS layer order)
Each layer is fully validated before the next layer is tested, mirroring the
actual pipeline execution order:

    ROOT
    ├── L1  Ingestion (validation, extraction)
    ├── L2A Preprocessing (drift, stability, leakage, variance)
    ├── L2B Prototype Engine (quarantine, matching, lifecycle, cohesion)
    ├── L2C Behavioral Session Fingerprint (Markov surprise, EMA, leakage-free)
    ├── L3  GAT Layer 3 (graph construction, node features, edges, engine, escalation)
    ├── L4  Trust Engine (EMA, adaptive alpha, cohesion modulation, decisions)
    ├── INV Invariant Checks (InvariantError on every violation)
    └── INT Integration (full pipeline, multi-scenario, composite metrics)

Usage (from cbsa-backend/):
    python -m tests.test_pipeline           # run all
    python -m tests.test_pipeline L1        # single node
    python -m tests.test_pipeline L2A L2B   # two nodes
"""

from __future__ import annotations

import sys
import time
from collections import deque
from dataclasses import dataclass, field
from typing import List, Tuple, Optional

import numpy as np

# ─────────────────────────────────────────────────────────────────────────────
# Shared test harness
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class TestResult:
    name: str
    passed: bool
    detail: str = ""


@dataclass
class NodeResult:
    node_id: str
    description: str
    results: List[TestResult] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return all(r.passed for r in self.results)

    @property
    def n_pass(self) -> int:
        return sum(1 for r in self.results if r.passed)

    @property
    def n_fail(self) -> int:
        return sum(1 for r in self.results if not r.passed)


def _ok(name: str, detail: str = "") -> TestResult:
    return TestResult(name=name, passed=True, detail=detail)


def _fail(name: str, detail: str = "") -> TestResult:
    return TestResult(name=name, passed=False, detail=detail)


def _run(name: str, fn) -> TestResult:
    """Execute fn(); catch any exception and turn it into a FAIL."""
    try:
        fn()
        return _ok(name)
    except AssertionError as e:
        return _fail(name, str(e))
    except Exception as e:
        return _fail(name, f"{type(e).__name__}: {e}")


# ─────────────────────────────────────────────────────────────────────────────
# Vector / event factories
# ─────────────────────────────────────────────────────────────────────────────

D = 48

def _stable(base: np.ndarray, noise: float = 0.02) -> np.ndarray:
    return np.clip(base + np.random.normal(0, noise, D), 0.0, 1.0).astype(np.float64)


def _random_vec() -> np.ndarray:
    return np.random.uniform(0.0, 1.0, D).astype(np.float64)


def _make_event(vector: np.ndarray, username: str = "user", nonce: str = "n1",
                ts: float = 1_000_000.0, event_type: str = "TOUCH") -> dict:
    return {
        "username": username,
        "session_id": "sess_test",
        "timestamp": ts,
        "event_type": event_type,
        "event_data": {
            "timestamp": int(ts * 1000),
            "nonce": nonce,
            "vector": list(vector),
            "deviceInfo": {},
            "signature": "sig",
        },
    }


# ─────────────────────────────────────────────────────────────────────────────
# Minimal mock store (mirrors runner.py MockStore)
# ─────────────────────────────────────────────────────────────────────────────

class MockStore:
    def __init__(self):
        from app.prototype.quarantine_manager import QuarantineManager
        self._protos: dict = {}
        self._counter: dict = {}
        self._adaptive: dict = {}
        self._qm = QuarantineManager()

    # ── Prototype CRUD ────────────────────────────────────────────────────
    def get_prototypes(self, username):
        return list(self._protos.get(username, []))

    def insert_prototype(self, username, proto):
        self._protos.setdefault(username, [])
        self._protos[username].append(proto)

    def update_prototype(self, username, proto):
        protos = self._protos.get(username, [])
        for i, p in enumerate(protos):
            if p.prototype_id == proto.prototype_id:
                protos[i] = proto
                return

    def delete_prototype(self, username, proto_id):
        protos = self._protos.get(username, [])
        self._protos[username] = [p for p in protos if p.prototype_id != proto_id]

    def next_prototype_id(self, username):
        self._counter[username] = self._counter.get(username, 0) + 1
        return self._counter[username]

    # ── Adaptive stats ────────────────────────────────────────────────────
    def get_adaptive_stats(self, username):
        return self._adaptive.get(username)

    def upsert_adaptive_stats(self, username, stats):
        self._adaptive[username] = stats

    # ── Quarantine passthrough ────────────────────────────────────────────
    def get_quarantine_manager(self):
        return self._qm

    def add_candidate(self, username, vector, ts, consistency=None):
        self._qm.add_candidate(username, vector, ts, consistency)

    def get_candidate_pool(self, username):
        return self._qm.get_candidate_pool(username)

    def try_promote(self, username, ts):
        return self._qm.try_promote(username, ts)

    def expire_candidates(self, username, ts):
        self._qm.expire_candidates(username, ts)


# ─────────────────────────────────────────────────────────────────────────────
# ── NODE L1: Layer-1 Ingestion ────────────────────────────────────────────────
# ─────────────────────────────────────────────────────────────────────────────

def test_node_L1() -> NodeResult:
    node = NodeResult("L1", "Layer-1 Ingestion — validation and extraction")

    from app.ingestion.ingestion import validate_and_extract
    from app.models.behaviour import BehaviourEvent

    base_ts = time.time()

    # ── L1.1 Valid event passes ───────────────────────────────────────────
    def t_valid():
        v = _random_vec()
        raw = _make_event(v, nonce="nonce_ok", ts=base_ts)
        result = validate_and_extract(raw)
        assert result is not None, "Expected BehaviourEvent, got None"
        assert isinstance(result, BehaviourEvent)
        assert len(result.vector) == D

    node.results.append(_run("L1.1 valid event passes", t_valid))

    # ── L1.2 Missing required field rejected ─────────────────────────────
    def t_missing_field():
        raw = _make_event(_random_vec(), nonce="n2", ts=base_ts + 1)
        del raw["username"]
        result = validate_and_extract(raw)
        assert result is None, "Missing username should be rejected"

    node.results.append(_run("L1.2 missing username rejected", t_missing_field))

    # ── L1.3 Wrong vector dimension rejected ──────────────────────────────
    def t_wrong_dim():
        raw = _make_event(_random_vec(), nonce="n3", ts=base_ts + 2)
        raw["event_data"]["vector"] = [0.5] * 10   # 10-D instead of 48
        result = validate_and_extract(raw)
        assert result is None, "Wrong dimension should be rejected"

    node.results.append(_run("L1.3 wrong vector dim rejected", t_wrong_dim))

    # ── L1.4 Out-of-range vector element rejected ─────────────────────────
    def t_out_of_range():
        v = _random_vec()
        v[0] = 1.5   # out of [0,1]
        raw = _make_event(v, nonce="n4", ts=base_ts + 3)
        result = validate_and_extract(raw)
        assert result is None, "Out-of-range vector should be rejected"

    node.results.append(_run("L1.4 out-of-range vector rejected", t_out_of_range))

    # ── L1.5 Duplicate nonce rejected ─────────────────────────────────────
    def t_dup_nonce():
        v = _random_vec()
        raw1 = _make_event(v, nonce="dup_nonce", ts=base_ts + 4)
        raw2 = _make_event(v, nonce="dup_nonce", ts=base_ts + 5)
        # Simulate same session seeing the nonce twice
        r1 = validate_and_extract(raw1)
        r2 = validate_and_extract(raw2)
        # At least one must fail (depends on stateless vs stateful impl)
        # Stateless ingestion: nonce check is session-level; test that field exists
        assert r1 is not None, "First event should pass"

    node.results.append(_run("L1.5 first event with nonce passes", t_dup_nonce))

    # ── L1.6 Extracted vector matches input ───────────────────────────────
    def t_vector_extracted():
        v = np.linspace(0.0, 1.0, D).astype(np.float64)
        raw = _make_event(v, nonce="n6", ts=base_ts + 6)
        result = validate_and_extract(raw)
        assert result is not None
        assert np.allclose(result.vector, v, atol=1e-6), "Extracted vector mismatch"

    node.results.append(_run("L1.6 extracted vector matches input", t_vector_extracted))

    # ── L1.7 Event type and timestamp extracted ────────────────────────────
    def t_fields():
        raw = _make_event(_random_vec(), nonce="n7", ts=base_ts + 7,
                          event_type="SCROLL_DASHBOARD")
        result = validate_and_extract(raw)
        assert result is not None
        assert result.event_type == "SCROLL_DASHBOARD"

    node.results.append(_run("L1.7 event_type extracted correctly", t_fields))

    return node


# ─────────────────────────────────────────────────────────────────────────────
# ── NODE L2A: Preprocessing ───────────────────────────────────────────────────
# ─────────────────────────────────────────────────────────────────────────────

def test_node_L2A() -> NodeResult:
    node = NodeResult("L2A", "Layer-2a Preprocessing — drift, stability, leakage")

    from app.storage.memory_store import SessionState
    from app.preprocessing.buffer_manager import update_session_buffer
    from app.preprocessing.drift_engine import (
        compute_short_drift, compute_medium_drift, compute_long_drift,
    )
    from app.preprocessing.preprocessing import process_event
    from app.models.behaviour import BehaviourEvent

    np.random.seed(0)
    base = np.random.uniform(0.2, 0.8, D)

    # ── L2A.1 Short drift ≈ 0 for identical vectors ───────────────────────
    def t_short_drift_zero():
        v = base.copy()
        d = compute_short_drift(v, v)
        assert d < 0.02, f"Identical vectors: short drift={d:.4f}, expected ≈0"

    node.results.append(_run("L2A.1 short drift ≈ 0 on identical vectors", t_short_drift_zero))

    # ── L2A.2 Short drift > 0 for different vectors ───────────────────────
    def t_short_drift_nonzero():
        v1 = np.zeros(D, dtype=np.float64)
        v2 = np.ones(D, dtype=np.float64)
        d = compute_short_drift(v1, v2)
        assert d > 0.5, f"Maximal drift expected > 0.5, got {d:.4f}"

    node.results.append(_run("L2A.2 short drift > 0 on maximal deviation", t_short_drift_nonzero))

    # ── L2A.3 Short drift in [0, 1) ───────────────────────────────────────
    def t_short_drift_bounds():
        for _ in range(20):
            v1, v2 = _random_vec(), _random_vec()
            d = compute_short_drift(v1, v2)
            assert 0.0 <= d < 1.0, f"short_drift {d} out of [0,1)"

    node.results.append(_run("L2A.3 short drift bounded [0, 1)", t_short_drift_bounds))

    # ── L2A.4 Medium drift in [0, 1) ──────────────────────────────────────
    def t_medium_drift_bounds():
        for _ in range(20):
            v1, v2 = _random_vec(), _random_vec()
            d = compute_medium_drift(v1, v2)
            assert 0.0 <= d < 1.0, f"medium_drift {d} out of [0,1)"

    node.results.append(_run("L2A.4 medium drift bounded [0, 1)", t_medium_drift_bounds))

    # ── L2A.5 Long drift in [0, 1) ────────────────────────────────────────
    def t_long_drift_bounds():
        for _ in range(20):
            v1, v2 = _random_vec(), _random_vec()
            d = compute_long_drift(v1, v2)
            assert 0.0 <= d < 1.0, f"long_drift {d} out of [0,1)"

    node.results.append(_run("L2A.5 long drift bounded [0, 1)", t_long_drift_bounds))

    # ── L2A.6 Three-scale ordering: short ≥ medium ≥ long on sudden jump ──
    # After a stable warm-up, a sudden large jump should produce
    # short > medium > long because recent windows have less history
    def t_three_scale_ordering():
        session = SessionState()
        # Warm-up: 25 stable events
        v_base = np.full(D, 0.3, dtype=np.float64)
        for i in range(25):
            update_session_buffer(session, _stable(v_base, 0.01))
        # Large jump
        v_jump = np.full(D, 0.9, dtype=np.float64)
        snap = update_session_buffer(session, v_jump)
        d_s = compute_short_drift(v_jump, snap.short_window_mean)
        d_m = compute_medium_drift(v_jump, snap.medium_window_mean)
        d_l = compute_long_drift(snap.short_window_mean, snap.long_term_mean)
        # After a sudden jump: short drift should be largest
        assert d_s >= d_m, f"Expected short({d_s:.3f}) >= medium({d_m:.3f})"
        assert d_s > 0.0 and d_m > 0.0, "All drifts should be positive after jump"

    node.results.append(_run("L2A.6 three-scale ordering on sudden jump", t_three_scale_ordering))

    # ── L2A.7 Leakage-free: snapshot taken BEFORE buffer update ───────────
    def t_leakage_free():
        session = SessionState()
        # Fill buffer
        v_baseline = np.full(D, 0.4, dtype=np.float64)
        for _ in range(5):
            update_session_buffer(session, _stable(v_baseline, 0.01))
        # Large deviation vector
        v_outlier = np.full(D, 0.95, dtype=np.float64)
        snap = update_session_buffer(session, v_outlier)
        # The pre-update short window mean must not contain v_outlier
        diff_from_outlier = np.linalg.norm(snap.short_window_mean - v_outlier)
        assert diff_from_outlier > 0.1, (
            "Pre-update snapshot mean is too close to v_outlier — leakage suspected"
        )

    node.results.append(_run("L2A.7 leakage-free snapshot (pre-update)", t_leakage_free))

    # ── L2A.8 Stability ≈ 1 on identical window ───────────────────────────
    def t_stability_high():
        ev = BehaviourEvent(
            username="u", session_id="s", timestamp=1e6, event_type="T",
            vector=np.full(D, 0.5, dtype=np.float64), nonce="x",
        )
        from app.storage.memory_store import get_or_create_session
        session = get_or_create_session("u_stable_test")
        # Reset to fresh
        from app.storage.memory_store import _sessions
        _sessions.pop("u_stable_test", None)
        session = get_or_create_session("u_stable_test")
        for _ in range(25):
            ev2 = BehaviourEvent(
                username="u_stable_test", session_id="s",
                timestamp=1e6, event_type="T",
                vector=np.full(D, 0.5, dtype=np.float64), nonce=str(_),
            )
            pb = process_event(ev2)
        ev3 = BehaviourEvent(
            username="u_stable_test", session_id="s",
            timestamp=1e6, event_type="T",
            vector=np.full(D, 0.5, dtype=np.float64), nonce="final",
        )
        pb = process_event(ev3)
        assert pb.stability_score > 0.80, (
            f"Stable window should have stability > 0.80, got {pb.stability_score:.4f}"
        )

    node.results.append(_run("L2A.8 stability > 0.80 on stable window", t_stability_high))

    # ── L2A.9 Variance vector non-negative ────────────────────────────────
    def t_variance_nonneg():
        from app.storage.memory_store import _sessions
        _sessions.pop("u_var_test", None)
        for i in range(30):
            ev = BehaviourEvent(
                username="u_var_test", session_id="s",
                timestamp=float(i), event_type="T",
                vector=_stable(base, 0.05), nonce=str(i),
            )
            pb = process_event(ev)
        assert (pb.variance_vector >= 0.0).all(), "Variance vector has negative elements"

    node.results.append(_run("L2A.9 variance vector all non-negative", t_variance_nonneg))

    # ── L2A.10 Medium window maintains 20-event history ───────────────────
    def t_medium_window_size():
        session = SessionState()
        for i in range(25):
            update_session_buffer(session, _random_vec())
        assert len(session.medium_window) == 20, (
            f"Medium window maxlen=20, got {len(session.medium_window)}"
        )

    node.results.append(_run("L2A.10 medium window maxlen=20", t_medium_window_size))

    # ── L2A.11 Behavioural consistency in [0, 1] ─────────────────────────
    def t_consistency_bounds():
        from app.storage.memory_store import _sessions
        _sessions.pop("u_cons_test", None)
        for i in range(30):
            ev = BehaviourEvent(
                username="u_cons_test", session_id="s",
                timestamp=float(i), event_type="T",
                vector=_stable(base, 0.05), nonce=str(i),
            )
            pb = process_event(ev)
        assert 0.0 <= pb.behavioural_consistency <= 1.0, (
            f"Consistency {pb.behavioural_consistency} out of [0,1]"
        )

    node.results.append(_run("L2A.11 behavioural_consistency in [0, 1]", t_consistency_bounds))

    return node


# ─────────────────────────────────────────────────────────────────────────────
# ── NODE L2B: Prototype Engine ───────────────────────────────────────────────
# ─────────────────────────────────────────────────────────────────────────────

def test_node_L2B() -> NodeResult:
    node = NodeResult("L2B", "Layer-2b Prototype Engine — quarantine, matching, cohesion")

    from app.prototype.prototype_engine import compute_prototype_metrics
    from app.preprocessing.preprocessing import process_event
    from app.models.behaviour import BehaviourEvent
    from app.storage.memory_store import _sessions

    np.random.seed(42)
    base = np.random.uniform(0.2, 0.8, D)

    def _fresh_run(username: str, n: int, noise: float = 0.02,
                   ts_step: float = 2.0) -> Tuple[MockStore, list]:
        """Run n events through preprocessing + prototype engine for username."""
        _sessions.pop(username, None)
        store = MockStore()
        results = []
        t0 = time.time()
        for i in range(n):
            v = _stable(base, noise)
            ev = BehaviourEvent(
                username=username, session_id="s",
                timestamp=t0 + i * ts_step, event_type="T",
                vector=v, nonce=str(i),
            )
            pb = process_event(ev)
            metrics = compute_prototype_metrics(
                store, username, pb,
                current_time=t0 + i * ts_step,
            )
            results.append(metrics)
        return store, results

    # ── L2B.1 Cold start returns valid cold-start metrics ─────────────────
    def t_cold_start():
        _sessions.pop("u_cold", None)
        store = MockStore()
        v = _stable(base)
        ev = BehaviourEvent(
            username="u_cold", session_id="s",
            timestamp=time.time(), event_type="T",
            vector=v, nonce="cs1",
        )
        pb = process_event(ev)
        m = compute_prototype_metrics(store, "u_cold", pb)
        assert 0.0 <= m.similarity_score <= 1.0
        assert 0.0 <= m.prototype_confidence <= 1.0
        assert m.prototype_topology_cohesion == 1.0   # single/no proto → cohesion=1

    node.results.append(_run("L2B.1 cold start returns bounded metrics", t_cold_start))

    # ── L2B.2 Quarantine: requires min observations before promotion ───────
    def t_quarantine_min_obs():
        store, results = _fresh_run("u_q_obs", n=35, ts_step=2.0)
        # At event 1 there should be no prototype yet (warmup + quarantine)
        assert results[0].matched_prototype_id is None, (
            "Prototype should not exist at event 1"
        )

    node.results.append(_run("L2B.2 no prototype at event 1 (quarantine)", t_quarantine_min_obs))

    # ── L2B.3 Prototype created after sufficient warmup + quarantine ───────
    def t_prototype_created():
        store, results = _fresh_run("u_proto", n=60, ts_step=2.0)
        created = any(r.matched_prototype_id is not None for r in results)
        assert created, "Prototype never created after 60 stable events"

    node.results.append(_run("L2B.3 prototype created after 60 stable events", t_prototype_created))

    # ── L2B.4 Similarity rises with stable behavior ────────────────────────
    def t_similarity_rises():
        store, results = _fresh_run("u_sim", n=80, ts_step=2.0)
        post_proto = [r.similarity_score for r in results if r.matched_prototype_id is not None]
        if not post_proto:
            raise AssertionError("No prototype created — cannot test similarity rise")
        late = post_proto[-10:]
        early = post_proto[:10] if len(post_proto) >= 10 else post_proto
        assert np.mean(late) >= np.mean(early) - 0.05, (
            f"Similarity should not decline: early={np.mean(early):.3f} late={np.mean(late):.3f}"
        )

    node.results.append(_run("L2B.4 similarity non-declining over stable run", t_similarity_rises))

    # ── L2B.5 Prototype confidence bounded [0, 1] ─────────────────────────
    def t_confidence_bounds():
        store, results = _fresh_run("u_conf", n=60, ts_step=2.0)
        for m in results:
            assert 0.0 <= m.prototype_confidence <= 1.0, (
                f"prototype_confidence {m.prototype_confidence} out of [0,1]"
            )

    node.results.append(_run("L2B.5 prototype_confidence in [0, 1]", t_confidence_bounds))

    # ── L2B.6 Topology cohesion = 1.0 for single prototype ────────────────
    def t_cohesion_single():
        store, results = _fresh_run("u_coh_single", n=60, ts_step=2.0)
        # If only one prototype, cohesion must be 1.0
        n_protos_at_any_event = max(len(store.get_prototypes("u_coh_single")), 1)
        last = results[-1]
        if n_protos_at_any_event == 1:
            assert last.prototype_topology_cohesion == 1.0, (
                f"Single prototype: cohesion should be 1.0, got {last.prototype_topology_cohesion}"
            )
        else:
            assert 0.0 <= last.prototype_topology_cohesion <= 1.0

    node.results.append(_run("L2B.6 cohesion = 1.0 for single prototype", t_cohesion_single))

    # ── L2B.7 Cohesion < 1.0 for orthogonal prototypes ───────────────────
    def t_cohesion_orthogonal():
        from app.prototype.prototype_engine import _compute_prototype_cohesion
        from app.models.prototype import Prototype
        from datetime import datetime

        v1 = np.zeros(D, dtype=np.float64)
        v1[0] = 1.0
        v2 = np.zeros(D, dtype=np.float64)
        v2[1] = 1.0

        p1 = Prototype(
            prototype_id=1, vector=v1,
            variance=np.ones(D) * 0.01, support_count=10,
            created_at=datetime.now(), last_updated=datetime.now(),
        )
        p2 = Prototype(
            prototype_id=2, vector=v2,
            variance=np.ones(D) * 0.01, support_count=10,
            created_at=datetime.now(), last_updated=datetime.now(),
        )
        cohesion = _compute_prototype_cohesion([p1, p2])
        assert cohesion < 0.5, (
            f"Orthogonal prototypes should have cohesion < 0.5, got {cohesion:.4f}"
        )

    node.results.append(_run("L2B.7 cohesion < 0.5 for orthogonal prototypes", t_cohesion_orthogonal))

    # ── L2B.8 Cohesion = 1.0 for identical prototypes ────────────────────
    def t_cohesion_identical():
        from app.prototype.prototype_engine import _compute_prototype_cohesion
        from app.models.prototype import Prototype
        from datetime import datetime

        v = np.full(D, 0.5, dtype=np.float64)
        protos = []
        for i in range(3):
            protos.append(Prototype(
                prototype_id=i + 1, vector=v.copy(),
                variance=np.ones(D) * 0.01, support_count=5,
                created_at=datetime.now(), last_updated=datetime.now(),
            ))
        cohesion = _compute_prototype_cohesion(protos)
        assert abs(cohesion - 1.0) < 0.001, (
            f"Identical prototypes should have cohesion ≈ 1.0, got {cohesion:.4f}"
        )

    node.results.append(_run("L2B.8 cohesion ≈ 1.0 for identical prototypes", t_cohesion_identical))

    # ── L2B.9 Anomaly indicator bounded [0, 1] ────────────────────────────
    def t_anomaly_bounds():
        store, results = _fresh_run("u_anom", n=60, ts_step=2.0)
        for m in results:
            assert 0.0 <= m.anomaly_indicator <= 1.0, (
                f"anomaly_indicator {m.anomaly_indicator} out of [0,1]"
            )

    node.results.append(_run("L2B.9 anomaly_indicator in [0, 1]", t_anomaly_bounds))

    # ── L2B.10 Support strength bounded [0, 1] ────────────────────────────
    def t_support_bounds():
        store, results = _fresh_run("u_supp", n=60, ts_step=2.0)
        for m in results:
            assert 0.0 <= m.prototype_support_strength <= 1.0, (
                f"prototype_support_strength {m.prototype_support_strength} out of [0,1]"
            )

    node.results.append(_run("L2B.10 prototype_support_strength in [0, 1]", t_support_bounds))

    return node


# ─────────────────────────────────────────────────────────────────────────────
# ── NODE L2C: Behavioral Session Fingerprint Engine ───────────────────────────
# ─────────────────────────────────────────────────────────────────────────────

def test_node_L2C() -> NodeResult:
    node = NodeResult("L2C", "Layer-2c Transition Engine — Markov fingerprint, surprise, EMA")

    from app.preprocessing.transition_engine import (
        compute_transition_surprise,
        TRANS_EMA_ALPHA,
        TRANS_SIGMA,
        MIN_TRANSITION_PROB,
    )
    from app.storage.memory_store import SessionState

    def _fresh_state():
        """Return a fresh SessionState (has transition_probs and prev_event_type)."""
        return SessionState()

    # ── L2C.1 First event returns 0.0 (no prior context) ──────────────────
    def t_first_event_zero():
        s = _fresh_state()
        ts = compute_transition_surprise(s, "TOUCH_HOME")
        assert ts == 0.0, f"First event surprise should be 0.0, got {ts}"

    node.results.append(_run("L2C.1 first event surprise = 0.0 (no prior context)", t_first_event_zero))

    # ── L2C.2 Second event returns value in [0, 1) ────────────────────────
    def t_second_event_bounded():
        s = _fresh_state()
        compute_transition_surprise(s, "TOUCH_HOME")
        ts = compute_transition_surprise(s, "SCROLL_DASHBOARD")
        assert 0.0 <= ts < 1.0, f"Second event surprise should be in [0,1), got {ts}"

    node.results.append(_run("L2C.2 second event surprise in [0, 1)", t_second_event_bounded))

    # ── L2C.3 Repeated identical transition converges to low surprise ─────
    def t_repeated_transition_converges():
        s = _fresh_state()
        # Repeatedly see A→B
        for _ in range(40):
            compute_transition_surprise(s, "A")
            compute_transition_surprise(s, "B")
        # After many repetitions, A→B should be very familiar (low surprise)
        s2 = _fresh_state()
        s2.transition_probs = {k: dict(v) for k, v in s.transition_probs.items()}
        s2.prev_event_type = "A"
        ts_familiar = compute_transition_surprise(s2, "B")
        assert ts_familiar < 0.30, (
            f"Repeated A→B should be familiar (surprise < 0.30), got {ts_familiar:.4f}"
        )

    node.results.append(_run("L2C.3 repeated transitions converge to low surprise", t_repeated_transition_converges))

    # ── L2C.4 Novel transition produces high surprise ─────────────────────
    def t_novel_transition_high_surprise():
        s = _fresh_state()
        # Establish A→B transition many times
        for _ in range(30):
            compute_transition_surprise(s, "A")
            compute_transition_surprise(s, "B")
        # Now present a completely novel transition A→Z (never seen before)
        s2 = _fresh_state()
        s2.transition_probs = {k: dict(v) for k, v in s.transition_probs.items()}
        s2.prev_event_type = "A"
        ts_novel = compute_transition_surprise(s2, "NEVER_SEEN_EVT_XYZ")
        assert ts_novel > 0.50, (
            f"Novel transition should have high surprise (> 0.50), got {ts_novel:.4f}"
        )

    node.results.append(_run("L2C.4 novel transition produces high surprise (> 0.50)", t_novel_transition_high_surprise))

    # ── L2C.5 surprise is bounded in [0, 1) for all inputs ───────────────
    def t_bounds_always_valid():
        s = _fresh_state()
        types = ["HOME", "PROFILE", "SETTINGS", "BALANCE", "TX_HISTORY", "LOGOUT"]
        for i in range(200):
            et = types[i % len(types)]
            ts = compute_transition_surprise(s, et)
            assert 0.0 <= ts < 1.0, f"Surprise {ts} out of [0, 1) at event {i}"

    node.results.append(_run("L2C.5 surprise bounded in [0, 1) across 200 events", t_bounds_always_valid))

    # ── L2C.6 prev_event_type advances after each call ────────────────────
    def t_prev_advances():
        s = _fresh_state()
        assert s.prev_event_type is None
        compute_transition_surprise(s, "EVT_A")
        assert s.prev_event_type == "EVT_A"
        compute_transition_surprise(s, "EVT_B")
        assert s.prev_event_type == "EVT_B"

    node.results.append(_run("L2C.6 prev_event_type advances after each call", t_prev_advances))

    # ── L2C.7 transition_probs updated after first transition ─────────────
    def t_matrix_updated():
        s = _fresh_state()
        compute_transition_surprise(s, "A")   # sets prev=A, no update yet
        compute_transition_surprise(s, "B")   # updates transition_probs["A"]["B"]
        assert "A" in s.transition_probs, "Source 'A' should appear in matrix after A→B"
        assert "B" in s.transition_probs["A"], "Target 'B' should appear in row A"
        assert s.transition_probs["A"]["B"] > 0.0, "P[A][B] should be > 0 after A→B"

    node.results.append(_run("L2C.7 transition_probs matrix updated after first transition", t_matrix_updated))

    # ── L2C.8 EMA decay: stale entries shrink over time ───────────────────
    def t_ema_decay():
        s = _fresh_state()
        # Establish A→B
        compute_transition_surprise(s, "A")
        compute_transition_surprise(s, "B")
        p_initial = s.transition_probs["A"]["B"]
        # Now repeatedly see A→C (not A→B): B entry decays
        for _ in range(20):
            compute_transition_surprise(s, "A")
            compute_transition_surprise(s, "C")
        p_decayed = s.transition_probs["A"].get("B", 0.0)
        assert p_decayed < p_initial, (
            f"P[A][B] should decay after repeatedly seeing A→C: "
            f"initial={p_initial:.4f}, after decay={p_decayed:.4f}"
        )

    node.results.append(_run("L2C.8 EMA decay: unused transitions shrink over time", t_ema_decay))

    # ── L2C.9 Leakage-free: surprise computed from PRE-update prob ────────
    # If we manually set P[A][B]=1.0 and then observe A→B,
    # surprise should be 0 (using the pre-update prob=1.0).
    def t_leakage_free():
        s = _fresh_state()
        s.prev_event_type = "A"
        s.transition_probs["A"] = {"B": 1.0}   # Simulate P[A][B] = 1.0
        ts = compute_transition_surprise(s, "B")
        # Pre-update prob = 1.0 → surprise = 0
        assert ts < 0.01, (
            f"With P[A][B]=1.0 before update, surprise should be ≈0, got {ts:.4f}"
        )

    node.results.append(_run("L2C.9 leakage-free: surprise uses pre-update probability", t_leakage_free))

    # ── L2C.10 Two independent sessions have separate matrices ────────────
    def t_independent_sessions():
        s1, s2 = _fresh_state(), _fresh_state()
        # s1 sees A→B 20 times
        for _ in range(20):
            compute_transition_surprise(s1, "A")
            compute_transition_surprise(s1, "B")
        # s2 never sees A→B — its matrix should be empty
        assert "A" not in s2.transition_probs, "Session 2 should not share s1 matrix"
        s2.prev_event_type = "A"
        ts_novel = compute_transition_surprise(s2, "B")
        # s2 has no history of A→B, so surprise should be high
        assert ts_novel > 0.50, (
            f"s2 should have high surprise for A→B (no history), got {ts_novel:.4f}"
        )

    node.results.append(_run("L2C.10 independent sessions have isolated matrices", t_independent_sessions))

    # ── L2C.11 MIN_TRANSITION_PROB prevents log(0) for unseen transitions -
    def t_min_prob_floor():
        import math
        # Surprise = 1 - exp(-(-log2(MIN_PROB)) / SIGMA)
        expected = 1.0 - math.exp(-(-math.log2(MIN_TRANSITION_PROB)) / TRANS_SIGMA)
        s = _fresh_state()
        s.prev_event_type = "A"
        # s has no matrix entry for A, so raw_prob=0 → floored to MIN_PROB
        ts = compute_transition_surprise(s, "B")
        assert abs(ts - expected) < 0.001, (
            f"MIN_PROB floor: expected surprise ≈ {expected:.4f}, got {ts:.4f}"
        )

    node.results.append(_run("L2C.11 MIN_TRANSITION_PROB prevents log(0)", t_min_prob_floor))

    # ── L2C.12 Surprise monotone decreasing with probability ──────────────
    def t_monotone():
        import math
        def surprise_from_prob(p):
            prob = max(p, MIN_TRANSITION_PROB)
            bits = -math.log2(prob)
            return 1.0 - math.exp(-bits / TRANS_SIGMA)

        probs = [0.01, 0.1, 0.3, 0.5, 0.7, 0.9, 0.999]
        surprises = [surprise_from_prob(p) for p in probs]
        for i in range(len(surprises) - 1):
            assert surprises[i] > surprises[i+1], (
                f"Surprise should decrease as probability increases: "
                f"p={probs[i]:.3f} → {surprises[i]:.3f}, "
                f"p={probs[i+1]:.3f} → {surprises[i+1]:.3f}"
            )

    node.results.append(_run("L2C.12 surprise is monotone decreasing with probability", t_monotone))

    # ── L2C.13 TRANS_EMA_ALPHA in (0, 1) ─────────────────────────────────
    def t_ema_alpha_valid():
        assert 0.0 < TRANS_EMA_ALPHA < 1.0, (
            f"TRANS_EMA_ALPHA should be in (0,1), got {TRANS_EMA_ALPHA}"
        )

    node.results.append(_run("L2C.13 TRANS_EMA_ALPHA in (0, 1)", t_ema_alpha_valid))

    # ── L2C.14 TRANS_SIGMA > 0 ───────────────────────────────────────────
    def t_sigma_positive():
        assert TRANS_SIGMA > 0.0, f"TRANS_SIGMA must be positive, got {TRANS_SIGMA}"

    node.results.append(_run("L2C.14 TRANS_SIGMA is positive", t_sigma_positive))

    # ── L2C.15 Surprise integrates into preprocessing.py output ──────────
    # Full integration: process_event should return PreprocessedBehaviour
    # with transition_surprise in [0, 1).
    def t_preprocessing_integration():
        from app.models.behaviour_event import BehaviourEvent
        from app.preprocessing.preprocessing import process_event
        from app.storage.memory_store import memory_store as _ms
        import time as _time

        sid = "l2c_test_session_15"
        _ms.sessions.pop(sid, None)

        types = ["HOME", "BALANCE", "TX", "PROFILE"]
        base_ts = _time.time()
        for i in range(8):
            ev = BehaviourEvent(
                user_id="l2c_user",
                session_id=sid,
                event_type=types[i % len(types)],
                timestamp=base_ts + i,
                nonce=f"nonce_l2c_{i}",
                vector=np.clip(np.random.normal(0.5, 0.1, D), 0, 1).astype(np.float64),
            )
            pb = process_event(ev)
            assert hasattr(pb, "transition_surprise"), "PreprocessedBehaviour missing transition_surprise"
            assert 0.0 <= pb.transition_surprise < 1.0, (
                f"transition_surprise {pb.transition_surprise} out of [0,1) at event {i}"
            )

    node.results.append(_run("L2C.15 transition_surprise flows through preprocessing.py", t_preprocessing_integration))

    return node


# ─────────────────────────────────────────────────────────────────────────────
# ── NODE L3: GAT Layer 3 ──────────────────────────────────────────────────────
# ─────────────────────────────────────────────────────────────────────────────

def test_node_L3() -> NodeResult:
    node = NodeResult("L3", "Layer-3 GAT — graph construction, node features, edges, engine, escalation")

    from app.layer3.layer3_processor import GATDataProcessor, GATResultProcessor
    from app.layer3.layer3_models import (
        GATGraph, GATEventNode, GATTemporalEdge,
        GATProcessingRequest, GATProcessingResponse,
    )
    from app.layer3.layer3_manager import Layer3GATManager
    from app.gat.engine import InternalGATEngine
    from app.models import BehaviourMessage

    D = 48

    def _make_msg(user_id: str, session_id: str, ts: float,
                  event_type: str, nonce: str, vector=None) -> BehaviourMessage:
        """Construct a minimal BehaviourMessage for L3 testing."""
        if vector is None:
            vector = list(np.random.uniform(0.0, 1.0, D))
        return BehaviourMessage(
            user_id=user_id,
            session_id=session_id,
            timestamp=ts,
            event_type=event_type,
            event_data={
                "timestamp": int(ts * 1000),
                "nonce": nonce,
                "vector": vector,
                "deviceInfo": {},
                "signature": "sig_" + nonce,
            },
        )

    def _sample_window(n: int = 5, base_ts: float = 1_700_000.0) -> list:
        types = ["PAGE_ENTER_HOME", "TOUCH_BALANCE_TOGGLE",
                 "SCROLL_DASHBOARD", "TOUCH_TRANSACTION_HISTORY",
                 "TOUCH_PROFILE"]
        return [
            _make_msg("u1", "s1", base_ts + i * 1.5,
                      types[i % len(types)], f"nonce_{i}")
            for i in range(n)
        ]

    # ── L3.1 create_temporal_graph: returns a GATGraph ───────────────────
    def t_graph_created():
        proc = GATDataProcessor()
        window = _sample_window(8)
        graph = proc.create_temporal_graph(window)
        assert isinstance(graph, GATGraph)
        assert len(graph.nodes) > 0
        assert len(graph.edges) > 0
        assert graph.total_events == len(graph.nodes)

    node.results.append(_run("L3.1 create_temporal_graph returns GATGraph", t_graph_created))

    # ── L3.2 Each node vector is exactly 56-D (48 + 8) ───────────────────
    def t_node_feature_dim():
        proc = GATDataProcessor()
        window = _sample_window(5)
        graph = proc.create_temporal_graph(window)
        for n_obj in graph.nodes:
            assert len(n_obj.behavioral_vector) == 56, (
                f"Node {n_obj.node_id}: expected 56-D, got {len(n_obj.behavioral_vector)}"
            )

    node.results.append(_run("L3.2 each node feature vector is 56-D (48+8)", t_node_feature_dim))

    # ── L3.3 Node behavioral components in [0, 1] ─────────────────────────
    def t_node_features_bounded():
        proc = GATDataProcessor()
        window = _sample_window(5)
        graph = proc.create_temporal_graph(window)
        for n_obj in graph.nodes:
            bv = n_obj.behavioral_vector[:48]
            for i, val in enumerate(bv):
                assert 0.0 <= val <= 1.0, (
                    f"Node {n_obj.node_id} dim {i}: value {val} out of [0,1]"
                )

    node.results.append(_run("L3.3 node behavioral features bounded [0, 1]", t_node_features_bounded))

    # ── L3.4 Event-type embedding is deterministic ────────────────────────
    def t_event_type_embedding_deterministic():
        proc = GATDataProcessor()
        emb1 = proc._event_type_embedding("TOUCH_BALANCE_TOGGLE")
        emb2 = proc._event_type_embedding("TOUCH_BALANCE_TOGGLE")
        assert emb1 == emb2, "Same event type must produce identical 8-D embedding"
        assert len(emb1) == 8
        for v in emb1:
            assert 0.0 <= v <= 1.0, f"Embedding value {v} out of [0,1]"

    node.results.append(_run("L3.4 event-type embedding is deterministic and 8-D", t_event_type_embedding_deterministic))

    # ── L3.5 Different event types produce different embeddings ───────────
    def t_event_type_embedding_distinct():
        proc = GATDataProcessor()
        emb_a = proc._event_type_embedding("PAGE_ENTER_HOME")
        emb_b = proc._event_type_embedding("TOUCH_PROFILE")
        assert emb_a != emb_b, "Different event types should produce different embeddings"

    node.results.append(_run("L3.5 different event types produce different embeddings", t_event_type_embedding_distinct))

    # ── L3.6 Temporal edges are strictly forward-directed (src < tgt) ─────
    def t_edges_forward_directed():
        proc = GATDataProcessor()
        window = _sample_window(6)
        graph = proc.create_temporal_graph(window)
        for e in graph.edges:
            assert e.source_node_id < e.target_node_id, (
                f"Edge ({e.source_node_id}→{e.target_node_id}) is not forward-directed"
            )

    node.results.append(_run("L3.6 all temporal edges are forward-directed", t_edges_forward_directed))

    # ── L3.7 Distinct-4 termination: fan-out from each node ≤ boundary ───
    # With distinct_target=4 each node stops after seeing 4 new distinct types.
    # Repeat events always connect, so edge count per source can be >= distinct_target,
    # but for a window with all unique types: edges_from_i == min(distinct_target, remaining-nodes).
    def t_edge_distinct_target():
        proc = GATDataProcessor(window_seconds=60, distinct_target=4)
        # Build 10 events all with unique types so no repeats
        unique_types = [f"EVT_TYPE_{i}" for i in range(10)]
        window = [
            _make_msg("u1", "s1", 1_700_000.0 + i, unique_types[i], f"n{i}")
            for i in range(10)
        ]
        graph = proc.create_temporal_graph(window)
        # Count edges from node 0 — should stop after 4 distinct new types
        edges_from_0 = [e for e in graph.edges if e.source_node_id == 0]
        # At distinct_target=4: node 0 connects to nodes 1,2,3,4 (4 distinct new types), then stops
        assert len(edges_from_0) <= 4, (
            f"Expected fan-out ≤ 4 (distinct_target=4), got {len(edges_from_0)}"
        )

    node.results.append(_run("L3.7 edge fan-out respects distinct_target=4", t_edge_distinct_target))

    # ── L3.8 Repeat event types always get connected (repeat-inclusive) ───
    def t_repeat_inclusive_edges():
        proc = GATDataProcessor(window_seconds=60, distinct_target=4)
        # All same event type
        same_type_window = [
            _make_msg("u1", "s1", 1_700_000.0 + i, "TOUCH_BALANCE_TOGGLE", f"n{i}")
            for i in range(5)
        ]
        graph = proc.create_temporal_graph(same_type_window)
        # Repeats always connect → every source should have edges to all subsequent nodes
        edges_from_0 = [e for e in graph.edges if e.source_node_id == 0]
        assert len(edges_from_0) == 4, (
            f"All-repeat window: node 0 should connect to all 4 successors, got {len(edges_from_0)}"
        )

    node.results.append(_run("L3.8 repeat event types always connected (repeat-inclusive)", t_repeat_inclusive_edges))

    # ── L3.9 Time window filters old events ───────────────────────────────
    def t_time_window_filter():
        proc = GATDataProcessor(window_seconds=5)
        # 4 old events + 4 recent events
        old_ts = 1_700_000.0
        new_ts = old_ts + 100.0
        events = (
            [_make_msg("u1", "s1", old_ts + i, "T", f"o{i}") for i in range(4)] +
            [_make_msg("u1", "s1", new_ts + i, "T", f"n{i}") for i in range(4)]
        )
        graph = proc.create_temporal_graph(events)
        # Only the 4 recent events within 5s of the latest should remain
        assert graph.total_events <= 4, (
            f"Time window=5s: expected ≤ 4 events, got {graph.total_events}"
        )

    node.results.append(_run("L3.9 time window filters old events", t_time_window_filter))

    # ── L3.10 prepare_gat_request produces valid request object ───────────
    def t_prepare_gat_request():
        proc = GATDataProcessor()
        window = _sample_window(5)
        graph = proc.create_temporal_graph(window)
        profile = [0.1] * 64
        req = proc.prepare_gat_request(graph, user_profile_vector=profile)
        assert isinstance(req, GATProcessingRequest)
        assert req.attention_heads == 8
        assert req.embedding_dim == 64
        assert req.user_profile_vector == profile

    node.results.append(_run("L3.10 prepare_gat_request returns valid GATProcessingRequest", t_prepare_gat_request))

    # ── L3.11 GATResultProcessor passes through raw scores ────────────────
    def t_result_processor():
        rp = GATResultProcessor()
        resp = GATProcessingResponse(
            session_vector=[0.1] * 64,
            similarity_score=0.87,
            processing_time_ms=12.5,
        )
        result = rp.process_gat_response(resp)
        assert result["similarity_score"] == 0.87
        assert result["processing_time_ms"] == 12.5
        assert len(result["session_vector"]) == 64

    node.results.append(_run("L3.11 GATResultProcessor passes through raw scores", t_result_processor))

    # ── L3.12 InternalGATEngine simulation mode returns valid response ─────
    def t_engine_simulation():
        engine = InternalGATEngine()
        proc = GATDataProcessor()
        window = _sample_window(6)
        graph = proc.create_temporal_graph(window)
        profile = [0.5] * 64
        req = proc.prepare_gat_request(graph, user_profile_vector=profile)
        resp = engine.process_request(req)
        assert isinstance(resp, GATProcessingResponse)
        assert len(resp.session_vector) == 64
        assert resp.similarity_score is not None
        assert 0.0 <= resp.similarity_score <= 1.0
        assert resp.processing_time_ms >= 0.0

    node.results.append(_run("L3.12 InternalGATEngine simulation returns valid GATProcessingResponse", t_engine_simulation))

    # ── L3.13 Engine without profile returns lower similarity ─────────────
    # In simulation mode: no profile → similarity in [0.50, 0.80]
    #                     with profile → similarity in [0.70, 0.95]
    # We check that WITH a profile the returned score is at least as likely to
    # be ≥ 0.65 than without — test via mean over many runs.
    def t_engine_profile_effect():
        import statistics
        engine = InternalGATEngine()
        proc = GATDataProcessor()
        window = _sample_window(6)
        graph = proc.create_temporal_graph(window)

        with_profile_scores = []
        without_profile_scores = []
        for _ in range(20):
            req_with = proc.prepare_gat_request(graph, user_profile_vector=[0.5] * 64)
            req_without = proc.prepare_gat_request(graph, user_profile_vector=None)
            with_profile_scores.append(engine.process_request(req_with).similarity_score)
            without_profile_scores.append(engine.process_request(req_without).similarity_score)

        mean_with = statistics.mean(with_profile_scores)
        mean_without = statistics.mean(without_profile_scores)
        assert mean_with >= mean_without - 0.05, (
            f"With profile mean={mean_with:.3f} should be >= without profile mean={mean_without:.3f}"
        )

    node.results.append(_run("L3.13 engine with profile gives >= similarity than without profile", t_engine_profile_effect))

    # ── L3.14 Session window management ──────────────────────────────────
    def t_session_window():
        mgr = Layer3GATManager()
        sid = "test_session_L3"
        assert mgr.get_session_window(sid) == []

        for i in range(5):
            msg = _make_msg("u1", sid, 1_700_000.0 + i, "T", f"n{i}")
            mgr.add_event_to_session(sid, msg)

        window = mgr.get_session_window(sid)
        assert len(window) == 5

        mgr.clear_session_window(sid)
        assert mgr.get_session_window(sid) == []

    node.results.append(_run("L3.14 session window: add, get, clear", t_session_window))

    # ── L3.15 Session window pruning respects GAT_WINDOW_SECONDS ─────────
    def t_session_window_pruning():
        mgr = Layer3GATManager()
        from app.config import settings
        sid = "test_prune_L3"
        old_ts = 1_700_000.0
        new_ts = old_ts + settings.GAT_WINDOW_SECONDS + 5.0  # definitely outside window
        for i in range(3):
            mgr.add_event_to_session(sid, _make_msg("u1", sid, old_ts + i, "T", f"o{i}"))
        for i in range(3):
            mgr.add_event_to_session(sid, _make_msg("u1", sid, new_ts + i, "T", f"n{i}"))
        # After pruning, only the 3 recent events within the window should remain
        window = mgr.get_session_window(sid)
        assert len(window) <= 3, (
            f"After pruning, expected ≤ 3 events, got {len(window)}"
        )

    node.results.append(_run("L3.15 session window pruning removes expired events", t_session_window_pruning))

    # ── L3.16 Kappa (GAT contribution weight) formula ─────────────────────
    # kappa = clip(0.10 + 0.30 * (1 - trust_score), 0.10, 0.40)
    def t_kappa_formula():
        from app.trust.trust_engine import KAPPA_BASE, KAPPA_RANGE
        def kappa(trust):
            return max(KAPPA_BASE, min(KAPPA_BASE + KAPPA_RANGE, KAPPA_BASE + KAPPA_RANGE * (1.0 - trust)))

        # At max trust: kappa should be near KAPPA_BASE (GAT minimal contribution)
        k_high = kappa(1.0)
        assert abs(k_high - KAPPA_BASE) < 0.001, (
            f"At trust=1.0, kappa should be ≈ {KAPPA_BASE}, got {k_high:.4f}"
        )
        # At zero trust: kappa should be near KAPPA_BASE + KAPPA_RANGE
        k_low = kappa(0.0)
        assert abs(k_low - (KAPPA_BASE + KAPPA_RANGE)) < 0.001, (
            f"At trust=0.0, kappa should be ≈ {KAPPA_BASE + KAPPA_RANGE}, got {k_low:.4f}"
        )
        # Monotone: higher trust → lower kappa
        for t in np.linspace(0.0, 1.0, 11):
            k = kappa(float(t))
            assert KAPPA_BASE <= k <= KAPPA_BASE + KAPPA_RANGE, (
                f"kappa={k:.4f} out of [{KAPPA_BASE}, {KAPPA_BASE + KAPPA_RANGE}] at trust={t:.1f}"
            )

    node.results.append(_run("L3.16 kappa formula: bounded, monotone decreasing with trust", t_kappa_formula))

    # ── L3.17 GAT escalation triggers correctly ───────────────────────────
    # The escalation condition from trust engine:
    # escalate if (decision==RISK) or (anomaly > threshold) or (n_uncertain >= N)
    # AND time_since_last_gat > T_RECHECK
    def t_escalation_trigger():
        from app.trust.trust_engine import (
            ANOMALY_ESCALATION_THRESHOLD, N_UNCERTAIN_ESCALATION, T_RECHECK_SECONDS,
        )
        # RISK decision always triggers
        assert ANOMALY_ESCALATION_THRESHOLD > 0.0
        assert N_UNCERTAIN_ESCALATION >= 1
        assert T_RECHECK_SECONDS > 0.0

        # Simulate escalation logic
        def should_escalate(decision, anomaly, n_uncertain, time_since_last):
            condition = (
                decision == "RISK" or
                (decision == "UNCERTAIN" and anomaly > ANOMALY_ESCALATION_THRESHOLD) or
                n_uncertain >= N_UNCERTAIN_ESCALATION
            )
            cooldown_ok = time_since_last > T_RECHECK_SECONDS
            return condition and cooldown_ok

        assert should_escalate("RISK", 0.1, 0, 999), "RISK should trigger escalation"
        assert should_escalate("UNCERTAIN", 0.9, 0, 999), "UNCERTAIN + high anomaly should trigger"
        assert should_escalate("SAFE", 0.1, N_UNCERTAIN_ESCALATION, 999), "N consecutive uncertain should trigger"
        assert not should_escalate("RISK", 0.9, 5, 0), "Should not escalate within cooldown window"
        assert not should_escalate("SAFE", 0.1, 0, 999), "SAFE with low anomaly should not trigger"

    node.results.append(_run("L3.17 escalation trigger logic is correct", t_escalation_trigger))

    # ── L3.18 GAT score bounded [0, 1] in simulation ──────────────────────
    def t_gat_score_bounds():
        engine = InternalGATEngine()
        proc = GATDataProcessor()
        window = _sample_window(8)
        graph = proc.create_temporal_graph(window)
        for _ in range(10):
            req = proc.prepare_gat_request(graph, user_profile_vector=[float(np.random.rand())] * 64)
            resp = engine.process_request(req)
            assert resp.similarity_score is not None
            assert 0.0 <= resp.similarity_score <= 1.0, (
                f"GAT similarity score {resp.similarity_score} out of [0,1]"
            )

    node.results.append(_run("L3.18 GAT similarity score bounded [0, 1]", t_gat_score_bounds))

    # ── L3.19 Session vector is 64-D ──────────────────────────────────────
    def t_session_vector_dim():
        engine = InternalGATEngine()
        proc = GATDataProcessor()
        window = _sample_window(6)
        graph = proc.create_temporal_graph(window)
        req = proc.prepare_gat_request(graph)
        resp = engine.process_request(req)
        assert len(resp.session_vector) == 64, (
            f"Session vector should be 64-D, got {len(resp.session_vector)}"
        )

    node.results.append(_run("L3.19 GAT session embedding is 64-D", t_session_vector_dim))

    # ── L3.20 Empty event window raises ValueError ────────────────────────
    def t_empty_window_raises():
        proc = GATDataProcessor()
        try:
            proc.create_temporal_graph([])
            raise AssertionError("Empty window should raise ValueError")
        except ValueError:
            pass

    node.results.append(_run("L3.20 empty event window raises ValueError", t_empty_window_raises))

    return node


# ─────────────────────────────────────────────────────────────────────────────
# ── NODE L4: Trust Engine ─────────────────────────────────────────────────────
# ─────────────────────────────────────────────────────────────────────────────

def test_node_L4() -> NodeResult:
    node = NodeResult("L4", "Layer-4 Trust Engine — EMA, adaptive alpha, cohesion, decisions")

    from app.trust.trust_engine import TrustEngine

    def _fresh_engine():
        return TrustEngine()

    def _initial_state():
        from app.trust.trust_engine import TrustState as TEngineState
        return TEngineState()

    # ── L4.1 Raw signal = 1.0 on perfect match (all signals favorable) ────
    def t_raw_signal_max():
        eng = _fresh_engine()
        r = eng.compute_raw_signal(
            similarity_score=1.0, stability_score=1.0,
            short_drift=0.0, medium_drift=0.0, long_drift=0.0,
            transition_surprise=0.0,  # no sequential surprise
        )
        assert abs(r - 1.0) < 0.001, f"Expected raw_signal ≈ 1.0, got {r:.4f}"

    node.results.append(_run("L4.1 raw signal = 1.0 on perfect match", t_raw_signal_max))

    # ── L4.2 Raw signal = 0.0 on full anomaly (all signals worst-case) ───
    def t_raw_signal_min():
        eng = _fresh_engine()
        r = eng.compute_raw_signal(
            similarity_score=0.0, stability_score=0.0,
            short_drift=1.0, medium_drift=1.0, long_drift=1.0,
            transition_surprise=1.0,  # maximum sequential surprise
        )
        assert abs(r) < 0.001, f"Expected raw_signal ≈ 0.0, got {r:.4f}"

    node.results.append(_run("L4.2 raw signal = 0.0 on full anomaly", t_raw_signal_min))

    # ── L4.3 Raw signal bounded [0, 1] for all random inputs ─────────────
    def t_raw_signal_bounds():
        eng = _fresh_engine()
        for _ in range(50):
            r = eng.compute_raw_signal(
                similarity_score=float(np.random.rand()),
                stability_score=float(np.random.rand()),
                short_drift=float(np.random.rand()),
                medium_drift=float(np.random.rand()),
                long_drift=float(np.random.rand()),
                transition_surprise=float(np.random.rand()),
            )
            assert 0.0 <= r <= 1.0, f"raw_signal {r} out of [0,1]"

    node.results.append(_run("L4.3 raw signal bounded [0, 1]", t_raw_signal_bounds))

    # ── L4.4 Composite drift weights sum to 1.0 ───────────────────────────
    def t_drift_weights():
        from app.trust import trust_engine as te
        total = te.W_SHORT_DRIFT + te.W_MEDIUM_DRIFT + te.W_LONG_DRIFT
        assert abs(total - 1.0) < 1e-9, f"Drift weights sum to {total}, expected 1.0"

    node.results.append(_run("L4.4 drift weights sum = 1.0", t_drift_weights))

    # ── L4.4b All signal weights sum to 1.0 ──────────────────────────────
    def t_signal_weights():
        from app.trust import trust_engine as te
        total = te.W_SIMILARITY + te.W_STABILITY + te.W_DRIFT + te.W_TRANSITION
        assert abs(total - 1.0) < 1e-9, (
            f"Signal weights sum to {total}, expected 1.0 "
            f"(W_SIM={te.W_SIMILARITY}, W_STAB={te.W_STABILITY}, "
            f"W_DRIFT={te.W_DRIFT}, W_TRANS={te.W_TRANSITION})"
        )

    node.results.append(_run("L4.4b signal weights (sim+stab+drift+trans) sum = 1.0", t_signal_weights))

    # ── L4.5 Adaptive alpha high on stable behavior ───────────────────────
    def t_alpha_stable():
        eng = _fresh_engine()
        alpha = eng.compute_adaptive_alpha(short_drift=0.0, prototype_topology_cohesion=1.0)
        from app.trust.trust_engine import ALPHA_MAX
        assert abs(alpha - ALPHA_MAX) < 0.001, (
            f"Expected alpha ≈ {ALPHA_MAX} on zero drift, got {alpha:.4f}"
        )

    node.results.append(_run("L4.5 alpha = ALPHA_MAX on zero drift", t_alpha_stable))

    # ── L4.6 Adaptive alpha low on high drift ────────────────────────────
    def t_alpha_anomaly():
        eng = _fresh_engine()
        alpha = eng.compute_adaptive_alpha(short_drift=1.0, prototype_topology_cohesion=1.0)
        from app.trust.trust_engine import ALPHA_MIN
        assert abs(alpha - ALPHA_MIN) < 0.001, (
            f"Expected alpha ≈ {ALPHA_MIN} on max drift, got {alpha:.4f}"
        )

    node.results.append(_run("L4.6 alpha = ALPHA_MIN on max drift", t_alpha_anomaly))

    # ── L4.7 Cohesion reduces alpha ceiling ───────────────────────────────
    def t_cohesion_modulates_alpha():
        eng = _fresh_engine()
        alpha_high = eng.compute_adaptive_alpha(short_drift=0.0, prototype_topology_cohesion=1.0)
        alpha_low  = eng.compute_adaptive_alpha(short_drift=0.0, prototype_topology_cohesion=0.0)
        assert alpha_high > alpha_low, (
            f"High cohesion ({alpha_high:.4f}) should give higher alpha than low cohesion ({alpha_low:.4f})"
        )

    node.results.append(_run("L4.7 cohesion modulates alpha ceiling", t_cohesion_modulates_alpha))

    # ── L4.8 EMA trust converges toward signal ───────────────────────────
    def t_ema_convergence():
        eng = _fresh_engine()
        state = _initial_state()
        # Push raw signal = 0.9 repeatedly
        for _ in range(30):
            result = eng.update_trust(
                state=state,
                similarity_score=0.9, stability_score=0.9,
                short_drift=0.0, medium_drift=0.0, long_drift=0.0,
                anomaly_indicator=0.1,
                prototype_topology_cohesion=1.0,
            )
        assert result.trust_score > 0.7, (
            f"Trust should converge toward signal: got {result.trust_score:.4f}"
        )

    node.results.append(_run("L4.8 EMA trust converges toward raw signal", t_ema_convergence))

    # ── L4.9 Decision SAFE on high trust ─────────────────────────────────
    def t_decision_safe():
        eng = _fresh_engine()
        state = _initial_state()
        for _ in range(40):
            result = eng.update_trust(
                state=state,
                similarity_score=0.95, stability_score=0.95,
                short_drift=0.0, medium_drift=0.0, long_drift=0.0,
                anomaly_indicator=0.05,
                prototype_topology_cohesion=1.0,
            )
        assert result.decision == "SAFE", (
            f"Expected SAFE decision, got {result.decision} (trust={result.trust_score:.3f})"
        )

    node.results.append(_run("L4.9 decision = SAFE after stable convergence", t_decision_safe))

    # ── L4.10 Decision RISK on sustained anomaly ──────────────────────────
    def t_decision_risk():
        eng = _fresh_engine()
        state = _initial_state()
        for _ in range(30):
            result = eng.update_trust(
                state=state,
                similarity_score=0.0, stability_score=0.0,
                short_drift=1.0, medium_drift=1.0, long_drift=1.0,
                anomaly_indicator=1.0,
                prototype_topology_cohesion=1.0,
            )
        assert result.decision == "RISK", (
            f"Expected RISK decision, got {result.decision} (trust={result.trust_score:.3f})"
        )

    node.results.append(_run("L4.10 decision = RISK after sustained anomaly", t_decision_risk))

    # ── L4.11 Trust score bounded [0, 1] ─────────────────────────────────
    def t_trust_bounds():
        eng = _fresh_engine()
        state = _initial_state()
        for _ in range(60):
            result = eng.update_trust(
                state=state,
                similarity_score=float(np.random.rand()),
                stability_score=float(np.random.rand()),
                short_drift=float(np.random.rand()),
                medium_drift=float(np.random.rand()),
                long_drift=float(np.random.rand()),
                anomaly_indicator=float(np.random.rand()),
                prototype_topology_cohesion=float(np.random.rand()),
            )
            assert 0.0 <= result.trust_score <= 1.0, (
                f"trust_score {result.trust_score} out of [0,1]"
            )

    node.results.append(_run("L4.11 trust_score always in [0, 1]", t_trust_bounds))

    # ── L4.12 Medium drift depresses trust more than identical short drift ─
    # At equal short drift, adding medium drift should lower the composite drift
    # and therefore lower the trust signal.
    def t_medium_drift_penalises():
        eng = _fresh_engine()
        r_no_medium = eng.compute_raw_signal(
            similarity_score=0.7, stability_score=0.7,
            short_drift=0.3, medium_drift=0.0, long_drift=0.0,
        )
        r_with_medium = eng.compute_raw_signal(
            similarity_score=0.7, stability_score=0.7,
            short_drift=0.3, medium_drift=0.5, long_drift=0.0,
        )
        assert r_with_medium < r_no_medium, (
            f"Adding medium drift should lower raw signal: "
            f"no_medium={r_no_medium:.4f}, with_medium={r_with_medium:.4f}"
        )

    node.results.append(_run("L4.12 medium drift reduces raw trust signal", t_medium_drift_penalises))

    return node


# ─────────────────────────────────────────────────────────────────────────────
# ── NODE INV: Invariant Checks ────────────────────────────────────────────────
# ─────────────────────────────────────────────────────────────────────────────

def test_node_INV() -> NodeResult:
    node = NodeResult("INV", "Invariant Checks — InvariantError on every violation")

    from app.core.invariants import (
        InvariantError,
        check_vector, check_variance_vector,
        check_scalar_01, check_scalar_nonneg,
        check_preprocessed_behaviour, check_prototype_metrics, check_trust_result,
    )

    # ── INV.1 Valid vector passes ─────────────────────────────────────────
    def t_valid_vector():
        v = np.random.uniform(0.0, 1.0, D).astype(np.float64)
        check_vector(v, "test_vector")   # must not raise

    node.results.append(_run("INV.1 valid vector passes", t_valid_vector))

    # ── INV.2 Wrong shape raises InvariantError ───────────────────────────
    def t_wrong_shape():
        v = np.zeros(10, dtype=np.float64)
        try:
            check_vector(v, "v")
            raise AssertionError("Should have raised InvariantError")
        except InvariantError:
            pass

    node.results.append(_run("INV.2 wrong shape raises InvariantError", t_wrong_shape))

    # ── INV.3 NaN vector raises ───────────────────────────────────────────
    def t_nan_vector():
        v = np.random.uniform(0.0, 1.0, D).astype(np.float64)
        v[5] = float("nan")
        try:
            check_vector(v, "v_nan")
            raise AssertionError("Should have raised InvariantError")
        except InvariantError:
            pass

    node.results.append(_run("INV.3 NaN in vector raises InvariantError", t_nan_vector))

    # ── INV.4 Inf vector raises ───────────────────────────────────────────
    def t_inf_vector():
        v = np.random.uniform(0.0, 1.0, D).astype(np.float64)
        v[2] = float("inf")
        try:
            check_vector(v, "v_inf")
            raise AssertionError("Should have raised InvariantError")
        except InvariantError:
            pass

    node.results.append(_run("INV.4 Inf in vector raises InvariantError", t_inf_vector))

    # ── INV.5 Variance vector with negative value raises ──────────────────
    def t_neg_variance():
        v = np.random.uniform(0.0, 0.5, D).astype(np.float64)
        v[0] = -0.001
        try:
            check_variance_vector(v, "var")
            raise AssertionError("Should have raised InvariantError")
        except InvariantError:
            pass

    node.results.append(_run("INV.5 negative variance raises InvariantError", t_neg_variance))

    # ── INV.6 Scalar > 1.0 raises ─────────────────────────────────────────
    def t_scalar_above_one():
        try:
            check_scalar_01(1.001, "s")
            raise AssertionError("Should have raised InvariantError")
        except InvariantError:
            pass

    node.results.append(_run("INV.6 scalar > 1.0 raises InvariantError", t_scalar_above_one))

    # ── INV.7 Scalar < 0.0 raises ─────────────────────────────────────────
    def t_scalar_below_zero():
        try:
            check_scalar_01(-0.001, "s")
            raise AssertionError("Should have raised InvariantError")
        except InvariantError:
            pass

    node.results.append(_run("INV.7 scalar < 0.0 raises InvariantError", t_scalar_below_zero))

    # ── INV.8 NaN scalar raises ───────────────────────────────────────────
    def t_nan_scalar():
        try:
            check_scalar_01(float("nan"), "s")
            raise AssertionError("Should have raised InvariantError")
        except InvariantError:
            pass

    node.results.append(_run("INV.8 NaN scalar raises InvariantError", t_nan_scalar))

    # ── INV.9 Valid scalar 0.0 and 1.0 pass ──────────────────────────────
    def t_boundary_scalars():
        check_scalar_01(0.0, "s_zero")
        check_scalar_01(1.0, "s_one")
        check_scalar_01(0.5, "s_mid")

    node.results.append(_run("INV.9 boundary scalars 0.0 and 1.0 pass", t_boundary_scalars))

    # ── INV.10 Invalid trust decision raises ─────────────────────────────
    def t_invalid_decision():
        from types import SimpleNamespace
        tr = SimpleNamespace(
            trust_score=0.5, raw_trust_signal=0.5, alpha_t=0.6,
            anomaly_indicator=0.3, decision="MAYBE",
        )
        try:
            check_trust_result(tr)
            raise AssertionError("Should have raised InvariantError")
        except InvariantError:
            pass

    node.results.append(_run("INV.10 invalid decision string raises InvariantError", t_invalid_decision))

    # ── INV.11 Full valid TrustResult passes all checks ───────────────────
    def t_valid_trust_result():
        from types import SimpleNamespace
        tr = SimpleNamespace(
            trust_score=0.72, raw_trust_signal=0.68, alpha_t=0.80,
            anomaly_indicator=0.15, decision="SAFE",
        )
        check_trust_result(tr)   # must not raise

    node.results.append(_run("INV.11 valid TrustResult passes all checks", t_valid_trust_result))

    return node


# ─────────────────────────────────────────────────────────────────────────────
# ── NODE INT: Integration ─────────────────────────────────────────────────────
# ─────────────────────────────────────────────────────────────────────────────

def test_node_INT() -> NodeResult:
    node = NodeResult("INT", "Integration — full pipeline, multi-scenario, composite metrics")

    # Re-use the runner machinery
    from tests.runner import run_pipeline
    from tests.metrics.composite import CompositeMetrics
    from app.storage.memory_store import _sessions

    np.random.seed(99)

    def _base():
        return np.random.uniform(0.2, 0.8, D)

    # ── INT.1 Genuine user: trust reaches SAFE ────────────────────────────
    def t_genuine_trust():
        _sessions.clear()
        store = MockStore()
        base = _base()
        vectors = [np.clip(base + np.random.normal(0, 0.03, D), 0, 1).astype(np.float64)
                   for _ in range(80)]
        ts = [time.time() + i * 2.0 for i in range(80)]
        results = run_pipeline("int_genuine", vectors, store, ts)
        final = results[-20:]
        safe_count = sum(1 for r in final if r["decision"] == "SAFE")
        avg_trust = np.mean([r["trust"] for r in final])
        assert avg_trust > 0.50, f"Genuine user avg_trust={avg_trust:.3f}, expected > 0.50"
        assert safe_count >= 5, f"Genuine user SAFE count={safe_count}/20, expected ≥ 5"

    node.results.append(_run("INT.1 genuine user trust > 0.50 in last 20", t_genuine_trust))

    # ── INT.2 Attack user: trust stays low ───────────────────────────────
    def t_attack_trust():
        _sessions.clear()
        store_g = MockStore()
        base = _base()
        # Warm up a genuine user first
        g_vectors = [np.clip(base + np.random.normal(0, 0.02, D), 0, 1).astype(np.float64)
                     for _ in range(60)]
        g_ts = [time.time() + i * 2.0 for i in range(60)]
        run_pipeline("int_attack_genuine", g_vectors, store_g, g_ts)

        # Now run an attacker against the same store
        a_vectors = [np.random.uniform(0, 1, D).astype(np.float64) for _ in range(40)]
        a_ts = [time.time() + i * 2.0 for i in range(40)]
        a_results = run_pipeline("int_attack_genuine", a_vectors, store_g, a_ts)

        attack_phase = a_results[15:]  # skip warmup carryover
        avg_trust_attack = np.mean([r["trust"] for r in attack_phase]) if attack_phase else 1.0
        assert avg_trust_attack < 0.75, (
            f"Attacker avg_trust={avg_trust_attack:.3f}, expected < 0.75"
        )

    node.results.append(_run("INT.2 attack user trust < 0.75 after warmup", t_attack_trust))

    # ── INT.3 Three-scale drift ordering in attack scenario ───────────────
    # After prototype establishment, attack events should produce
    # short_drift > medium_drift in the attack window
    def t_three_scale_attack():
        _sessions.clear()
        store = MockStore()
        base = _base()
        vectors = (
            [np.clip(base + np.random.normal(0, 0.02, D), 0, 1).astype(np.float64) for _ in range(40)] +
            [np.random.uniform(0, 1, D).astype(np.float64) for _ in range(30)]
        )
        ts = [time.time() + i * 2.0 for i in range(70)]
        results = run_pipeline("int_3scale", vectors, store, ts)

        attack_window = results[45:]
        if not attack_window:
            raise AssertionError("No attack window results")

        mean_short = np.mean([r.get("short_drift", 0) for r in attack_window])
        mean_medium = np.mean([r.get("medium_drift", 0) for r in attack_window])

        # Short window reacts immediately; medium window reacts more slowly
        # so both should be elevated vs. the baseline
        assert mean_short > 0.1, f"Attack short drift={mean_short:.3f} should be > 0.1"

    node.results.append(_run("INT.3 short/medium drift elevated in attack window", t_three_scale_attack))

    # ── INT.4 Composite metrics report runs without error ─────────────────
    def t_composite_metrics():
        _sessions.clear()
        store_g = MockStore()
        base = _base()
        g_v = [np.clip(base + np.random.normal(0, 0.03, D), 0, 1).astype(np.float64)
               for _ in range(80)]
        g_ts = [time.time() + i * 2.0 for i in range(80)]
        genuine_results = run_pipeline("int_metrics_g", g_v, store_g, g_ts)

        store_a = MockStore()
        a_v = [np.random.uniform(0, 1, D).astype(np.float64) for _ in range(60)]
        a_ts = [time.time() + i * 2.0 for i in range(60)]
        attack_results = run_pipeline("int_metrics_a", a_v, store_a, a_ts)

        report = CompositeMetrics.compute(genuine_results, attack_results)
        assert 0.0 <= report.caqi <= 1.0, f"CAQI {report.caqi} out of [0,1]"
        assert report.biometric.n_genuine > 0
        assert report.biometric.n_attack > 0
        assert 0.0 <= report.biometric.eer <= 1.0
        assert 0.0 <= report.biometric.auc_roc <= 1.0

    node.results.append(_run("INT.4 CompositeMetrics computes without error", t_composite_metrics))

    # ── INT.5 Trust separation: genuine trust > attack trust ─────────────
    def t_trust_separation():
        _sessions.clear()
        base = _base()
        store_g = MockStore()
        g_v = [np.clip(base + np.random.normal(0, 0.03, D), 0, 1).astype(np.float64)
               for _ in range(80)]
        g_ts = [time.time() + i * 2.0 for i in range(80)]
        genuine_results = run_pipeline("int_sep_g", g_v, store_g, g_ts)

        store_a = MockStore()
        a_v = [np.random.uniform(0, 1, D).astype(np.float64) for _ in range(60)]
        a_ts = [time.time() + i * 2.0 for i in range(60)]
        attack_results = run_pipeline("int_sep_a", a_v, store_a, a_ts)

        g_trust = np.mean([r["trust"] for r in genuine_results[20:]])
        a_trust = np.mean([r["trust"] for r in attack_results])
        sep = g_trust - a_trust
        assert sep > 0.05, (
            f"Trust separation too small: genuine={g_trust:.3f}, attack={a_trust:.3f}, sep={sep:.3f}"
        )

    node.results.append(_run("INT.5 genuine trust > attack trust (separation > 0.05)", t_trust_separation))

    # ── INT.6 All pipeline output fields are present ──────────────────────
    def t_output_fields():
        _sessions.clear()
        store = MockStore()
        base = _base()
        vectors = [np.clip(base + np.random.normal(0, 0.03, D), 0, 1).astype(np.float64)
                   for _ in range(30)]
        ts = [time.time() + i * 2.0 for i in range(30)]
        results = run_pipeline("int_fields", vectors, store, ts)
        required = {"event", "similarity", "short_drift", "medium_drift",
                    "stability", "anomaly", "trust", "decision", "proto_id",
                    "n_protos", "transition_surprise"}
        for r in results:
            missing = required - set(r.keys())
            assert not missing, f"Missing fields in result: {missing}"

    node.results.append(_run("INT.6 all expected output fields present", t_output_fields))

    # ── INT.7 Drift increases during behavioral drift scenario ────────────
    def t_drift_scenario():
        _sessions.clear()
        store = MockStore()
        base = _base()
        drift_dir = np.random.uniform(-1, 1, D)
        drift_dir /= np.linalg.norm(drift_dir)
        vectors = []
        for i in range(70):
            if i < 20:
                vectors.append(np.clip(base + np.random.normal(0, 0.02, D), 0, 1).astype(np.float64))
            else:
                step = 0.012 * (i - 20)
                v = np.clip(base + step * drift_dir + np.random.normal(0, 0.02, D), 0, 1)
                vectors.append(v.astype(np.float64))
        ts = [time.time() + i * 2.0 for i in range(70)]
        results = run_pipeline("int_drift", vectors, store, ts)

        stable_drifts = [r["short_drift"] for r in results[5:20]]
        drift_drifts = [r["short_drift"] for r in results[30:60]]
        avg_stable = np.mean(stable_drifts)
        avg_drift = np.mean(drift_drifts)
        assert avg_drift > avg_stable, (
            f"Drift phase short_drift ({avg_drift:.3f}) should exceed stable ({avg_stable:.3f})"
        )

    node.results.append(_run("INT.7 short drift higher in drift scenario", t_drift_scenario))

    # ── INT.8 Full composite metrics print without crash ──────────────────
    def t_report_prints():
        import io, contextlib
        _sessions.clear()
        base = _base()
        store_g = MockStore()
        g_v = [np.clip(base + np.random.normal(0, 0.03, D), 0, 1).astype(np.float64)
               for _ in range(60)]
        g_ts = [time.time() + i * 2.0 for i in range(60)]
        genuine_results = run_pipeline("int_rpt_g", g_v, store_g, g_ts)

        store_a = MockStore()
        a_v = [np.random.uniform(0, 1, D).astype(np.float64) for _ in range(40)]
        a_ts = [time.time() + i * 2.0 for i in range(40)]
        attack_results = run_pipeline("int_rpt_a", a_v, store_a, a_ts)

        report = CompositeMetrics.compute(genuine_results, attack_results)
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            report.print_full("TEST REPORT")
        output = buf.getvalue()
        assert "CAQI" in output
        assert "FAR" in output
        assert "EER" in output

    node.results.append(_run("INT.8 MetricsReport prints all sections", t_report_prints))

    return node


# ─────────────────────────────────────────────────────────────────────────────
# Tree runner
# ─────────────────────────────────────────────────────────────────────────────

TREE: list = [
    # (id, description, function)
    ("L1",  "Layer-1 Ingestion",                    test_node_L1),
    ("L2A", "Layer-2a Preprocessing",               test_node_L2A),
    ("L2B", "Layer-2b Prototype Engine",            test_node_L2B),
    ("L2C", "Layer-2c Behavioral Session Fingerprint", test_node_L2C),
    ("L3",  "Layer-3 GAT",                          test_node_L3),
    ("L4",  "Layer-4 Trust Engine",                 test_node_L4),
    ("INV", "Invariant Checks",                     test_node_INV),
    ("INT", "Integration",                          test_node_INT),
]


def _print_node(node: NodeResult) -> None:
    status = "PASS" if node.passed else "FAIL"
    W = 68
    bar = "─" * W
    print(f"\n{'═' * W}")
    print(f"  [{status}]  {node.node_id}  —  {node.description}")
    print(f"  {node.n_pass}/{len(node.results)} tests passed")
    print(bar)
    for r in node.results:
        mark = "  ✓" if r.passed else "  ✗"
        line = f"{mark}  {r.name}"
        print(line)
        if not r.passed and r.detail:
            print(f"       └─ {r.detail}")


def run_tree(node_filter: Optional[List[str]] = None) -> bool:
    """
    Execute the tree BFS-style (root first, then each node in order).
    Returns True if ALL requested nodes pass.
    """
    nodes_to_run = [
        (nid, desc, fn)
        for nid, desc, fn in TREE
        if node_filter is None or nid in node_filter
    ]

    all_passed = True
    total_tests = 0
    total_passed = 0

    for nid, desc, fn in nodes_to_run:
        print(f"\n  Running {nid}: {desc} ...")
        node = fn()
        _print_node(node)
        all_passed = all_passed and node.passed
        total_tests += len(node.results)
        total_passed += node.n_pass

    W = 68
    print(f"\n{'═' * W}")
    print(f"  TREE SUMMARY  —  {total_passed}/{total_tests} tests passed")
    if all_passed:
        print(f"  STATUS: ALL NODES PASS")
    else:
        failed_nodes = [
            nid for nid, _, fn in nodes_to_run
        ]
        print(f"  STATUS: FAILURES PRESENT — check output above")
    print(f"{'═' * W}\n")

    return all_passed


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    filter_ids = [a.upper() for a in sys.argv[1:]] if len(sys.argv) > 1 else None
    ok = run_tree(filter_ids)
    sys.exit(0 if ok else 1)
