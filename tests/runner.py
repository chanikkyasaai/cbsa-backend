"""
CBSA Test Harness Runner

Simulates event streams through the Layer-2/Layer-4 pipeline directly —
no WebSocket, no HTTP, no Cosmos DB required.

Usage (from cbsa-backend/):
    python -m tests.runner                  # run all scenarios + validation report
    python -m tests.runner genuine          # single scenario
    python -m tests.runner drift attack     # multiple scenarios

Scenarios:
  genuine     - Stable user; expects similarity↑, trust→SAFE
  drift       - Gradual behavioral drift; expects drift↑ then stabilize
  attack      - Random adversarial vectors; expects trust→RISK/UNCERTAIN
  cold_start  - New user; trust≈0.5→UNCERTAIN until prototype created
  failure     - Invalid inputs (bad vector, duplicate nonce, timestamp); expects rejection

Output:
  Per-event metrics table (Similarity, Drift, Stability, Trust, Decision)
  PASS/FAIL verdict per scenario
  SYSTEM VALIDATION REPORT with FAR/FRR/EER when all scenarios run
"""

from __future__ import annotations

import sys
import time
from typing import Optional

import numpy as np

# ---------------------------------------------------------------------------
# Inline mock store (no CosmosDB, no SQLite)
# ---------------------------------------------------------------------------

class MockStore:
    """
    Minimal in-memory store satisfying the prototype_engine interface.

    Supports get/insert/update/delete prototype, plus the adaptive fields
    and Cosmos-backed quarantine interfaces.
    """

    def __init__(self) -> None:
        from app.prototype.quarantine_manager import QuarantineManager
        self._protos: dict = {}          # username -> list[Prototype]
        self._counter: dict = {}
        self._adaptive: dict = {}        # username -> {sim_mean, sim_m2, sim_count, ...}
        self._qm = QuarantineManager()   # Use in-memory quarantine

    # ── Prototype interface ──────────────────────────────────────────────────

    def get_prototypes(self, username: str):
        return list(self._protos.get(username, []))

    def insert_prototype(self, username: str, vector, variance, support_count: int) -> int:
        from datetime import datetime
        from app.models.prototype import Prototype
        n = self._counter.get(username, 0) + 1
        self._counter[username] = n
        p = Prototype(
            prototype_id=n,
            vector=vector.copy(),
            variance=variance.copy(),
            support_count=support_count,
            created_at=datetime.utcnow(),
            last_updated=datetime.utcnow(),
        )
        self._protos.setdefault(username, []).append(p)
        return n

    def update_prototype(self, prototype) -> None:
        for username, protos in self._protos.items():
            for i, p in enumerate(protos):
                if p.prototype_id == prototype.prototype_id:
                    protos[i] = prototype
                    return

    def delete_prototype(self, proto_id, username: str = "") -> None:
        if username and username in self._protos:
            self._protos[username] = [p for p in self._protos[username] if p.prototype_id != proto_id]
        else:
            for u in self._protos:
                self._protos[u] = [p for p in self._protos[u] if p.prototype_id != proto_id]

    # ── Adaptive fields ──────────────────────────────────────────────────────

    def get_user_adaptive_fields(self, username: str) -> Optional[dict]:
        return self._adaptive.get(username)

    def update_user_adaptive_fields(self, username: str, similarity: float, drift: float) -> None:
        """Welford online update for both similarity and drift distributions."""
        d = self._adaptive.setdefault(username, {
            "sim_count": 0, "sim_mean": 0.0, "sim_m2": 0.0,
            "drift_count": 0, "drift_mean": 0.0, "drift_m2": 0.0,
            "adaptive_sigma": 1.0392304845413265,
        })
        # Similarity
        d["sim_count"] += 1
        delta = similarity - d["sim_mean"]
        d["sim_mean"] += delta / d["sim_count"]
        d["sim_m2"] += delta * (similarity - d["sim_mean"])
        # Drift
        d["drift_count"] += 1
        ddelta = drift - d["drift_mean"]
        d["drift_mean"] += ddelta / d["drift_count"]
        d["drift_m2"] += ddelta * (drift - d["drift_mean"])
        # Adaptive sigma: mean + 0.674 * std (75th percentile)
        if d["drift_count"] > 1:
            drift_var = d["drift_m2"] / d["drift_count"]
            d["adaptive_sigma"] = d["drift_mean"] + 0.674 * float(np.sqrt(drift_var))

    def submit_quarantine_candidate(self, username: str, vector, current_time: float):
        return self._qm.submit(username=username, vector=vector, current_time=current_time)


# ---------------------------------------------------------------------------
# Scenario helpers
# ---------------------------------------------------------------------------

def _make_stable_vector(base: np.ndarray, noise_std: float = 0.02) -> np.ndarray:
    """Return base + small Gaussian noise, clipped to [0, 1]."""
    return np.clip(base + np.random.normal(0, noise_std, size=base.shape), 0.0, 1.0)


def _make_drifted_vector(base: np.ndarray, drift_direction: np.ndarray, step: float) -> np.ndarray:
    return np.clip(base + step * drift_direction, 0.0, 1.0)


def _make_random_vector() -> np.ndarray:
    return np.random.uniform(0, 1, size=48).astype(np.float64)


# ---------------------------------------------------------------------------
# Core pipeline runner
# ---------------------------------------------------------------------------

def run_pipeline(username: str, vectors, store: MockStore, timestamps=None):
    """
    Feed a list of 48-D vectors through the full pipeline.
    Returns list of per-event result dicts.
    """
    # Clear any existing session state between scenario runs
    from app.storage.memory_store import memory_store as _ms
    _ms.sessions.pop(f"test-{username}", None)

    from app.trust.trust_engine import TrustEngine, TrustState
    from app.prototype.prototype_engine import compute_prototype_metrics
    from app.models.behaviour_event import BehaviourEvent

    trust_engine = TrustEngine()
    trust_state = TrustState()

    t0 = time.time() if timestamps is None else timestamps[0]
    results = []

    for i, vec in enumerate(vectors):
        t = t0 + i * 2.0 if timestamps is None else timestamps[i]

        # Build a minimal BehaviourEvent
        event = BehaviourEvent(
            user_id=username,
            session_id=f"test-{username}",
            event_type="touch",
            timestamp=t,
            nonce=f"nonce-{i}",
            vector=vec.astype(np.float64),
        )

        # Layer-2: preprocessing
        from app.preprocessing.preprocessing import process_event
        preprocessed = process_event(event)

        # Layer-2: prototype engine (pass event timestamp for quarantine timing)
        metrics = compute_prototype_metrics(store, username, preprocessed, current_time=t)

        # Layer-4: trust engine
        trust_result = trust_engine.update_trust(
            state=trust_state,
            similarity_score=metrics.similarity_score,
            stability_score=metrics.stability_score,
            short_drift=metrics.short_drift,
            long_drift=metrics.long_drift,
            anomaly_indicator=metrics.anomaly_indicator,
            current_time=t,
        )

        results.append({
            "event": i + 1,
            "similarity": metrics.similarity_score,
            "short_drift": metrics.short_drift,
            "stability": metrics.stability_score,
            "anomaly": metrics.anomaly_indicator,
            "trust": trust_result.trust_score,
            "decision": trust_result.decision,
            "proto_id": metrics.matched_prototype_id,
            "n_protos": len(store.get_prototypes(username)),
        })

    return results


# ---------------------------------------------------------------------------
# Scenarios
# ---------------------------------------------------------------------------

def scenario_genuine(n_events: int = 60):
    """
    Stable user: 48-D vectors drawn from a fixed Gaussian.
    Expected outcome:
      - similarity rises and stabilises above 0.75
      - trust reaches SAFE (> 0.65)
    """
    print("\n" + "=" * 70)
    print("SCENARIO 1: Genuine User (stable behavior)")
    print("=" * 70)

    np.random.seed(42)
    base = np.random.uniform(0.2, 0.8, size=48).astype(np.float64)
    vectors = [_make_stable_vector(base, noise_std=0.03) for _ in range(n_events)]
    # Use spread timestamps (>=30s span for quarantine promotion)
    timestamps = [time.time() + i * 2.0 for i in range(n_events)]

    store = MockStore()
    results = run_pipeline("genuine_user", vectors, store, timestamps)

    _print_table(results)

    # Verdict
    final_20 = results[-20:]
    avg_sim = np.mean([r["similarity"] for r in final_20])
    avg_trust = np.mean([r["trust"] for r in final_20])
    safe_count = sum(1 for r in final_20 if r["decision"] == "SAFE")

    print(f"\nFinal 20 events — avg_similarity={avg_sim:.3f}, avg_trust={avg_trust:.3f}, "
          f"SAFE={safe_count}/20")

    passed = avg_trust > 0.55 and safe_count >= 10
    print(f"VERDICT: {'PASS' if passed else 'FAIL'} "
          f"(expect avg_trust > 0.55 and >= 10 SAFE in last 20)")
    return passed


def scenario_drift(n_events: int = 80):
    """
    Gradual behavioral drift: vector moves slowly away from initial state.
    Expected outcome:
      - short_drift rises during middle phase
      - trust dips toward UNCERTAIN as drift grows
      - if drift stabilises, trust partially recovers
    """
    print("\n" + "=" * 70)
    print("SCENARIO 2: Behavioral Drift (gradual change)")
    print("=" * 70)

    np.random.seed(123)
    base = np.random.uniform(0.3, 0.7, size=48).astype(np.float64)
    drift_dir = np.random.uniform(-1, 1, size=48).astype(np.float64)
    drift_dir = drift_dir / np.linalg.norm(drift_dir)

    vectors = []
    for i in range(n_events):
        if i < 20:
            # Stable phase
            vectors.append(_make_stable_vector(base, noise_std=0.02))
        elif i < 50:
            # Drift phase: 30 events of gradual drift
            step = 0.015 * (i - 20)
            vectors.append(_make_drifted_vector(base, drift_dir, step) + np.random.normal(0, 0.02, 48))
        else:
            # Re-stabilise at new location
            new_base = np.clip(base + 0.45 * drift_dir, 0, 1)
            vectors.append(_make_stable_vector(new_base, noise_std=0.02))

    vectors = [np.clip(v, 0, 1).astype(np.float64) for v in vectors]
    timestamps = [time.time() + i * 2.0 for i in range(n_events)]

    store = MockStore()
    results = run_pipeline("drift_user", vectors, store, timestamps)

    _print_table(results)

    # Verdict: drift phase should have elevated short_drift
    drift_phase = results[20:50]
    stable_phase = results[0:20]
    avg_drift_in_drift = np.mean([r["short_drift"] for r in drift_phase])
    avg_drift_stable = np.mean([r["short_drift"] for r in stable_phase[5:]])  # skip warmup

    print(f"\nDrift phase avg short_drift={avg_drift_in_drift:.3f}, "
          f"stable phase avg short_drift={avg_drift_stable:.3f}")

    passed = avg_drift_in_drift > avg_drift_stable
    print(f"VERDICT: {'PASS' if passed else 'FAIL'} "
          f"(expect drift_phase short_drift > stable_phase short_drift)")
    return passed


def scenario_attack(n_events: int = 50):
    """
    Attack simulation: attacker injects random behavioral vectors.
    Expected outcome:
      - similarity collapses to near 0
      - trust falls to RISK (< 0.40)
    """
    print("\n" + "=" * 70)
    print("SCENARIO 3: Attack (random/adversarial vectors)")
    print("=" * 70)

    np.random.seed(77)
    base = np.random.uniform(0.3, 0.7, size=48).astype(np.float64)

    vectors = []
    for i in range(n_events):
        if i < 15:
            # Legitimate warmup: establish prototype
            vectors.append(_make_stable_vector(base, noise_std=0.02))
        else:
            # Attack: random vectors (no behavioral coherence)
            vectors.append(_make_random_vector())

    timestamps = [time.time() + i * 2.0 for i in range(n_events)]

    store = MockStore()
    results = run_pipeline("attack_user", vectors, store, timestamps)

    _print_table(results)

    # Verdict: attack phase should have low trust
    attack_phase = results[20:]  # give quarantine time to promote
    avg_trust_attack = np.mean([r["trust"] for r in attack_phase]) if attack_phase else 1.0
    risk_count = sum(1 for r in attack_phase if r["decision"] == "RISK")

    print(f"\nAttack phase avg_trust={avg_trust_attack:.3f}, RISK={risk_count}/{len(attack_phase)}")

    passed = avg_trust_attack < 0.65
    print(f"VERDICT: {'PASS' if passed else 'FAIL'} "
          f"(expect attack phase avg_trust < 0.65)")
    return passed


# ---------------------------------------------------------------------------
# Table printer
# ---------------------------------------------------------------------------

def _print_table(results: list) -> None:
    header = (
        f"{'Evt':>4} | {'Sim':>5} | {'Drift':>5} | {'Stab':>5} | "
        f"{'Anom':>5} | {'Trust':>5} | {'Decision':<10} | {'Protos':>6}"
    )
    print(header)
    print("-" * len(header))
    for r in results:
        print(
            f"{r['event']:>4} | {r['similarity']:>5.3f} | {r['short_drift']:>5.3f} | "
            f"{r['stability']:>5.3f} | {r['anomaly']:>5.3f} | {r['trust']:>5.3f} | "
            f"{r['decision']:<10} | {r['n_protos']:>6}"
        )


# ---------------------------------------------------------------------------
# Prototype analytics (Task 6)
# ---------------------------------------------------------------------------

def print_user_prototype_stats(username: str, store: MockStore) -> None:
    """
    Print a summary of the prototype set for a user.

    Columns: prototype_id, support_count, vector_mean, vector_std
    """
    protos = store.get_prototypes(username)
    print(f"\n--- Prototype stats for '{username}' ({len(protos)} prototype(s)) ---")
    if not protos:
        print("  (no prototypes)")
        return
    print(f"  {'ID':>4} | {'Support':>7} | {'Vec mean':>8} | {'Vec std':>8}")
    print("  " + "-" * 38)
    for p in protos:
        vec_mean = float(np.mean(p.vector))
        vec_std  = float(np.std(p.vector))
        print(f"  {p.prototype_id:>4} | {p.support_count:>7} | {vec_mean:>8.4f} | {vec_std:>8.4f}")


# ---------------------------------------------------------------------------
# Cold-start scenario (Task 5)
# ---------------------------------------------------------------------------

def scenario_cold_start(n_events: int = 40):
    """
    Brand-new user: no prototype exists at start.
    Expected outcome:
      - First ~16 events: similarity=0, trust ≈ 0.5 (UNCERTAIN/no prototype yet)
      - After quarantine promotion: similarity rises, trust moves toward SAFE
      - Final 10 events: at least some SAFE or upward trust trend
    """
    print("\n" + "=" * 70)
    print("SCENARIO 4: Cold Start (new user, no prototype)")
    print("=" * 70)

    np.random.seed(99)
    base = np.random.uniform(0.2, 0.8, size=48).astype(np.float64)
    vectors = [_make_stable_vector(base, noise_std=0.02) for _ in range(n_events)]
    timestamps = [time.time() + i * 2.0 for i in range(n_events)]

    store = MockStore()
    results = run_pipeline("cold_start_user", vectors, store, timestamps)

    _print_table(results)
    print_user_prototype_stats("cold_start_user", store)

    # Early phase: expect low/uncertain trust (prototype not yet created)
    early = results[:15]
    avg_trust_early = np.mean([r["trust"] for r in early])

    # Late phase: prototype should exist, trust rising
    late = results[-10:]
    avg_trust_late = np.mean([r["trust"] for r in late])
    n_protos_final = results[-1]["n_protos"]

    print(f"\nEarly (1-15) avg_trust={avg_trust_early:.3f}, "
          f"Late (final 10) avg_trust={avg_trust_late:.3f}, "
          f"final proto_count={n_protos_final}")

    passed = n_protos_final >= 1 and avg_trust_late > avg_trust_early
    print(f"VERDICT: {'PASS' if passed else 'FAIL'} "
          f"(expect prototype created and late trust > early trust)")
    return passed


# ---------------------------------------------------------------------------
# Failure scenario (Task 5)
# ---------------------------------------------------------------------------

def scenario_failure():
    """
    Invalid-input rejection tests via validate_and_extract().

    Sub-tests:
      F1 - wrong vector length (47 dims)  → ValueError
      F2 - out-of-range vector (value > 1) → ValueError
      F3 - duplicate nonce                 → ValueError
      F4 - non-monotonic timestamp         → ValueError

    PASS if all 4 sub-tests raise ValueError as expected.
    """
    print("\n" + "=" * 70)
    print("SCENARIO 5: Failure / Invalid Inputs")
    print("=" * 70)

    from app.ingestion.ingestion import validate_and_extract
    from app.storage.memory_store import memory_store as _ms

    # Isolate session so nonce/timestamp state is fresh
    session_id = "test-failure-session"
    _ms.sessions.pop(session_id, None)

    t_base = time.time()
    sub_results = {}

    # ── F1: wrong vector length ───────────────────────────────────────────────
    try:
        validate_and_extract({
            "user_id": "failure_user",
            "session_id": session_id,
            "timestamp": t_base + 1.0,
            "event_type": "touch",
            "event_data": {
                "nonce": "nonce-f1",
                "vector": [0.5] * 47,   # 47 dims instead of 48
            },
        })
        sub_results["F1_wrong_length"] = False
        print("  F1 wrong_length : FAIL (expected ValueError, got none)")
    except ValueError as e:
        sub_results["F1_wrong_length"] = True
        print(f"  F1 wrong_length : PASS ({e})")

    # ── F2: out-of-range vector value ─────────────────────────────────────────
    _ms.sessions.pop(session_id, None)
    try:
        validate_and_extract({
            "user_id": "failure_user",
            "session_id": session_id,
            "timestamp": t_base + 1.0,
            "event_type": "touch",
            "event_data": {
                "nonce": "nonce-f2",
                "vector": [1.5] + [0.5] * 47,  # first element > 1
            },
        })
        sub_results["F2_out_of_range"] = False
        print("  F2 out_of_range : FAIL (expected ValueError, got none)")
    except ValueError as e:
        sub_results["F2_out_of_range"] = True
        print(f"  F2 out_of_range : PASS ({e})")

    # ── F3: duplicate nonce ───────────────────────────────────────────────────
    _ms.sessions.pop(session_id, None)
    good_payload = {
        "user_id": "failure_user",
        "session_id": session_id,
        "timestamp": t_base + 1.0,
        "event_type": "touch",
        "event_data": {
            "nonce": "nonce-dup",
            "vector": [0.5] * 48,
        },
    }
    try:
        validate_and_extract(good_payload)  # first time: OK
        validate_and_extract({              # second time with same nonce
            **good_payload,
            "timestamp": t_base + 2.0,
        })
        sub_results["F3_duplicate_nonce"] = False
        print("  F3 dup_nonce    : FAIL (expected ValueError, got none)")
    except ValueError as e:
        sub_results["F3_duplicate_nonce"] = True
        print(f"  F3 dup_nonce    : PASS ({e})")

    # ── F4: non-monotonic timestamp ───────────────────────────────────────────
    _ms.sessions.pop(session_id, None)
    try:
        validate_and_extract({
            "user_id": "failure_user",
            "session_id": session_id,
            "timestamp": t_base + 5.0,
            "event_type": "touch",
            "event_data": {"nonce": "nonce-ts1", "vector": [0.5] * 48},
        })
        validate_and_extract({
            "user_id": "failure_user",
            "session_id": session_id,
            "timestamp": t_base + 3.0,  # earlier than previous → reject
            "event_type": "touch",
            "event_data": {"nonce": "nonce-ts2", "vector": [0.5] * 48},
        })
        sub_results["F4_non_monotonic_ts"] = False
        print("  F4 non_mono_ts  : FAIL (expected ValueError, got none)")
    except ValueError as e:
        sub_results["F4_non_monotonic_ts"] = True
        print(f"  F4 non_mono_ts  : PASS ({e})")

    passed = all(sub_results.values())
    print(f"\nVERDICT: {'PASS' if passed else 'FAIL'} "
          f"({sum(sub_results.values())}/{len(sub_results)} sub-tests passed)")
    return passed


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

SCENARIOS = {
    "genuine":    scenario_genuine,
    "drift":      scenario_drift,
    "attack":     scenario_attack,
    "cold_start": scenario_cold_start,
    "failure":    scenario_failure,
}


def main():
    # Ensure cbsa-backend is on sys.path
    import os
    backend_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if backend_root not in sys.path:
        sys.path.insert(0, backend_root)

    args = sys.argv[1:]
    if args:
        to_run = {k: v for k, v in SCENARIOS.items() if k in args}
        if not to_run:
            print(f"Unknown scenario(s): {args}. Choose from: {list(SCENARIOS.keys())}")
            sys.exit(1)
    else:
        to_run = SCENARIOS

    # ── Per-scenario raw results (for evaluation) ─────────────────────────────
    _raw: dict = {}   # name -> list[dict] (per-event results from run_pipeline)

    scenario_verdicts: dict = {}
    for name, fn in to_run.items():
        try:
            # Intercept run_pipeline output for genuine/attack scenarios
            if name == "genuine":
                np.random.seed(42)
                base = np.random.uniform(0.2, 0.8, size=48).astype(np.float64)
                vectors = [_make_stable_vector(base, noise_std=0.03) for _ in range(60)]
                timestamps = [time.time() + i * 2.0 for i in range(60)]
                store = MockStore()
                _raw["genuine"] = run_pipeline("genuine_user_eval", vectors, store, timestamps)
                # Re-run the display scenario (seed is consumed; that's fine for verdicts)
                scenario_verdicts[name] = fn()
            elif name == "attack":
                np.random.seed(77)
                base = np.random.uniform(0.3, 0.7, size=48).astype(np.float64)
                vectors = []
                for i in range(50):
                    if i < 15:
                        vectors.append(_make_stable_vector(base, noise_std=0.02))
                    else:
                        vectors.append(_make_random_vector())
                timestamps = [time.time() + i * 2.0 for i in range(50)]
                store = MockStore()
                _raw["attack"] = run_pipeline("attack_user_eval", vectors, store, timestamps)
                scenario_verdicts[name] = fn()
            else:
                scenario_verdicts[name] = fn()
        except Exception:
            import traceback
            print(f"\nScenario '{name}' raised an exception:")
            traceback.print_exc()
            scenario_verdicts[name] = False

    # ── Per-scenario PASS/FAIL summary ────────────────────────────────────────
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    all_passed = True
    for name, passed in scenario_verdicts.items():
        status = "PASS" if passed else "FAIL"
        print(f"  {name:<12}: {status}")
        if not passed:
            all_passed = False

    # ── System Validation Report (Task 10) ───────────────────────────────────
    if "genuine" in _raw and "attack" in _raw:
        from tests.evaluation import evaluate
        eval_result = evaluate(_raw["genuine"], _raw["attack"])
        eval_result.print_report("SYSTEM VALIDATION REPORT")

    sys.exit(0 if all_passed else 1)


if __name__ == "__main__":
    main()
