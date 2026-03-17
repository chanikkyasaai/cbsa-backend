import io
import json
import logging
import time
import asyncio
import uuid
import zipfile
from pathlib import Path
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, Header
from fastapi.responses import JSONResponse, FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import ValidationError
from typing import List, Dict, Any, Optional

from app.config import settings, configure_logging
from app.ingestion.ingestion import validate_and_extract
from app.preprocessing.preprocessing import process_event
from app.prototype.prototype_engine import compute_prototype_metrics
from app.trust.trust_engine import trust_engine
from app.core.invariants import (
    check_preprocessed_behaviour,
    check_prototype_metrics,
    check_trust_result,
    InvariantError,
)
from app.logging.structured_logger import structured_logger
from app.storage.sqlite_store import SQLiteStore, DB_PATH
from app.storage.cosmos_prototype_store import cosmos_prototype_store
from app.storage.memory_store import memory_store
from app.models import BehaviourMessage, ServerResponse, LoginRequest, LoginResponse, TrainRequest
from app.api.websocket_manager import ConnectionManager
from app.layer3.layer3_manager import Layer3GATManager
from app.azure.enrollment_store import enrollment_store, ENROLLMENT_DURATION_SECONDS
from app.azure.behavioral_logger import behavioral_logger, BEHAVIORAL_LOG_DIR
from app.gat.trainer import triplet_trainer, CHECKPOINT_PATH
from app.azure.cosmos_logger import cosmos_logger
from app.azure.cosmos_profile_store import cosmos_profile_store
from app.azure.blob_model_store import blob_model_store

configure_logging()
logger = logging.getLogger(__name__)

app = FastAPI(
    title=settings.APP_NAME,
    version=settings.VERSION,
)

BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"
EVENT_MAP_PATH = BASE_DIR.parent / "EVENT_FLOW_MAP.json"

app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

behaviour_manager = ConnectionManager()
monitor_manager = ConnectionManager()
gat_manager = Layer3GATManager()  # Initialize GAT manager


async def _session_sweeper() -> None:
    """
    Background asyncio task that evicts inactive sessions every 60 seconds.
    Sessions idle for > SESSION_TTL_SECONDS (600s) are removed from memory_store.
    """
    from app.storage.memory_store import SESSION_TTL_SECONDS
    sweep_interval = min(60.0, SESSION_TTL_SECONDS / 5)
    while True:
        await asyncio.sleep(sweep_interval)
        try:
            evicted = memory_store.evict_expired_sessions()
            if evicted:
                logger.info("Session sweeper: evicted %d expired session(s)", evicted)
        except Exception as exc:
            logger.error("Session sweeper error: %s", exc)


@app.on_event("startup")
async def startup_event() -> None:
    # Always initialise the SQLite store (used as fallback in DEBUG_MODE).
    app.state.sqlite_store = SQLiteStore(DB_PATH)
    # CosmosPrototypeStore is a module-level singleton; expose it on app.state too.
    app.state.prototype_store = cosmos_prototype_store
    # Start background session TTL sweeper
    asyncio.create_task(_session_sweeper())


# ---------------------------------------------------------------------------
# Authorization dependency
# ---------------------------------------------------------------------------

def _verify_admin_token(authorization: Optional[str] = Header(default=None)) -> None:
    """
    Require a valid Bearer token for destructive / training endpoints.
    If ADMIN_TOKEN is not configured in the environment, all requests are rejected
    to prevent accidental exposure of admin operations.
    """
    expected_token = settings.ADMIN_TOKEN.strip()
    if not expected_token:
        raise HTTPException(
            status_code=503,
            detail="Admin endpoints are disabled: ADMIN_TOKEN is not configured",
        )
    if not authorization or authorization != f"Bearer {expected_token}":
        raise HTTPException(
            status_code=401,
            detail="Unauthorized: valid Authorization: Bearer <token> header required",
        )


# Per-session tracking of the last GAT inference timestamp.
# Used to enforce the configured interval between GAT calls.
_last_gat_inference_time: Dict[str, float] = {}


def load_event_flow_map() -> dict:
    try:
        with EVENT_MAP_PATH.open("r", encoding="utf-8") as file:
            return json.load(file)
    except FileNotFoundError:
        logger.error("EVENT_FLOW_MAP.json not found at %s", EVENT_MAP_PATH)
        return {"eventFlowMap": {}}


EVENT_FLOW_MAP = load_event_flow_map()


@app.get("/")
async def root():
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/health")
async def health():
    """
    Health check endpoint.
    Verifies all Cosmos DB connections and fails if any are unavailable.
    """
    cosmos_checks = {
        "cosmos_logger": {
            "enabled": cosmos_logger._enabled,
            "connected": cosmos_logger._container is not None,
        },
        "cosmos_profile_store": {
            "enabled": cosmos_profile_store._enabled,
            "connected": cosmos_profile_store._container is not None,
        },
        "enrollment_store": {
            "enabled": enrollment_store._enabled,
            "connected": enrollment_store._container is not None,
        },
        "cosmos_prototype_store": {
            "enabled": cosmos_prototype_store._enabled,
            "connected": cosmos_prototype_store._proto_container is not None,
        },
        "cosmos_behavioral_logger": {
            "enabled": behavioral_logger._enabled,
            "connected": behavioral_logger._container is not None,
        },
    }
    
    # Check if Cosmos is expected to be configured
    cosmos_expected = bool(settings.COSMOS_ENDPOINT and settings.COSMOS_KEY)
    
    # Determine overall health status
    all_healthy = True
    errors = []
    
    if cosmos_expected:
        for service, check in cosmos_checks.items():
            if not check["enabled"] or not check["connected"]:
                all_healthy = False
                errors.append(f"{service}: not connected")
    
    status_code = 200 if all_healthy else 503
    
    return JSONResponse(
        status_code=status_code,
        content={
            "status": "healthy" if all_healthy else "unhealthy",
            "active_connections": behaviour_manager.get_connection_count(),
            "monitor_connections": monitor_manager.get_connection_count(),
            "cosmos_db": cosmos_checks,
            "cosmos_expected": cosmos_expected,
            "errors": errors if errors else None,
        }
    )


# ================== Authentication Endpoints ==================

@app.post("/login", response_model=LoginResponse)
async def login(request: LoginRequest):
    """
    Login endpoint.
    - Accepts a username (no real auth — any user can log in).
    - Checks if a trained profile exists for the user.
    - If not, marks the user as enrolling and starts the 5-minute enrollment timer.
    - Returns enrollment status with time remaining.
    """
    username = request.username.strip()
    if not username:
        raise HTTPException(status_code=400, detail="Username cannot be empty")

    # Check if a trained profile already exists
    state = enrollment_store.get_or_create(username)

    if enrollment_store.has_profile(username) or state.enrolled:
        enrollment_store.mark_enrolled(username)
        return LoginResponse(
            username=username,
            status="enrolled",
            message="Welcome back! Profile found.",
        )

    # Not enrolled yet — start / resume enrollment session timer
    enrollment_store.start_session(username)

    remaining = state.seconds_remaining()
    accumulated = state.accumulated_seconds

    if remaining <= 0:
        # Timer already expired — enrollment complete (awaiting training)
        return LoginResponse(
            username=username,
            status="enrolling",
            message="Enrollment data collected. Run /train to build your profile.",
            seconds_remaining=0.0,
            accumulated_seconds=accumulated,
            total_seconds=ENROLLMENT_DURATION_SECONDS,
        )

    return LoginResponse(
        username=username,
        status="enrolling",
        message=f"Enrollment in progress. {remaining:.0f}s remaining.",
        seconds_remaining=remaining,
        accumulated_seconds=accumulated,
        total_seconds=ENROLLMENT_DURATION_SECONDS,
    )


@app.post("/logout")
async def logout(body: Dict[str, Any]):
    """
    Logout endpoint — saves accumulated enrollment time for the session.
    """
    username = body.get("username", "").strip()
    if not username:
        raise HTTPException(status_code=400, detail="username required")
    enrollment_store.end_session(username)
    return {"status": "ok", "message": "Session ended. Enrollment time saved."}


@app.post("/train")
async def train_profile(request: TrainRequest, authorization: Optional[str] = Header(default=None)):
    """
    Trigger triplet training for a user (or all users).
    After training, a profile is stored in data/profiles/<user_id>_profile.json.
    Set force=True to retrain even if a profile already exists.

    Requires ``Authorization: Bearer <ADMIN_TOKEN>`` header.
    """
    _verify_admin_token(authorization)
    import asyncio

    loop = asyncio.get_event_loop()

    if request.user_id:
        # Train a single user
        user_id_str: str = request.user_id  # narrow Optional[str] → str
        result = await loop.run_in_executor(
            None, lambda: triplet_trainer.train_user(user_id_str, force=request.force)
        )
        if result.get("status") == "success":
            enrollment_store.mark_enrolled(request.user_id)
        return result
    else:
        # Train all users
        results = await loop.run_in_executor(
            None, lambda: triplet_trainer.train_all(force=request.force)
        )
        for r in results:
            if r.get("status") == "success":
                uid = r.get("user_id")
                if uid:
                    enrollment_store.mark_enrolled(uid)
        return {"results": results, "total": len(results)}


@app.get("/event-flow-map")
async def event_flow_map():
    return JSONResponse(content=EVENT_FLOW_MAP)


@app.websocket(settings.WEBSOCKET_ENDPOINT)
async def websocket_behaviour_endpoint(websocket: WebSocket):
    await behaviour_manager.connect(websocket)
    client_id = id(websocket)
    logger.info(f"Client {client_id} connected to behaviour stream")

    # Per-connection tracking for enrollment notifications
    ws_user_id: str | None = None
    ws_session_id: str | None = None
    enrollment_notified = False

    try:
        while True:
            try:
                data = await websocket.receive_json()

                # ---------- Engine processing (v1 layers) ----------
                event = None
                preprocessed = None
                warmup_state = None

                try:
                    # ── PIPELINE: validate → preprocess (snapshot → drift → buffer) ──
                    event = validate_and_extract(data)
                    preprocessed = process_event(event)

                    # Runtime invariant guard — catch data quality issues early
                    try:
                        check_preprocessed_behaviour(preprocessed)
                    except InvariantError as inv_err:
                        logger.error("Invariant violation in preprocessed output: %s", inv_err)

                    warmup_state = await asyncio.to_thread(
                        cosmos_prototype_store.collect_warmup_window,
                        event.user_id,
                        preprocessed.window_vector,
                    )

                    logger.info(
                        "Processed event username=%s session=%s type=%s short_drift=%.4f long_drift=%.4f warmup=%s",
                        event.user_id,
                        event.session_id,
                        event.event_type,
                        preprocessed.short_drift,
                        preprocessed.long_drift,
                        warmup_state.get("warmup", False),
                    )

                except Exception as engine_error:
                    logger.warning("Engine processing error for client %s: %s", client_id, engine_error)

                # ---------- Behavioral logging / enrollment / GAT ----------
                try:
                    behaviour_msg = BehaviourMessage(**data)

                    user_id = behaviour_msg.user_id or "unknown"
                    session_id = behaviour_msg.session_id or "unknown"
                    ws_session_id = session_id

                    logger.info(
                        "Received message from user %s (session: %s, client: %s)",
                        user_id, session_id, client_id,
                    )

                    # ---- Behavioral data logging ----
                    if user_id != "unknown":
                        behavioral_logger.log_event(user_id, session_id, data)

                    # ---- Enrollment session tracking ----
                    if user_id != "unknown" and ws_user_id != user_id:
                        ws_user_id = user_id
                        enrollment_notified = False
                        enrollment_store.start_session(user_id)

                    # Check enrollment completion and push notification once
                    if ws_user_id and not enrollment_notified:
                        state = enrollment_store.get_or_create(ws_user_id)
                        if state.is_enrollment_complete() and not state.enrolled:
                            enrollment_store.end_session(ws_user_id)
                            enrollment_notified = True
                            await behaviour_manager.send_personal_message(
                                {
                                    "type": "enrollment_status",
                                    "status": "enrollment_complete",
                                    "message": "Enrollment data collection complete. Profile training is now available.",
                                    "user_id": ws_user_id,
                                    "server_timestamp": time.time(),
                                },
                                websocket,
                            )
                            logger.info(f"Enrollment complete notification sent to user {ws_user_id}")

                    # ---- GAT Layer 3 processing ----
                    # Always collect events for the session window.
                    gat_manager.add_event_to_session(session_id, behaviour_msg)

                    gat_similarity = None
                    gat_result: Dict[str, Any] = {}

                    # Only run GAT inference for enrolled users (skip during enrollment).
                    user_is_enrolled = (
                        user_id != "unknown"
                        and enrollment_store.get_enrollment_status(user_id).get("status") == "enrolled"
                    )

                    # NOTE: Periodic GAT inference removed.
                    # Layer-4 (TrustEngine) now drives GAT escalation via
                    # event-driven uncertainty detection. See trust_engine.py
                    # for full escalation logic documentation.
                    # The _last_gat_inference_time dict is retained for
                    # backward compatibility with disconnect cleanup only.
                    if user_is_enrolled:
                        pass  # GAT escalation handled by Layer-4 below

                    # ---- Monitor broadcast ----
                    event_type_str = event.event_type if event else (data.get("event_type") if isinstance(data, dict) else "unknown")
                    monitor_message = {
                        "receivedAt": time.time(),
                        "eventType": event_type_str,
                        "payload": data.get("event_data") if isinstance(data, dict) else None,
                        "deviceInfo": (
                            data.get("event_data", {}).get("deviceInfo")
                            if isinstance(data, dict) and isinstance(data.get("event_data"), dict)
                            else None
                        ),
                        "signature": data.get("signature") if isinstance(data, dict) else None,
                        "raw": data,
                        "gatSimilarity": gat_similarity,
                        "layer3Processed": gat_similarity is not None,
                    }

                    try:
                        await monitor_manager.broadcast(monitor_message)
                    except Exception as e:
                        logger.error("Failed to broadcast to monitor clients: %s", e, exc_info=True)

                    # ---- Send response to client ----
                    engine_metrics: Dict[str, Any] = {}
                    if warmup_state is not None and bool(warmup_state.get("warmup", False)):
                        response = {
                            "status": "WARMUP",
                            "collected_windows": int(warmup_state.get("collected_windows", 0)),
                        }
                    elif preprocessed is not None and event is not None:
                        # ── PIPELINE: prototype matching ──────────────────────
                        metrics = await asyncio.to_thread(
                            compute_prototype_metrics,
                            cosmos_prototype_store,
                            event.user_id,
                            preprocessed,
                            event.timestamp,
                        )

                        # Runtime invariant guard on Layer-2 output
                        try:
                            check_prototype_metrics(metrics)
                        except InvariantError as inv_err:
                            logger.error("Invariant violation in prototype metrics: %s", inv_err)

                        # ── PIPELINE: Layer-4 Trust Engine ────────────────────
                        # Retrieve per-session TrustState from memory_store
                        session_state = memory_store.get_or_create_session(
                            event.session_id
                        )
                        _t_now = time.time()
                        trust_result = trust_engine.update_trust(
                            state=session_state.trust_state,
                            similarity_score=metrics.similarity_score,
                            stability_score=metrics.stability_score,
                            short_drift=metrics.short_drift,
                            long_drift=metrics.long_drift,
                            anomaly_indicator=metrics.anomaly_indicator,
                            gat_similarity=gat_similarity,
                            current_time=_t_now,
                        )

                        # Runtime invariant guard on Layer-4 output
                        try:
                            check_trust_result(trust_result)
                        except InvariantError as inv_err:
                            logger.error("Invariant violation in trust result: %s", inv_err)

                        # Task 8: Drift vs Trust validation debug log
                        # Emit structured line so reviewers can verify drift↑ → trust↓
                        logger.debug(
                            "DRIFT_TRUST user=%s event=%d short_drift=%.4f trust=%.4f decision=%s",
                            event.user_id,
                            session_state.trust_state.event_count,
                            metrics.short_drift,
                            trust_result.trust_score,
                            trust_result.decision,
                        )

                        # ── PIPELINE: GAT escalation (Layer-3) ───────────────
                        # If Layer-4 says escalate and we haven't already run GAT,
                        # trigger GAT processing now.
                        if (
                            trust_result.escalate_to_layer3
                            and not gat_result
                            and user_is_enrolled
                        ):
                            session_window = gat_manager.get_session_window(session_id)
                            if len(session_window) >= 5:
                                logger.info(
                                    "Layer-4 escalation triggered for session %s "
                                    "(decision=%s, anomaly=%.3f)",
                                    session_id,
                                    trust_result.decision,
                                    trust_result.anomaly_indicator,
                                )
                                gat_result = await gat_manager.process_escalated_session(
                                    session_id, session_window
                                )
                                _raw_gat = gat_result.get("similarity_score")
                                # Task 7: validate GAT output before use — no fallback randoms
                                if (
                                    _raw_gat is not None
                                    and isinstance(_raw_gat, (int, float))
                                    and 0.0 <= float(_raw_gat) <= 1.0
                                ):
                                    gat_similarity = float(_raw_gat)
                                    logger.info(
                                        "GAT layer3 result: session=%s gat_score=%.4f",
                                        session_id, gat_similarity,
                                    )
                                else:
                                    gat_similarity = None
                                    if _raw_gat is not None:
                                        logger.warning(
                                            "GAT returned invalid score %s for session %s — ignored",
                                            _raw_gat, session_id,
                                        )
                                # Re-run trust update with GAT augmentation
                                if gat_similarity is not None:
                                    trust_result = trust_engine.update_trust(
                                        state=session_state.trust_state,
                                        similarity_score=metrics.similarity_score,
                                        stability_score=metrics.stability_score,
                                        short_drift=metrics.short_drift,
                                        long_drift=metrics.long_drift,
                                        anomaly_indicator=metrics.anomaly_indicator,
                                        gat_similarity=gat_similarity,
                                        current_time=time.time(),
                                    )

                        # ── PIPELINE: Structured event log (final step) ───────
                        try:
                            await asyncio.to_thread(
                                structured_logger.log,
                                event.user_id,
                                event.session_id,
                                event.timestamp,
                                event.event_type,
                                metrics,
                                trust_result,
                            )
                        except Exception as log_err:
                            logger.error("Structured logger error: %s", log_err)

                        engine_metrics = {
                            # Layer-2 metrics
                            "similarityScore": metrics.similarity_score,
                            "shortDrift": metrics.short_drift,
                            "longDrift": metrics.long_drift,
                            "stabilityScore": metrics.stability_score,
                            "prototypeConfidence": metrics.prototype_confidence,
                            "behaviouralConsistency": metrics.behavioural_consistency,
                            "prototypeSupportStrength": metrics.prototype_support_strength,
                            "anomalyIndicator": metrics.anomaly_indicator,
                            "matchedPrototypeId": metrics.matched_prototype_id,
                            # Layer-4 metrics
                            "trustScore": trust_result.trust_score,
                            "decision": trust_result.decision,
                            "escalatedToLayer3": trust_result.escalate_to_layer3,
                        }
                        response = {
                            # Layer-2 output (full rich vector)
                            "similarity_score": metrics.similarity_score,
                            "short_drift": metrics.short_drift,
                            "long_drift": metrics.long_drift,
                            "stability_score": metrics.stability_score,
                            "prototype_confidence": metrics.prototype_confidence,
                            "behavioural_consistency": metrics.behavioural_consistency,
                            "prototype_support_strength": metrics.prototype_support_strength,
                            "anomaly_indicator": metrics.anomaly_indicator,
                            "matched_prototype_id": metrics.matched_prototype_id,
                            # Layer-4 output
                            "trust_score": trust_result.trust_score,
                            "decision": trust_result.decision,
                            "raw_trust_signal": trust_result.raw_trust_signal,
                            # Layer-3 GAT (Task 7)
                            "layer3_used": trust_result.gat_augmented,
                            "gat_score": gat_similarity,   # float or null
                        }
                    else:
                        # Engine failed — acknowledge receipt without metrics
                        response = {"status": "received"}

                    # ---- Cosmos DB computation log ----
                    cosmos_gat_record: Dict[str, Any] = {}
                    if gat_result:
                        cosmos_gat_record = {
                            "similarityScore": gat_result.get("similarity_score"),
                            "sessionVector": gat_result.get("session_vector"),
                            "processingTimeMs": gat_result.get("processing_time_ms"),
                            "graphEvents": gat_result.get("graph_events"),
                            "graphDuration": gat_result.get("graph_duration"),
                        }
                    try:
                        await asyncio.to_thread(
                            cosmos_logger.log_computation,
                            user_id,
                            session_id,
                            event_type_str,
                            engine_metrics,
                            cosmos_gat_record,
                        )
                    except Exception as cosmos_err:
                        logger.error("Cosmos DB logging error: %s", cosmos_err)

                    await behaviour_manager.send_personal_message(response, websocket)
                except ValueError as e:
                    logger.warning("Validation error from client %s: %s", client_id, e)
                    await behaviour_manager.send_personal_message({"error": str(e)}, websocket)
                    
            except ValueError as e:
                logger.error(f"JSON decode error from client {client_id}: {e}")
                error_response = {
                    "status": "error",
                    "message": "Malformed JSON",
                }
                await behaviour_manager.send_personal_message(error_response, websocket)
                
    except WebSocketDisconnect:
        behaviour_manager.disconnect(websocket)
        if ws_user_id:
            enrollment_store.end_session(ws_user_id)
        if ws_session_id:
            _last_gat_inference_time.pop(ws_session_id, None)
        logger.info(f"Client {client_id} disconnected normally")
        
    except Exception as e:
        logger.error(f"Unexpected error with client {client_id}: {e}", exc_info=True)
        behaviour_manager.disconnect(websocket)
        if ws_user_id:
            enrollment_store.end_session(ws_user_id)
        if ws_session_id:
            _last_gat_inference_time.pop(ws_session_id, None)


@app.websocket("/ws/monitor")
async def websocket_monitor(websocket: WebSocket):
    await monitor_manager.connect(websocket)
    client_id = id(websocket)
    logger.info(f"Monitor client {client_id} connected")

    try:
        while True:
            try:
                await websocket.receive_text()
            except ValueError:
                pass
    except WebSocketDisconnect:
        monitor_manager.disconnect(websocket)
        logger.info(f"Monitor client {client_id} disconnected normally")
    except Exception as e:
        logger.error(f"Unexpected error with monitor client {client_id}: {e}", exc_info=True)
        monitor_manager.disconnect(websocket)


# ================== Layer 3 GAT API Endpoints ==================

@app.post("/gat/process")
async def process_gat_session(request_data: Dict[str, Any]):
    """
    Manually trigger GAT processing for a session
    Used for testing or external escalation requests
    """
    try:
        session_id = request_data.get("session_id")
        if not session_id:
            raise HTTPException(status_code=400, detail="session_id required")
        
        session_window = gat_manager.get_session_window(session_id)
        if not session_window:
            raise HTTPException(status_code=404, detail="No session data found")
        
        auth_result = await gat_manager.process_escalated_session(session_id, session_window)
        return auth_result
        
    except Exception as e:
        logger.error(f"GAT processing API error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/gat/enroll")
async def enroll_user_profile(enrollment_data: Dict[str, Any]):
    """
    Enroll a user profile from verified behavioral sessions
    """
    try:
        user_id = enrollment_data.get("user_id")
        session_data = enrollment_data.get("verified_sessions", [])
        
        if not user_id:
            raise HTTPException(status_code=400, detail="user_id required")
        if not session_data:
            raise HTTPException(status_code=400, detail="verified_sessions required")
        
        # Convert session data to BehaviourMessage objects
        verified_sessions = []
        for session in session_data:
            session_messages = []
            for event_data in session:
                try:
                    msg = BehaviourMessage(**event_data)
                    session_messages.append(msg)
                except Exception as e:
                    logger.warning(f"Skipping invalid event in enrollment: {e}")
            if session_messages:
                verified_sessions.append(session_messages)
        
        if not verified_sessions:
            raise HTTPException(status_code=400, detail="No valid sessions found")
        
        enrollment_result = await gat_manager.enroll_user_session(user_id, verified_sessions)
        return enrollment_result
        
    except Exception as e:
        logger.error(f"User enrollment API error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/gat/profile/{user_id}")
async def get_user_profile(user_id: str):
    """
    Retrieve user behavioral profile
    """
    try:
        profile = await gat_manager.profile_manager.get_user_profile(user_id)
        if not profile:
            raise HTTPException(status_code=404, detail="User profile not found")
        
        return {
            "user_id": profile.user_id,
            "profile_dimensions": len(profile.profile_vector),
            "enrollment_sessions": profile.enrollment_sessions,
            "last_updated": profile.last_updated.isoformat(),
            "profile_confidence": profile.profile_confidence
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Profile retrieval API error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/gat/sessions/{session_id}")
async def get_session_window(session_id: str):
    """
    Get current session window data
    """
    session_window = gat_manager.get_session_window(session_id)
    return {
        "session_id": session_id,
        "event_count": len(session_window),
        "events": [
            {
                "timestamp": event.timestamp,
                "event_type": getattr(event, 'event_type', 'unknown'),
                "user_id": event.user_id,
                "session_id": event.session_id
            }
            for event in session_window
        ]
    }


@app.delete("/gat/sessions/{session_id}")
async def clear_session_window(session_id: str):
    """
    Clear session window data
    """
    gat_manager.clear_session_window(session_id)
    return {"message": f"Session window cleared for {session_id}"}


@app.get("/gat/stats")
async def get_gat_stats():
    """
    Get GAT processing statistics
    """
    active_sessions = len(gat_manager.session_windows)
    total_events = sum(len(window) for window in gat_manager.session_windows.values())
    
    return {
        "active_sessions": active_sessions,
        "total_events_buffered": total_events,
        "window_seconds": settings.GAT_WINDOW_SECONDS,
        "debug_mode": settings.DEBUG_MODE,
        "cloud_endpoint": settings.GAT_CLOUD_ENDPOINT,
    }


# ================== Admin Endpoints ==================

@app.delete("/admin/user/{user_id}")
async def delete_user_data(user_id: str, authorization: Optional[str] = Header(default=None)):
    """
    Delete **all** data associated with a user:
      - Cosmos DB prototype store (prototypes + behaviour logs for this user)
      - SQLite user row, prototypes, behaviour_logs (debug mode)
      - Behavioral log file (JSONL)
      - User profile (Cosmos DB + local disk)
      - Enrollment state
      - In-memory GAT session windows for the user
      - Cosmos DB computation logs for the user

    Requires ``Authorization: Bearer <ADMIN_TOKEN>`` header.
    """
    _verify_admin_token(authorization)
    results: Dict[str, Any] = {"user_id": user_id}

    # 1. Cosmos prototype store (prototypes + behaviour logs)
    try:
        cosmos_prototype_store.delete_user(user_id)
        results["cosmos_prototype_store"] = "cleared"
    except Exception as e:
        logger.error("Failed to clear Cosmos prototype data for %s: %s", user_id, e)
        results["cosmos_prototype_store"] = f"error: {e}"

    # 2. SQLite: prototypes + behaviour_logs + user row
    try:
        store: SQLiteStore = app.state.sqlite_store
        with store._connect() as conn:
            conn.execute("DELETE FROM behaviour_logs WHERE username = ?", (user_id,))
            conn.execute("DELETE FROM prototypes WHERE username = ?", (user_id,))
            conn.execute("DELETE FROM users WHERE username = ?", (user_id,))
            conn.commit()
        results["sqlite"] = "cleared"
    except Exception as e:
        logger.error("Failed to clear SQLite data for %s: %s", user_id, e)
        results["sqlite"] = f"error: {e}"

    # 3. Behavioral log file (Cosmos + local JSONL)
    try:
        behavioral_logger.delete_user_log(user_id)
        results["behavioral_log"] = "cleared"
    except Exception as e:
        results["behavioral_log"] = f"error: {e}"

    # 4. User profile (Cosmos + local disk)
    try:
        cosmos_profile_store.delete_profile(user_id)
        results["profile"] = "cleared"
    except Exception as e:
        results["profile"] = f"error: {e}"

    # 5. Enrollment state
    try:
        if user_id in enrollment_store._states:
            del enrollment_store._states[user_id]
            enrollment_store._save()
        results["enrollment"] = "cleared"
    except Exception as e:
        results["enrollment"] = f"error: {e}"

    # 6. In-memory GAT sessions belonging to this user
    try:
        sessions_cleared = 0
        for sid in list(gat_manager.session_windows.keys()):
            window = gat_manager.session_windows[sid]
            if window and getattr(window[0], "user_id", None) == user_id:
                del gat_manager.session_windows[sid]
                sessions_cleared += 1
        results["gat_sessions_cleared"] = sessions_cleared
    except Exception as e:
        results["gat_sessions"] = f"error: {e}"

    # 7. Cosmos DB computation logs for this user
    try:
        deleted_count = _delete_cosmos_logs_for_user(user_id)
        results["cosmos_computation_logs"] = f"deleted {deleted_count}"
    except Exception as e:
        results["cosmos_computation_logs"] = f"error: {e}"

    # 8. In-memory profile manager
    try:
        if user_id in gat_manager.profile_manager.profiles:
            del gat_manager.profile_manager.profiles[user_id]
        results["in_memory_profile"] = "cleared"
    except Exception as e:
        results["in_memory_profile"] = f"error: {e}"

    return results


@app.delete("/admin/truncate")
async def truncate_all_data(authorization: Optional[str] = Header(default=None)):
    """
    Delete **all** data for every user:
      - Cosmos DB prototype store (prototypes + behaviour logs)
      - SQLite tables (users, prototypes, behaviour_logs)
      - All behavioral log files (Cosmos + local JSONL)
      - All user profiles (Cosmos DB + local disk)
      - Enrollment store
      - All in-memory GAT session windows
      - All Cosmos DB computation logs
      - All model checkpoints (Blob Storage + local disk)

    Requires ``Authorization: Bearer <ADMIN_TOKEN>`` header.
    """
    _verify_admin_token(authorization)
    results: Dict[str, Any] = {}

    # 1. Cosmos prototype store (prototypes + behaviour logs)
    try:
        counts = cosmos_prototype_store.delete_all()
        results["cosmos_prototype_store"] = f"deleted {counts.get('prototypes_deleted', 0)} prototypes + {counts.get('logs_deleted', 0)} logs"
    except Exception as e:
        results["cosmos_prototype_store"] = f"error: {e}"

    # 2. SQLite
    try:
        store: SQLiteStore = app.state.sqlite_store
        with store._connect() as conn:
            conn.execute("DELETE FROM behaviour_logs")
            conn.execute("DELETE FROM prototypes")
            conn.execute("DELETE FROM users")
            conn.commit()
        results["sqlite"] = "truncated"
    except Exception as e:
        results["sqlite"] = f"error: {e}"

    # 3. Behavioral logs (Cosmos + local JSONL files)
    try:
        count = behavioral_logger.delete_all_logs()
        results["behavioral_logs"] = f"deleted {count} Cosmos docs + local files"
    except Exception as e:
        results["behavioral_logs"] = f"error: {e}"

    # 4. Profiles (Cosmos + local)
    try:
        n = cosmos_profile_store.delete_all_profiles()
        results["profiles"] = f"deleted {n}"
    except Exception as e:
        results["profiles"] = f"error: {e}"

    # 5. Enrollment store
    try:
        enrollment_store._states.clear()
        enrollment_store._save()
        results["enrollment"] = "truncated"
    except Exception as e:
        results["enrollment"] = f"error: {e}"

    # 6. GAT session windows
    try:
        gat_manager.session_windows.clear()
        gat_manager.profile_manager.profiles.clear()
        results["gat_sessions"] = "truncated"
    except Exception as e:
        results["gat_sessions"] = f"error: {e}"

    # 7. Cosmos computation logs
    try:
        deleted_count = _delete_all_cosmos_logs()
        results["cosmos_computation_logs"] = f"deleted {deleted_count}"
    except Exception as e:
        results["cosmos_computation_logs"] = f"error: {e}"

    # 8. Model checkpoints (Blob + local)
    try:
        blob_count = blob_model_store.delete_all_models()
        local_count = 0
        if settings.DEBUG_MODE:
            local_ckpt_dir = CHECKPOINT_PATH.parent
            if local_ckpt_dir.exists():
                for p in local_ckpt_dir.glob("*.pt"):
                    p.unlink(missing_ok=True)
                    local_count += 1
                for p in local_ckpt_dir.glob("*.pth"):
                    p.unlink(missing_ok=True)
                    local_count += 1
        results["model_checkpoints"] = f"deleted {blob_count} blobs + {local_count} local"
    except Exception as e:
        results["model_checkpoints"] = f"error: {e}"

    return results


# ---------------------------------------------------------------------------
# Helpers for Cosmos computation-log cleanup
# ---------------------------------------------------------------------------

def _delete_cosmos_logs_for_user(user_id: str) -> int:
    """Delete all computation-log docs for a given userId."""
    container = cosmos_logger._container
    if container is None:
        return 0
    count = 0
    try:
        items = list(
            container.query_items(
                query="SELECT c.id FROM c WHERE c.userId = @uid",
                parameters=[{"name": "@uid", "value": user_id}],
                partition_key=user_id,
            )
        )
        for item in items:
            try:
                container.delete_item(item=item["id"], partition_key=user_id)
                count += 1
            except Exception as exc:
                logger.debug("Failed to delete computation log %s: %s", item["id"], exc)
    except Exception as exc:
        logger.error("Failed to delete Cosmos logs for %s: %s", user_id, exc)
    return count


def _delete_all_cosmos_logs() -> int:
    """Delete every document in the computation-log container."""
    container = cosmos_logger._container
    if container is None:
        return 0
    count = 0
    try:
        items = list(
            container.query_items(
                query="SELECT c.id, c.userId FROM c",
                enable_cross_partition_query=True,
            )
        )
        for item in items:
            try:
                container.delete_item(
                    item=item["id"], partition_key=item["userId"]
                )
                count += 1
            except Exception as exc:
                logger.debug("Failed to delete computation log %s: %s", item["id"], exc)
    except Exception as exc:
        logger.error("Failed to truncate Cosmos computation logs: %s", exc)
    return count


# ---------------------------------------------------------------------------
# Legacy data migration endpoint
# ---------------------------------------------------------------------------

@app.post("/admin/upload-legacy")
async def upload_legacy_data(authorization: Optional[str] = Header(default=None)):
    """
    Migrate existing local data (cbsa.db SQLite file and per-user JSONL logs)
    to Cosmos DB.

    - Reads every user row + prototypes + behaviour_logs from cbsa.db
    - Reads every *.jsonl file in data/behavioral_logs/
    - Upserts everything into the Cosmos DB prototype-store and behaviour-logs containers.

    This endpoint is idempotent: re-running it will upsert (not duplicate) data
    that was already migrated.

    Requires ``Authorization: Bearer <ADMIN_TOKEN>`` header.
    """
    _verify_admin_token(authorization)

    if not cosmos_prototype_store._enabled:
        raise HTTPException(
            status_code=503,
            detail="Cosmos DB prototype store is not configured – cannot migrate",
        )

    results: Dict[str, Any] = {}

    # 1. Read SQLite and upload to Cosmos
    sqlite_users_migrated = 0
    sqlite_errors = 0
    try:
        sqlite_store: SQLiteStore = app.state.sqlite_store
        # Get all usernames from SQLite
        with sqlite_store._connect() as conn:
            rows = conn.execute("SELECT username FROM users").fetchall()
        usernames = [row["username"] for row in rows]

        for username in usernames:
            try:
                user_data = sqlite_store.export_user(username)
                cosmos_prototype_store.import_user(user_data)
                sqlite_users_migrated += 1
            except Exception as exc:
                logger.error("Failed to migrate SQLite user %s: %s", username, exc)
                sqlite_errors += 1

        results["sqlite_migration"] = {
            "users_migrated": sqlite_users_migrated,
            "errors": sqlite_errors,
        }
    except Exception as exc:
        results["sqlite_migration"] = {"error": str(exc)}

    # 2. Read JSONL behavioral logs and upload to Cosmos
    jsonl_users_migrated = 0
    jsonl_events_migrated = 0
    jsonl_errors = 0
    try:
        if BEHAVIORAL_LOG_DIR.exists():
            for jsonl_path in BEHAVIORAL_LOG_DIR.glob("*.jsonl"):
                user_id = jsonl_path.stem
                try:
                    events: list = []
                    with jsonl_path.open("r", encoding="utf-8") as f:
                        for line in f:
                            line = line.strip()
                            if line:
                                try:
                                    events.append(json.loads(line))
                                except json.JSONDecodeError:
                                    pass

                    if cosmos_prototype_store._logs_container is not None:
                        for event in events:
                            event_user_id = event.get("user_id", user_id)
                            try:
                                cosmos_prototype_store._logs_container.upsert_item(
                                    {
                                        "id": str(uuid.uuid4()),
                                        "userId": event_user_id,
                                        "sessionId": str(event.get("session_id", "")),
                                        "eventTimestamp": float(event.get("logged_at") or event.get("timestamp", 0.0)),
                                        "eventType": str(event.get("event_type", "")),
                                        "vectorJson": "[]",
                                        "shortDrift": 0.0,
                                        "longDrift": 0.0,
                                        "stabilityScore": 0.0,
                                        "createdAt": str(event.get("logged_at", "")),
                                        "rawEvent": event,
                                    }
                                )
                                jsonl_events_migrated += 1
                            except Exception:
                                jsonl_errors += 1
                    jsonl_users_migrated += 1
                except Exception as exc:
                    logger.error("Failed to migrate JSONL log for %s: %s", user_id, exc)
                    jsonl_errors += 1

        results["jsonl_migration"] = {
            "users_migrated": jsonl_users_migrated,
            "events_migrated": jsonl_events_migrated,
            "errors": jsonl_errors,
        }
    except Exception as exc:
        results["jsonl_migration"] = {"error": str(exc)}

    return results


# ---------------------------------------------------------------------------
# Cosmos DB snapshot helpers & endpoints
# ---------------------------------------------------------------------------

DUMP_ROOT = Path(__file__).resolve().parent.parent / "data" / "cosmos_dump"

_COSMOS_CONTAINERS = [
    # (settings-key-name-for-logging, partition_key_field, live_container_object_getter)
    # Evaluated lazily inside helpers so late-binding works.
]


def _get_containers():
    """Return list of (label, pk_field, container) tuples at call time."""
    return [
        (settings.COSMOS_CONTAINER,                "userId", cosmos_logger._container),
        (settings.COSMOS_PROFILES_CONTAINER,       "userId", cosmos_profile_store._container),
        (settings.COSMOS_ENROLLMENT_CONTAINER,     "userId", enrollment_store._container),
        (settings.COSMOS_PROTOTYPE_CONTAINER,      "userId", cosmos_prototype_store._proto_container),
        (settings.COSMOS_BEHAVIOUR_LOGS_CONTAINER, "userId", cosmos_prototype_store._logs_container),
    ]


def _query_all_containers() -> Dict[str, Any]:
    """
    Query every Cosmos container.
    Returns a dict mapping container_name → list[dict] of all documents.
    Nothing is written to disk here.
    """
    containers = _get_containers()
    logger.info("cosmos_dump: starting query across %d containers", len(containers))
    result: Dict[str, Any] = {}

    for container_name, pk_field, container in containers:
        if container is None:
            logger.warning("cosmos_dump: container '%s' is not connected – skipping", container_name)
            result[container_name] = None
            continue

        logger.info("cosmos_dump: querying container '%s' …", container_name)
        try:
            items = list(
                container.query_items(
                    query="SELECT * FROM c",
                    enable_cross_partition_query=True,
                )
            )
            logger.info(
                "cosmos_dump: container '%s' returned %d document(s)", container_name, len(items)
            )
            result[container_name] = {"pk_field": pk_field, "items": items}
        except Exception as exc:
            logger.error(
                "cosmos_dump: failed to query container '%s': %s", container_name, exc
            )
            result[container_name] = {"error": str(exc)}

    logger.info("cosmos_dump: all containers queried")
    return result


def _write_dump_to_disk(query_result: Dict[str, Any]) -> Dict[str, Any]:
    """
    Write pre-fetched query results to DUMP_ROOT/<container>/<pk_value>.json.
    Returns a per-container summary dict.
    """
    summary: Dict[str, Any] = {}
    logger.info("cosmos_dump: writing files to %s", DUMP_ROOT)

    for container_name, data in query_result.items():
        if data is None:
            summary[container_name] = {"skipped": "container not connected"}
            continue
        if "error" in data:
            summary[container_name] = {"error": data["error"]}
            continue

        pk_field = data["pk_field"]
        items = data["items"]
        container_dir = DUMP_ROOT / container_name
        container_dir.mkdir(parents=True, exist_ok=True)

        # Group by partition key value
        groups: Dict[str, list] = {}
        for item in items:
            pk_value = str(item.get(pk_field, "__unknown__"))
            groups.setdefault(pk_value, []).append(item)

        files_written = 0
        for pk_value, docs in groups.items():
            safe_name = "".join(
                ch if ch.isalnum() or ch in "-_." else "_" for ch in pk_value
            )
            out_path = container_dir / f"{safe_name}.json"
            out_path.write_text(json.dumps(docs, indent=2, default=str), encoding="utf-8")
            files_written += 1
            logger.debug(
                "cosmos_dump: wrote %s (%d doc(s))", out_path.relative_to(DUMP_ROOT.parent), len(docs)
            )

        logger.info(
            "cosmos_dump: container '%s' → %d partition file(s) written", container_name, files_written
        )
        summary[container_name] = {
            "total_documents": len(items),
            "partition_key_field": pk_field,
            "files_written": files_written,
            "output_dir": str(container_dir),
        }

    logger.info("cosmos_dump: local write complete")
    return summary


def _build_zip_from_query(query_result: Dict[str, Any]) -> bytes:
    """
    Build an in-memory zip from pre-fetched query results.
    Structure: cosmos_dump/<container>/<pk_value>.json
    Nothing is written to disk.
    """
    logger.info("cosmos_dump: building in-memory zip …")
    buf = io.BytesIO()
    total_files = 0

    with zipfile.ZipFile(buf, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
        for container_name, data in query_result.items():
            if data is None or "error" in data:
                logger.warning(
                    "cosmos_dump: skipping container '%s' in zip (%s)",
                    container_name,
                    "not connected" if data is None else data.get("error"),
                )
                continue

            pk_field = data["pk_field"]
            items = data["items"]

            groups: Dict[str, list] = {}
            for item in items:
                pk_value = str(item.get(pk_field, "__unknown__"))
                groups.setdefault(pk_value, []).append(item)

            for pk_value, docs in groups.items():
                safe_name = "".join(
                    ch if ch.isalnum() or ch in "-_." else "_" for ch in pk_value
                )
                arc_path = f"cosmos_dump/{container_name}/{safe_name}.json"
                zf.writestr(arc_path, json.dumps(docs, indent=2, default=str))
                total_files += 1
                logger.debug("cosmos_dump: zipped %s (%d doc(s))", arc_path, len(docs))

    zip_bytes = buf.getvalue()
    logger.info(
        "cosmos_dump: zip complete – %d file(s), %.1f KB", total_files, len(zip_bytes) / 1024
    )
    return zip_bytes


@app.post("/admin/cosmos-dump/download")
async def admin_cosmos_dump_download(authorization: Optional[str] = Header(default=None)):
    """
    **Production endpoint** – reads every document from every Cosmos DB
    container, packages everything into an in-memory zip, and streams it as
    ``cosmos_dump.zip`` directly to the caller's device.  Nothing is written
    to the server's disk.

    Zip structure::

        cosmos_dump/
          <container-name>/
            <partition-key-value>.json

    Requires ``Authorization: Bearer <ADMIN_TOKEN>`` header.
    """
    _verify_admin_token(authorization)
    logger.info("cosmos_dump/download: request received")

    query_result = await asyncio.to_thread(_query_all_containers)
    zip_bytes = await asyncio.to_thread(_build_zip_from_query, query_result)

    logger.info("cosmos_dump/download: streaming zip to client (%.1f KB)", len(zip_bytes) / 1024)
    return StreamingResponse(
        io.BytesIO(zip_bytes),
        media_type="application/zip",
        headers={"Content-Disposition": "attachment; filename=cosmos_dump.zip"},
    )


@app.post("/admin/cosmos-dump")
async def admin_cosmos_dump(authorization: Optional[str] = Header(default=None)):
    """
    **Debug endpoint** – reads every document from every Cosmos DB container
    and writes them to the server's local filesystem under::

        data/cosmos_dump/<container-name>/<partition-key-value>.json

    Each file is a JSON array of all documents sharing that partition-key
    value.  Existing files are overwritten on each call.  Nothing is streamed
    back to the caller beyond a JSON summary.

    Only available when ``DEBUG_MODE=True`` in settings.
    Requires ``Authorization: Bearer <ADMIN_TOKEN>`` header.
    """
    _verify_admin_token(authorization)
    if not settings.DEBUG_MODE:
        raise HTTPException(
            status_code=403,
            detail="Local-disk dump is only available in DEBUG_MODE. Use POST /admin/cosmos-dump/download instead.",
        )

    logger.info("cosmos_dump: local dump request received (DEBUG_MODE)")
    query_result = await asyncio.to_thread(_query_all_containers)
    summary = await asyncio.to_thread(_write_dump_to_disk, query_result)
    logger.info("cosmos_dump: local dump finished – summary: %s", summary)
    return {"dump_root": str(DUMP_ROOT), "containers": summary}
