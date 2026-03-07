import json
import logging
import time
import asyncio
from pathlib import Path
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import ValidationError
from typing import List, Dict, Any

from app.config import settings, configure_logging
from app.engine.ingestion import validate_and_extract
from app.engine.preprocessing import process_event
from app.engine.prototype_engine import compute_prototype_metrics
from app.storage.sqlite_store import SQLiteStore, DB_PATH
from app.models import BehaviourMessage, ServerResponse, LoginRequest, LoginResponse, TrainRequest
from app.websocket_manager import ConnectionManager
from app.layer3_manager import Layer3GATManager
from app.enrollment_store import enrollment_store, ENROLLMENT_DURATION_SECONDS
from app.behavioral_logger import behavioral_logger
from app.triplet_trainer import triplet_trainer
from app.cosmos_logger import cosmos_logger

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


@app.on_event("startup")
async def startup_event() -> None:
    app.state.sqlite_store = SQLiteStore(DB_PATH)


async def simulate_layer2_decision(session_id: str, behaviour_msg: BehaviourMessage) -> bool:
    """
    Simulate Layer 2 consistency engine decision for escalation
    In real implementation, this would be replaced by actual Layer 2 logic
    """
    import random
    
    # Simulate escalation based on various factors
    event_type = getattr(behaviour_msg, 'event_type', 'unknown')
    
    # Higher escalation probability for certain event patterns
    escalation_triggers = [
        "TOUCH_BALANCE_TOGGLE",  # Financial operations
        "TOUCH_TRANSACTION_HISTORY",  # Sensitive data access
        "PAGE_ENTER_MORE"  # Navigation patterns
    ]
    
    # Check session window size - escalate every ~10-15 events for demonstration
    window_size = len(gat_manager.get_session_window(session_id))
    
    if event_type in escalation_triggers:
        # 30% chance of escalation for sensitive events
        return random.random() < 0.3
    elif window_size > 0 and window_size % 12 == 0:
        # Escalate every 12th event for demo purposes
        return True
    else:
        # 5% baseline escalation rate
        return random.random() < 0.05


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
    return JSONResponse(
        content={
            "status": "healthy",
            "active_connections": behaviour_manager.get_connection_count(),
            "monitor_connections": monitor_manager.get_connection_count(),
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
async def train_profile(request: TrainRequest):
    """
    Trigger triplet training for a user (or all users).
    After training, a profile is stored in data/profiles/<user_id>_profile.json.
    Set force=True to retrain even if a profile already exists.
    """
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
                    event = validate_and_extract(data)
                    preprocessed = process_event(event)

                    try:
                        await asyncio.to_thread(
                            app.state.sqlite_store.insert_behaviour_log,
                            event.user_id,
                            event.session_id,
                            event.timestamp,
                            event.event_type,
                            preprocessed.window_vector,
                            preprocessed.short_drift,
                            preprocessed.long_drift,
                            preprocessed.stability_score,
                        )
                    except Exception as log_error:
                        logger.error("Failed to persist behaviour log: %s", log_error, exc_info=True)

                    warmup_state = await asyncio.to_thread(
                        app.state.sqlite_store.collect_warmup_window,
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
                    gat_manager.add_event_to_session(session_id, behaviour_msg)

                    should_escalate = await simulate_layer2_decision(session_id, behaviour_msg)

                    gat_similarity = None
                    gat_result: Dict[str, Any] = {}
                    if should_escalate:
                        session_window = gat_manager.get_session_window(session_id)
                        if len(session_window) >= 5:
                            logger.info(f"Escalating session {session_id} to Layer 3 GAT processing")
                            gat_result = await gat_manager.process_escalated_session(session_id, session_window)
                            gat_similarity = gat_result.get("similarity_score")

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
                        metrics = await asyncio.to_thread(
                            compute_prototype_metrics,
                            app.state.sqlite_store,
                            event.user_id,
                            preprocessed,
                        )
                        engine_metrics = {
                            "similarityScore": metrics.similarity_score,
                            "shortDrift": metrics.short_drift,
                            "longDrift": metrics.long_drift,
                            "stabilityScore": metrics.stability_score,
                            "matchedPrototypeId": metrics.matched_prototype_id,
                        }
                        response = {
                            "similarity_score": metrics.similarity_score,
                            "short_drift": metrics.short_drift,
                            "long_drift": metrics.long_drift,
                            "stability_score": metrics.stability_score,
                            "matched_prototype_id": metrics.matched_prototype_id,
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
        logger.info(f"Client {client_id} disconnected normally")
        
    except Exception as e:
        logger.error(f"Unexpected error with client {client_id}: {e}", exc_info=True)
        behaviour_manager.disconnect(websocket)
        if ws_user_id:
            enrollment_store.end_session(ws_user_id)


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
