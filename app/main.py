import json
import logging
import time
from pathlib import Path
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import ValidationError

from app.config import settings, configure_logging
from app.models import BehaviourMessage, ServerResponse
from app.websocket_manager import ConnectionManager

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
message_counter = 0


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


@app.get("/event-flow-map")
async def event_flow_map():
    return JSONResponse(content=EVENT_FLOW_MAP)


@app.websocket(settings.WEBSOCKET_ENDPOINT)
async def websocket_behaviour_endpoint(websocket: WebSocket):
    global message_counter
    await behaviour_manager.connect(websocket)
    client_id = id(websocket)
    logger.info(f"Client {client_id} connected to behaviour stream")

    try:
        while True:
            try:
                data = await websocket.receive_json()
                
                try:
                    behaviour_msg = BehaviourMessage(**data)
                    message_counter += 1
                    
                    user_id = behaviour_msg.user_id or "unknown"
                    session_id = behaviour_msg.session_id or "unknown"
                    payload = None
                    event_type = None
                    device_info = None

                    if isinstance(data, dict):
                        payload = data.get("payload") or data.get("event_data")
                        event_type = data.get("event_type")

                        if isinstance(payload, dict):
                            event_type = event_type or payload.get("eventType")
                            device_info = payload.get("deviceInfo")
                    
                    logger.info(
                        f"Received message #{message_counter} from user {user_id} "
                        f"(session: {session_id}, event: {event_type or 'unknown'}, client: {client_id})"
                    )
                    logger.info(f"Data received: {data}")

                    monitor_message = {
                        "receivedAt": time.time(),
                        "eventType": event_type,
                        "payload": payload,
                        "deviceInfo": device_info,
                        "signature": data.get("signature") if isinstance(data, dict) else None,
                        "raw": data,
                    }

                    try:
                        await monitor_manager.broadcast(monitor_message)
                    except Exception as e:
                        logger.error("Failed to broadcast to monitor clients: %s", e, exc_info=True)
                    
                    response = ServerResponse(
                        status="received",
                        server_timestamp=time.time(),
                        message_id=message_counter,
                    )
                    
                    await behaviour_manager.send_personal_message(response.dict(), websocket)
                    
                except ValidationError as e:
                    logger.warning(f"Validation error from client {client_id}: {e}")
                    error_response = {
                        "status": "error",
                        "message": "Invalid data format",
                        "errors": e.errors(),
                    }
                    await behaviour_manager.send_personal_message(error_response, websocket)
                    
            except ValueError as e:
                logger.error(f"JSON decode error from client {client_id}: {e}")
                error_response = {
                    "status": "error",
                    "message": "Malformed JSON",
                }
                await behaviour_manager.send_personal_message(error_response, websocket)
                
    except WebSocketDisconnect:
        behaviour_manager.disconnect(websocket)
        logger.info(f"Client {client_id} disconnected normally")
        
    except Exception as e:
        logger.error(f"Unexpected error with client {client_id}: {e}", exc_info=True)
        behaviour_manager.disconnect(websocket)


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
