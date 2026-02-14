import logging
import time
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse
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

manager = ConnectionManager()
message_counter = 0


@app.get("/")
async def root():
    return JSONResponse(
        content={
            "message": "Backend running",
            "app": settings.APP_NAME,
            "version": settings.VERSION,
            "active_connections": manager.get_connection_count(),
        }
    )


@app.get("/health")
async def health():
    return JSONResponse(
        content={
            "status": "healthy",
            "active_connections": manager.get_connection_count(),
        }
    )


@app.websocket(settings.WEBSOCKET_ENDPOINT)
async def websocket_behaviour_endpoint(websocket: WebSocket):
    global message_counter
    await manager.connect(websocket)
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
                    
                    logger.info(
                        f"Received message #{message_counter} from user {user_id} "
                        f"(session: {session_id}, client: {client_id})"
                    )
                    logger.info(f"Data received: {data}")
                    
                    response = ServerResponse(
                        status="received",
                        server_timestamp=time.time(),
                        message_id=message_counter,
                    )
                    
                    await manager.send_personal_message(response.dict(), websocket)
                    
                except ValidationError as e:
                    logger.warning(f"Validation error from client {client_id}: {e}")
                    error_response = {
                        "status": "error",
                        "message": "Invalid data format",
                        "errors": e.errors(),
                    }
                    await manager.send_personal_message(error_response, websocket)
                    
            except ValueError as e:
                logger.error(f"JSON decode error from client {client_id}: {e}")
                error_response = {
                    "status": "error",
                    "message": "Malformed JSON",
                }
                await manager.send_personal_message(error_response, websocket)
                
    except WebSocketDisconnect:
        manager.disconnect(websocket)
        logger.info(f"Client {client_id} disconnected normally")
        
    except Exception as e:
        logger.error(f"Unexpected error with client {client_id}: {e}", exc_info=True)
        manager.disconnect(websocket)
