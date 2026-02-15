# CBSA Backend - Continuous Behavioral Streaming API

A lightweight, production-ready FastAPI backend for real-time behavioral data streaming from mobile applications.


## Features

- **WebSocket-based streaming**: Real-time continuous data ingestion
- **Multi-client support**: Handle multiple simultaneous connections
- **Flexible data ingestion**: Accepts any JSON structure, no strict schema enforced
- **Live monitoring UI**: Web page that visualizes event flow traversal
- **Structured logging**: Comprehensive logging for monitoring
- **Error handling**: Graceful handling of malformed data and disconnections
- **Production-ready**: Clean, modular, and scalable architecture

## Requirements

- Python 3.10 or higher
- pip

## Project Structure

```
backend/
│
├── app/
│   ├── __init__.py
│   ├── main.py                 # FastAPI application and WebSocket endpoint
│   ├── websocket_manager.py    # Connection manager for multiple clients
│   ├── models.py               # Pydantic models for data validation
│   └── config.py               # Application configuration
│
├── requirements.txt
└── README.md
```

## Setup Instructions

### 1. Create Virtual Environment

```bash
python -m venv venv
```

### 2. Activate Virtual Environment

**Windows:**
```bash
venv\Scripts\activate
```

**macOS/Linux:**
```bash
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

## Running the Server

Start the server with auto-reload enabled:

```bash
uvicorn app.main:app --reload
```

For production deployment:

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

The server will start on `http://localhost:8000`

Open `http://localhost:8000` to view the live event flow monitor.

## API Endpoints

### HTTP Endpoints

#### `GET /`
Serves the live event flow monitoring web page.

#### `GET /health`
Returns server health status and active connection count.

#### `GET /event-flow-map`
Serves the event flow map JSON used by the monitoring UI.

### WebSocket Endpoint

#### `WS /ws/behaviour`
WebSocket endpoint for continuous behavioral data streaming.

#### `WS /ws/monitor`
WebSocket endpoint used by the web UI to receive live events.


## Data Format

### Client Message Format

You can send **any valid JSON object**. There is no enforced schema. Example messages:

```json
{"user_id": "user123", "timestamp": 1708041234.567, "session_id": "session_abc_123", "features": {"touch_velocity": 45.2}}
```

```json
{"foo": "bar", "arbitrary": [1,2,3], "nested": {"a": 1}}
```

```json
{"anything": "goes", "number": 42}
```


### Server Response Format

The server acknowledges each message with:

```json
{
  "status": "received",
  "server_timestamp": 1708041234.890,
  "message_id": 1
}
```

### Error Response Format

If malformed JSON is received:

```json
{
  "status": "error",
  "message": "Malformed JSON"
}
```

## Testing the WebSocket Connection


### Using wscat

Install wscat globally:

```bash
npm install -g wscat
```

Connect to the WebSocket endpoint:

```bash
wscat -c ws://localhost:8000/ws/behaviour
```

Send any JSON message, for example:

```json
{"foo": "bar", "random": 123}
```

### Using Python

```python
import asyncio
import websockets
import json
import time

async def test_websocket():
    uri = "ws://localhost:8000/ws/behaviour"
    
    async with websockets.connect(uri) as websocket:

        message = {"foo": "bar", "random": 123}
        await websocket.send(json.dumps(message))
        response = await websocket.recv()
        print(f"Response: {response}")

asyncio.run(test_websocket())
```

### Using JavaScript/React Native

```javascript
const ws = new WebSocket('ws://localhost:8000/ws/behaviour');

ws.onopen = () => {
  console.log('Connected to server');
  

  const message = {foo: 'bar', random: 123};
  ws.send(JSON.stringify(message));
};

ws.onmessage = (event) => {
  console.log('Server response:', JSON.parse(event.data));
};

ws.onerror = (error) => {
  console.error('WebSocket error:', error);
};

ws.onclose = () => {
  console.log('Disconnected from server');
};
```

## Configuration

Edit `app/config.py` to modify server settings:

- `APP_NAME`: Application name
- `VERSION`: Application version
- `HOST`: Server host (default: 0.0.0.0)
- `PORT`: Server port (default: 8000)
- `LOG_LEVEL`: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
- `WEBSOCKET_ENDPOINT`: WebSocket route path

## Logging

The application uses structured logging with the following format:

```
2026-02-14 10:30:45 - app.main - INFO - Client 123456 connected to behaviour stream
2026-02-14 10:30:46 - app.main - INFO - Received message #1 from user user123 (session: session_abc, client: 123456)
```

## Production Deployment

For production deployment, consider:

1. **Use a process manager** like systemd or supervisord
2. **Configure reverse proxy** (nginx or traefik)
3. **Enable HTTPS/WSS** for secure connections
4. **Set appropriate log levels** (INFO or WARNING)
5. **Monitor active connections** via the `/health` endpoint
6. **Configure uvicorn workers** for better performance:

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers 4
```

## Architecture Notes

This backend serves as the **real-time ingestion layer** for a multi-layer behavioral authentication system. It is designed to be:

- **Lightweight**: No database, no ML, no unnecessary complexity
- **Stable**: Comprehensive error handling and graceful disconnection
- **Scalable**: Supports multiple concurrent clients efficiently
- **Modular**: Clean separation of concerns for easy maintenance

## License

MIT
