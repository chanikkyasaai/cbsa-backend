# CBSA Backend API Specification

## Base URL
```
http://localhost:8000
```
*(Replace with actual backend URL in production)*

---

## REST Endpoints

### 1. GET `/`
**Description**: Root endpoint - get server information

**Request**: None

**Response**:
```json
{
  "message": "Backend running",
  "app": "CBSA Backend",
  "version": "1.0.0",
  "active_connections": 0
}
```

---

### 2. GET `/health`
**Description**: Health check endpoint

**Request**: None

**Response**:
```json
{
  "status": "healthy",
  "active_connections": 0
}
```

---

## WebSocket Endpoint

### 3. WebSocket `/ws/behaviour`
**Description**: Persistent WebSocket connection for streaming behavioral data

**Connection URL**: `ws://localhost:8000/ws/behaviour`

**Outgoing Message (Client → Server)**:
```json
{
  "user_id": "string (optional)",
  "session_id": "string (optional)",
  "timestamp": 1708012800.123,
  "event_type": "string (required)",
  "event_data": {}
}
```

**Success Response (Server → Client)**:
```json
{
  "status": "received",
  "server_timestamp": 1708012800.456,
  "message_id": 1
}
```

**Error Response (Server → Client)**:
```json
{
  "status": "error",
  "message": "Invalid data format",
  "errors": []
}
```

---

## Notes
- WebSocket connection must be established and kept alive
- All WebSocket messages must be valid JSON strings
- `timestamp` should be Unix timestamp in seconds (not milliseconds)
- Use `Date.now() / 1000` in JavaScript to get proper timestamp format
