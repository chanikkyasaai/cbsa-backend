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
- **Azure Cosmos DB** – Computation logs + user profile storage (separate containers)
- **Azure Blob Storage** – Model checkpoint (`.pth`) persistence
- **Admin endpoints** – Per-user data wipe and full system truncate

## Requirements

- Python 3.10 or higher
- pip

## Project Structure

```
backend/
│
├── app/
│   ├── __init__.py
│   ├── main.py                  # FastAPI application, WebSocket + admin endpoints
│   ├── config.py                # Application configuration & env vars
│   ├── websocket_manager.py     # Connection manager for multiple clients
│   ├── models/                  # Pydantic models for data validation
│   ├── cosmos_logger.py         # Cosmos DB computation-log writer
│   ├── cosmos_profile_store.py  # Cosmos DB user-profile store (64-D vectors)
│   ├── blob_model_store.py      # Azure Blob Storage for .pth model files
│   ├── triplet_trainer.py       # Triplet-loss training & profile creation
│   ├── gat_engine.py            # In-process GAT inference engine
│   ├── enrollment_store.py      # Enrollment state tracker
│   ├── behavioral_logger.py     # Per-user JSONL event logger
│   ├── layer3_manager.py        # GAT processing integration
│   ├── layer3_cloud.py          # GAT cloud interface & profile manager
│   ├── layer3_processor.py      # GAT data/result processors
│   ├── layer3_models.py         # Layer 3 Pydantic models
│   ├── engine/                  # Core engine (ingestion, preprocessing, similarity)
│   ├── gat/                     # GAT neural network components
│   └── storage/                 # SQLite store, memory store
│
├── data/
│   ├── profiles/                # Local-disk user profiles (fallback)
│   ├── behavioral_logs/         # Per-user JSONL behavioral logs
│   └── checkpoints/             # Local-disk model checkpoints (fallback)
│
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
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
- `DEBUG_MODE`: **`True`** (default) for development, **`False`** for production

#### `DEBUG_MODE` behaviour

| Area | `DEBUG_MODE = True` (dev) | `DEBUG_MODE = False` (prod) |
|---|---|---|
| User profiles (64-D) | Cosmos DB **+** local `data/profiles/` | **Cosmos DB only** – no local files |
| Model checkpoints (`.pth`) | Blob Storage **+** local `data/checkpoints/` | **Blob Storage only** – local disk is not read or written |
| GAT model loading | Blob → local fallback | **Blob only** |

> **Important:** In production, set `DEBUG_MODE=False` and make sure
> `COSMOS_ENDPOINT`/`COSMOS_KEY` and `AZURE_STORAGE_CONNECTION_STRING` are
> configured. Without them the profile and model stores will be completely
> disabled.

### Environment Variables

The following environment variables configure Azure integrations. All are
**optional** – if absent, the corresponding feature is silently disabled.
In development (`DEBUG_MODE=True`) the app falls back to local-disk storage;
in production (`DEBUG_MODE=False`) **only** Azure stores are used.

| Variable | Required for | Default | Description |
|---|---|---|---|
| `COSMOS_ENDPOINT` | Cosmos DB | *(empty – disabled)* | Azure Cosmos DB account URI |
| `COSMOS_KEY` | Cosmos DB | *(empty – disabled)* | Cosmos DB primary or secondary key |
| `COSMOS_DATABASE` | Cosmos DB | `cbsa-logs` | Database name (shared by all containers) |
| `COSMOS_CONTAINER` | Computation logs | `computation-logs` | Container for per-event computation logs |
| `COSMOS_PROFILES_CONTAINER` | User profiles | `user-profiles` | **Separate** container for 64-D user baseline profiles |
| `AZURE_STORAGE_CONNECTION_STRING` | Blob Storage | *(empty – disabled)* | Full connection string for the Storage Account |
| `AZURE_STORAGE_CONTAINER` | Blob Storage | `cbsa-models` | Blob container for `.pth` model checkpoints |

> **Tip:** In Azure App Service, set these as *Application settings* so they're injected as environment variables at runtime.

### Azure Setup Steps (you need to do this)

#### 1. Create a Cosmos DB Account (if you don't have one)

```bash
# Create a Cosmos DB account (SQL / NoSQL API)
az cosmosdb create \
  --name <YOUR_COSMOSDB_ACCOUNT> \
  --resource-group cbsa-rg \
  --kind GlobalDocumentDB \
  --default-consistency-level Session

# Get the primary key
az cosmosdb keys list \
  --name <YOUR_COSMOSDB_ACCOUNT> \
  --resource-group cbsa-rg \
  --query primaryMasterKey -o tsv
```

The database and containers are auto-created on first run. Two containers will
be created inside the same database (`cbsa-logs` by default):

| Container | Partition Key | Purpose |
|---|---|---|
| `computation-logs` | `/userId` | Per-event engine metrics + GAT scores |
| `user-profiles` | `/userId` | 64-D baseline profile vectors |

#### 2. Create an Azure Storage Account + Blob Container

```bash
# Create a storage account
az storage account create \
  --name <YOUR_STORAGE_ACCOUNT> \
  --resource-group cbsa-rg \
  --location eastus \
  --sku Standard_LRS

# Get the connection string
az storage account show-connection-string \
  --name <YOUR_STORAGE_ACCOUNT> \
  --resource-group cbsa-rg \
  --query connectionString -o tsv
```

The blob container (`cbsa-models` by default) is auto-created on first run.
It stores `.pth` model checkpoint files. On app startup, the GAT engine tries
to download the latest checkpoint from Blob Storage; if unavailable it falls
back to local files or random initialisation.

#### 3. Set the Environment Variables

For **local development**, create a `.env` file (already in `.gitignore`):

```bash
# .env
COSMOS_ENDPOINT=https://<YOUR_COSMOSDB_ACCOUNT>.documents.azure.com:443/
COSMOS_KEY=<YOUR_COSMOS_PRIMARY_KEY>
COSMOS_DATABASE=cbsa-logs
COSMOS_CONTAINER=computation-logs
COSMOS_PROFILES_CONTAINER=user-profiles
AZURE_STORAGE_CONNECTION_STRING=DefaultEndpointsProtocol=https;AccountName=...
AZURE_STORAGE_CONTAINER=cbsa-models
```

For **Azure App Service**, add the same values as Application settings:

```bash
az webapp config appsettings set \
  --resource-group cbsa-rg \
  --name <YOUR_WEBAPP_NAME> \
  --settings \
    COSMOS_ENDPOINT="https://<ACCOUNT>.documents.azure.com:443/" \
    COSMOS_KEY="<KEY>" \
    COSMOS_DATABASE="cbsa-logs" \
    COSMOS_CONTAINER="computation-logs" \
    COSMOS_PROFILES_CONTAINER="user-profiles" \
    AZURE_STORAGE_CONNECTION_STRING="<CONN_STRING>" \
    AZURE_STORAGE_CONTAINER="cbsa-models"
```

For **GitHub Actions**, add these as repository secrets and reference them in your workflow:

| Secret | Value |
|---|---|
| `COSMOS_ENDPOINT` | Cosmos DB URI |
| `COSMOS_KEY` | Cosmos DB key |
| `AZURE_STORAGE_CONNECTION_STRING` | Storage Account connection string |

## Admin Endpoints

### `DELETE /admin/user/{user_id}`

Deletes **all** data associated with a single user:

- SQLite user row, prototypes, behaviour_logs
- Behavioral log file (`.jsonl`)
- User profile (Cosmos DB `user-profiles` container + local disk)
- Enrollment state
- In-memory GAT session windows
- Cosmos DB computation logs for that user

```bash
curl -X DELETE http://localhost:8000/admin/user/alice
```

### `DELETE /admin/truncate`

Deletes **all** data for every user across the entire system:

- SQLite tables (users, prototypes, behaviour_logs)
- All behavioral log files
- All user profiles (Cosmos DB + local disk)
- Enrollment store
- All in-memory GAT session windows
- All Cosmos DB computation logs
- All model checkpoints (Blob Storage + local disk)

```bash
curl -X DELETE http://localhost:8000/admin/truncate
```

> ⚠️ **These endpoints are destructive.** In production, protect them with
> authentication middleware or remove them entirely.

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

## Local Verification with Docker Compose

Before merging or touching any Azure secrets, you can verify that both containers build and start correctly on your machine using Docker Compose:

```bash
# Build both images and start the services in the foreground (Ctrl-C to stop)
docker compose up --build
```

| Service | URL |
|---|---|
| Main backend | <http://localhost:8000/health> |
| GAT service  | <http://localhost:8001/health> |

Both endpoints should return a JSON health response, confirming the images build correctly and the apps listen on the expected ports.

To run in the background:

```bash
docker compose up --build -d
# check logs for a specific service
docker compose logs -f gat-service
# tear down
docker compose down
```

Once both `/health` endpoints respond successfully you can be confident the Dockerfiles are correct, and you can proceed to merge and configure the Azure secrets to deploy.

## Deploying to Azure

This repository includes a `Dockerfile` and a GitHub Actions workflow (`.github/workflows/azure-deploy.yml`) to deploy the backend to **Azure App Service for Containers**.

### Prerequisites

1. **Azure CLI** installed locally (`az` command).
2. An **Azure subscription**.
3. A **resource group**, **Azure Container Registry (ACR)**, and an **Azure App Service** (Linux, Docker container) created in your subscription.

### One-time Azure Setup

```bash
# 1. Create a resource group
az group create --name cbsa-rg --location eastus

# 2. Create an Azure Container Registry
az acr create --resource-group cbsa-rg --name <YOUR_ACR_NAME> --sku Basic --admin-enabled true

# 3. Create an App Service plan (Linux)
az appservice plan create --name cbsa-plan --resource-group cbsa-rg --is-linux --sku B1

# 4a. Create the main backend Web App (port 8000)
az webapp create \
  --resource-group cbsa-rg \
  --plan cbsa-plan \
  --name <YOUR_WEBAPP_NAME> \
  --deployment-container-image-name <YOUR_ACR_NAME>.azurecr.io/cbsa-backend:latest

# 4b. Create the GAT service Web App (port 8001)
az webapp create \
  --resource-group cbsa-rg \
  --plan cbsa-plan \
  --name <YOUR_GAT_WEBAPP_NAME> \
  --deployment-container-image-name <YOUR_ACR_NAME>.azurecr.io/cbsa-gat-service:latest

# Tell Azure which port the GAT service container listens on
az webapp config appsettings set \
  --resource-group cbsa-rg \
  --name <YOUR_GAT_WEBAPP_NAME> \
  --settings WEBSITES_PORT=8001

# 5. Configure both web apps to pull from ACR
az webapp config container set \
  --name <YOUR_WEBAPP_NAME> \
  --resource-group cbsa-rg \
  --docker-custom-image-name <YOUR_ACR_NAME>.azurecr.io/cbsa-backend:latest \
  --docker-registry-server-url https://<YOUR_ACR_NAME>.azurecr.io \
  --docker-registry-server-user $(az acr credential show --name <YOUR_ACR_NAME> --query username -o tsv) \
  --docker-registry-server-password $(az acr credential show --name <YOUR_ACR_NAME> --query passwords[0].value -o tsv)

az webapp config container set \
  --name <YOUR_GAT_WEBAPP_NAME> \
  --resource-group cbsa-rg \
  --docker-custom-image-name <YOUR_ACR_NAME>.azurecr.io/cbsa-gat-service:latest \
  --docker-registry-server-url https://<YOUR_ACR_NAME>.azurecr.io \
  --docker-registry-server-user $(az acr credential show --name <YOUR_ACR_NAME> --query username -o tsv) \
  --docker-registry-server-password $(az acr credential show --name <YOUR_ACR_NAME> --query passwords[0].value -o tsv)

# 6. Create a service principal for GitHub Actions
az ad sp create-for-rbac \
  --name cbsa-github-actions \
  --role contributor \
  --scopes /subscriptions/<YOUR_SUBSCRIPTION_ID>/resourceGroups/cbsa-rg \
  --sdk-auth
```

### GitHub Secrets

Add the following secrets to your GitHub repository (**Settings → Secrets and variables → Actions**):

| Secret | Used by | Value |
|---|---|---|
| `AZURE_CREDENTIALS` | both services | Full JSON output from the `az ad sp create-for-rbac` command above |
| `REGISTRY_LOGIN_SERVER` | both services | `<YOUR_ACR_NAME>.azurecr.io` |
| `REGISTRY_USERNAME` | both services | ACR admin username (from `az acr credential show`) |
| `REGISTRY_PASSWORD` | both services | ACR admin password (from `az acr credential show`) |
| `AZURE_WEBAPP_NAME` | main backend | The name you chose for the main backend Web App |
| `AZURE_GAT_WEBAPP_NAME` | GAT service | The name you chose for the GAT service Web App (port 8001) |

### CI/CD Pipeline

Once the secrets are configured, every push to the `main` branch will automatically run two parallel jobs:

1. **`build-and-deploy`** — builds the main backend image → pushes to ACR → deploys to `AZURE_WEBAPP_NAME`.
2. **`build-and-deploy-gat`** — builds the GAT service image from `gat-service/` → pushes to ACR → deploys to `AZURE_GAT_WEBAPP_NAME`.

You can also trigger a deployment manually from **Actions → Build and Deploy to Azure App Service → Run workflow**.

### Manual Deployment (without CI/CD)

```bash
# Log in to ACR
az acr login --name <YOUR_ACR_NAME>

# --- Main backend ---
docker build -t <YOUR_ACR_NAME>.azurecr.io/cbsa-backend:latest .
docker push <YOUR_ACR_NAME>.azurecr.io/cbsa-backend:latest
az webapp restart --name <YOUR_WEBAPP_NAME> --resource-group cbsa-rg

# --- GAT service ---
docker build -t <YOUR_ACR_NAME>.azurecr.io/cbsa-gat-service:latest ./gat-service
docker push <YOUR_ACR_NAME>.azurecr.io/cbsa-gat-service:latest
az webapp restart --name <YOUR_GAT_WEBAPP_NAME> --resource-group cbsa-rg
```

After deployment, the service is available at:

```
# Main backend
https://<YOUR_WEBAPP_NAME>.azurewebsites.net

# GAT service
https://<YOUR_GAT_WEBAPP_NAME>.azurewebsites.net
```

WebSocket connections use `wss://` automatically via the Azure App Service TLS termination:

```
wss://<YOUR_WEBAPP_NAME>.azurewebsites.net/ws/behaviour
```

## Architecture Notes

This backend serves as the **real-time ingestion layer** for a multi-layer behavioral authentication system. It is designed to be:

- **Lightweight**: No database, no ML, no unnecessary complexity
- **Stable**: Comprehensive error handling and graceful disconnection
- **Scalable**: Supports multiple concurrent clients efficiently
- **Modular**: Clean separation of concerns for easy maintenance

## License

MIT
