# CBSA Backend — Setup & Run Guide

## First-Time Setup (once)

```bash
cd cbsa-backend
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Starting a New Terminal Session

```bash
cd cbsa-backend
.\.venv\Scripts\Activate.ps1
# Prompt should show (.venv)
```

## Start the Backend (port 8000)

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

## GAT Inference Engine

The GAT engine runs in-process — no separate service is needed. PyTorch and
`torch_geometric` must be installed once:

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install torch_geometric
```

The engine initialises automatically at startup. If no trained checkpoint is
available it falls back to random-weight inference (useful during development).

## Useful Endpoints

| Purpose         | URL                                           |
|-----------------|-----------------------------------------------|
| REST API root   | http://localhost:8000                         |
| Health check    | http://localhost:8000/health                  |
| Event monitor   | http://localhost:8000/static/index.html       |
| Behaviour WS    | ws://localhost:8000/ws/behaviour              |
| Monitor WS      | ws://localhost:8000/ws/monitor                |

## Mobile App Configuration

In `cbsa-app`, set the backend IP to your machine's local IP (e.g. `192.168.x.x`)
on the login screen's configuration panel. Port: `8000`.
