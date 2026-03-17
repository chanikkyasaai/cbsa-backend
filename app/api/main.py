"""
app.api.main — FastAPI application entry point.

Re-exports the `app` FastAPI instance.

To start the server:
    uvicorn app.api.main:app --reload
    uvicorn app.main:app --reload
"""
from app.main import app  # noqa: F401

__all__ = ["app"]
