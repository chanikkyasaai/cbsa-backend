from pydantic import BaseModel, Field
from typing import Dict, Any, Optional


class BehaviourMessage(BaseModel):
    user_id: Optional[str] = Field(None, description="Unique user identifier")
    timestamp: Optional[float] = Field(None, description="Unix timestamp")
    session_id: Optional[str] = Field(None, description="Session identifier")
    features: Optional[Dict[str, Any]] = Field(None, description="Flexible feature data")

    class Config:
        extra = "allow"


class ServerResponse(BaseModel):
    status: str
    server_timestamp: float
    message_id: int


class LoginRequest(BaseModel):
    username: str = Field(..., description="Username typed by the user")


class LoginResponse(BaseModel):
    username: str
    status: str  # "enrolling" | "enrolled"
    message: str
    seconds_remaining: Optional[float] = None
    accumulated_seconds: Optional[float] = None
    total_seconds: Optional[float] = None


class TrainRequest(BaseModel):
    user_id: Optional[str] = Field(None, description="Train for specific user; if None, train all users")
    force: bool = Field(False, description="Force retrain even if profile exists")
