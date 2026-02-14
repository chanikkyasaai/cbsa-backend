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
