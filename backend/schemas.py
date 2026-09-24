from datetime import datetime
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, ConfigDict, Field, computed_field
from uuid import UUID

# User schemas
class UserCreate(BaseModel):
    """Schema for creating a user."""
    username: str = Field(..., max_length=100)
    email: Optional[str] = Field(None, max_length=255)

class UserResponse(BaseModel):
    """Schema for user response."""
    id: UUID
    username: str
    email: Optional[str]
    created_at: datetime
    
    model_config = ConfigDict(from_attributes=True)

# Enrollment schemas
class EnrollRequest(BaseModel):
    """Schema for enrollment request."""
    user_id: Optional[str] = None

class EnrollResponse(BaseModel):
    """Schema for enrollment response."""
    user_id: Optional[str]
    sample_id: str
    embedding_size: int
    message: str

# Verification schemas
class VerifyRequest(BaseModel):
    """Schema for verification request."""
    pass

class VerifyResponse(BaseModel):
    """Schema for verification response."""
    user_id: Optional[str]
    similarity: float
    is_same_speaker: bool
    risk_score: float
    risk_level: str
    verdict: str
    confidence: float

# Detection schemas
class DetectionResultResponse(BaseModel):
    """Schema for detection result."""
    id: UUID
    session_id: str
    timestamp: datetime
    risk_score: float
    verdict: str
    risk_level: str
    features_json: Optional[Dict[str, Any]]
    audio_duration: Optional[float]
    model_used: Optional[str]
    
    @computed_field
    @property
    def risk_level_color(self) -> str:
        """Computed field for risk level color."""
        if self.risk_level.lower() == "low":
            return "green"
        elif self.risk_level.lower() == "medium":
            return "yellow"
        elif self.risk_level.lower() == "high":
            return "red"
        return "gray"

    model_config = ConfigDict(from_attributes=True)

class UploadResponse(DetectionResultResponse):
    """Schema for upload response."""
    filename: str
    file_size_bytes: int

class HistoryResponse(BaseModel):
    """Schema for history response."""
    results: List[DetectionResultResponse]
    total: int
    page: int
    limit: int
    pages: int

# WebSocket schemas
class WSMessage(BaseModel):
    """Schema for WebSocket message."""
    type: str
    session_id: str
    timestamp: datetime
    data: Dict[str, Any]

# System schemas
class ErrorResponse(BaseModel):
    """Schema for error response."""
    error: str
    detail: Optional[str] = None

class HealthResponse(BaseModel):
    """Schema for health check response."""
    status: str
    database: str
    models_loaded: bool
    version: str = "1.0.0"
