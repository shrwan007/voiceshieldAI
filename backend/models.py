from sqlalchemy import Column, String, Float, Integer, DateTime, Text, ForeignKey, JSON
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from sqlalchemy import text
from backend.database import Base

class User(Base):
    """User model."""
    __tablename__ = "users"

    id = Column(UUID(as_uuid=True), primary_key=True, server_default=text("gen_random_uuid()"))
    username = Column(String(100), unique=True, nullable=False)
    email = Column(String(255), unique=True, nullable=True)
    created_at = Column(DateTime, server_default=func.now())
    
    samples = relationship("VoiceSample", back_populates="user")
    detections = relationship("DetectionResult", back_populates="user")

class VoiceSample(Base):
    """Voice sample model for speaker enrollment."""
    __tablename__ = "voice_samples"

    id = Column(UUID(as_uuid=True), primary_key=True, server_default=text("gen_random_uuid()"))
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=True)
    file_path = Column(Text, nullable=False)
    embedding = Column(JSON, nullable=True)
    sample_rate = Column(Integer, default=16000)
    duration = Column(Float, nullable=True)
    created_at = Column(DateTime, server_default=func.now())
    
    user = relationship("User", back_populates="samples")

class DetectionResult(Base):
    """Detection result model."""
    __tablename__ = "detection_results"

    id = Column(UUID(as_uuid=True), primary_key=True, server_default=text("gen_random_uuid()"))
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=True)
    session_id = Column(String(255), nullable=False, index=True)
    timestamp = Column(DateTime, server_default=func.now(), index=True)
    risk_score = Column(Float, nullable=False)
    verdict = Column(String(50), nullable=False)  # "real", "synthetic", "cloned"
    risk_level = Column(String(20), nullable=False)  # "Low", "Medium", "High"
    features_json = Column(JSON, nullable=True)
    audio_duration = Column(Float, nullable=True)
    model_used = Column(String(100), nullable=True)
    
    user = relationship("User", back_populates="detections")
