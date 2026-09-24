import os
from functools import lru_cache
from typing import List, Optional
import yaml
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field

def load_yaml_config(file_path: str = "config.yaml") -> dict:
    """Load configuration from a YAML file."""
    if not os.path.exists(file_path):
        return {}
    with open(file_path, "r") as f:
        try:
            return yaml.safe_load(f) or {}
        except Exception:
            return {}

class Settings(BaseSettings):
    """Application settings, loaded from env vars, .env file, and config.yaml."""
    # Database
    database_url: str = Field(..., env="DATABASE_URL")
    
    # Application
    secret_key: str = Field(..., env="SECRET_KEY")
    debug: bool = True
    
    # Audio
    sample_rate: int = 16000
    silence_threshold_db: float = -40.0
    min_audio_length_sec: float = 1.0
    max_audio_length_sec: float = 30.0
    chunk_size_bytes: int = 4096
    
    # Risk Thresholds
    low_threshold: float = 0.35
    medium_threshold: float = 0.65
    high_threshold: float = 0.85
    
    # Models
    detection_model: str = "speechbrain/aasist-spoof-detection"
    speaker_model: str = "speechbrain/spkrec-ecapa-voxceleb"
    device: str = "cpu"
    speaker_similarity_threshold: float = 0.75
    
    # Storage
    audio_storage_path: str = "audio_storage"
    models_cache_path: str = "models_cache"
    max_file_size_mb: int = 50
    
    # API
    history_default_limit: int = 20
    cors_origins: List[str] = ["http://localhost:5173", "http://localhost:3000"]
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Override with yaml config if available
        yaml_data = load_yaml_config()
        if "audio" in yaml_data:
            self.sample_rate = yaml_data["audio"].get("sample_rate", self.sample_rate)
            self.silence_threshold_db = yaml_data["audio"].get("silence_threshold_db", self.silence_threshold_db)
            self.min_audio_length_sec = yaml_data["audio"].get("min_audio_length_sec", self.min_audio_length_sec)
            self.max_audio_length_sec = yaml_data["audio"].get("max_audio_length_sec", self.max_audio_length_sec)
            self.chunk_size_bytes = yaml_data["audio"].get("chunk_size_bytes", self.chunk_size_bytes)
            
        if "risk" in yaml_data:
            self.low_threshold = yaml_data["risk"].get("low_threshold", self.low_threshold)
            self.medium_threshold = yaml_data["risk"].get("medium_threshold", self.medium_threshold)
            self.high_threshold = yaml_data["risk"].get("high_threshold", self.high_threshold)
            
        if "model" in yaml_data:
            self.detection_model = yaml_data["model"].get("detection_model", self.detection_model)
            self.speaker_model = yaml_data["model"].get("speaker_model", self.speaker_model)
            self.device = yaml_data["model"].get("device", self.device)
            self.speaker_similarity_threshold = yaml_data["model"].get("speaker_similarity_threshold", self.speaker_similarity_threshold)
            
        if "storage" in yaml_data:
            self.audio_storage_path = yaml_data["storage"].get("audio_storage_path", self.audio_storage_path)
            self.models_cache_path = yaml_data["storage"].get("models_cache_path", self.models_cache_path)
            self.max_file_size_mb = yaml_data["storage"].get("max_file_size_mb", self.max_file_size_mb)
            
        if "api" in yaml_data:
            self.history_default_limit = yaml_data["api"].get("history_default_limit", self.history_default_limit)
            self.cors_origins = yaml_data["api"].get("cors_origins", self.cors_origins)

@lru_cache()
def get_settings() -> Settings:
    """Return a cached Settings instance."""
    # We might need to dummy env vars if running locally without .env for testing
    import os
    if not os.getenv("DATABASE_URL"):
        os.environ["DATABASE_URL"] = "postgresql+asyncpg://voiceshield:voiceshield_pass@localhost:5432/voiceshield_db"
    if not os.getenv("SECRET_KEY"):
        os.environ["SECRET_KEY"] = "default-secret-key-min-32-chars-long"
    return Settings()

settings = get_settings()
