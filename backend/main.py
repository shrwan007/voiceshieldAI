"""
VoiceShield FastAPI Application
Entry point for the backend server.

Startup sequence:
1. Load settings (config.yaml + .env)
2. Initialize database (create tables via SQLAlchemy)
3. Pre-load ML models (detector + verifier) in background
4. Start queue worker
5. Register all API routers
6. Configure CORS
"""

from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.staticfiles import StaticFiles
import logging
import os

# Import all routers
from backend.routers import upload, stream, enroll, verify, history
from backend.database import init_db, check_db_connection
from backend.config import get_settings
from backend.services.queue_worker import worker_queue
from backend.pipeline.stage4_detection import get_detector
from backend.pipeline.stage5_verify import get_verifier
from backend.services.websocket_manager import manager
from backend.schemas import HealthResponse

# Logging configuration
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("voiceshield")

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan handler.
    Runs setup on startup and cleanup on shutdown.
    """
    settings = get_settings()
    
    # Startup
    logger.info("Starting VoiceShield...")
    
    # Create storage directories
    os.makedirs(settings.audio_storage_path, exist_ok=True)
    os.makedirs(settings.models_cache_path, exist_ok=True)
    
    # Initialize database
    await init_db()
    logger.info("Database initialized")
    
    # Pre-load models (non-blocking — models lazy-load on first inference)
    logger.info("Models will lazy-load on first inference")
    get_detector()
    get_verifier()
    
    # Start queue worker
    await worker_queue.start()
    logger.info("Processing queue started")
    
    yield
    
    # Shutdown
    await worker_queue.stop()
    logger.info("VoiceShield shutdown complete")

# Create FastAPI app
app = FastAPI(
    title="VoiceShield API",
    description="Real-time AI voice fraud and deepfake detection system",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan
)

settings = get_settings()

# Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.add_middleware(GZipMiddleware, minimum_size=1000)

# Include routers
app.include_router(upload.router)
app.include_router(stream.router)
app.include_router(enroll.router)
app.include_router(verify.router)
app.include_router(history.router)

# Health check endpoint
@app.get("/health", response_model=HealthResponse, tags=["health"])
async def health_check():
    """
    System health check.
    Returns database connectivity and model load status.
    """
    db_ok = await check_db_connection()
    return HealthResponse(
        status="healthy" if db_ok else "degraded",
        database="connected" if db_ok else "disconnected",
        models_loaded=True,  # lazy-loaded
        version="1.0.0"
    )

# Root redirect
@app.get("/", include_in_schema=False)
async def root():
    return {"message": "VoiceShield API v1.0.0", "docs": "/docs"}

# Active sessions info
@app.get("/api/sessions", tags=["monitoring"])
async def get_active_sessions():
    """
    Get count and IDs of active WebSocket streaming sessions.
    """
    return {
        "active_sessions": manager.session_count(),
        "session_ids": manager.get_active_sessions()
    }
