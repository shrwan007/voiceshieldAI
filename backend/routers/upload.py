"""
POST /api/upload — Audio File Upload Endpoint
Accepts an audio file, runs the full 6-stage pipeline, persists the result,
and returns the detection verdict + risk score.
"""

from fastapi import APIRouter, File, UploadFile, Form, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from typing import Optional
import uuid

from backend.config import get_settings
from backend.database import get_db
from backend.models import VoiceSample, DetectionResult
from backend.schemas import UploadResponse, DetectionResultResponse
from backend.pipeline import run_full_pipeline
from backend.services.queue_worker import worker_queue

router = APIRouter(prefix="/api", tags=["upload"])


@router.post("/upload", response_model=DetectionResultResponse)
async def upload_audio(
    file: UploadFile = File(...),
    user_id: Optional[str] = Form(None),
    session_id: Optional[str] = Form(None),
    db: AsyncSession = Depends(get_db),
    settings=Depends(get_settings),
):
    """
    Upload an audio file for deepfake detection.

    Accepts: WAV, MP3, FLAC, OGG, M4A
    Max size: 50MB

    Returns the detection result with risk score and verdict.
    If user_id is provided and that user has enrolled samples, also runs speaker verification.
    """
    file_bytes = await file.read()
    file_size_bytes = len(file_bytes)

    if file_size_bytes == 0:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    if file_size_bytes > settings.max_file_size_mb * 1024 * 1024:
        raise HTTPException(
            status_code=400,
            detail=f"File too large. Maximum size is {settings.max_file_size_mb}MB."
        )

    # Fetch enrolled speaker embedding if user_id provided
    enrolled_embedding = None
    if user_id:
        try:
            result = await db.execute(
                select(VoiceSample)
                .where(VoiceSample.user_id == user_id)
                .order_by(VoiceSample.created_at.desc())
                .limit(1)
            )
            sample = result.scalar_one_or_none()
            if sample and sample.embedding:
                enrolled_embedding = sample.embedding
        except Exception as e:
            print(f"[upload] Could not fetch enrolled embedding: {e}")

    # Generate session_id if not provided
    effective_session_id = session_id or f"upload-{uuid.uuid4().hex[:8]}"

    # Run the full 6-stage pipeline (async — runs in same event loop)
    try:
        pipeline_result = await run_full_pipeline(
            file_bytes=file_bytes,
            filename=file.filename or "uploaded_audio",
            user_id=user_id,
            enrolled_embedding=enrolled_embedding,
            session_id=effective_session_id,
            ws_manager=None,  # no WS push for file upload
        )
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Pipeline error: {str(e)}")

    risk = pipeline_result.get("risk", {})
    detection = pipeline_result.get("detection", {})
    audio_meta = pipeline_result.get("audio_metadata", {})
    features = pipeline_result.get("features", {})

    # Persist DetectionResult to database
    new_result = DetectionResult(
        user_id=user_id,
        session_id=effective_session_id,
        risk_score=risk.get("risk_score", 0.0),
        verdict=risk.get("verdict", "unknown"),
        risk_level=risk.get("risk_level", "Low"),
        features_json=features,
        audio_duration=audio_meta.get("duration", 0.0),
        model_used=detection.get("model_used", "unknown"),
    )
    db.add(new_result)
    await db.commit()
    await db.refresh(new_result)

    return new_result


@router.get("/upload/status/{job_id}")
async def get_upload_status(job_id: str):
    """
    Get the status of a queued upload job.
    Returns job_id, status, and result when completed.
    """
    status_info = await worker_queue.get_job_status(job_id)
    if status_info.get("error") == "Job not found":
        raise HTTPException(status_code=404, detail="Job not found")
    return status_info
