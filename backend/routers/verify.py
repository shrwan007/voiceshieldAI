"""
POST /api/verify — Speaker Verification Endpoint
Verify if an audio sample matches an enrolled user's voice.
Returns similarity score + combined risk assessment.
"""

from fastapi import APIRouter, File, UploadFile, Form, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from typing import Optional
import uuid

from backend.config import get_settings
from backend.database import get_db
from backend.models import VoiceSample, DetectionResult
from backend.schemas import VerifyResponse
from backend.pipeline.stage1_capture import ingest_upload
from backend.pipeline.stage2_preprocess import preprocess
from backend.pipeline.stage3_features import extract_features
from backend.pipeline.stage4_detection import run_detection_async
from backend.pipeline.stage5_verify import run_verification_async
from backend.pipeline.stage6_alert import compute_risk_score

router = APIRouter(prefix="/api", tags=["verify"])


@router.post("/verify", response_model=VerifyResponse)
async def verify_speaker(
    file: UploadFile = File(...),
    user_id: str = Form(...),
    db: AsyncSession = Depends(get_db),
    settings=Depends(get_settings),
):
    """
    Compare submitted audio against enrolled samples for user_id.

    Process:
    1. Load most recent enrolled sample embedding for user
    2. Run stages 1-4 on submitted audio
    3. Run stage 5 verification against enrolled embedding
    4. Run stage 6 to compute combined risk
    5. Persist DetectionResult
    6. Return VerifyResponse

    Returns 404 if user has no enrolled samples.
    """
    # Fetch most recent enrolled embedding
    result = await db.execute(
        select(VoiceSample)
        .where(VoiceSample.user_id == user_id)
        .order_by(VoiceSample.created_at.desc())
        .limit(1)
    )
    sample = result.scalar_one_or_none()
    if sample is None or not sample.embedding:
        raise HTTPException(
            status_code=404,
            detail=f"No enrolled voice samples found for user {user_id}. "
                   "Please enroll a genuine voice sample first via POST /api/enroll."
        )

    enrolled_embedding = sample.embedding

    # Read submitted audio
    file_bytes = await file.read()
    if not file_bytes:
        raise HTTPException(status_code=400, detail="Uploaded audio file is empty.")

    # Stage 1: Ingest
    try:
        audio_data = await ingest_upload(
            file_bytes,
            file.filename or "verify_audio.wav",
            target_sr=settings.sample_rate,
        )
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))

    # Stage 2: Preprocess
    processed_audio = preprocess(audio_data)

    # Stage 3: Features
    features = extract_features(processed_audio)

    # Stage 4: Detection
    detection = await run_detection_async(processed_audio, features)

    # Stage 5: Verification against enrolled embedding
    verification = await run_verification_async(processed_audio, enrolled_embedding)

    # Stage 6: Combined risk score
    risk_result = compute_risk_score(detection, verification, settings)

    # Persist DetectionResult
    new_result = DetectionResult(
        user_id=user_id,
        session_id=f"verify-{uuid.uuid4().hex[:8]}",
        risk_score=risk_result.risk_score,
        verdict=risk_result.verdict,
        risk_level=risk_result.risk_level,
        features_json=features.to_dict(),
        audio_duration=processed_audio.duration,
        model_used=detection.model_used,
    )
    db.add(new_result)
    await db.commit()

    return VerifyResponse(
        user_id=user_id,
        similarity=verification.similarity,
        is_same_speaker=verification.is_same_speaker,
        risk_score=risk_result.risk_score,
        risk_level=risk_result.risk_level,
        verdict=risk_result.verdict,
        confidence=detection.confidence,
    )
