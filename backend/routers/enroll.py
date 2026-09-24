"""
POST /api/enroll — Speaker Enrollment Endpoint
Allows storing a genuine voice sample for a user.
The speaker embedding is extracted and stored for future verification.
"""

from fastapi import APIRouter, File, UploadFile, Form, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from typing import Optional
import uuid
import os

from backend.config import get_settings
from backend.database import get_db
from backend.models import User, VoiceSample
from backend.schemas import EnrollResponse
from backend.pipeline.stage1_capture import ingest_upload
from backend.pipeline.stage2_preprocess import preprocess
from backend.pipeline.stage5_verify import extract_embedding_async
from backend.utils.audio_utils import save_audio_file

router = APIRouter(prefix="/api", tags=["enroll"])


@router.post("/enroll", response_model=EnrollResponse)
async def enroll_speaker(
    file: UploadFile = File(...),
    username: str = Form(...),
    email: Optional[str] = Form(None),
    user_id: Optional[str] = Form(None),
    db: AsyncSession = Depends(get_db),
    settings=Depends(get_settings),
):
    """
    Enroll a genuine voice sample for speaker verification.

    If user_id provided: adds sample to existing user.
    If only username provided: creates a new user, then adds sample.

    Process:
    1. Read audio file and run stages 1-2 (capture + preprocess)
    2. Extract speaker embedding via Stage 5
    3. Create/get User record in DB
    4. Save audio file to storage
    5. Create VoiceSample record with embedding
    """
    file_bytes = await file.read()
    if not file_bytes:
        raise HTTPException(status_code=400, detail="Uploaded audio file is empty.")

    # Stage 1: Ingest audio
    try:
        audio_data = await ingest_upload(
            file_bytes,
            file.filename or "enrolled_sample.wav",
            target_sr=settings.sample_rate,
        )
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))

    # Stage 2: Preprocess
    processed_audio = preprocess(audio_data)

    # Stage 5: Extract speaker embedding
    try:
        embedding = await extract_embedding_async(processed_audio)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Embedding extraction failed: {e}")

    # Resolve or create User
    effective_user_id = user_id
    if effective_user_id:
        result = await db.execute(select(User).where(User.id == effective_user_id))
        user = result.scalar_one_or_none()
        if user is None:
            raise HTTPException(status_code=404, detail=f"User {effective_user_id} not found.")
    else:
        # Create new user
        user = User(username=username, email=email)
        db.add(user)
        await db.flush()  # get the generated ID
        effective_user_id = str(user.id)

    # Save audio file to storage
    file_path = save_audio_file(
        processed_audio.audio,
        processed_audio.sample_rate,
        directory=settings.audio_storage_path,
        filename=f"enroll_{effective_user_id}_{uuid.uuid4().hex[:8]}.wav",
    )

    # Create VoiceSample record
    sample = VoiceSample(
        user_id=effective_user_id,
        file_path=file_path,
        embedding=embedding,
        sample_rate=processed_audio.sample_rate,
        duration=processed_audio.duration,
    )
    db.add(sample)
    await db.commit()
    await db.refresh(sample)

    return EnrollResponse(
        user_id=effective_user_id,
        sample_id=str(sample.id),
        embedding_size=len(embedding),
        message=f"Voice sample enrolled successfully for user '{username}'.",
    )


@router.get("/enroll/{user_id}/samples")
async def get_user_samples(
    user_id: str,
    db: AsyncSession = Depends(get_db),
):
    """
    Get all enrolled samples for a user.
    Returns list of sample metadata (no embeddings, those are internal).
    """
    result = await db.execute(
        select(VoiceSample)
        .where(VoiceSample.user_id == user_id)
        .order_by(VoiceSample.created_at.desc())
    )
    samples = result.scalars().all()

    return {
        "user_id": user_id,
        "samples": [
            {
                "id": str(s.id),
                "file_path": os.path.basename(s.file_path),
                "sample_rate": s.sample_rate,
                "duration": s.duration,
                "created_at": s.created_at.isoformat() if s.created_at else None,
            }
            for s in samples
        ],
        "total": len(samples),
    }


@router.delete("/enroll/{user_id}/samples/{sample_id}", status_code=204)
async def delete_sample(
    user_id: str,
    sample_id: str,
    db: AsyncSession = Depends(get_db),
):
    """
    Delete an enrolled voice sample.
    Also removes the audio file from storage if it exists.
    """
    result = await db.execute(
        select(VoiceSample).where(
            VoiceSample.id == sample_id,
            VoiceSample.user_id == user_id,
        )
    )
    sample = result.scalar_one_or_none()
    if sample is None:
        raise HTTPException(status_code=404, detail="Voice sample not found.")

    # Remove audio file from disk
    try:
        if sample.file_path and os.path.exists(sample.file_path):
            os.remove(sample.file_path)
    except Exception as e:
        print(f"[enroll] Could not delete audio file {sample.file_path}: {e}")

    await db.delete(sample)
    await db.commit()
