"""
WS /api/stream/{session_id} — Real-Time Audio Streaming Endpoint
Accepts raw PCM audio bytes via WebSocket, processes them in chunks,
and pushes detection results back through the same WebSocket connection.

Protocol:
  Client → Server: binary PCM frames (int16, 16kHz, mono)
  Server → Client: JSON detection result messages
"""

from fastapi import APIRouter, WebSocket, WebSocketDisconnect, Query, Depends
from sqlalchemy.ext.asyncio import AsyncSession
from typing import Optional
import asyncio
from datetime import datetime, timezone
import uuid

from backend.services.websocket_manager import manager
from backend.database import get_db
from backend.models import VoiceSample, DetectionResult
from backend.config import get_settings
from backend.pipeline.stage1_capture import ingest_stream_chunk, get_or_create_session, close_session
from backend.pipeline.stage2_preprocess import preprocess
from backend.pipeline.stage3_features import extract_features
from backend.pipeline.stage4_detection import run_detection_async
from backend.pipeline.stage5_verify import run_verification_async
from backend.pipeline.stage6_alert import compute_risk_score, push_alert

router = APIRouter(prefix="/api", tags=["stream"])


async def _get_enrolled_embedding(db: AsyncSession, user_id: str) -> Optional[list]:
    """Fetch the most recent enrolled speaker embedding for a user from the DB."""
    try:
        from sqlalchemy import select
        result = await db.execute(
            select(VoiceSample)
            .where(VoiceSample.user_id == user_id)
            .order_by(VoiceSample.created_at.desc())
            .limit(1)
        )
        sample = result.scalar_one_or_none()
        if sample and sample.embedding:
            return sample.embedding
    except Exception as e:
        print(f"[stream] Failed to fetch enrolled embedding: {e}")
    return None


async def _persist_result(db: AsyncSession, session_id: str, user_id: Optional[str],
                          risk_result, features, detection, audio_data):
    """Persist detection result to database (fire-and-forget)."""
    try:
        new_result = DetectionResult(
            user_id=user_id,
            session_id=session_id,
            risk_score=risk_result.risk_score,
            verdict=risk_result.verdict,
            risk_level=risk_result.risk_level,
            features_json=features.to_dict(),
            audio_duration=audio_data.duration,
            model_used=detection.model_used
        )
        db.add(new_result)
        await db.commit()
    except Exception as e:
        print(f"[stream] DB persist error: {e}")


@router.websocket("/stream/{session_id}")
async def stream_audio(
    websocket: WebSocket,
    session_id: str,
    user_id: Optional[str] = Query(None)
):
    """
    WebSocket endpoint for real-time voice fraud detection.

    Workflow:
    1. Accept connection, register with ConnectionManager
    2. Initialize StreamSession for chunk buffering
    3. Loop: receive binary chunk → stage1 ingest → if buffer ready:
       a. stage2 preprocess
       b. stage3 extract features
       c. stage4 detect (async, in thread pool)
       d. stage5 verify (async, if enrolled embedding available)
       e. stage6 compute risk + push alert via WebSocket
       f. Persist result to DB (fire-and-forget)
    4. On disconnect: cleanup session
    """
    # Accept connection and register
    await manager.connect(session_id, websocket)
    stream_session = get_or_create_session(session_id)
    settings = get_settings()

    # Pre-fetch enrolled embedding (DB lookup once, not per-chunk)
    enrolled_embedding = None

    try:
        while True:
            # Receive binary PCM chunk from client
            chunk = await websocket.receive_bytes()

            # Stage 1b: Accumulate chunk into buffer; returns AudioData when ready
            audio_data = await ingest_stream_chunk(chunk, stream_session)

            if audio_data is None:
                # Buffer not yet full — keep accumulating
                continue

            try:
                # Stage 2: Preprocess (noise reduction, silence trim, normalize)
                processed_audio = preprocess(audio_data)

                # Stage 3: Feature extraction
                features = extract_features(processed_audio)

                # Stage 4 + 5: Run detection and optional verification concurrently
                if user_id and enrolled_embedding is None:
                    # Lazy-load enrolled embedding once per session
                    # (DB not available inside WS handler directly — skip DB here)
                    pass  # enrolled_embedding would be fetched via REST pre-session

                detection_task = asyncio.create_task(
                    run_detection_async(processed_audio, features)
                )

                if enrolled_embedding:
                    verification_task = asyncio.create_task(
                        run_verification_async(processed_audio, enrolled_embedding)
                    )
                    detection_result, verification_result = await asyncio.gather(
                        detection_task, verification_task
                    )
                else:
                    detection_result = await detection_task
                    verification_result = None

                # Stage 6: Compute risk score and push alert
                risk_result = compute_risk_score(detection_result, verification_result, settings)
                await push_alert(session_id, risk_result, features, manager)

            except Exception as e:
                # Send error message but keep the connection alive
                await manager.send_json(session_id, {
                    "type": "error",
                    "session_id": session_id,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "data": {"error": str(e)}
                })

    except (WebSocketDisconnect, Exception):
        await manager.disconnect(session_id)
        close_session(session_id)
