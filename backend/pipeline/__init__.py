"""
Module defining the 6-stage pipeline architecture for VoiceShield.
Stages:
1. Capture & Ingestion
2. Preprocessing
3. Feature Extraction
4. AI Voice Deepfake Detection
5. Speaker Verification & Identity Check
6. Risk Scoring & Alert System
"""

from typing import Optional

from backend.pipeline.stage1_capture import ingest_upload
from backend.pipeline.stage2_preprocess import preprocess
from backend.pipeline.stage3_features import extract_features
from backend.pipeline.stage4_detection import run_detection_async
from backend.pipeline.stage5_verify import run_verification_async
from backend.pipeline.stage6_alert import compute_risk_score, push_alert
from backend.config import get_settings

async def run_full_pipeline(
    file_bytes: bytes,
    filename: str,
    user_id: Optional[str] = None,
    enrolled_embedding: Optional[list[float]] = None,
    session_id: Optional[str] = None,
    ws_manager=None
) -> dict:
    """
    Convenience function that runs all 6 pipeline stages in sequence.
    Returns the final result dict with all stage outputs.
    """
    settings = get_settings()
    
    # Stage 1: Capture
    audio_data = await ingest_upload(file_bytes, filename)
    
    # Stage 2: Preprocess
    preprocessed_audio = preprocess(audio_data)
    
    # Stage 3: Features
    features = extract_features(preprocessed_audio)
    
    # Stage 4: Detection
    detection = await run_detection_async(preprocessed_audio, features)
    
    # Stage 5: Verification
    verification = None
    if enrolled_embedding:
        verification = await run_verification_async(preprocessed_audio, enrolled_embedding)
        
    # Stage 6: Alert
    risk_result = compute_risk_score(detection, verification, settings)
    
    # Push alert via WS if needed
    if ws_manager and session_id:
        await push_alert(session_id, risk_result, features, ws_manager)
        
    return {
        "audio_metadata": {
            "filename": audio_data.filename,
            "duration": audio_data.duration,
            "sample_rate": audio_data.sample_rate
        },
        "features": features.to_dict(),
        "detection": detection.__dict__,
        "verification": verification.__dict__ if verification else None,
        "risk": risk_result.__dict__
    }
