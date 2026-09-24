"""
Stage 6 — Risk Scoring & Alert System
Responsibility:
  1. Combine detection + verification outputs into a unified risk score (0.0–1.0)
  2. Map score to risk level: Low / Medium / High
  3. Push WebSocket alert to frontend when risk >= high_threshold
"""

from dataclasses import dataclass
from typing import Optional
from datetime import datetime, timezone

from backend.pipeline.stage4_detection import DetectionOutput
from backend.pipeline.stage5_verify import VerificationOutput
from backend.pipeline.stage3_features import FeatureSet

@dataclass
class RiskResult:
    risk_score: float
    risk_level: str
    verdict: str
    confidence: float
    risk_color: str
    alert_triggered: bool

def compute_risk_score(
    detection: DetectionOutput,
    verification: Optional[VerificationOutput],
    settings
) -> RiskResult:
    """Combine detection and verification into unified risk score."""
    score = detection.raw_score
    
    if verification and not verification.is_same_speaker:
        score += 0.15
        
    if detection.verdict == "cloned" and verification:
        score += (1.0 - verification.similarity) * 0.1
        
    score = max(0.0, min(1.0, score))
    
    risk_level, risk_color = risk_score_to_level(score, settings)
    alert_triggered = score >= settings.high_threshold
    
    return RiskResult(
        risk_score=score,
        risk_level=risk_level,
        verdict=detection.verdict,
        confidence=detection.confidence,
        risk_color=risk_color,
        alert_triggered=alert_triggered
    )

def risk_score_to_level(score: float, settings) -> tuple[str, str]:
    """Map 0-1 score to (risk_level, risk_color)."""
    if score >= settings.high_threshold:
        return "High", "red"
    elif score >= settings.medium_threshold:
        return "Medium", "yellow"
    else:
        return "Low", "green"

async def push_alert(
    session_id: str,
    result: RiskResult,
    features: FeatureSet,
    ws_manager
) -> None:
    """Push detection result via WebSocket to connected frontend clients."""
    timestamp = datetime.now(timezone.utc).isoformat()
    
    result_msg = {
        "type": "detection_result",
        "session_id": session_id,
        "timestamp": timestamp,
        "data": {
            "risk_score": result.risk_score,
            "risk_level": result.risk_level,
            "risk_color": result.risk_color,
            "verdict": result.verdict,
            "confidence": result.confidence,
            "alert_triggered": result.alert_triggered,
            "features": features.to_dict()
        }
    }
    
    await ws_manager.send_json(session_id, result_msg)
    
    if result.alert_triggered:
        alert_msg = {
            "type": "alert",
            "session_id": session_id,
            "timestamp": timestamp,
            "data": result_msg["data"]
        }
        await ws_manager.send_json(session_id, alert_msg)
