"""
GET /api/history — Detection History Endpoint
Paginated retrieval of past detection results.
Supports filtering by user_id, risk_level, verdict, date range.
"""

from fastapi import APIRouter, Query, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func, and_
from typing import Optional
from datetime import datetime
import math

from backend.database import get_db
from backend.models import DetectionResult
from backend.schemas import HistoryResponse, DetectionResultResponse

router = APIRouter(prefix="/api", tags=["history"])


def _build_filters(
    user_id: Optional[str],
    risk_level: Optional[str],
    verdict: Optional[str],
    session_id: Optional[str],
    start_date: Optional[datetime],
    end_date: Optional[datetime],
):
    """Build SQLAlchemy filter conditions from query parameters."""
    conditions = []
    if user_id:
        conditions.append(DetectionResult.user_id == user_id)
    if risk_level:
        conditions.append(DetectionResult.risk_level == risk_level)
    if verdict:
        conditions.append(DetectionResult.verdict == verdict)
    if session_id:
        conditions.append(DetectionResult.session_id == session_id)
    if start_date:
        conditions.append(DetectionResult.timestamp >= start_date)
    if end_date:
        conditions.append(DetectionResult.timestamp <= end_date)
    return conditions


@router.get("/history/stats/summary")
async def get_history_summary(
    user_id: Optional[str] = Query(None),
    db: AsyncSession = Depends(get_db),
):
    """
    Get aggregate detection statistics.
    Returns total counts, breakdown by verdict and risk level, average score,
    and recent high-risk results.

    NOTE: This route MUST be declared before /history/{result_id} to avoid
    'stats' being caught as a result_id path param.
    """
    conditions = []
    if user_id:
        conditions.append(DetectionResult.user_id == user_id)

    # Total count
    total_result = await db.execute(
        select(func.count(DetectionResult.id)).where(and_(*conditions))
    )
    total = total_result.scalar() or 0

    # Average risk score
    avg_result = await db.execute(
        select(func.avg(DetectionResult.risk_score)).where(and_(*conditions))
    )
    avg_score = float(avg_result.scalar() or 0.0)

    # Count by verdict
    by_verdict = {}
    for verdict in ["real", "synthetic", "cloned"]:
        cnt_result = await db.execute(
            select(func.count(DetectionResult.id)).where(
                and_(*conditions, DetectionResult.verdict == verdict)
            )
        )
        by_verdict[verdict] = cnt_result.scalar() or 0

    # Count by risk level
    by_risk_level = {}
    for level in ["Low", "Medium", "High"]:
        cnt_result = await db.execute(
            select(func.count(DetectionResult.id)).where(
                and_(*conditions, DetectionResult.risk_level == level)
            )
        )
        by_risk_level[level] = cnt_result.scalar() or 0

    # Recent high-risk (last 5)
    recent_result = await db.execute(
        select(DetectionResult)
        .where(and_(*conditions, DetectionResult.risk_level == "High"))
        .order_by(DetectionResult.timestamp.desc())
        .limit(5)
    )
    recent_high = recent_result.scalars().all()

    return {
        "total_detections": total,
        "total_scans": total,
        "high_risk_count": by_risk_level.get("High", 0),
        "avg_risk_score": round(avg_score, 4),
        "by_verdict": by_verdict,
        "by_risk_level": by_risk_level,
        "recent_high_risk": [
            {
                "id": str(r.id),
                "session_id": r.session_id,
                "timestamp": r.timestamp.isoformat() if r.timestamp else None,
                "risk_score": r.risk_score,
                "verdict": r.verdict,
            }
            for r in recent_high
        ],
    }


@router.get("/history", response_model=HistoryResponse)
async def get_history(
    page: int = Query(1, ge=1),
    limit: int = Query(20, ge=1, le=100),
    user_id: Optional[str] = Query(None),
    risk_level: Optional[str] = Query(None),
    verdict: Optional[str] = Query(None),
    session_id: Optional[str] = Query(None),
    start_date: Optional[datetime] = Query(None),
    end_date: Optional[datetime] = Query(None),
    db: AsyncSession = Depends(get_db),
):
    """
    Get paginated detection history with optional filters.
    Results ordered by timestamp descending (newest first).
    """
    conditions = _build_filters(user_id, risk_level, verdict, session_id, start_date, end_date)

    # Count total matching records
    count_result = await db.execute(
        select(func.count(DetectionResult.id)).where(and_(*conditions))
    )
    total = count_result.scalar() or 0
    pages = math.ceil(total / limit) if total > 0 else 1

    # Fetch page
    offset = (page - 1) * limit
    results_query = await db.execute(
        select(DetectionResult)
        .where(and_(*conditions))
        .order_by(DetectionResult.timestamp.desc())
        .offset(offset)
        .limit(limit)
    )
    results = results_query.scalars().all()

    return HistoryResponse(
        results=[DetectionResultResponse.model_validate(r) for r in results],
        total=total,
        page=page,
        limit=limit,
        pages=pages,
    )


@router.get("/history/{result_id}", response_model=DetectionResultResponse)
async def get_detection_result(
    result_id: str,
    db: AsyncSession = Depends(get_db),
):
    """
    Get a single detection result by ID.
    Returns 404 if not found.
    """
    result = await db.execute(
        select(DetectionResult).where(DetectionResult.id == result_id)
    )
    record = result.scalar_one_or_none()
    if record is None:
        raise HTTPException(status_code=404, detail="Detection result not found.")
    return DetectionResultResponse.model_validate(record)


@router.delete("/history/{result_id}", status_code=204)
async def delete_detection_result(
    result_id: str,
    db: AsyncSession = Depends(get_db),
):
    """
    Delete a detection result by ID.
    Returns 204 No Content on success.
    """
    result = await db.execute(
        select(DetectionResult).where(DetectionResult.id == result_id)
    )
    record = result.scalar_one_or_none()
    if record is None:
        raise HTTPException(status_code=404, detail="Detection result not found.")
    await db.delete(record)
    await db.commit()
