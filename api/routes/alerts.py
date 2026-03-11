"""
SafeWatch API — Alert Routes
──────────────────────────────
Alert history retrieval and management endpoints.
"""

from __future__ import annotations
import time
from typing import Optional

from fastapi import APIRouter, Query
from fastapi.responses import JSONResponse
from loguru import logger

from safewatch.logger import SafeWatchLogger

router = APIRouter()

_sw_logger = SafeWatchLogger(
    log_dir="data/logs",
    alert_archive="data/alerts/alerts.jsonl",
)


@router.get("")
async def get_alerts(
    limit: int = Query(default=50, ge=1, le=500),
    severity: Optional[str] = Query(default=None, regex="^(low|medium|high)$"),
    track_id: Optional[int] = Query(default=None),
):
    """
    Retrieve recent alerts from the archive.

    Parameters
    ----------
    limit   : Maximum number of alerts to return (default 50)
    severity: Filter by severity level (low / medium / high)
    track_id: Filter by person track ID
    """
    alerts = _sw_logger.get_recent_alerts(limit=limit * 3)  # Fetch more, then filter

    if severity:
        alerts = [a for a in alerts if a.get("severity") == severity]
    if track_id is not None:
        alerts = [a for a in alerts if a.get("track_id") == track_id]

    alerts = alerts[:limit]

    return JSONResponse({
        "count": len(alerts),
        "alerts": alerts,
        "generated_at": time.time(),
    })


@router.get("/stats")
async def get_alert_stats():
    """Return aggregate statistics about all recorded alerts."""
    all_alerts = _sw_logger.get_recent_alerts(limit=10000)

    severity_counts = {"low": 0, "medium": 0, "high": 0}
    unique_persons = set()
    for a in all_alerts:
        sev = a.get("severity", "low")
        if sev in severity_counts:
            severity_counts[sev] += 1
        if "track_id" in a:
            unique_persons.add(a["track_id"])

    return JSONResponse({
        "total_alerts": len(all_alerts),
        "severity_breakdown": severity_counts,
        "unique_persons_flagged": len(unique_persons),
        "generated_at": time.time(),
    })


@router.delete("")
async def clear_alerts():
    """Clear all stored alerts (admin operation)."""
    _sw_logger.clear_alerts()
    return JSONResponse({"status": "cleared", "message": "All alerts have been deleted."})
