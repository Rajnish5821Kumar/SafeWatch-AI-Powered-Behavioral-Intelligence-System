"""
SafeWatch API — Analytics Routes
──────────────────────────────────
Classroom-level aggregated statistics and time-series data endpoints.
"""

from __future__ import annotations
import time
import random
import math
from typing import List

from fastapi import APIRouter, Query
from fastapi.responses import JSONResponse

router = APIRouter()

# In-memory rolling analytics store (populated by video_processor in production)
_analytics_store = {
    "emotion_distribution": {
        "neutral": 0.45, "happy": 0.20, "sad": 0.12,
        "angry": 0.07, "fear": 0.05, "surprise": 0.08, "disgust": 0.03
    },
    "engagement_score": 0.67,
    "anomaly_rate": 0.08,
    "active_persons": 28,
    "avg_fps": 13.2,
    "timeline": [],          # Populated below
    "session_start": time.time(),
}

# Generate synthetic timeline for demo (last 60 data points @ 10s intervals)
_start = time.time() - 600
_analytics_store["timeline"] = [
    {
        "t": _start + i * 10,
        "engagement": round(0.55 + 0.20 * math.sin(i * 0.3) + random.uniform(-0.05, 0.05), 3),
        "anomaly_rate": round(max(0, 0.05 + 0.08 * math.cos(i * 0.2) + random.uniform(-0.02, 0.03)), 3),
        "persons": max(10, int(28 + 5 * math.sin(i * 0.15) + random.randint(-2, 2))),
    }
    for i in range(60)
]


def update_analytics(frame_result_dict: dict) -> None:
    """Called by video processor to update live analytics."""
    _analytics_store["active_persons"] = frame_result_dict.get("n_persons", 0)
    _analytics_store["avg_fps"] = frame_result_dict.get("fps", 0.0)

    emo = frame_result_dict.get("emotion_summary", {})
    if emo:
        _analytics_store["emotion_distribution"] = emo

    anomalies = frame_result_dict.get("anomalies", {})
    n_alerts = sum(1 for v in anomalies.values() if v.get("alert"))
    n_total = max(1, len(anomalies))
    _analytics_store["anomaly_rate"] = round(n_alerts / n_total, 3)

    # Append to timeline
    _analytics_store["timeline"].append({
        "t": time.time(),
        "engagement": round(1.0 - _analytics_store["anomaly_rate"], 3),
        "anomaly_rate": _analytics_store["anomaly_rate"],
        "persons": _analytics_store["active_persons"],
    })
    # Keep only last 300 points
    if len(_analytics_store["timeline"]) > 300:
        _analytics_store["timeline"] = _analytics_store["timeline"][-300:]


@router.get("/summary")
async def get_summary():
    """Return current classroom-level behavioral summary."""
    return JSONResponse({
        "active_persons": _analytics_store["active_persons"],
        "engagement_score": _analytics_store["engagement_score"],
        "anomaly_rate": _analytics_store["anomaly_rate"],
        "avg_fps": _analytics_store["avg_fps"],
        "emotion_distribution": _analytics_store["emotion_distribution"],
        "session_duration_sec": round(time.time() - _analytics_store["session_start"], 1),
        "generated_at": time.time(),
    })


@router.get("/timeline")
async def get_timeline(
    points: int = Query(default=60, ge=5, le=300),
):
    """Return time-series engagement and anomaly rate data for charting."""
    timeline = _analytics_store["timeline"][-points:]
    return JSONResponse({
        "points": len(timeline),
        "data": timeline,
        "generated_at": time.time(),
    })


@router.get("/emotions")
async def get_emotion_distribution():
    """Return current classroom emotion distribution."""
    return JSONResponse({
        "distribution": _analytics_store["emotion_distribution"],
        "generated_at": time.time(),
    })
