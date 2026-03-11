"""
SafeWatch — Structured Logger
──────────────────────────────
Loguru-based structured logging with:
- Per-frame model output logging (DEBUG level)
- Alert archival to JSONL file
- System metrics tracking (FPS, latency per stage)
- Log rotation and size management
"""

from __future__ import annotations
import json
import time
import os
from pathlib import Path
from typing import Dict, List, Optional
from collections import deque
import threading

from loguru import logger

from safewatch.insight_generator import SafetyInsight


class SafeWatchLogger:
    """
    Centralized structured logger for SafeWatch.

    Writes human-readable logs via loguru and machine-readable
    JSONL archives for alert replay and post-hoc analysis.

    Example
    -------
    >>> sw_logger = SafeWatchLogger(log_dir="data/logs", alert_archive="data/alerts/alerts.jsonl")
    >>> sw_logger.log_frame(frame_id=1, n_persons=12, fps=14.2)
    >>> sw_logger.log_alert(insight)
    """

    def __init__(
        self,
        log_dir: str = "data/logs",
        alert_archive: str = "data/alerts/alerts.jsonl",
        log_level: str = "INFO",
        max_file_size_mb: int = 50,
    ):
        self.log_dir = Path(log_dir)
        self.alert_archive = Path(alert_archive)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.alert_archive.parent.mkdir(parents=True, exist_ok=True)

        # Configure loguru
        log_file = self.log_dir / "safewatch_{time:YYYY-MM-DD}.log"
        logger.add(
            str(log_file),
            level=log_level,
            rotation=f"{max_file_size_mb} MB",
            retention="14 days",
            format="{time:YYYY-MM-DD HH:mm:ss} | {level:<8} | {name}:{line} | {message}",
            enqueue=True,  # Thread-safe async logging
        )

        # Metrics tracking
        self._fps_window: deque = deque(maxlen=60)
        self._stage_latencies: Dict[str, deque] = {
            "detection": deque(maxlen=60),
            "tracking": deque(maxlen=60),
            "pose": deque(maxlen=60),
            "emotion": deque(maxlen=60),
            "anomaly": deque(maxlen=60),
            "total": deque(maxlen=60),
        }
        self._alert_count = 0
        self._frame_count = 0
        self._lock = threading.Lock()

        logger.info("SafeWatchLogger initialized")

    def log_frame(
        self,
        frame_id: int,
        n_persons: int,
        n_alerts: int = 0,
        fps: float = 0.0,
        stage_latencies: Optional[Dict[str, float]] = None,
    ) -> None:
        """Log per-frame processing summary."""
        with self._lock:
            self._frame_count += 1
            if fps > 0:
                self._fps_window.append(fps)
            if stage_latencies:
                for stage, lat in stage_latencies.items():
                    if stage in self._stage_latencies:
                        self._stage_latencies[stage].append(lat)

        logger.debug(
            f"Frame {frame_id:05d} | persons={n_persons:3d} | alerts={n_alerts} | fps={fps:.1f}"
        )

    def log_alert(self, insight: SafetyInsight) -> None:
        """Persist alert to JSONL archive and log to console."""
        with self._lock:
            self._alert_count += 1

        alert_dict = insight.to_dict()
        alert_dict["logged_at"] = time.time()

        # Append to JSONL archive
        try:
            with open(self.alert_archive, "a", encoding="utf-8") as f:
                f.write(json.dumps(alert_dict) + "\n")
        except Exception as e:
            logger.error(f"Failed to write alert to archive: {e}")

        logger.warning(
            f"ALERT [{insight.severity.upper()}] Track {insight.track_id}: "
            f"{insight.headline} (score={insight.anomaly_score:.2f})"
        )

    def log_system_metrics(self, extra: Optional[Dict] = None) -> None:
        """Log rolling system performance metrics."""
        with self._lock:
            avg_fps = (
                sum(self._fps_window) / len(self._fps_window)
                if self._fps_window else 0.0
            )
            metrics = {
                "avg_fps": round(avg_fps, 1),
                "frames_processed": self._frame_count,
                "total_alerts": self._alert_count,
            }
            for stage, window in self._stage_latencies.items():
                if window:
                    metrics[f"avg_latency_{stage}_ms"] = round(
                        sum(window) / len(window) * 1000, 1
                    )
            if extra:
                metrics.update(extra)

        logger.info(f"System Metrics: {json.dumps(metrics)}")
        return metrics

    def get_recent_alerts(self, limit: int = 50) -> List[dict]:
        """Read the most recent N alerts from the JSONL archive."""
        if not self.alert_archive.exists():
            return []
        alerts = []
        try:
            with open(self.alert_archive, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        alerts.append(json.loads(line))
            # Return most recent first
            return list(reversed(alerts[-limit:]))
        except Exception as e:
            logger.error(f"Failed to read alert archive: {e}")
            return []

    def clear_alerts(self) -> None:
        """Clear the alert archive (for testing/reset)."""
        if self.alert_archive.exists():
            self.alert_archive.unlink()
        logger.info("Alert archive cleared")

    @property
    def stats(self) -> dict:
        with self._lock:
            return {
                "frames_processed": self._frame_count,
                "total_alerts": self._alert_count,
                "avg_fps": (
                    sum(self._fps_window) / len(self._fps_window)
                    if self._fps_window else 0.0
                ),
            }
