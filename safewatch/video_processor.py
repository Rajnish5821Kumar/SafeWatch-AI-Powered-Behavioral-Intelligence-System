"""
SafeWatch — Video Processing Pipeline Orchestrator
────────────────────────────────────────────────────
Ties together detection → tracking → pose → emotion → profiling →
anomaly detection → insight generation into a single processing loop.

Key optimisations:
  - Keyframe-selective processing (skip frames with low visual change)
  - Configurable max FPS cap
  - Async-friendly generator interface
  - Per-stage latency telemetry
"""

from __future__ import annotations
import cv2
import numpy as np
import time
import threading
import queue
from pathlib import Path
from typing import Dict, List, Optional, Generator, Tuple
from dataclasses import dataclass, field

import yaml
from loguru import logger

from safewatch.detector import PersonDetector
from safewatch.tracker import MultiObjectTracker
from safewatch.pose_estimator import PoseEstimator
from safewatch.emotion_analyzer import EmotionAnalyzer
from safewatch.behavioral_profiler import BehavioralProfiler
from safewatch.anomaly_detector import AnomalyDetector, AnomalyResult
from safewatch.insight_generator import InsightGenerator, SafetyInsight
from safewatch.logger import SafeWatchLogger


@dataclass
class FrameResult:
    """All analysis results for a single video frame."""
    frame_id: int
    timestamp: float
    frame: Optional[np.ndarray]              # Annotated frame (if enabled)
    n_persons: int
    tracked_ids: List[int]
    anomaly_results: Dict[int, AnomalyResult]
    insights: List[SafetyInsight]
    fps: float
    stage_latencies: Dict[str, float] = field(default_factory=dict)
    emotion_summary: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict:
        """Serialisable summary (no raw frame data)."""
        return {
            "frame_id": self.frame_id,
            "timestamp": self.timestamp,
            "n_persons": self.n_persons,
            "tracked_ids": self.tracked_ids,
            "fps": round(self.fps, 1),
            "anomalies": {
                str(tid): {
                    "score": round(r.combined_score, 3),
                    "severity": r.severity,
                    "alert": r.is_alert,
                }
                for tid, r in self.anomaly_results.items()
            },
            "alerts": [i.to_dict() for i in self.insights],
            "emotion_summary": {k: round(v, 3) for k, v in self.emotion_summary.items()},
            "stage_latencies_ms": {
                k: round(v * 1000, 1) for k, v in self.stage_latencies.items()
            },
        }


class VideoProcessor:
    """
    Main SafeWatch processing pipeline.

    Usage
    -----
    >>> vp = VideoProcessor.from_config("config.yaml")
    >>> for result in vp.process_video("classroom.mp4"):
    ...     print(result.to_dict())
    """

    def __init__(
        self,
        detector: PersonDetector,
        tracker: MultiObjectTracker,
        pose_estimator: PoseEstimator,
        emotion_analyzer: EmotionAnalyzer,
        profiler: BehavioralProfiler,
        anomaly_detector: AnomalyDetector,
        insight_generator: InsightGenerator,
        sw_logger: SafeWatchLogger,
        max_fps: int = 15,
        keyframe_diff_threshold: float = 0.015,
        output_annotated: bool = True,
    ):
        self.detector = detector
        self.tracker = tracker
        self.pose_estimator = pose_estimator
        self.emotion_analyzer = emotion_analyzer
        self.profiler = profiler
        self.anomaly_detector = anomaly_detector
        self.insight_generator = insight_generator
        self.logger = sw_logger

        self.max_fps = max_fps
        self.keyframe_diff_threshold = keyframe_diff_threshold
        self.output_annotated = output_annotated

        self._prev_gray: Optional[np.ndarray] = None
        self._frame_interval = 1.0 / max_fps
        self._last_frame_ts = 0.0
        self._is_running = False

        # Latest result for WebSocket push
        self._latest_result: Optional[FrameResult] = None
        self._result_queue: queue.Queue = queue.Queue(maxsize=10)
        self._lock = threading.Lock()

    # ──────────────────────────────────────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────────────────────────────────────

    @classmethod
    def from_config(cls, config_path: str = "config.yaml") -> "VideoProcessor":
        """
        Factory: build a fully-configured VideoProcessor from YAML config.
        """
        with open(config_path, "r") as f:
            cfg = yaml.safe_load(f)

        det_cfg = cfg["detection"]
        trk_cfg = cfg["tracking"]
        pose_cfg = cfg["pose"]
        emo_cfg = cfg["emotion"]
        prof_cfg = cfg["profiling"]
        anom_cfg = cfg["anomaly"]
        ins_cfg = cfg["insights"]
        vid_cfg = cfg["processing"]
        log_cfg = cfg["logging"]

        detector = PersonDetector(
            model_path=det_cfg["model"],
            confidence_threshold=det_cfg["confidence_threshold"],
            iou_threshold=det_cfg["iou_threshold"],
            device=det_cfg["device"],
            input_size=det_cfg["input_size"],
            min_box_area=trk_cfg.get("min_box_area", 100),
        )
        tracker = MultiObjectTracker(
            track_threshold=trk_cfg["track_threshold"],
            match_threshold=trk_cfg["match_threshold"],
            track_buffer=trk_cfg["track_buffer"],
            frame_rate=vid_cfg["max_fps"],
        )
        pose_estimator = PoseEstimator(
            model_path=pose_cfg["model"],
            device=det_cfg["device"],
            min_keypoint_confidence=pose_cfg["min_keypoint_confidence"],
        )
        emotion_analyzer = EmotionAnalyzer(
            smoothing_alpha=emo_cfg["smoothing_alpha"],
            min_face_size=emo_cfg["min_face_size"],
        )
        profiler = BehavioralProfiler(
            window_seconds=prof_cfg["window_seconds"],
        )
        anomaly_detector = AnomalyDetector(
            alert_threshold=anom_cfg["alert_threshold"],
            contamination=anom_cfg["contamination"],
            lstm_seq_len=anom_cfg["lstm_sequence_len"],
            baseline_path=anom_cfg.get("baseline_path"),
            lstm_model_path=anom_cfg.get("lstm_model_path"),
        )
        insight_generator = InsightGenerator(
            min_alert_interval_sec=ins_cfg["min_alert_interval_sec"],
        )
        sw_logger = SafeWatchLogger(
            log_dir=log_cfg["log_dir"],
            alert_archive=log_cfg["alert_archive"],
            log_level=log_cfg["level"],
            max_file_size_mb=log_cfg["max_file_size_mb"],
        )

        return cls(
            detector=detector,
            tracker=tracker,
            pose_estimator=pose_estimator,
            emotion_analyzer=emotion_analyzer,
            profiler=profiler,
            anomaly_detector=anomaly_detector,
            insight_generator=insight_generator,
            sw_logger=sw_logger,
            max_fps=vid_cfg["max_fps"],
            keyframe_diff_threshold=vid_cfg["keyframe_diff_threshold"],
            output_annotated=vid_cfg["output_annotated"],
        )

    def process_video(
        self,
        video_path: str,
        max_frames: Optional[int] = None,
    ) -> Generator[FrameResult, None, None]:
        """
        Process a video file frame by frame, yielding FrameResult for each
        processed keyframe.

        Parameters
        ----------
        video_path : str
            Path to MP4/AVI/etc.
        max_frames : Optional[int]
            Stop after this many frames (for testing).
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise IOError(f"Cannot open video: {video_path}")

        source_fps = cap.get(cv2.CAP_PROP_FPS) or 25
        process_every_n = max(1, int(source_fps / self.max_fps))

        logger.info(
            f"Processing video: {video_path} | source_fps={source_fps:.1f} | "
            f"processing every {process_every_n} frames"
        )

        frame_id = 0
        processed_count = 0
        self._is_running = True

        try:
            while self._is_running:
                ret, frame = cap.read()
                if not ret:
                    break

                frame_id += 1
                if frame_id % process_every_n != 0:
                    continue

                if max_frames and processed_count >= max_frames:
                    break

                # Keyframe selection: skip frames with low visual change
                if self._is_similar_to_previous(frame):
                    continue

                t_start = time.perf_counter()
                result = self._process_frame(frame, frame_id)
                t_total = time.perf_counter() - t_start

                result.fps = 1.0 / max(t_total, 1e-4)
                result.stage_latencies["total"] = t_total

                # Update latest result for WebSocket
                with self._lock:
                    self._latest_result = result

                # Log
                self.logger.log_frame(
                    frame_id=frame_id,
                    n_persons=result.n_persons,
                    n_alerts=len(result.insights),
                    fps=result.fps,
                    stage_latencies=result.stage_latencies,
                )
                for insight in result.insights:
                    self.logger.log_alert(insight)

                processed_count += 1
                yield result

        finally:
            cap.release()
            self._is_running = False
            self.logger.log_system_metrics({"source_video": video_path})
            logger.info(f"Video processing complete. Frames processed: {processed_count}")

    def process_frame(self, frame: np.ndarray, frame_id: int = 0) -> FrameResult:
        """Process a single BGR frame (for real-time/stream use)."""
        return self._process_frame(frame, frame_id)

    def stop(self) -> None:
        self._is_running = False

    @property
    def latest_result(self) -> Optional[FrameResult]:
        with self._lock:
            return self._latest_result

    # ──────────────────────────────────────────────────────────────────────────
    # Private pipeline
    # ──────────────────────────────────────────────────────────────────────────

    def _process_frame(self, frame: np.ndarray, frame_id: int) -> FrameResult:
        latencies: Dict[str, float] = {}

        # Stage 1 — Detection
        t0 = time.perf_counter()
        detections = self.detector.detect(frame, frame_id=frame_id)
        latencies["detection"] = time.perf_counter() - t0

        # Stage 2 — Tracking
        t0 = time.perf_counter()
        tracked_persons = self.tracker.update(
            detections, frame_id=frame_id, frame_shape=frame.shape[:2]
        )
        latencies["tracking"] = time.perf_counter() - t0

        # Stage 3 — Pose Estimation
        t0 = time.perf_counter()
        pose_map = self.pose_estimator.estimate(frame, tracked_persons, frame_id)
        latencies["pose"] = time.perf_counter() - t0

        # Stage 4 — Emotion Analysis
        t0 = time.perf_counter()
        emotion_map = self.emotion_analyzer.analyze(frame, tracked_persons)
        latencies["emotion"] = time.perf_counter() - t0

        # Stage 5 — Behavioral Profiling
        self.profiler.update(tracked_persons, pose_map, emotion_map, frame_id)

        # Stage 6 — Anomaly Detection
        t0 = time.perf_counter()
        feature_vectors = self.profiler.get_feature_vectors(
            active_ids=[p.track_id for p in tracked_persons]
        )
        anomaly_results = self.anomaly_detector.score_all(feature_vectors)
        latencies["anomaly"] = time.perf_counter() - t0

        # Stage 7 — Insight Generation
        insights = self.insight_generator.generate(
            anomaly_results, self.profiler, emotion_map
        )

        # Annotate frame
        annotated_frame = None
        if self.output_annotated and frame is not None:
            annotated_frame = self._annotate(
                frame, tracked_persons, pose_map, anomaly_results
            )

        emotion_summary = self.emotion_analyzer.classroom_emotion_summary()

        return FrameResult(
            frame_id=frame_id,
            timestamp=time.time(),
            frame=annotated_frame,
            n_persons=len(tracked_persons),
            tracked_ids=[p.track_id for p in tracked_persons],
            anomaly_results=anomaly_results,
            insights=insights,
            fps=0.0,  # filled by caller
            stage_latencies=latencies,
            emotion_summary=emotion_summary,
        )

    def _is_similar_to_previous(self, frame: np.ndarray) -> bool:
        """Skip frames with low visual change using MSE of grayscale frames."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
        if self._prev_gray is None:
            self._prev_gray = gray
            return False
        mse = float(np.mean((gray - self._prev_gray) ** 2))
        self._prev_gray = gray
        return mse < self.keyframe_diff_threshold

    def _annotate(self, frame, tracked_persons, pose_map, anomaly_results) -> np.ndarray:
        """Overlay tracks, skeletons, and anomaly scores on frame."""
        out = self.tracker.draw_tracks(frame, tracked_persons)
        out = self.pose_estimator.draw_skeletons(out, pose_map)
        out = self.emotion_analyzer.draw_emotions(out, tracked_persons)

        # Draw anomaly score badges
        for person in tracked_persons:
            ar = anomaly_results.get(person.track_id)
            if ar and ar.is_alert:
                x1, y1, _, _ = person.bbox
                sev_colors = {
                    "low": (0, 200, 255),
                    "medium": (0, 100, 255),
                    "high": (0, 0, 220),
                }
                color = sev_colors.get(ar.severity, (180, 180, 180))
                label = f"⚠ {ar.severity.upper()} {ar.combined_score:.2f}"
                cv2.putText(out, label, (x1, y1 - 24),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2, cv2.LINE_AA)

        # FPS counter overlay
        cv2.putText(out, "SafeWatch | LIVE", (10, 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 180), 2, cv2.LINE_AA)
        return out
