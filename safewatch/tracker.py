"""
SafeWatch — Multi-Object Tracker
──────────────────────────────────
Wraps ByteTrack (via supervision) to maintain persistent person IDs
across frames. Handles occlusion robustly in high-density classroom scenes.
"""

from __future__ import annotations
import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
import cv2

import supervision as sv
from loguru import logger

from safewatch.detector import Detection


@dataclass
class TrackedPerson:
    """A tracked person with persistent ID and trajectory history."""
    track_id: int
    bbox: Tuple[int, int, int, int]   # (x1, y1, x2, y2)
    confidence: float
    frame_id: int
    zone_id: Optional[str] = None

    # Trajectory: list of (frame_id, center_x, center_y)
    trajectory: List[Tuple[int, int, int]] = field(default_factory=list)

    @property
    def center(self) -> Tuple[int, int]:
        x1, y1, x2, y2 = self.bbox
        return ((x1 + x2) // 2, (y1 + y2) // 2)

    @property
    def width(self) -> int:
        return self.bbox[2] - self.bbox[0]

    @property
    def height(self) -> int:
        return self.bbox[3] - self.bbox[1]

    def update_trajectory(self, frame_id: int):
        cx, cy = self.center
        self.trajectory.append((frame_id, cx, cy))
        # Keep only the last 300 positions (~20s at 15fps)
        if len(self.trajectory) > 300:
            self.trajectory = self.trajectory[-300:]

    def motion_velocity(self) -> float:
        """Average pixel displacement per frame over recent trajectory."""
        if len(self.trajectory) < 2:
            return 0.0
        positions = [(t[1], t[2]) for t in self.trajectory[-30:]]
        deltas = [
            np.sqrt((positions[i][0] - positions[i-1][0])**2 +
                    (positions[i][1] - positions[i-1][1])**2)
            for i in range(1, len(positions))
        ]
        return float(np.mean(deltas)) if deltas else 0.0

    def is_stationary(self, velocity_threshold: float = 2.0) -> bool:
        return self.motion_velocity() < velocity_threshold


class MultiObjectTracker:
    """
    ByteTrack-based multi-person tracker using supervision library.

    Maintains persistent track IDs across frames and enriches each
    tracked person with trajectory data for downstream behavioral analysis.

    Example
    -------
    >>> tracker = MultiObjectTracker()
    >>> tracked = tracker.update(detections, frame_id=42)
    """

    def __init__(
        self,
        track_threshold: float = 0.40,
        match_threshold: float = 0.80,
        frame_rate: int = 15,
        track_buffer: int = 50,
    ):
        self.track_threshold = track_threshold
        self.match_threshold = match_threshold
        self.frame_rate = frame_rate
        self.track_buffer = track_buffer

        self._tracker = sv.ByteTracker(
            track_activation_threshold=track_threshold,
            lost_track_buffer=track_buffer,
            minimum_matching_threshold=match_threshold,
            frame_rate=frame_rate,
        )

        # Persistent history keyed by track_id
        self._person_history: Dict[int, TrackedPerson] = {}
        self._active_ids: set = set()

        logger.info(
            f"ByteTrack initialized | threshold={track_threshold} "
            f"buffer={track_buffer} fps={frame_rate}"
        )

    def update(
        self,
        detections: List[Detection],
        frame_id: int = 0,
        frame_shape: Optional[Tuple[int, int]] = None,
    ) -> List[TrackedPerson]:
        """
        Update tracker with current frame detections.

        Parameters
        ----------
        detections : List[Detection]
            Person detections from the current frame.
        frame_id : int
            Current frame index.
        frame_shape : Optional[Tuple[int, int]]
            (height, width) of frame — used for coordinate normalisation.

        Returns
        -------
        List[TrackedPerson]
            Active tracked persons with persistent IDs and trajectories.
        """
        if not detections:
            self._active_ids.clear()
            return []

        # Convert to supervision Detections format
        bboxes = np.array([d.bbox for d in detections], dtype=float)  # (N, 4)
        confs = np.array([d.confidence for d in detections], dtype=float)
        class_ids = np.zeros(len(detections), dtype=int)

        sv_dets = sv.Detections(
            xyxy=bboxes,
            confidence=confs,
            class_id=class_ids,
        )

        # Run ByteTrack update
        tracked_sv = self._tracker.update_with_detections(sv_dets)

        tracked_persons: List[TrackedPerson] = []
        new_active_ids: set = set()

        for i in range(len(tracked_sv)):
            xyxy = tracked_sv.xyxy[i].astype(int)
            bbox = (int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3]))
            conf = float(tracked_sv.confidence[i]) if tracked_sv.confidence is not None else 1.0
            track_id = int(tracked_sv.tracker_id[i])

            # Retrieve or create history
            if track_id not in self._person_history:
                self._person_history[track_id] = TrackedPerson(
                    track_id=track_id,
                    bbox=bbox,
                    confidence=conf,
                    frame_id=frame_id,
                )
            else:
                self._person_history[track_id].bbox = bbox
                self._person_history[track_id].confidence = conf
                self._person_history[track_id].frame_id = frame_id

            person = self._person_history[track_id]
            person.update_trajectory(frame_id)
            tracked_persons.append(person)
            new_active_ids.add(track_id)

        self._active_ids = new_active_ids
        return tracked_persons

    def get_person_history(self, track_id: int) -> Optional[TrackedPerson]:
        return self._person_history.get(track_id)

    @property
    def active_count(self) -> int:
        return len(self._active_ids)

    @property
    def total_tracks_seen(self) -> int:
        return len(self._person_history)

    def draw_tracks(
        self,
        frame: np.ndarray,
        tracked_persons: List[TrackedPerson],
        draw_trajectory: bool = True,
    ) -> np.ndarray:
        """Annotate frame with track IDs, bboxes, and motion trails."""
        out = frame.copy()
        colors = [
            (0, 200, 255), (0, 255, 128), (255, 100, 100),
            (200, 0, 255), (255, 200, 0), (0, 128, 255),
        ]

        for person in tracked_persons:
            color = colors[person.track_id % len(colors)]
            x1, y1, x2, y2 = person.bbox
            cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
            label = f"ID:{person.track_id}"
            cv2.putText(out, label, (x1, y1 - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA)

            # Draw trajectory trail
            if draw_trajectory and len(person.trajectory) > 1:
                pts = [(t[1], t[2]) for t in person.trajectory[-25:]]
                for j in range(1, len(pts)):
                    alpha = j / len(pts)
                    trail_color = tuple(int(c * alpha) for c in color)
                    cv2.line(out, pts[j-1], pts[j], trail_color, 1)

        return out
