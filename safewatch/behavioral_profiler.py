"""
SafeWatch — Behavioral Profiler
─────────────────────────────────
Aggregates per-person pose + emotion features over a rolling time window.
Produces a compact behavioral feature vector for anomaly detection.
Tracks social isolation score based on pairwise inter-person distances.
"""

from __future__ import annotations
import numpy as np
from collections import deque
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Deque
import time

from loguru import logger


@dataclass
class BehavioralProfile:
    """
    Rolling behavioral profile for one person.
    Accumulates pose and emotion samples over a sliding window.
    """
    track_id: int
    window_seconds: float = 30.0

    # Sample queues: (timestamp, feature_array)
    pose_samples: Deque = field(default_factory=lambda: deque(maxlen=450))
    emotion_samples: Deque = field(default_factory=lambda: deque(maxlen=450))
    velocity_samples: Deque = field(default_factory=lambda: deque(maxlen=450))

    # Derived aggregated features (updated on each flush)
    avg_motion_velocity: float = 0.0
    motion_entropy: float = 0.0
    avg_emotion_valence: float = 0.0
    dominant_emotion_dist: Dict[str, float] = field(default_factory=dict)
    avg_slouch_score: float = 0.0
    avg_head_tilt: float = 0.0
    isolation_score: float = 0.0   # 0=well-integrated, 1=isolated
    arm_raise_freq: float = 0.0    # Fraction of frames with arm raised
    total_frames: int = 0

    # Metadata
    first_seen_ts: float = field(default_factory=time.time)
    last_updated_ts: float = field(default_factory=time.time)

    def to_feature_vector(self) -> np.ndarray:
        """
        Compact 12-dimensional feature vector for anomaly scoring.
        All fields normalised to [0, 1] or [-1, 1].
        """
        return np.array([
            np.clip(self.avg_motion_velocity / 20.0, 0.0, 1.0),
            np.clip(self.motion_entropy, 0.0, 1.0),
            np.clip((self.avg_emotion_valence + 1.0) / 2.0, 0.0, 1.0),
            np.clip(self.avg_slouch_score, 0.0, 1.0),
            np.clip(self.avg_head_tilt / 45.0, -1.0, 1.0),
            np.clip(self.isolation_score, 0.0, 1.0),
            np.clip(self.arm_raise_freq, 0.0, 1.0),
            self.dominant_emotion_dist.get("angry", 0.0),
            self.dominant_emotion_dist.get("fear", 0.0),
            self.dominant_emotion_dist.get("sad", 0.0),
            self.dominant_emotion_dist.get("happy", 0.0),
            self.dominant_emotion_dist.get("neutral", 0.0),
        ], dtype=np.float32)

    def seconds_in_scene(self) -> float:
        return time.time() - self.first_seen_ts

    def is_stale(self, max_gap_seconds: float = 5.0) -> bool:
        return (time.time() - self.last_updated_ts) > max_gap_seconds


class BehavioralProfiler:
    """
    Maintains rolling behavioral profiles for all active tracked persons.

    Each person gets a profile accumulating pose and emotion readings.
    Every N frames the profile is flushed into a compact feature vector
    suitable for IsolationForest / LSTM anomaly detection.

    Example
    -------
    >>> profiler = BehavioralProfiler(window_seconds=30)
    >>> profiler.update(tracked_persons, pose_map, emotion_map, frame_id=42)
    >>> vectors = profiler.get_feature_vectors()
    """

    def __init__(
        self,
        window_seconds: float = 30.0,
        flush_every_n_frames: int = 5,
    ):
        self.window_seconds = window_seconds
        self.flush_every = flush_every_n_frames
        self._profiles: Dict[int, BehavioralProfile] = {}
        self._frame_count = 0

    def update(
        self,
        tracked_persons: list,
        pose_map: dict,
        emotion_map: dict,
        frame_id: int = 0,
    ) -> None:
        """
        Ingest one frame's worth of tracking + pose + emotion data.

        Parameters
        ----------
        tracked_persons : list of TrackedPerson
        pose_map : dict, track_id → PoseFeatures
        emotion_map : dict, track_id → EmotionReading
        frame_id : int
        """
        self._frame_count += 1
        now = time.time()

        for person in tracked_persons:
            tid = person.track_id
            if tid not in self._profiles:
                self._profiles[tid] = BehavioralProfile(
                    track_id=tid,
                    window_seconds=self.window_seconds,
                )

            profile = self._profiles[tid]
            profile.last_updated_ts = now
            profile.total_frames += 1

            # Velocity sample
            profile.velocity_samples.append(person.motion_velocity())

            # Pose sample
            pose = pose_map.get(tid)
            if pose is not None and pose.is_valid():
                profile.pose_samples.append(pose.to_feature_vector())

            # Emotion sample
            emo = emotion_map.get(tid)
            if emo is not None:
                profile.emotion_samples.append(emo.to_feature_vector())

        # Compute isolation scores using pairwise distances
        self._update_isolation_scores(tracked_persons)

        # Flush aggregate stats periodically
        if self._frame_count % self.flush_every == 0:
            self._flush_aggregates(tracked_persons, pose_map, emotion_map)

    def _update_isolation_scores(self, tracked_persons: list) -> None:
        """Compute per-person isolation score from pairwise centroid distances."""
        if len(tracked_persons) < 2:
            for p in tracked_persons:
                if p.track_id in self._profiles:
                    self._profiles[p.track_id].isolation_score = 1.0
            return

        centers = np.array([list(p.center) for p in tracked_persons], dtype=float)
        ids = [p.track_id for p in tracked_persons]

        for i, tid in enumerate(ids):
            others = np.delete(centers, i, axis=0)
            dists = np.linalg.norm(others - centers[i], axis=1)
            min_dist = float(np.min(dists))

            # Normalise: 200px = isolated threshold
            iso_score = float(np.clip(min_dist / 200.0, 0.0, 1.0))
            if tid in self._profiles:
                # Smooth with previous value
                prev = self._profiles[tid].isolation_score
                self._profiles[tid].isolation_score = 0.8 * prev + 0.2 * iso_score

    def _flush_aggregates(
        self, tracked_persons: list, pose_map: dict, emotion_map: dict
    ) -> None:
        """Recompute aggregate statistics from rolling sample queues."""
        for person in tracked_persons:
            tid = person.track_id
            if tid not in self._profiles:
                continue
            profile = self._profiles[tid]

            # Motion
            if profile.velocity_samples:
                vels = list(profile.velocity_samples)
                profile.avg_motion_velocity = float(np.mean(vels))
                # Motion entropy: high = erratic, low = stationary or smooth
                hist, _ = np.histogram(vels, bins=10, range=(0, 30), density=True)
                hist = hist + 1e-9
                profile.motion_entropy = float(
                    -np.sum(hist * np.log(hist)) / np.log(10)
                )

            # Pose aggregates
            if profile.pose_samples:
                pose_arr = np.array(list(profile.pose_samples))
                profile.avg_head_tilt = float(pose_arr[:, 0].mean() * 45.0)
                profile.avg_slouch_score = float(pose_arr[:, 2].mean())
                profile.arm_raise_freq = float(
                    np.mean((pose_arr[:, 3] > 0.4) | (pose_arr[:, 4] > 0.4))
                )

            # Emotion aggregates
            if profile.emotion_samples:
                emo_arr = np.array(list(profile.emotion_samples))
                # Last column is valence
                profile.avg_emotion_valence = float(emo_arr[:, -1].mean())
                # Columns 0-6 are emotion scores
                emo_means = emo_arr[:, :7].mean(axis=0)
                emotion_labels = ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"]
                profile.dominant_emotion_dist = {
                    k: float(v) for k, v in zip(emotion_labels, emo_means)
                }

    def get_feature_vectors(
        self, active_ids: Optional[List[int]] = None
    ) -> Dict[int, np.ndarray]:
        """
        Returns behavioral feature vectors for all (or specified) active persons.

        Returns
        -------
        Dict[int, np.ndarray]
            track_id → 12-dim feature vector
        """
        result = {}
        ids = active_ids if active_ids else list(self._profiles.keys())
        for tid in ids:
            if tid in self._profiles:
                result[tid] = self._profiles[tid].to_feature_vector()
        return result

    def get_profile(self, track_id: int) -> Optional[BehavioralProfile]:
        return self._profiles.get(track_id)

    def cleanup_stale(self, max_gap_seconds: float = 10.0) -> None:
        """Remove profiles for persons no longer in scene."""
        stale = [tid for tid, p in self._profiles.items() if p.is_stale(max_gap_seconds)]
        for tid in stale:
            del self._profiles[tid]
        if stale:
            logger.debug(f"Cleaned up {len(stale)} stale profiles")

    @property
    def active_profile_count(self) -> int:
        return len(self._profiles)
