"""
SafeWatch — Emotion Analyzer
──────────────────────────────
Detects facial emotions for each tracked person using the FER library.
Applies exponential smoothing to stabilise predictions over time.
Uses a privacy-first approach: face crops are never stored to disk.
"""

from __future__ import annotations
import numpy as np
import cv2
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from loguru import logger


EMOTIONS = ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"]

# Valence scores: negative → positive mapping for behavioral risk scoring
EMOTION_VALENCE: Dict[str, float] = {
    "angry":    -1.0,
    "disgust":  -0.8,
    "fear":     -0.9,
    "sad":      -0.7,
    "surprise":  0.2,
    "neutral":   0.1,
    "happy":     1.0,
}


@dataclass
class EmotionReading:
    """Smoothed emotion state for one person."""
    track_id: int
    dominant_emotion: str = "neutral"
    scores: Dict[str, float] = field(default_factory=lambda: {e: 0.0 for e in EMOTIONS})
    valence: float = 0.0           # [-1, 1] — negative = distress
    confidence: float = 0.0        # Face detection confidence
    frames_analyzed: int = 0

    def update(self, new_scores: Dict[str, float], alpha: float = 0.3):
        """Exponential smoothing update."""
        for emo in EMOTIONS:
            prev = self.scores.get(emo, 0.0)
            self.scores[emo] = alpha * new_scores.get(emo, 0.0) + (1 - alpha) * prev
        self.dominant_emotion = max(self.scores, key=self.scores.get)
        self.valence = EMOTION_VALENCE.get(self.dominant_emotion, 0.0)
        self.frames_analyzed += 1

    def risk_level(self) -> str:
        """Map valence to a simple 3-tier risk label."""
        if self.valence >= 0.0:
            return "low"
        elif self.valence >= -0.7:
            return "medium"
        return "high"

    def to_feature_vector(self) -> np.ndarray:
        return np.array([self.scores.get(e, 0.0) for e in EMOTIONS] + [self.valence],
                        dtype=np.float32)


class EmotionAnalyzer:
    """
    FER-based facial emotion recognizer with temporal smoothing.

    Privacy-first: face crops are processed in-memory and never persisted.
    Supports per-person smoothed emotion states tracked across frames.

    Example
    -------
    >>> analyzer = EmotionAnalyzer(smoothing_alpha=0.3)
    >>> readings = analyzer.analyze(frame, tracked_persons)
    """

    def __init__(
        self,
        smoothing_alpha: float = 0.30,
        min_face_size: int = 32,
        detect_faces: bool = True,
    ):
        self.smoothing_alpha = smoothing_alpha
        self.min_face_size = min_face_size
        self.detect_faces = detect_faces

        self._emotion_states: Dict[int, EmotionReading] = {}

        # Lazy-load FER to avoid slow import at module level
        self._fer = None
        logger.info("EmotionAnalyzer ready (FER will be loaded on first use)")

    def _get_fer(self):
        if self._fer is None:
            try:
                from fer import FER
                self._fer = FER(mtcnn=False)
                logger.info("FER model loaded successfully")
            except ImportError:
                logger.warning("FER not installed — emotion analysis disabled")
                self._fer = "disabled"
        return self._fer if self._fer != "disabled" else None

    def analyze(
        self,
        frame: np.ndarray,
        tracked_persons: list,  # List[TrackedPerson]
    ) -> Dict[int, EmotionReading]:
        """
        Analyze emotions for each tracked person.

        Parameters
        ----------
        frame : np.ndarray
            BGR video frame.
        tracked_persons : list
            TrackedPerson objects with bbox and track_id.

        Returns
        -------
        Dict[int, EmotionReading]
            Smoothed emotion state per track_id.
        """
        fer_model = self._get_fer()

        for person in tracked_persons:
            track_id = person.track_id

            # Ensure state exists
            if track_id not in self._emotion_states:
                self._emotion_states[track_id] = EmotionReading(track_id=track_id)

            if fer_model is None:
                # Return neutral fallback
                continue

            # Extract face crop from upper region of person bbox
            face_crop = self._extract_face_region(frame, person.bbox)
            if face_crop is None:
                continue

            try:
                raw_results = fer_model.detect_emotions(face_crop)
                if not raw_results:
                    continue

                # Take the highest-confidence face detection
                best = max(raw_results, key=lambda r: sum(r["emotions"].values()))
                new_scores = best["emotions"]
                self._emotion_states[track_id].update(new_scores, self.smoothing_alpha)
            except Exception as e:
                logger.debug(f"Emotion analysis error for track {track_id}: {e}")

        return self._emotion_states

    def _extract_face_region(
        self, frame: np.ndarray, bbox: Tuple[int, int, int, int]
    ) -> Optional[np.ndarray]:
        """
        Crop the expected face region from the person bounding box.
        Assumes face occupies roughly the top 30% of the person box.
        """
        x1, y1, x2, y2 = bbox
        h = y2 - y1
        w = x2 - x1

        # Top 30% of the person bbox = likely face
        face_y2 = y1 + int(h * 0.32)
        face_x1 = x1 + int(w * 0.15)
        face_x2 = x2 - int(w * 0.15)

        # Clamp to frame bounds
        H, W = frame.shape[:2]
        face_x1 = max(0, face_x1)
        face_y1 = max(0, y1)
        face_x2 = min(W, face_x2)
        face_y2 = min(H, face_y2)

        if (face_x2 - face_x1) < self.min_face_size or (face_y2 - face_y1) < self.min_face_size:
            return None

        return frame[face_y1:face_y2, face_x1:face_x2]

    def get_state(self, track_id: int) -> EmotionReading:
        """Get or create emotion state for a track."""
        if track_id not in self._emotion_states:
            self._emotion_states[track_id] = EmotionReading(track_id=track_id)
        return self._emotion_states[track_id]

    def classroom_emotion_summary(self) -> Dict[str, float]:
        """
        Aggregate emotion distribution across all active persons.
        Returns proportion of each emotion across the classroom.
        """
        if not self._emotion_states:
            return {e: 0.0 for e in EMOTIONS}

        totals = {e: 0.0 for e in EMOTIONS}
        count = len(self._emotion_states)
        for state in self._emotion_states.values():
            for emo in EMOTIONS:
                totals[emo] += state.scores.get(emo, 0.0)

        return {e: v / count for e, v in totals.items()}

    def draw_emotions(
        self,
        frame: np.ndarray,
        tracked_persons: list,
    ) -> np.ndarray:
        """Overlay emotion labels on tracked persons."""
        out = frame.copy()
        emotion_colors = {
            "happy": (0, 220, 100),
            "neutral": (180, 180, 180),
            "sad": (200, 100, 50),
            "angry": (0, 0, 220),
            "fear": (0, 180, 220),
            "surprise": (220, 180, 0),
            "disgust": (100, 50, 200),
        }
        for person in tracked_persons:
            state = self._emotion_states.get(person.track_id)
            if state is None:
                continue
            x1, y1, x2, y2 = person.bbox
            emo = state.dominant_emotion
            col = emotion_colors.get(emo, (180, 180, 180))
            cv2.putText(
                out, f"{emo} {state.valence:+.1f}",
                (x1, y2 + 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, col, 1, cv2.LINE_AA
            )
        return out
