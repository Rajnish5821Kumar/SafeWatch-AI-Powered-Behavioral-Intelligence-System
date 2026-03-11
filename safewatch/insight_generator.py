"""
SafeWatch — Insight Generator (Explainable AI)
───────────────────────────────────────────────
Translates raw anomaly scores + behavioral profiles into
human-readable, contextual safety insights for educators.
This is the XAI (Explainable AI) layer of SafeWatch.
"""

from __future__ import annotations
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional
from loguru import logger

from safewatch.anomaly_detector import AnomalyResult


@dataclass
class SafetyInsight:
    """A single generated safety insight / alert."""
    track_id: int
    severity: str        # "low", "medium", "high"
    headline: str        # Short educator-facing title
    description: str     # Full human-readable explanation
    evidence: List[str]  # List of contributing behavioral signals
    recommended_action: str
    timestamp: float = field(default_factory=time.time)
    anomaly_score: float = 0.0

    def to_dict(self) -> dict:
        return {
            "track_id": self.track_id,
            "severity": self.severity,
            "headline": self.headline,
            "description": self.description,
            "evidence": self.evidence,
            "recommended_action": self.recommended_action,
            "timestamp": self.timestamp,
            "anomaly_score": round(self.anomaly_score, 3),
            "id": f"alert_{self.track_id}_{int(self.timestamp)}",
        }


class InsightGenerator:
    """
    Rule-based + template NLG insight generator.

    Converts anomaly scores and behavioral features into plain-language
    alerts that non-technical educators can immediately understand and act on.

    Example
    -------
    >>> gen = InsightGenerator()
    >>> insights = gen.generate(anomaly_results, profiler, emotion_map)
    """

    # Debounce: don't re-alert the same person within this many seconds
    ALERT_COOLDOWN_SEC = 30.0

    SEVERITY_COLORS = {
        "low": "#F59E0B",
        "medium": "#EF4444",
        "high": "#7C3AED",
    }

    def __init__(self, min_alert_interval_sec: float = 30.0):
        self.min_alert_interval = min_alert_interval_sec
        self._last_alert_ts: Dict[int, float] = {}

    def generate(
        self,
        anomaly_results: Dict[int, AnomalyResult],
        profiler,             # BehavioralProfiler
        emotion_map: dict,    # Dict[int, EmotionReading]
    ) -> List[SafetyInsight]:
        """
        Generate human-readable insights for anomalous persons.

        Only produces alerts for persons above the alert threshold
        and respects the per-person debounce cooldown.
        """
        insights: List[SafetyInsight] = []
        now = time.time()

        for track_id, result in anomaly_results.items():
            if not result.is_alert:
                continue

            # Debounce check
            last_ts = self._last_alert_ts.get(track_id, 0.0)
            if (now - last_ts) < self.min_alert_interval:
                continue

            # Build contextual evidence list
            profile = profiler.get_profile(track_id)
            emotion = emotion_map.get(track_id)

            evidence = self._gather_evidence(result, profile, emotion)
            headline, description, action = self._compose_narrative(
                track_id, result, profile, emotion
            )

            insight = SafetyInsight(
                track_id=track_id,
                severity=result.severity,
                headline=headline,
                description=description,
                evidence=evidence,
                recommended_action=action,
                anomaly_score=result.combined_score,
            )

            insights.append(insight)
            self._last_alert_ts[track_id] = now
            logger.warning(
                f"[{result.severity.upper()}] Alert for Track {track_id} "
                f"(score={result.combined_score:.2f}): {headline}"
            )

        return insights

    # ──────────────────────────────────────────────────────────────────────────
    # Private narrative construction helpers
    # ──────────────────────────────────────────────────────────────────────────

    def _gather_evidence(self, result: AnomalyResult, profile, emotion) -> List[str]:
        """Build a list of contributing behavioral signals as plain English strings."""
        evidence = []

        if profile:
            if profile.isolation_score > 0.65:
                evidence.append(
                    f"High social isolation score ({profile.isolation_score:.0%}) — "
                    "person is positioned significantly apart from peers."
                )
            if profile.avg_motion_velocity < 1.5:
                evidence.append(
                    "Near-zero movement for extended period — possible disengagement or distress."
                )
            elif profile.avg_motion_velocity > 15.0:
                evidence.append(
                    f"Elevated motion velocity ({profile.avg_motion_velocity:.1f} px/frame) — "
                    "unusual agitation or restlessness detected."
                )
            if profile.avg_slouch_score > 0.65:
                evidence.append(
                    f"Persistent slouched posture ({profile.avg_slouch_score:.0%}) — "
                    "may indicate fatigue, low energy, or discomfort."
                )
            if profile.avg_head_tilt > 25:
                evidence.append(
                    f"Significant head tilt ({profile.avg_head_tilt:.0f}°) over extended period."
                )
            if profile.motion_entropy > 0.75:
                evidence.append(
                    "High motion entropy — erratic, unpredictable movement pattern detected."
                )

        if emotion:
            if emotion.dominant_emotion in ("angry", "fear", "sad"):
                evidence.append(
                    f"Dominant facial expression: {emotion.dominant_emotion} "
                    f"(valence={emotion.valence:+.1f}) sustained over observation window."
                )

        if result.lstm_score > 0.5:
            evidence.append(
                f"LSTM temporal model flagged sustained behavioral deviation "
                f"(reconstruction error score: {result.lstm_score:.2f})."
            )

        if not evidence:
            evidence.append(
                f"Combined behavioral anomaly score exceeded threshold "
                f"({result.combined_score:.2f} > threshold)."
            )

        return evidence

    def _compose_narrative(
        self,
        track_id: int,
        result: AnomalyResult,
        profile,
        emotion,
    ):
        """Generate headline, description, and recommended action."""

        tid = track_id
        score = result.combined_score
        sev = result.severity

        # Pick headline template based on primary signal
        iso_score = profile.isolation_score if profile else 0.0
        vel = profile.avg_motion_velocity if profile else 0.0
        emo_name = emotion.dominant_emotion if emotion else "neutral"

        if iso_score > 0.65 and emo_name in ("sad", "fear"):
            headline = f"Student {tid} — Possible signs of social withdrawal and distress"
            description = (
                f"Our system has detected that Student ID {tid} has been positioned away "
                f"from peers and showing '{emo_name}' facial expressions for an extended period. "
                f"This combination may indicate emotional distress, social isolation, or a personal "
                f"difficulty that warrants a quiet check-in by an educator."
            )
            action = "Consider privately checking in with this student to ask how they are doing."

        elif vel > 15.0:
            headline = f"Student {tid} — Elevated agitation or restlessness detected"
            description = (
                f"Student ID {tid} is showing significantly elevated and erratic movement "
                f"patterns compared to classroom norms. This may indicate emotional dysregulation, "
                f"anxiety, or an escalating interpersonal situation. Immediate but calm attention "
                f"from a teacher or counsellor is recommended."
            )
            action = "Observe the student and consider a calm, non-confrontational engagement."

        elif emo_name == "angry" and sev in ("medium", "high"):
            headline = f"Student {tid} — Sustained anger expression detected"
            description = (
                f"Student ID {tid} has exhibited facial expressions consistent with anger "
                f"for a prolonged period (anomaly score: {score:.2f}). "
                f"Combined with abnormal behavioral patterns, this may signal an interpersonal "
                f"conflict or unresolved frustration."
            )
            action = "Monitor closely. If the student appears distressed, involve school counselling."

        elif profile and profile.avg_slouch_score > 0.7 and vel < 2.0:
            headline = f"Student {tid} — Prolonged disengagement and fatigue signals"
            description = (
                f"Student ID {tid} has been largely stationary with a consistently slouched "
                f"posture for over {int(profile.seconds_in_scene())}s in the current session. "
                f"This pattern may indicate extreme fatigue, illness, or emotional withdrawal."
            )
            action = "Check if the student needs a break, medical attention, or emotional support."

        else:
            headline = f"Student {tid} — Unusual behavioral pattern detected ({sev} severity)"
            description = (
                f"Our behavioral AI has flagged Student ID {tid} with an anomaly score of "
                f"{score:.2f}. The combination of motion, posture, and emotional signals "
                f"deviates significantly from typical classroom behavior norms. "
                f"Human review is recommended."
            )
            action = "Review the annotated video evidence and use educator judgement to determine next steps."

        return headline, description, action
