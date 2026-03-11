"""
SafeWatch — Pose Estimator
───────────────────────────
Uses YOLOv8-Pose to extract 17-keypoint COCO skeletons per tracked person.
Computes higher-level posture features: head tilt, shoulder symmetry,
slouch score, and estimated arm position.
"""

from __future__ import annotations
import numpy as np
import cv2
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple

from ultralytics import YOLO
from loguru import logger


# COCO 17-keypoint indices
KP = {
    "nose": 0, "left_eye": 1, "right_eye": 2,
    "left_ear": 3, "right_ear": 4,
    "left_shoulder": 5, "right_shoulder": 6,
    "left_elbow": 7, "right_elbow": 8,
    "left_wrist": 9, "right_wrist": 10,
    "left_hip": 11, "right_hip": 12,
    "left_knee": 13, "right_knee": 14,
    "left_ankle": 15, "right_ankle": 16,
}

SKELETON_EDGES = [
    (KP["nose"], KP["left_eye"]), (KP["nose"], KP["right_eye"]),
    (KP["left_eye"], KP["left_ear"]), (KP["right_eye"], KP["right_ear"]),
    (KP["left_shoulder"], KP["right_shoulder"]),
    (KP["left_shoulder"], KP["left_elbow"]),
    (KP["right_shoulder"], KP["right_elbow"]),
    (KP["left_elbow"], KP["left_wrist"]),
    (KP["right_elbow"], KP["right_wrist"]),
    (KP["left_shoulder"], KP["left_hip"]),
    (KP["right_shoulder"], KP["right_hip"]),
    (KP["left_hip"], KP["right_hip"]),
    (KP["left_hip"], KP["left_knee"]),
    (KP["right_hip"], KP["right_knee"]),
    (KP["left_knee"], KP["left_ankle"]),
    (KP["right_knee"], KP["right_ankle"]),
]


@dataclass
class PoseFeatures:
    """Extracted posture features for one person."""
    track_id: int
    keypoints: np.ndarray           # (17, 3) — x, y, confidence
    head_tilt_deg: float = 0.0      # Head rotation in degrees
    shoulder_asymmetry: float = 0.0 # Shoulder height difference (normalised)
    slouch_score: float = 0.0       # 0=upright, 1=heavily slouched
    arm_raise_left: float = 0.0     # Left arm raise ratio [0-1]
    arm_raise_right: float = 0.0    # Right arm raise ratio [0-1]
    body_orientation: str = "frontal"  # "frontal", "side", "back"
    overall_confidence: float = 0.0    # Mean keypoint confidence

    def is_valid(self, min_confidence: float = 0.35) -> bool:
        return self.overall_confidence >= min_confidence

    def to_feature_vector(self) -> np.ndarray:
        """Returns a compact feature vector for behavioral profiling."""
        orientation_enc = {"frontal": 0.0, "side": 0.5, "back": 1.0}
        return np.array([
            self.head_tilt_deg / 45.0,      # normalise to [-1, 1]
            self.shoulder_asymmetry,
            self.slouch_score,
            self.arm_raise_left,
            self.arm_raise_right,
            orientation_enc.get(self.body_orientation, 0.0),
            self.overall_confidence,
        ], dtype=np.float32)


class PoseEstimator:
    """
    YOLOv8-Pose skeleton estimator.

    Runs pose estimation per person crop (or full-frame with person filtering)
    and computes rich posture features for downstream behavioral modeling.

    Example
    -------
    >>> estimator = PoseEstimator("yolov8n-pose.pt")
    >>> features = estimator.estimate(frame, tracked_persons)
    """

    def __init__(
        self,
        model_path: str = "yolov8n-pose.pt",
        device: str = "cuda",
        min_keypoint_confidence: float = 0.35,
        input_size: int = 640,
    ):
        self.min_kp_conf = min_keypoint_confidence
        self.device = device

        logger.info(f"Loading pose model: {model_path} on {device}")
        self.model = YOLO(model_path)
        self.model.to(device)

    def estimate(
        self,
        frame: np.ndarray,
        tracked_persons: list,  # List[TrackedPerson]
        frame_id: int = 0,
    ) -> Dict[int, PoseFeatures]:
        """
        Run pose estimation on the full frame and associate results with track IDs.

        Parameters
        ----------
        frame : np.ndarray
            BGR video frame.
        tracked_persons : list
            List of TrackedPerson objects with track_id and bbox.
        frame_id : int
            Current frame index.

        Returns
        -------
        Dict[int, PoseFeatures]
            Map of track_id → PoseFeatures.
        """
        results = self.model.predict(
            source=frame,
            imgsz=640,
            conf=0.30,
            device=self.device,
            verbose=False,
        )

        pose_map: Dict[int, PoseFeatures] = {}

        if not results or results[0].keypoints is None:
            return pose_map

        kps_data = results[0].keypoints.data.cpu().numpy()  # (N_poses, 17, 3)
        xyxy_data = results[0].boxes.xyxy.cpu().numpy()     # (N_poses, 4)

        # Match each pose result to the nearest tracked person by IoU
        for pose_idx in range(len(kps_data)):
            kps = kps_data[pose_idx]          # (17, 3)
            pose_box = xyxy_data[pose_idx]    # (4,) x1,y1,x2,y2

            best_track_id, best_iou = self._match_pose_to_track(
                pose_box, tracked_persons
            )
            if best_track_id is None or best_iou < 0.30:
                continue

            features = self._compute_features(kps, best_track_id)
            pose_map[best_track_id] = features

        return pose_map

    # ──────────────────────────────────────────────────────────────────────────
    # Private helpers
    # ──────────────────────────────────────────────────────────────────────────

    def _match_pose_to_track(
        self,
        pose_box: np.ndarray,
        tracked_persons: list,
    ) -> Tuple[Optional[int], float]:
        """Find the tracked person whose bounding box best overlaps the pose box."""
        best_id = None
        best_iou = 0.0
        for person in tracked_persons:
            iou = self._box_iou(pose_box, np.array(person.bbox))
            if iou > best_iou:
                best_iou = iou
                best_id = person.track_id
        return best_id, best_iou

    @staticmethod
    def _box_iou(a: np.ndarray, b: np.ndarray) -> float:
        ax1, ay1, ax2, ay2 = a
        bx1, by1, bx2, by2 = b
        ix1, iy1 = max(ax1, bx1), max(ay1, by1)
        ix2, iy2 = min(ax2, bx2), min(ay2, by2)
        inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
        if inter == 0:
            return 0.0
        area_a = (ax2 - ax1) * (ay2 - ay1)
        area_b = (bx2 - bx1) * (by2 - by1)
        return inter / (area_a + area_b - inter + 1e-6)

    def _compute_features(self, kps: np.ndarray, track_id: int) -> PoseFeatures:
        """Derive posture metrics from raw (17, 3) keypoint array."""
        avg_conf = float(np.mean(kps[:, 2]))

        def kp(name: str) -> Tuple[float, float, float]:
            idx = KP[name]
            return kps[idx, 0], kps[idx, 1], kps[idx, 2]

        # Head tilt — angle between left_ear → right_ear
        le_x, le_y, le_c = kp("left_ear")
        re_x, re_y, re_c = kp("right_ear")
        head_tilt = 0.0
        if le_c > self.min_kp_conf and re_c > self.min_kp_conf:
            head_tilt = float(np.degrees(np.arctan2(re_y - le_y, re_x - le_x)))

        # Shoulder asymmetry — normalised height difference
        ls_x, ls_y, ls_c = kp("left_shoulder")
        rs_x, rs_y, rs_c = kp("right_shoulder")
        shoulder_asym = 0.0
        if ls_c > self.min_kp_conf and rs_c > self.min_kp_conf:
            shoulder_asym = abs(ls_y - rs_y) / max(abs(ls_x - rs_x), 1.0)
            shoulder_asym = min(shoulder_asym, 1.0)

        # Slouch — ratio of torso length vs ideal
        lh_x, lh_y, lh_c = kp("left_hip")
        rh_x, rh_y, rh_c = kp("right_hip")
        slouch = 0.0
        if (ls_c > self.min_kp_conf and rs_c > self.min_kp_conf and
                lh_c > self.min_kp_conf and rh_c > self.min_kp_conf):
            torso_h = abs(((ls_y + rs_y) / 2) - ((lh_y + rh_y) / 2))
            torso_w = abs(ls_x - rs_x)
            ideal_ratio = 1.8
            actual_ratio = torso_h / max(torso_w, 1.0)
            slouch = float(np.clip(1.0 - (actual_ratio / ideal_ratio), 0.0, 1.0))

        # Arm raises
        lw_x, lw_y, lw_c = kp("left_wrist")
        rw_x, rw_y, rw_c = kp("right_wrist")
        arm_raise_l = 0.0
        arm_raise_r = 0.0
        if ls_c > self.min_kp_conf and lw_c > self.min_kp_conf:
            # Wrist above shoulder → raise = 1.0
            arm_raise_l = float(np.clip((ls_y - lw_y) / max(abs(ls_y - lh_y), 1.0), 0.0, 1.0))
        if rs_c > self.min_kp_conf and rw_c > self.min_kp_conf:
            arm_raise_r = float(np.clip((rs_y - rw_y) / max(abs(rs_y - rh_y), 1.0), 0.0, 1.0))

        # Body orientation: use ear visibility
        orientation = "frontal"
        if le_c < 0.2 or re_c < 0.2:
            orientation = "side"
        if le_c < 0.15 and re_c < 0.15:
            orientation = "back"

        return PoseFeatures(
            track_id=track_id,
            keypoints=kps,
            head_tilt_deg=head_tilt,
            shoulder_asymmetry=shoulder_asym,
            slouch_score=slouch,
            arm_raise_left=arm_raise_l,
            arm_raise_right=arm_raise_r,
            body_orientation=orientation,
            overall_confidence=avg_conf,
        )

    def draw_skeletons(
        self,
        frame: np.ndarray,
        pose_map: Dict[int, PoseFeatures],
        color: Tuple[int, int, int] = (0, 255, 200),
    ) -> np.ndarray:
        """Overlay COCO skeletons on the frame."""
        out = frame.copy()
        for track_id, feat in pose_map.items():
            kps = feat.keypoints
            # Draw keypoints
            for idx in range(17):
                x, y, c = int(kps[idx, 0]), int(kps[idx, 1]), kps[idx, 2]
                if c > self.min_kp_conf:
                    cv2.circle(out, (x, y), 3, color, -1)
            # Draw skeleton edges
            for p, q in SKELETON_EDGES:
                xp, yp, cp = int(kps[p, 0]), int(kps[p, 1]), kps[p, 2]
                xq, yq, cq = int(kps[q, 0]), int(kps[q, 1]), kps[q, 2]
                if cp > self.min_kp_conf and cq > self.min_kp_conf:
                    cv2.line(out, (xp, yp), (xq, yq), color, 1, cv2.LINE_AA)
        return out
