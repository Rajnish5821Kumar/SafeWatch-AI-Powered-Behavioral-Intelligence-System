"""
SafeWatch — Person Detector
────────────────────────────
Uses YOLOv8 to detect persons in video frames with GPU acceleration.
Supports batched inference and confidence filtering.
"""

from __future__ import annotations
import numpy as np
import cv2
from dataclasses import dataclass, field
from typing import List, Tuple, Optional
from pathlib import Path

from ultralytics import YOLO
from loguru import logger


@dataclass
class Detection:
    """A single person detection result."""
    bbox: Tuple[int, int, int, int]   # (x1, y1, x2, y2)
    confidence: float
    class_id: int = 0                 # 0 = person in COCO
    frame_id: int = 0

    @property
    def center(self) -> Tuple[int, int]:
        x1, y1, x2, y2 = self.bbox
        return ((x1 + x2) // 2, (y1 + y2) // 2)

    @property
    def area(self) -> int:
        x1, y1, x2, y2 = self.bbox
        return max(0, x2 - x1) * max(0, y2 - y1)

    @property
    def width(self) -> int:
        return self.bbox[2] - self.bbox[0]

    @property
    def height(self) -> int:
        return self.bbox[3] - self.bbox[1]


class PersonDetector:
    """
    YOLOv8-based person detector.

    Example
    -------
    >>> detector = PersonDetector("yolov8n.pt", device="cuda")
    >>> detections = detector.detect(frame)
    """

    PERSON_CLASS_ID = 0

    def __init__(
        self,
        model_path: str = "yolov8n.pt",
        confidence_threshold: float = 0.40,
        iou_threshold: float = 0.50,
        device: str = "cuda",
        input_size: int = 640,
        min_box_area: int = 100,
    ):
        self.confidence_threshold = confidence_threshold
        self.iou_threshold = iou_threshold
        self.device = device
        self.input_size = input_size
        self.min_box_area = min_box_area

        logger.info(f"Loading detection model: {model_path} on {device}")
        self.model = YOLO(model_path)
        self.model.to(device)

        self._frame_count = 0
        self._total_detections = 0

    def detect(self, frame: np.ndarray, frame_id: int = 0) -> List[Detection]:
        """
        Run person detection on a single BGR frame.

        Parameters
        ----------
        frame : np.ndarray
            BGR image array from OpenCV.
        frame_id : int
            Frame index for tracking lineage.

        Returns
        -------
        List[Detection]
            Detected person bounding boxes with confidence scores.
        """
        self._frame_count += 1

        results = self.model.predict(
            source=frame,
            imgsz=self.input_size,
            conf=self.confidence_threshold,
            iou=self.iou_threshold,
            classes=[self.PERSON_CLASS_ID],
            device=self.device,
            verbose=False,
        )

        detections: List[Detection] = []
        if not results or results[0].boxes is None:
            return detections

        boxes = results[0].boxes
        for i in range(len(boxes)):
            xyxy = boxes.xyxy[i].cpu().numpy().astype(int)
            conf = float(boxes.conf[i].cpu().numpy())
            x1, y1, x2, y2 = xyxy
            bbox = (int(x1), int(y1), int(x2), int(y2))
            det = Detection(bbox=bbox, confidence=conf, frame_id=frame_id)
            if det.area >= self.min_box_area:
                detections.append(det)

        self._total_detections += len(detections)
        return detections

    def detect_batch(self, frames: List[np.ndarray]) -> List[List[Detection]]:
        """Run detection on a batch of frames for higher GPU throughput."""
        results = self.model.predict(
            source=frames,
            imgsz=self.input_size,
            conf=self.confidence_threshold,
            iou=self.iou_threshold,
            classes=[self.PERSON_CLASS_ID],
            device=self.device,
            verbose=False,
        )

        batch_detections: List[List[Detection]] = []
        for frame_id, result in enumerate(results):
            frame_dets = []
            if result.boxes is not None:
                boxes = result.boxes
                for i in range(len(boxes)):
                    xyxy = boxes.xyxy[i].cpu().numpy().astype(int)
                    conf = float(boxes.conf[i].cpu().numpy())
                    x1, y1, x2, y2 = xyxy
                    det = Detection(
                        bbox=(int(x1), int(y1), int(x2), int(y2)),
                        confidence=conf,
                        frame_id=frame_id,
                    )
                    if det.area >= self.min_box_area:
                        frame_dets.append(det)
            batch_detections.append(frame_dets)

        return batch_detections

    def draw_detections(self, frame: np.ndarray, detections: List[Detection]) -> np.ndarray:
        """Draw bounding boxes on frame for debugging."""
        out = frame.copy()
        for det in detections:
            x1, y1, x2, y2 = det.bbox
            cv2.rectangle(out, (x1, y1), (x2, y2), (0, 255, 0), 2)
            label = f"Person {det.confidence:.2f}"
            cv2.putText(out, label, (x1, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (0, 255, 0), 1, cv2.LINE_AA)
        return out

    @property
    def stats(self) -> dict:
        return {
            "frames_processed": self._frame_count,
            "total_detections": self._total_detections,
            "avg_detections_per_frame": (
                self._total_detections / max(1, self._frame_count)
            ),
        }
