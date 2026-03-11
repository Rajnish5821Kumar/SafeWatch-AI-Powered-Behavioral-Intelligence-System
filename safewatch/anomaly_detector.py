"""
SafeWatch — Anomaly Detector
──────────────────────────────
Dual-model behavioral anomaly detection:
  1. IsolationForest  — fast, real-time scoring on current feature vectors
  2. LSTM Autoencoder — temporal anomaly over 60-second behavioral sequences

Both produce normalised anomaly scores [0, 1].
"""

from __future__ import annotations
import numpy as np
import pickle
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
from pathlib import Path

from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from loguru import logger


@dataclass
class AnomalyResult:
    """Anomaly scoring result for one person."""
    track_id: int
    iforest_score: float     # [0, 1] — higher = more anomalous
    lstm_score: float        # [0, 1] — from LSTM Autoencoder (if available)
    combined_score: float    # Weighted combination
    is_alert: bool
    severity: str            # "low", "medium", "high"

    @property
    def primary_score(self) -> float:
        return self.combined_score


class IsolationForestDetector:
    """
    Scikit-learn IsolationForest wrapper.

    Trained on a baseline of normal classroom behavior.
    Supports online partial fitting for continuous adaptation.
    """

    def __init__(
        self,
        contamination: float = 0.05,
        n_estimators: int = 200,
        max_samples: str = "auto",
        random_state: int = 42,
    ):
        self.contamination = contamination
        self.scaler = StandardScaler()
        self.model = IsolationForest(
            contamination=contamination,
            n_estimators=n_estimators,
            max_samples=max_samples,
            random_state=random_state,
            warm_start=False,
        )
        self._is_fitted = False
        self._baseline_buffer: List[np.ndarray] = []
        self._baseline_min_samples = 100   # Need N samples before fitting

    def feed_baseline(self, feature_vectors: List[np.ndarray]) -> None:
        """Accumulate baseline normal-behavior samples."""
        self._baseline_buffer.extend(feature_vectors)
        if len(self._baseline_buffer) >= self._baseline_min_samples and not self._is_fitted:
            self._fit()

    def _fit(self) -> None:
        X = np.array(self._baseline_buffer)
        self.scaler.fit(X)
        X_scaled = self.scaler.transform(X)
        self.model.fit(X_scaled)
        self._is_fitted = True
        logger.info(f"IsolationForest fitted on {len(X)} baseline samples")

    def score(self, feature_vector: np.ndarray) -> float:
        """
        Returns anomaly score in [0, 1].
        0 = completely normal, 1 = highly anomalous.
        """
        if not self._is_fitted:
            return 0.0

        x = feature_vector.reshape(1, -1)
        x_scaled = self.scaler.transform(x)

        # decision_function: negative anomalous, positive normal
        raw_score = self.model.decision_function(x_scaled)[0]
        # Convert: raw∈[-0.5, 0.5] approx → normalise to [0,1] inverted
        normalised = float(np.clip(0.5 - raw_score, 0.0, 1.0))
        return normalised

    def save(self, path: str) -> None:
        with open(path, "wb") as f:
            pickle.dump({"model": self.model, "scaler": self.scaler}, f)
        logger.info(f"IsolationForest saved to {path}")

    def load(self, path: str) -> None:
        if not os.path.exists(path):
            logger.warning(f"Baseline model not found at {path}, will train from scratch")
            return
        with open(path, "rb") as f:
            data = pickle.load(f)
        self.model = data["model"]
        self.scaler = data["scaler"]
        self._is_fitted = True
        logger.info(f"IsolationForest loaded from {path}")


class LSTMAutoencoderDetector:
    """
    LSTM Autoencoder for temporal sequence anomaly detection.

    Detects sustained behavioral anomalies over a 60-second window
    by measuring reconstruction error of the behavioral time series.
    """

    def __init__(
        self,
        sequence_length: int = 60,
        feature_dim: int = 12,
        model_path: Optional[str] = None,
    ):
        self.seq_len = sequence_length
        self.feature_dim = feature_dim
        self.model_path = model_path
        self._model = None
        self._sequences: Dict[int, List[np.ndarray]] = {}  # track_id → history
        self._threshold: float = 0.05   # Reconstruction MSE threshold

        if model_path and os.path.exists(model_path):
            self._load_model(model_path)
        else:
            self._build_model()

    def _build_model(self):
        """Build and compile LSTM Autoencoder architecture."""
        try:
            import tensorflow as tf
            from tensorflow import keras

            inp = keras.Input(shape=(self.seq_len, self.feature_dim))
            # Encoder
            x = keras.layers.LSTM(64, activation="tanh", return_sequences=True)(inp)
            x = keras.layers.Dropout(0.2)(x)
            x = keras.layers.LSTM(32, activation="tanh", return_sequences=False)(x)
            # Bottleneck
            encoded = keras.layers.Dense(16, activation="relu")(x)
            # Decoder
            x = keras.layers.RepeatVector(self.seq_len)(encoded)
            x = keras.layers.LSTM(32, activation="tanh", return_sequences=True)(x)
            x = keras.layers.Dropout(0.2)(x)
            x = keras.layers.LSTM(64, activation="tanh", return_sequences=True)(x)
            decoded = keras.layers.TimeDistributed(
                keras.layers.Dense(self.feature_dim, activation="sigmoid")
            )(x)

            self._model = keras.Model(inp, decoded)
            self._model.compile(optimizer="adam", loss="mse")
            logger.info("LSTM Autoencoder built (untrained — will score 0 until trained)")

        except ImportError:
            logger.warning("TensorFlow not available — LSTM Autoencoder disabled")
            self._model = None

    def _load_model(self, path: str):
        try:
            from tensorflow import keras
            self._model = keras.models.load_model(path)
            logger.info(f"LSTM Autoencoder loaded from {path}")
        except Exception as e:
            logger.warning(f"Failed to load LSTM model: {e}")
            self._build_model()

    def train(self, normal_sequences: np.ndarray, epochs: int = 30, batch_size: int = 32):
        """
        Train the autoencoder on sequences of normal classroom behavior.

        Parameters
        ----------
        normal_sequences : np.ndarray
            Shape (N, seq_len, feature_dim) — normal behavior sequences.
        """
        if self._model is None:
            logger.warning("LSTM model not available, skipping training")
            return

        logger.info(f"Training LSTM Autoencoder on {len(normal_sequences)} sequences...")
        self._model.fit(
            normal_sequences, normal_sequences,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=0.1,
            verbose=1,
            callbacks=[
                __import__("tensorflow").keras.callbacks.EarlyStopping(
                    patience=5, restore_best_weights=True
                )
            ],
        )
        if self.model_path:
            self._model.save(self.model_path)
            logger.info(f"LSTM model saved to {self.model_path}")

    def feed(self, track_id: int, feature_vector: np.ndarray) -> None:
        """Append one frame's feature vector to this person's sequence buffer."""
        if track_id not in self._sequences:
            self._sequences[track_id] = []
        self._sequences[track_id].append(feature_vector)
        # Keep only last seq_len samples
        if len(self._sequences[track_id]) > self.seq_len:
            self._sequences[track_id] = self._sequences[track_id][-self.seq_len:]

    def score(self, track_id: int) -> float:
        """
        Returns temporal anomaly score for a person.
        Returns 0.0 if insufficient history or model unavailable.
        """
        if self._model is None:
            return 0.0

        seq = self._sequences.get(track_id, [])
        if len(seq) < self.seq_len:
            return 0.0

        x = np.array(seq[-self.seq_len:], dtype=np.float32).reshape(1, self.seq_len, self.feature_dim)
        x_recon = self._model.predict(x, verbose=0)
        mse = float(np.mean((x - x_recon) ** 2))
        return float(np.clip(mse / self._threshold, 0.0, 1.0))


class AnomalyDetector:
    """
    Combined anomaly detection pipeline.

    Runs both IsolationForest (real-time) and LSTM Autoencoder (temporal)
    and produces a final weighted anomaly score per person.

    Example
    -------
    >>> detector = AnomalyDetector(alert_threshold=0.72)
    >>> results = detector.score_all(feature_vectors)
    """

    SEVERITY_THRESHOLDS = {
        "low":    (0.72, 0.82),
        "medium": (0.82, 0.92),
        "high":   (0.92, 1.01),
    }

    def __init__(
        self,
        alert_threshold: float = 0.72,
        contamination: float = 0.05,
        lstm_seq_len: int = 60,
        feature_dim: int = 12,
        baseline_path: Optional[str] = None,
        lstm_model_path: Optional[str] = None,
        iforest_weight: float = 0.6,
        lstm_weight: float = 0.4,
    ):
        self.alert_threshold = alert_threshold
        self.iforest_weight = iforest_weight
        self.lstm_weight = lstm_weight

        self.iforest = IsolationForestDetector(contamination=contamination)
        self.lstm = LSTMAutoencoderDetector(
            sequence_length=lstm_seq_len,
            feature_dim=feature_dim,
            model_path=lstm_model_path,
        )

        if baseline_path:
            self.iforest.load(baseline_path)

        logger.info(
            f"AnomalyDetector ready | alert_threshold={alert_threshold} "
            f"iforest_w={iforest_weight} lstm_w={lstm_weight}"
        )

    def score_all(
        self,
        feature_vectors: Dict[int, np.ndarray],
    ) -> Dict[int, AnomalyResult]:
        """
        Score all active persons and produce AnomalyResult per track.

        Parameters
        ----------
        feature_vectors : Dict[int, np.ndarray]
            From BehavioralProfiler.get_feature_vectors()

        Returns
        -------
        Dict[int, AnomalyResult]
        """
        # Feed baseline to IsolationForest
        self.iforest.feed_baseline(list(feature_vectors.values()))

        results: Dict[int, AnomalyResult] = {}

        for track_id, fv in feature_vectors.items():
            # Feed temporal model
            self.lstm.feed(track_id, fv)

            if_score = self.iforest.score(fv)
            lstm_score = self.lstm.score(track_id)

            combined = (
                self.iforest_weight * if_score +
                self.lstm_weight * lstm_score
            )
            combined = float(np.clip(combined, 0.0, 1.0))

            is_alert = combined >= self.alert_threshold
            severity = self._get_severity(combined)

            results[track_id] = AnomalyResult(
                track_id=track_id,
                iforest_score=if_score,
                lstm_score=lstm_score,
                combined_score=combined,
                is_alert=is_alert,
                severity=severity,
            )

        return results

    def _get_severity(self, score: float) -> str:
        for sev, (lo, hi) in self.SEVERITY_THRESHOLDS.items():
            if lo <= score < hi:
                return sev
        return "none"
