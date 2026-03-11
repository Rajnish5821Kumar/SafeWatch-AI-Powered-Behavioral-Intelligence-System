# SafeWatch 🛡️
### AI-Powered Behavioral Intelligence System for Educational Institutions

> *"A silent guardian — detecting behavioral risks before they escalate."*

SafeWatch is a **production-grade, end-to-end computer vision pipeline** that analyzes classroom CCTV footage in real time to identify students who may be experiencing emotional distress, social isolation, or behavioral anomalies. Built for the Sentio Mind mission of creating safer, more empathetic learning environments.

---

## 🏗️ Architecture Overview

```
CCTV Video Feed
     │
     ▼
┌─────────────────────────────────────────────────────────┐
│                    VideoProcessor                        │
│                                                         │
│  1. PersonDetector (YOLOv8)   ─→ Bounding Boxes        │
│  2. MultiObjectTracker (ByteTrack) ─→ Track IDs        │
│  3. PoseEstimator (YOLOv8-Pose) ─→ 17-KP Skeleton     │
│  4. EmotionAnalyzer (FER) ─→ Smoothed Emotion State   │
│  5. BehavioralProfiler ─→ 12-dim Feature Vector       │
│  6. AnomalyDetector (IsolationForest + LSTM AE) ─→ Score│
│  7. InsightGenerator (XAI) ─→ Human-Readable Alert    │
└─────────────────────────────────────────────────────────┘
     │                               │
     ▼                               ▼
FastAPI Backend                 Structured Logger
(REST + WebSocket)              (JSONL Alert Archive)
     │
     ▼
Real-Time Dashboard
(Glassmorphism UI · Chart.js · WebSocket)
```

---

## 🧠 AI Pipeline Modules

| Module | Technology | Purpose |
|--------|-----------|---------|
| `detector.py` | YOLOv8 (CUDA) | Multi-person detection |
| `tracker.py` | ByteTrack | Persistent track IDs across frames |
| `pose_estimator.py` | YOLOv8-Pose | 17-KP skeleton + posture metrics |
| `emotion_analyzer.py` | FER (CNN) | Facial emotion with exponential smoothing |
| `behavioral_profiler.py` | Custom | Rolling 30s behavioral feature aggregation |
| `anomaly_detector.py` | IsolationForest + LSTM AE | Dual-model anomaly scoring [0–1] |
| `insight_generator.py` | Rule-based NLG (XAI) | Human-readable educator alerts |
| `video_processor.py` | Async orchestrator | Keyframe-selective pipeline execution |
| `logger.py` | Loguru | JSONL alert archive + latency tracking |

---

## 📁 Project Structure

```
SafeWatch/
├── safewatch/
│   ├── detector.py          # YOLOv8 person detection
│   ├── tracker.py           # ByteTrack MOT
│   ├── pose_estimator.py    # Pose estimation + posture features
│   ├── emotion_analyzer.py  # FER emotion + temporal smoothing
│   ├── behavioral_profiler.py  # Feature vector aggregation
│   ├── anomaly_detector.py  # IsolationForest + LSTM Autoencoder
│   ├── insight_generator.py # XAI alert generation
│   ├── video_processor.py   # Pipeline orchestrator
│   └── logger.py            # Structured logging
├── api/
│   ├── main.py              # FastAPI app + WebSocket
│   └── routes/
│       ├── stream.py        # Video upload + background processing
│       ├── alerts.py        # Alert history API
│       └── analytics.py     # Classroom statistics API
├── frontend/
│   ├── index.html           # Real-time monitoring dashboard
│   ├── style.css            # Glassmorphism dark-mode UI
│   └── app.js               # WebSocket client + Chart.js
├── data/
│   ├── alerts/              # JSONL alert archive
│   └── logs/                # Rotating log files
├── models/                  # Saved ML models (LSTM AE, IsolationForest)
├── config.yaml              # Full pipeline configuration
└── requirements.txt         # Python dependencies
```

---

## 🚀 Quick Start

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

> **GPU acceleration**: Install PyTorch with CUDA support first:
> ```bash
> pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
> ```

### 2. Start the API server
```bash
uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
```

### 3. Open the dashboard
Navigate to **[http://localhost:8000](http://localhost:8000)**

Or open `frontend/index.html` directly in a browser (demo mode with synthetic data).

### 4. Run analysis on a video
```python
from safewatch.video_processor import VideoProcessor

processor = VideoProcessor.from_config("config.yaml")
for result in processor.process_video("classroom.mp4"):
    print(f"Frame {result.frame_id}: {result.n_persons} persons, "
          f"{len(result.insights)} alerts")
```

Or via the API:
```bash
curl -X POST http://localhost:8000/api/upload \
     -F "file=@classroom.mp4"
```

---

## 🔌 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/` | Serve dashboard |
| `GET` | `/health` | System health check |
| `WS`  | `/ws/stream` | Real-time frame results (WebSocket) |
| `POST` | `/api/upload` | Upload video for analysis |
| `GET` | `/api/status` | Processing pipeline status |
| `GET` | `/api/alerts` | Alert history (filterable) |
| `GET` | `/api/alerts/stats` | Aggregate alert statistics |
| `GET` | `/api/analytics/summary` | Classroom-level metrics |
| `GET` | `/api/analytics/timeline` | Time-series chart data |
| `GET` | `/api/analytics/emotions` | Emotion distribution |
| `GET` | `/docs` | Swagger API documentation |

---

## ⚙️ Configuration

All pipeline parameters are controlled via `config.yaml`:

```yaml
detection:
  model: "yolov8n.pt"
  confidence_threshold: 0.40
  device: "cuda"            # Switch to "cpu" if no GPU

anomaly:
  alert_threshold: 0.72     # Lower = more sensitive
  contamination: 0.05       # Expected anomaly fraction

processing:
  max_fps: 15               # Maximum processing rate
  keyframe_diff_threshold: 0.015  # Skip similar frames
```

---

## 🔬 Anomaly Detection

SafeWatch uses a **dual-model approach**:

### IsolationForest (Real-time)
- Scores instantaneous behavioral feature vectors
- Trained on baseline normal classroom behavior
- Fast: < 5ms per frame

### LSTM Autoencoder (Temporal)
- Detects sustained behavioral anomalies over 60-second sequences
- High reconstruction error = deviation from learned normal patterns
- Catches slow-onset distress signals IsolationForest misses

**Combined score** = 0.6 × IsolationForest + 0.4 × LSTM

---

## 🔏 Privacy by Design

- **No face storage**: Face crops are processed in-memory and never persisted
- **Aggregated outputs only**: Only emotion labels and aggregate scores are stored
- **No biometric identification**: Tracking by temporal position, not facial recognition
- **Configurable retention**: Log files auto-rotate with configurable retention period
- **On-premise deployment**: All processing runs locally — no cloud API calls

---

## 📊 Behavioral Features Extracted

Per tracked person, over a 30-second rolling window:

| Feature | Description |
|---------|-------------|
| `avg_motion_velocity` | Mean pixel displacement per frame |
| `motion_entropy` | Movement randomness (erratic = high) |
| `avg_emotion_valence` | Average emotional positivity [-1, 1] |
| `avg_slouch_score` | Posture deviation from upright [0, 1] |
| `avg_head_tilt` | Mean head tilt in degrees |
| `isolation_score` | Distance from nearest peer (normalized) |
| `arm_raise_freq` | Fraction of frames with arm raised |
| `dominant_emotion_*` | Per-emotion average scores (6 features) |

---

## 🛠️ Tech Stack

| Layer | Technologies |
|-------|-------------|
| **CV / Detection** | YOLOv8, OpenCV |
| **Tracking** | ByteTrack (via supervision) |
| **Pose Estimation** | YOLOv8-Pose |
| **Emotion Recognition** | FER (CNN-based) |
| **Anomaly Detection** | scikit-learn IsolationForest, TensorFlow LSTM |
| **Backend API** | FastAPI, Uvicorn, WebSockets |
| **Frontend** | Vanilla JS, Chart.js, CSS Glassmorphism |
| **Logging** | Loguru, JSONL |
| **Configuration** | PyYAML |

---

## 📄 License
MIT License — Educational and research use.

---

*Built with ❤️ for the Sentio Mind mission — AI-powered behavioral intelligence for educational safety.*
