"""
SafeWatch — FastAPI Main Application
──────────────────────────────────────
Serves the REST API and WebSocket stream endpoint.
"""

from __future__ import annotations
import asyncio
import json
import time
import threading
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse
import uvicorn
from loguru import logger

from api.routes.alerts import router as alerts_router
from api.routes.analytics import router as analytics_router
from api.routes.stream import router as stream_router, get_processor


# ─── App initialization ───────────────────────────────────────────────────────

app = FastAPI(
    title="SafeWatch API",
    description="AI-powered behavioral intelligence system for educational institutions",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(alerts_router, prefix="/api/alerts", tags=["Alerts"])
app.include_router(analytics_router, prefix="/api/analytics", tags=["Analytics"])
app.include_router(stream_router, prefix="/api", tags=["Stream"])

# Serve frontend static files
frontend_path = Path(__file__).parent.parent / "frontend"
if frontend_path.exists():
    app.mount("/static", StaticFiles(directory=str(frontend_path)), name="static")


# ─── Core Endpoints ───────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def serve_dashboard():
    """Serve the frontend dashboard."""
    index_path = frontend_path / "index.html"
    if index_path.exists():
        return HTMLResponse(content=index_path.read_text(encoding="utf-8"))
    return HTMLResponse("<h1>SafeWatch API — Frontend not found</h1>")


@app.get("/health")
async def health_check():
    """System health check endpoint."""
    processor = get_processor()
    return JSONResponse({
        "status": "ok",
        "service": "SafeWatch AI",
        "version": "1.0.0",
        "timestamp": time.time(),
        "pipeline_ready": processor is not None,
        "models": {
            "detector": "yolov8n.pt",
            "pose": "yolov8n-pose.pt",
            "tracker": "ByteTrack",
            "anomaly": "IsolationForest + LSTM Autoencoder",
        },
    })


# ─── WebSocket Real-Time Stream ───────────────────────────────────────────────

connected_clients: list = []
latest_payload: Optional[dict] = None


@app.websocket("/ws/stream")
async def websocket_stream(websocket: WebSocket):
    """
    WebSocket endpoint for real-time dashboard updates.
    Pushes live frame analysis events to all connected frontends.
    """
    await websocket.accept()
    connected_clients.append(websocket)
    logger.info(f"WebSocket client connected. Total clients: {len(connected_clients)}")

    try:
        # Send last known state immediately on connect
        if latest_payload:
            await websocket.send_json(latest_payload)

        while True:
            # Keep alive with heartbeat ping
            await asyncio.sleep(5)
            await websocket.send_json({"type": "heartbeat", "ts": time.time()})

    except WebSocketDisconnect:
        connected_clients.remove(websocket)
        logger.info(f"WebSocket disconnected. Remaining: {len(connected_clients)}")
    except Exception as e:
        if websocket in connected_clients:
            connected_clients.remove(websocket)
        logger.debug(f"WebSocket error: {e}")


async def broadcast_frame_result(result_dict: dict) -> None:
    """Broadcast a frame result to all connected WebSocket clients."""
    global latest_payload
    payload = {"type": "frame_result", "data": result_dict}
    latest_payload = payload

    dead = []
    for ws in connected_clients:
        try:
            await ws.send_json(payload)
        except Exception:
            dead.append(ws)

    for ws in dead:
        if ws in connected_clients:
            connected_clients.remove(ws)


# ─── Startup / Shutdown ───────────────────────────────────────────────────────

@app.on_event("startup")
async def startup_event():
    logger.info("🚀 SafeWatch API starting up...")
    logger.info("📡 WebSocket stream available at ws://localhost:8000/ws/stream")
    logger.info("📊 Dashboard available at http://localhost:8000")
    logger.info("📖 API docs at http://localhost:8000/docs")


@app.on_event("shutdown")
async def shutdown_event():
    logger.info("SafeWatch API shutting down...")


if __name__ == "__main__":
    uvicorn.run(
        "api.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info",
    )
