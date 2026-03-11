"""
SafeWatch API — Stream Routes
──────────────────────────────
Handles video upload and real-time streaming analysis.
"""

from __future__ import annotations
import asyncio
import os
import tempfile
import time
import threading
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse
from loguru import logger

router = APIRouter()

# Singleton processor (lazy-initialized)
_processor = None
_processor_lock = threading.Lock()


def get_processor():
    global _processor
    return _processor


def _initialize_processor():
    global _processor
    if _processor is not None:
        return _processor
    with _processor_lock:
        if _processor is not None:
            return _processor
        try:
            from safewatch.video_processor import VideoProcessor
            logger.info("Initializing SafeWatch pipeline from config.yaml...")
            _processor = VideoProcessor.from_config("config.yaml")
            logger.info("SafeWatch pipeline ready")
        except Exception as e:
            logger.error(f"Failed to initialize pipeline: {e}")
            _processor = None
    return _processor


# Processing state
_processing_state = {
    "is_processing": False,
    "current_video": None,
    "frames_processed": 0,
    "start_time": None,
    "last_result": None,
}


@router.post("/upload")
async def upload_video(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
):
    """
    Upload a video file for SafeWatch analysis.
    Analysis runs in the background; results stream via WebSocket.
    """
    if _processing_state["is_processing"]:
        raise HTTPException(
            status_code=409,
            detail="Another video is currently being processed. Please wait."
        )

    if not file.content_type or "video" not in file.content_type:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid file type: {file.content_type}. Please upload a video file."
        )

    # Save upload to temp file
    suffix = Path(file.filename or "video.mp4").suffix
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    try:
        contents = await file.read()
        tmp.write(contents)
        tmp.close()
        video_path = tmp.name
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save upload: {e}")

    # Ensure processor is ready
    processor = _initialize_processor()

    background_tasks.add_task(
        _run_video_analysis,
        video_path=video_path,
        original_filename=file.filename or "video.mp4",
    )

    return JSONResponse({
        "status": "accepted",
        "message": f"Processing started for '{file.filename}'",
        "stream_url": "ws://localhost:8000/ws/stream",
        "size_bytes": len(contents),
    })


async def _run_video_analysis(video_path: str, original_filename: str):
    """Background task: run video through SafeWatch pipeline."""
    from api.main import broadcast_frame_result

    processor = get_processor()
    if processor is None:
        processor = _initialize_processor()

    _processing_state["is_processing"] = True
    _processing_state["current_video"] = original_filename
    _processing_state["frames_processed"] = 0
    _processing_state["start_time"] = time.time()

    logger.info(f"Starting analysis of: {original_filename}")

    try:
        for result in processor.process_video(video_path, max_frames=1000):
            _processing_state["frames_processed"] += 1
            _processing_state["last_result"] = result.to_dict()

            # Broadcast to all WebSocket clients
            await broadcast_frame_result(result.to_dict())
            await asyncio.sleep(0)  # Yield to event loop

    except Exception as e:
        logger.error(f"Video analysis error: {e}")
    finally:
        _processing_state["is_processing"] = False
        _processing_state["current_video"] = None
        # Cleanup temp file
        try:
            os.unlink(video_path)
        except Exception:
            pass
        logger.info(f"Analysis complete. Frames: {_processing_state['frames_processed']}")


@router.get("/status")
async def get_processing_status():
    """Get current processing pipeline status."""
    return JSONResponse({
        "is_processing": _processing_state["is_processing"],
        "current_video": _processing_state["current_video"],
        "frames_processed": _processing_state["frames_processed"],
        "elapsed_seconds": (
            time.time() - _processing_state["start_time"]
            if _processing_state["start_time"] else 0
        ),
        "pipeline_ready": get_processor() is not None,
    })
