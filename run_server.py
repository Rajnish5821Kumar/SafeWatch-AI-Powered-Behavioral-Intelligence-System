"""
SafeWatch — Convenience server entry point.
Run this from the project root:
    python run_server.py
"""
import sys
import os

# Ensure project root is on path
sys.path.insert(0, os.path.dirname(__file__))

import uvicorn

if __name__ == "__main__":
    print("""
╔══════════════════════════════════════════════════════════════╗
║          SafeWatch — AI Behavioral Intelligence System       ║
║──────────────────────────────────────────────────────────────║
║  Dashboard  : http://localhost:8000                          ║
║  API Docs   : http://localhost:8000/docs                     ║
║  WebSocket  : ws://localhost:8000/ws/stream                  ║
╚══════════════════════════════════════════════════════════════╝
    """)
    uvicorn.run(
        "api.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info",
    )
