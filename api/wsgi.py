"""
WSGI entry point for Flight-Guard API.
Loads models before serving; gunicorn points here.

Usage:
    gunicorn --bind 0.0.0.0:8000 --workers 2 api.wsgi:application
"""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure project root is on sys.path so `from models.xxx import ...` works
_ROOT = Path(__file__).parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from loguru import logger

from api.app import _load_models, app

logger.info("WSGI startup — loading Flight-Guard models…")
try:
    _load_models()
    logger.success("Models loaded. Flight-Guard API is ready.")
except Exception as exc:
    logger.warning("Model loading failed (API will return mock scores): {}", exc)

application = app
