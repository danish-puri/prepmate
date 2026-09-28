"""Vercel's entrypoint. Vercel looks for a FastAPI `app` in a root app.py, and
the real one lives in backend/main.py."""

from backend.main import app  # noqa: F401
