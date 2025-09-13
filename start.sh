#!/usr/bin/env bash
set -e
# Render will provide $PORT; bind to 0.0.0.0 so it’s public.
uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}
