#!/usr/bin/env bash
set -e
# Bind to the port Render provides and to all interfaces.
uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}
