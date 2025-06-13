#!/bin/bash

# Ensure script to exit on error
set -e

PORT=${PORT:-8501}  # Default to 8501 if PORT is not set

echo "Starting FastAPI..."
uvicorn src.api.main:app --host 0.0.0.0 --port 8000 &

echo "Starting Streamlit..."
streamlit run src/streamlit_app.py \
    --server.port=$PORT \
    --server.address=0.0.0.0 \
    --server.enableCORS=false \
    --server.enableXsrfProtection=false

# Wait for all background processes
wait