# ==========================================
# Phase 1: Compile the React UI
# ==========================================
FROM node:20-alpine AS frontend-builder
WORKDIR /app/frontend

# Install dependencies (only copy package.json first for cache optimization)
COPY frontend/package.json frontend/package-lock.json* ./
RUN npm install

# Copy all React source code and build it natively
COPY frontend/ .
RUN npm run build

# ==========================================
# Phase 2: Assemble Python Backend
# ==========================================
FROM python:3.10-slim

# Install system-level dependencies for TensorFlow and OpenCV if needed
RUN apt-get update && apt-get install -y \
    build-essential \
    libhdf5-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy python dependencies explicitly
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy backend python code, ML models, and reference logic
# (best_weights.weights.h5 may be an LFS pointer stub at this point — fixed below)
COPY . .

# Download real model weights from HuggingFace Space, overwriting any LFS pointer stub.
# The file is 49MB and lives at https://huggingface.co/spaces/srikarp/dermaAI
RUN python -c "\
from huggingface_hub import hf_hub_download; \
import shutil, os; \
print('Downloading model weights from HuggingFace...'); \
path = hf_hub_download(repo_id='srikarp/dermaAI', filename='best_weights.weights.h5', repo_type='space'); \
shutil.copy(path, '/app/best_weights.weights.h5'); \
size = os.path.getsize('/app/best_weights.weights.h5'); \
print(f'Downloaded: {size / 1024 / 1024:.1f} MB'); \
assert size > 1_000_000, 'ERROR: Downloaded file is too small — likely an LFS pointer stub!'"

# Securely copy the Production Ready files from Phase 1 into Flask's reach
# `app.py` is configured to look inside `frontend/dist`
COPY --from=frontend-builder /app/frontend/dist /app/frontend/dist

# Suppress verbose TF logs and reduce startup memory overhead
ENV TF_CPP_MIN_LOG_LEVEL=3
ENV TF_ENABLE_ONEDNN_OPTS=0
ENV MALLOC_ARENA_MAX=2

# Render injects a PORT environment variable at runtime
EXPOSE 10000

# --preload: loads the app ONCE then forks workers (saves ~300MB RAM vs each worker loading TF)
# --timeout 180: gives TensorFlow enough time to initialize on cold start
# --workers 1: single worker to stay within free-tier 512MB RAM
CMD ["gunicorn", "--bind", "0.0.0.0:10000", "--workers", "1", "--timeout", "180", "--preload", "app:app"]