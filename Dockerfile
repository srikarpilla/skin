# ==========================================
# Phase 1: Compile the React UI
# ==========================================
FROM node:18-alpine AS frontend-builder
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
COPY . .

# Securely copy the Production Ready files from Phase 1 into Flask's reach
# `app.py` is configured to look inside `frontend/dist`
COPY --from=frontend-builder /app/frontend/dist /app/frontend/dist

# Render injects a PORT environment variable at runtime
EXPOSE 10000

# Specify how Gunicorn should efficiently run the app logic across CPU threads
# Set timeout high so TensorFlow has ample time to load dynamically
CMD ["gunicorn", "--bind", "0.0.0.0:10000", "--workers", "1", "--timeout", "120", "app:app"]