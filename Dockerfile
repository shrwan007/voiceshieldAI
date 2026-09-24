# ==============================================================================
# VoiceShield AI — Backend Dockerfile
# Base: Python 3.11-slim
# Includes audio processing libraries (libsndfile, ffmpeg) and PyTorch dependencies
# ==============================================================================

FROM python:3.11-slim

# Prevent Python from writing .pyc files and enable unbuffered output
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Install OS-level dependencies for audio processing and C runtime libraries
RUN apt-get update && apt-get install -y --no-install-recommends \
    libsndfile1 \
    ffmpeg \
    libgomp1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Python requirements
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy application source code
COPY . .

# Ensure storage and model caching directories exist
RUN mkdir -p audio_storage models_cache

# Expose FastAPI application port
EXPOSE 8000

# Start Uvicorn ASGI server
CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000"]
