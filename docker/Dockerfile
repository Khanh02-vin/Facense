# ============================================================
# Facense — Face Preference Learning Pipeline
# ============================================================

FROM python:3.11-slim

LABEL maintainer="Facense"
LABEL description="Face preference learning: frame extraction, feature extraction, preference model"

WORKDIR /workspace

# ----- System dependencies -----
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# ----- Python dependencies -----
# Core ML / data
RUN pip install --no-cache-dir \
    torch==2.3.0 \
    torchvision==0.18.0 \
    --index-url https://download.pytorch.org/whl/cpu

RUN pip install --no-cache-dir \
    transformers>=4.30.0 \
    timm>=0.9.0 \
    pillow>=10.0.0 \
    numpy>=1.24.0 \
    pandas>=2.0.0 \
    scipy>=1.10.0 \
    scikit-learn>=1.3.0 \
    tqdm>=4.60.0 \
    matplotlib>=3.7.0

# Vision / CV
RUN pip install --no-cache-dir \
    opencv-python-headless>=4.8.0 \
    mediapipe>=0.10.0,\<0.10.15 \
    protobuf\<4

# Serving layer (optional — used by serve.py)
RUN pip install --no-cache-dir \
    fastapi>=0.100.0 \
    uvicorn[standard]>=0.23.0 \
    pydantic>=2.0.0

# ============================================================
# Entry point — overridable via docker-compose command or CLI
# ============================================================
CMD ["python", "src/run_pipeline.py"]
