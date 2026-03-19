# BeatHole AI — ACE-Step 1.5 RunPod Worker
FROM nvidia/cuda:12.8.0-cudnn-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
# HF_HOME: HuggingFace cache (tokenizers, etc.)
ENV HF_HOME=/app/models
# ACESTEP_PROJECT_ROOT: ACE-Step stores checkpoints under {root}/checkpoints/
# initialize_service() downloads models here automatically on first run.
ENV ACESTEP_PROJECT_ROOT=/app

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 python3.11-dev python3-pip python3.11-venv \
    git ffmpeg libsndfile1 curl ca-certificates && \
    rm -rf /var/lib/apt/lists/* && \
    ln -sf /usr/bin/python3.11 /usr/bin/python3 && \
    ln -sf /usr/bin/python3 /usr/bin/python && \
    pip install --upgrade pip

# 1. Install torch (required by nano-vllm/flash-attn at build time)
RUN pip install --no-cache-dir \
    --extra-index-url https://download.pytorch.org/whl/cu128 \
    torch==2.10.0+cu128 torchaudio==2.10.0+cu128

# 2. Install nano-vllm with --no-build-isolation so flash-attn can find torch
RUN pip install --no-cache-dir --no-build-isolation \
    git+https://github.com/GeeeekExplorer/nano-vllm.git

# 3. Install ACE-Step with --no-deps to skip PyPI nano-vllm resolution
RUN pip install --no-cache-dir --no-deps \
    git+https://github.com/ace-step/ACE-Step-1.5.git

# 4. Install ACE-Step's remaining dependencies (known from its pyproject.toml)
RUN pip install --no-cache-dir \
    --extra-index-url https://download.pytorch.org/whl/cu128 \
    accelerate diffusers diskcache einops "fastapi>=0.110.0" \
    "gradio==6.2.0" "lightning>=2.0.0" loguru lycoris-lora \
    matplotlib modelscope huggingface_hub soundfile \
    vector_quantize_pytorch "typer-slim>=0.21.1"

# Pin transformers to the version ACE-Step 1.5 requires (>=4.51.0,<4.58.0).
# modelscope above may pull transformers 5.x which breaks ACE-Step internals.
RUN pip install --no-cache-dir "transformers>=4.51.0,<4.58.0"

# 5. Install runpod + numpy
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 6. Basic Pitch — audio-to-MIDI transcription for real stem notes
RUN pip install --no-cache-dir basic-pitch

COPY handler.py .

CMD ["python", "-u", "handler.py"]
