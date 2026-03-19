# BeatHole AI — ACE-Step 1.5 RunPod Worker
FROM nvidia/cuda:12.8.0-cudnn-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV HF_HOME=/app/models
ENV ACESTEP_CHECKPOINT_DIR=/app/models/ace-step-v1.5

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 python3.11-dev python3-pip python3.11-venv \
    git ffmpeg libsndfile1 curl ca-certificates && \
    rm -rf /var/lib/apt/lists/* && \
    ln -sf /usr/bin/python3.11 /usr/bin/python3 && \
    ln -sf /usr/bin/python3 /usr/bin/python && \
    pip install --upgrade pip

# Install ACE-Step 1.5 — it pins torch==2.10.0+cu128 internally
# Add PyTorch cu128 index so pip can find it
COPY requirements.txt .
# nano-vllm is a dependency of ACE-Step 1.5 but not on PyPI — install from source first
RUN pip install --no-cache-dir \
    git+https://github.com/GeeeekExplorer/nano-vllm.git
RUN pip install --no-cache-dir \
    --extra-index-url https://download.pytorch.org/whl/cu128 \
    -r requirements.txt

COPY handler.py .

CMD ["python", "-u", "handler.py"]
