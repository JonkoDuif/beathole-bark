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
    python3.10 python3.10-dev python3-pip python3.10-venv \
    git ffmpeg libsndfile1 curl ca-certificates && \
    rm -rf /var/lib/apt/lists/* && \
    ln -sf /usr/bin/python3.10 /usr/bin/python3 && \
    ln -sf /usr/bin/python3 /usr/bin/python && \
    pip install --upgrade pip

# 1. Install torch first for CUDA runtime compatibility
RUN pip install --no-cache-dir \
    --extra-index-url https://download.pytorch.org/whl/cu128 \
    torch==2.10.0+cu128 torchaudio==2.10.0+cu128

# 2. Install ACE-Step the official way from the upstream repo instead of a
# hand-assembled dependency stack. This keeps its LLM runtime closer to upstream.
RUN git clone --depth 1 https://github.com/ace-step/ACE-Step.git /opt/ACE-Step && \
    pip install --no-cache-dir -e /opt/ACE-Step

# 3. Install runpod + project extras
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 4. Basic Pitch — audio-to-MIDI transcription for real stem notes
RUN pip install --no-cache-dir basic-pitch

# 5. Demucs — neural source separator (replaces ACE-Step extract for clean stems)
RUN pip install --no-cache-dir demucs
# Pre-download the 6-stem model at build time (avoids cold-start download)
RUN python -c "from demucs.pretrained import get_model; get_model('htdemucs_6s')"

COPY handler.py .

CMD ["python", "-u", "handler.py"]
