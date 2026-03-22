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

# 2. Clone ACE-Step 1.5 so we can install the exact bundled nano-vllm runtime
# that ACE-Step's LLM handler expects.
RUN git clone --depth 1 https://github.com/ace-step/ACE-Step-1.5.git /opt/ACE-Step-1.5

# 3. Install ACE-Step's bundled nano-vllm with --no-build-isolation so torch is
# already available at build time. This matches the path ACE-Step logs refer to.
RUN pip uninstall -y nano-vllm || true && \
    pip install --no-cache-dir --no-build-isolation /opt/ACE-Step-1.5/acestep/third_parts/nano-vllm

# 4. Install ACE-Step with --no-deps to keep the package layout compatible with
# the current handler imports (`acestep.inference`, `acestep.llm_inference`).
RUN pip install --no-cache-dir --no-deps \
    git+https://github.com/ace-step/ACE-Step-1.5.git

# 5. Install ACE-Step's remaining dependencies.
RUN pip install --no-cache-dir \
    --extra-index-url https://download.pytorch.org/whl/cu128 \
    accelerate diffusers diskcache einops "fastapi>=0.110.0" \
    "gradio==6.2.0" "lightning>=2.0.0" loguru lycoris-lora \
    matplotlib modelscope huggingface_hub soundfile \
    vector_quantize_pytorch "typer-slim>=0.21.1"

# Keep transformers on the ACE-Step 1.5 compatible range.
RUN pip install --no-cache-dir "transformers>=4.51.0,<4.58.0"

# 6. Install runpod + project extras
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 7. Basic Pitch — audio-to-MIDI transcription for real stem notes
RUN pip install --no-cache-dir basic-pitch

# 8. Demucs — neural source separator (replaces ACE-Step extract for clean stems)
RUN pip install --no-cache-dir demucs
# Pre-download the 6-stem model at build time (avoids cold-start download)
RUN python -c "from demucs.pretrained import get_model; get_model('htdemucs_6s')"

COPY handler.py .

CMD ["python", "-u", "handler.py"]
