# H100 full-experiment environment (pipeline_build_plan.md's H100 execution
# order: steps 1-2 at scale, 4, 5, 6, 8). Requires `docker run --gpus all`
# (nvidia-container-toolkit on the host) — this image alone does not give the
# container GPU access.
#
# requirements-h100.txt is still an UNPINNED PLACEHOLDER (see its header
# comment) — pin it against this image's actual CUDA/driver combination
# before trusting any number produced inside this container.

FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.10 python3.10-venv python3-pip git vim \
        curl wget less unzip tmux htop nvtop \
        rsync openssh-client ca-certificates \
        build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

# Environment only — no repo code baked in. Clone the repo yourself once
# you're on the H100 box (git is installed above) and `pip install -e .`
# from pipeline/ there, e.g.:
#
#   cp .env.example .env   # fill in HF_TOKEN (Llama-3.3-70B is gated) + paths
#   docker run --gpus all -it \
#     -v /path/on/h100/hf_cache:/hf_cache \
#     --env-file .env \
#     qcd-h100 bash
#   # inside the container:
#   git clone <repo-url> repo && cd repo/pipeline && pip install -e .
COPY requirements-h100.txt ./
RUN python3.10 -m pip install --upgrade pip \
    && python3.10 -m pip install -r requirements-h100.txt

ENV HF_HOME=/hf_cache \
    HF_HUB_CACHE=/hf_cache

CMD ["bash"]
