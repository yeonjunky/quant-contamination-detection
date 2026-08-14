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
    software-properties-common \
    && add-apt-repository -y ppa:deadsnakes/ppa \
    && apt-get update && apt-get install -y --no-install-recommends \
    python3.12 python3.12-venv python3-pip git vim \
    curl wget less unzip tmux htop nvtop \
    rsync openssh-client ca-certificates \
    build-essential \
    && rm -rf /var/lib/apt/lists/* \
    && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.12 1

WORKDIR /workspace

# Environment only — no repo code baked in. Clone the repo yourself once
# you're on the H100 box (git is installed above) and `pip install -e .`
# from pipeline/ there, e.g.:
#
#   cp .env.example .env   # fill in HF_TOKEN (meta-llama/Llama-3.1-8B-Instruct
#                           # is gated) + paths
#   docker run --gpus all -it \
#     -v /path/on/h100/hf_cache:/workspace/hf_cache \
#     --env-file .env \
#     qcd-h100 bash
#   # inside the container:
#   git clone <repo-url> repo && cd repo/pipeline && pip install -e .

RUN python3 -m ensurepip --upgrade \
    && python3 -m pip install --upgrade pip

ENV HF_HOME=/workspace/hf_cache \
    HF_HUB_CACHE=/workspace/hf_cache

CMD ["bash"]
