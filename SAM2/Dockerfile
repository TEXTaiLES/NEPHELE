# Use *devel* so nvcc is available to build CUDA extensions
FROM nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04

# ---------- ENV (system-level) ----------
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/home/user/.cache/huggingface \
    MPLBACKEND=TkAgg \
    CUDA_HOME=/usr/local/cuda \
    TORCH_CUDA_ARCH_LIST=8.6 \
    IN_MNT=/data/in \
    OUT_MNT=/data/out \
    INDEX_ROOT=/data/out/__index_cache__

# ---------- System deps ----------
RUN apt-get update && apt-get install -y --no-install-recommends \
    git curl ca-certificates build-essential ninja-build cmake \
    python3 python3-pip python3-dev python3-venv \
    ffmpeg libgl1 libglib2.0-0 \
    python3-tk tk x11-apps \
    libx11-xcb1 libxkbcommon-x11-0 libxrender1 libxext6 libxfixes3 libxi6 libxtst6 libsm6 \
 && rm -rf /var/lib/apt/lists/*

# ---------- Create non-root user (MATCH HOST UID/GID at build) ----------
ARG UID=1000
ARG GID=1000
ARG UNAME=user
RUN groupadd -g ${GID} ${UNAME} || true \
 && useradd -m -u ${UID} -g ${GID} -s /bin/bash ${UNAME}
RUN mkdir -p /workspace ${IN_MNT} ${OUT_MNT} && chown -R ${UID}:${GID} /workspace ${IN_MNT} ${OUT_MNT}

# Default umask so new files keep group write
RUN echo 'umask 0002' >> /etc/profile

# ---------- System-wide Python deps (as root) ----------
ENV PIP_BREAK_SYSTEM_PACKAGES=1
RUN python3 -m pip install --upgrade pip && \
    pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cu121 \
      torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 && \
    pip install --no-cache-dir \
      numpy pillow opencv-python matplotlib tqdm \
      einops timm pycocotools imageio[ffmpeg] av pyyaml

# ---------- SAM2 system-wide ----------
RUN git clone --depth=1 https://github.com/facebookresearch/segment-anything-2.git /opt/segment-anything-2
WORKDIR /opt/segment-anything-2
RUN pip install --no-cache-dir --upgrade setuptools wheel
# non-editable install
RUN pip install --no-cache-dir --no-build-isolation .

RUN pip install --no-cache-dir flask==3.0.3

# ---------- Switch to non-root user and set HOME-aware env ----------
USER ${UNAME}
ENV HOME=/home/${UNAME}
ENV PYTHONUSERBASE=$HOME/.local
ENV PATH=$HOME/.local/bin:$PATH

WORKDIR /workspace
# (optional) convenience app link
RUN mkdir -p "$HOME/app" && ln -s "$HOME/app" /workspace/app || true

# Keep group-writable on interactive shells and CMD
RUN echo 'umask 0002' >> "$HOME/.bashrc"

CMD ["/bin/bash", "-lc", "umask 0002 && exec bash"]
