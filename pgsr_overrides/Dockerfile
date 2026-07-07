FROM nvidia/cuda:11.8.0-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

ENV TZ=UTC

ENV PATH="/opt/conda/bin:${PATH}"

ENV TORCH_CUDA_ARCH_LIST="6.0;6.1;7.0;7.5;8.0;8.6+PTX"

RUN apt-get update && apt-get install -y --no-install-recommends \
git wget build-essential cmake ninja-build g++ \
libgl1-mesa-glx libegl1-mesa libegl1 libglu1-mesa \
libxrandr2 libxinerama1 libxcursor1 libxi6 \
ca-certificates curl python3-pip \
&& apt-get clean && rm -rf /var/lib/apt/lists/*

# Miniconda with Python 3.8
RUN wget -q https://repo.anaconda.com/miniconda/Miniconda3-py38_23.5.2-0-Linux-x86_64.sh -O miniconda.sh \
&& bash miniconda.sh -b -p /opt/conda \
&& rm miniconda.sh

SHELL ["/bin/bash", "-c"]

WORKDIR /app

# Clone PGSR with submodules
RUN git clone https://github.com/zju3dv/PGSR.git --recursive .

# PyTorch 2.0.1 + CUDA 11.8
RUN source /opt/conda/etc/profile.d/conda.sh && \
conda create -n pgsr python=3.8 -y && \
conda activate pgsr && \
pip install --upgrade pip && \
pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 \
--index-url https://download.pytorch.org/whl/cu118

# pytorch3d: try prebuilt wheel first, fall back to source build
RUN source /opt/conda/etc/profile.d/conda.sh && conda activate pgsr && \
pip install fvcore iopath && \
pip install \
"pytorch3d @ https://dl.fbaipublicfiles.com/pytorch3d/packaging/wheels/py38_cu118_pyt201/pytorch3d-0.7.4-cp38-cp38-linux_x86_64.whl" \
|| (echo "Prebuilt wheel failed, building from source..." && \
FORCE_CUDA=1 pip install "git+https://github.com/facebookresearch/pytorch3d.git@stable")

# Other requirements (excluding pytorch3d — already installed above)
RUN source /opt/conda/etc/profile.d/conda.sh && conda activate pgsr && \
pip install open3d plyfile "opencv-python-headless" lpips trimesh xatlas

# CUDA submodules
RUN source /opt/conda/etc/profile.d/conda.sh && conda activate pgsr && \
pip install submodules/diff-plane-rasterization

RUN source /opt/conda/etc/profile.d/conda.sh && conda activate pgsr && \
pip install submodules/simple-knn

# Apply project-specific overrides on top of the upstream PGSR source.
# Kept after the heavy pip/CUDA layers so editing an override only
# invalidates these cheap COPY layers on rebuild.
COPY train.py render.py bake_texture.py ./
COPY scene/ ./scene/

# Entrypoint script
RUN printf '#!/bin/bash\nsource /opt/conda/etc/profile.d/conda.sh\nconda activate pgsr\nexec "$@"\n' \
> /app/entrypoint.sh && chmod +x /app/entrypoint.sh

ARG USER_ID=1000

ARG GROUP_ID=1000

RUN groupadd -g ${GROUP_ID} pgsruser 2>/dev/null || true && \
useradd -m -u ${USER_ID} -g ${GROUP_ID} -s /bin/bash pgsruser 2>/dev/null || true && \
chown -R ${USER_ID}:${GROUP_ID} /app

USER pgsruser

RUN printf '#!/bin/bash\nsource /opt/conda/etc/profile.d/conda.sh\nconda activate pgsr\nexec "$@"\n' \
> /app/entrypoint.sh && chmod +x /app/entrypoint.sh

ENTRYPOINT ["/app/entrypoint.sh"]

CMD ["tail", "-f", "/dev/null"]
