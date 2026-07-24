# Fast-PGSR (FastGS-accelerated PGSR) image — fully isolated from pgsr:local.
#
# FastGS ships its OWN accelerated train.py/render.py and a different, older
# CUDA/PyTorch stack than our stock PGSR image, so it cannot share the `pgsr`
# conda env. This image builds a dedicated `fast-pgsr` env matching FastGS's
# environment.yml (CUDA 11.6 / Python 3.7.13 / PyTorch 1.12.1) and compiles the
# four CUDA extensions FastGS needs.
#
# The ONLY project override applied on top of upstream FastGS is
# scene/dataset_readers.py (patched to read COLMAP from a flat `sparse/` dir,
# matching how run_pipeline.sh already stages COLMAP output). FastGS's own
# train.py/render.py are the accelerated versions and are deliberately left
# untouched.
FROM nvidia/cuda:11.6.2-devel-ubuntu20.04

ENV DEBIAN_FRONTEND=noninteractive

ENV TZ=UTC

ENV PATH="/opt/conda/bin:${PATH}"

# FastGS requires Compute Capability 7.0+ (see its README hardware notes).
ENV TORCH_CUDA_ARCH_LIST="7.0;7.5;8.0;8.6+PTX"

RUN apt-get update && apt-get install -y --no-install-recommends \
git wget build-essential cmake ninja-build g++ \
libgl1-mesa-glx libegl1-mesa libegl1 libglu1-mesa \
libxrandr2 libxinerama1 libxcursor1 libxi6 \
ca-certificates curl python3-pip \
&& apt-get clean && rm -rf /var/lib/apt/lists/*

# Miniconda with Python 3.8 base (the fast-pgsr env pins Python 3.7.13 below)
RUN wget -q https://repo.anaconda.com/miniconda/Miniconda3-py38_23.5.2-0-Linux-x86_64.sh -O miniconda.sh \
&& bash miniconda.sh -b -p /opt/conda \
&& rm miniconda.sh

SHELL ["/bin/bash", "-c"]

WORKDIR /app

# Clone FastGS at the fast-pgsr branch. FastGS vendors its four CUDA extensions
# as plain directories under submodules/ (no .gitmodules), so a shallow branch
# clone pulls everything needed; --recurse-submodules is a harmless no-op.
RUN git clone --branch fast-pgsr --single-branch \
https://github.com/fastgs/FastGS.git . \
&& git submodule update --init --recursive || true

# fast-pgsr conda env: Python 3.7.13, then PyTorch 1.12.1 / tv 0.13.1 (cu116).
# PyTorch is installed via pip from the official cu116 wheel index rather than
# `conda install`, because conda's SAT solver thrashes for many minutes on this
# old pinned combo. The pip wheels bundle their own CUDA runtime; nvcc for the
# CUDA-extension builds below comes from the CUDA 11.6 devel base image, so a
# separate conda cudatoolkit package is not needed.
RUN source /opt/conda/etc/profile.d/conda.sh && \
conda create -n fast-pgsr python=3.7.13 -y && \
conda activate fast-pgsr && \
pip install --upgrade pip && \
pip install torch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 \
--index-url https://download.pytorch.org/whl/cu116

# pytorch3d (imported by scene/gaussian_model.py): try the prebuilt wheel that
# matches py37/cu116/pyt1121, fall back to a source build if the wheel is gone.
RUN source /opt/conda/etc/profile.d/conda.sh && conda activate fast-pgsr && \
pip install fvcore iopath && \
pip install \
"pytorch3d @ https://dl.fbaipublicfiles.com/pytorch3d/packaging/wheels/py37_cu116_pyt1121/pytorch3d-0.7.2-cp37-cp37m-linux_x86_64.whl" \
|| (echo "Prebuilt pytorch3d wheel failed, building from source..." && \
FORCE_CUDA=1 pip install "git+https://github.com/facebookresearch/pytorch3d.git@stable")

# Remaining Python requirements (FastGS requirements.txt minus pytorch3d)
RUN source /opt/conda/etc/profile.d/conda.sh && conda activate fast-pgsr && \
pip install open3d plyfile "opencv-python-headless" lpips trimesh xatlas tqdm

# Four CUDA extensions FastGS needs. diff-gaussian-rasterization_fastgs and
# fused-ssim are the FastGS-specific accelerators; the other two are shared
# with stock PGSR but rebuilt here against this env's torch/CUDA.
RUN source /opt/conda/etc/profile.d/conda.sh && conda activate fast-pgsr && \
pip install submodules/diff-plane-rasterization

RUN source /opt/conda/etc/profile.d/conda.sh && conda activate fast-pgsr && \
pip install submodules/simple-knn

RUN source /opt/conda/etc/profile.d/conda.sh && conda activate fast-pgsr && \
pip install submodules/diff-gaussian-rasterization_fastgs

RUN source /opt/conda/etc/profile.d/conda.sh && conda activate fast-pgsr && \
pip install submodules/fused-ssim

# Apply the single project-specific override on top of upstream FastGS. Kept
# after the heavy pip/CUDA layers so editing it only invalidates this cheap
# COPY layer on rebuild. NOTE: we intentionally do NOT copy train.py/render.py
# — FastGS's own versions are the accelerated ones we want.
COPY scene/dataset_readers.py ./scene/dataset_readers.py

ARG USER_ID=1000

ARG GROUP_ID=1000

RUN groupadd -g ${GROUP_ID} fastpgsruser 2>/dev/null || true && \
useradd -m -u ${USER_ID} -g ${GROUP_ID} -s /bin/bash fastpgsruser 2>/dev/null || true && \
chown -R ${USER_ID}:${GROUP_ID} /app

USER fastpgsruser

RUN printf '#!/bin/bash\nsource /opt/conda/etc/profile.d/conda.sh\nconda activate fast-pgsr\nexec "$@"\n' \
> /app/entrypoint.sh && chmod +x /app/entrypoint.sh

ENTRYPOINT ["/app/entrypoint.sh"]

CMD ["tail", "-f", "/dev/null"]
