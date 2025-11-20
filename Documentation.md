# nephele — Detailed Technical Documentation

## Overview

nephele is a two-stage image-to-3D pipeline that combines interactive 2D segmentation (SAM2) with GPU-accelerated Gaussian-splatting reconstruction (SuGaR). The goal is to produce clean, background-free 3D geometry from photographic datasets with minimal manual effort: precise foreground masks are generated with SAM2 and are converted to dense, continuous 3D meshes by SuGaR.

This document describes architecture, data flow, installation, configuration, usage examples, internals, recommended parameters, troubleshooting, and future improvements.

---

## Table of Contents

1. Overview
2. Architecture and Data Flow
3. Data Formats and Directory Layout
4. Installation
   - Requirements
   - Repositories and dependencies
   - Docker + GPU setup
5. Configuration & Environment
6. Pipeline Workflow (detailed)
7. Implementation Details
   - SAM2: annotation and mask generation
   - SuGaR: pointcloud, splats, and surface extraction
8. Performance & Tuning
9. Troubleshooting
10. Limitations and Known Issues
11. Roadmap & Future Work
12. References

---

## Quick links

- [Overview](#overview)
- [Architecture & Data Flow](#architecture-and-data-flow)
- [Data Formats & Layout](#data-formats-and-directory-layout)
- [Installation](#installation)
- [Pipeline Workflow](#pipeline-workflow-detailed)
- [Troubleshooting](#troubleshooting)

---

## 2. Architecture and Data Flow

High-level components:

- **Input:** photographic images (single or multi-view).
- **SAM2:** interactive segmentation (user-provided positive/negative points) → binary masks.
- **Mask-to-pointcloud:** convert mask pixels into 3D samples (single-view heuristics or multi-view triangulation if camera poses available).
- **SuGaR:** represent points as Gaussian splats, aggregate density field, extract surface (Poisson / Marching Cubes).
- **Postprocessing:** smoothing, decimation, normals, optional texture baking.

Data flow diagram (logical):

images → preprocessing → SAM2 annotation → masks → point sampling → splatting (SuGaR) → density field → surface extraction → mesh → postprocessing

---

## 3. Data Formats and Directory Layout

Recommended repository layout (root = $nephele_PATH):

nephele/
- data/
  - datasets/
    - <dataset_name>/
      - images/            # **.jpg** images
      - cameras/           # optional camera intrinsics / extrinsics (JSON, COLMAP files, etc.)
- externals/
  - SAM2/                 # optional clone/submodule for SAM integration
  - SuGaR/                # optional clone/submodule for SuGaR
- colmap/                # COLMAP wrapper for SfM and dense stereo (included in this repo)
- scripts/                # glue scripts and CLI wrappers (e.g. annotation, conversion, reconstruction)
- results/
  - <dataset_name>/
    - masks/
    - pointclouds/
    - splats/
    - meshes/
- docs/
  - README.md (this file)

File conventions (brief):
- **Images**: `.jpg` (RGB). Use unique, zero-padded filenames for ordering.
- **Masks**: binary PNG or `.npy` arrays with same HxW as source image (`0` background, `1` foreground).
- **Pointclouds**: PLY or `.npy` arrays `(x,y,z[,r,g,b])`.
- **Meshes**: `.ply` or `.obj` with normals; UVs/textures optional.

> Note: the `scripts/` directory contains _tool wrappers_. Avoid hard-coded references to non-existent filenames — adapt commands below to the tools available in your copy of the repo.

---

## 4. Installation

### Requirements

- **OS**: Linux (Ubuntu tested).
- **GPU**: NVIDIA with compatible drivers (recommended).
- **Containers**: Docker + NVIDIA Container Toolkit for GPU passthrough.
- **Python**: 3.8+ (for local utilities).
- **Hardware**: 8GB+ RAM (16GB recommended), disk for data.

### Clone the project

Replace `<repo-url>` with the URL for this repository and clone it to your machine:

```bash
git clone <repo-url> $nephele_PATH
cd $nephele_PATH
```



### Docker & NVIDIA toolkit (Ubuntu example)

```bash
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | \
  sudo tee /etc/apt/sources.list.d/nvidia-docker.list
sudo apt-get update
sudo apt-get install -y nvidia-docker2
sudo systemctl restart docker
```

Verify GPU inside a container:

```bash
docker run --rm --gpus all nvidia/cuda:12.1-base nvidia-smi
```

If using GUI-based SAM2 UIs, ensure X11 socket and DISPLAY forwarded and that host allows the container to connect to X server (e.g. xhost +local:root).

---

## 5. Configuration & Environment

Set environment variables used by scripts/wrappers:

- nephele_PATH — path to the project root (absolute).


Example:

```bash
export SAM_FIT_SUGAR_PATH="/home/you/nephele"

```

---

## 6. Pipeline Workflow (detailed)

1. **Prepare dataset:** place images under data/datasets/<dataset_name>/.
2. **Preprocess images (optional):** resizing, color normalization, or lens correction.
3. **Annotate using SAM2 UI (interactive):** add positive and negative points. Save masks to results/<dataset>/masks/.
   - Positive points: indicate foreground (object).
   - Negative points: indicate background/holes.
4. **Convert masks to sampled pointclouds using SFM.**
   - Recommended for accurate geometry: use calibrated multi-view images and triangulate mask outlines or dense stereo.
5. **Run SuGaR reconstruction:** create Gaussian splats per sample, aggregate, then extract surface.
6. **Postprocess mesh:** smoothing, decimation, normal recomputation, and optional texture baking.

---
## 7. Implementation Details

### SAM2: annotation and mask generation


- **SAM Video Prediction (mask propagation):** when input is a video or ordered image sequence, use a propagation tool to automatically extend annotations across frames. Typical workflow:
  1. Annotate key frames (first frame or several keyframes).
  2. Run mask propagation to predict masks for intermediate frames.
  3. Inspect and correct poor predictions by adding corrective points and re-propagating if necessary.
  4. Save resulting masks into `results/<dataset>/video_masks/` as a time-ordered set of PNGs or .npy files.



**Interaction & mask generation**  
- Supported prompt types: **points**, **bounding boxes**, and **masks** on any selected frame.  
- **Left-click** adds a *positive (foreground)* point, **right-click** a *negative (background)* one.  
- After a prompt, SAM2 produces a **binary segmentation mask** for that frame (saved as `.png` or `.npy`).  
- Internally, the model performs:
  1. **Seed-based region growth:** expands the foreground region from annotated seeds using visual similarity and edge-aware refinement.  
  2. **Contour refinement:** aligns mask boundaries with local gradients and object edges.  
  3. **Memory encoding:** the resulting mask is stored in the temporal memory, linking it to the object ID and spatial context for future frames.


**Video Propagation / Prediction**  
When SAM2 is used for a video or image sequence, segmentation can be propagated automatically:

1. **Initialization:**  
   Annotate one or more *key frames* with points, boxes, or masks using  
   ```python
   frame_idx, obj_ids, masks = predictor.add_new_points_or_box(state, frame_idx, obj_id, points, labels)

- **Integration with 3D pipeline:** per-frame masks from video propagation can be converted to per-frame point samples and then fused in SuGaR (temporal fusion or multi-view triangulation) to improve geometry stability and coverage.

References and resources

- **Segment Anything Model (official):** https://github.com/facebookresearch/segment-anything
- **Example community projects (search GitHub):** "Video-SAM", "VideoSAM", "sam-video" — these provide mask propagation implementations and GUI wrappers for temporal annotation.

### SuGaR: Surface-Aligned Gaussian Splatting for Efficient 3D Mesh Reconstruction

**Reference:**  
Guédon & Lepetit, *"SuGaR: Surface-Aligned Gaussian Splatting for Efficient 3D Mesh Reconstruction and High-Quality Mesh Rendering"*, CVPR 2024.  
[[Paper](https://openaccess.thecvf.com/content/CVPR2024/papers/Guedon_SuGaR_Surface-Aligned_Gaussian_Splatting_for_Efficient_3D_Mesh_Reconstruction_and_CVPR_2024_paper.pdf)] · [[GitHub](https://github.com/Anttwo/SuGaR)]

---

### 📘 Overview
SuGaR extends the **3D Gaussian Splatting (3DGS)** framework to enable **accurate and efficient surface reconstruction**.  
While standard 3DGS optimizes unstructured volumetric Gaussians, SuGaR adds geometric priors to **align Gaussians with scene surfaces**, enabling high-fidelity **mesh extraction** and **hybrid mesh-Gaussian rendering**.

---

### Core Concept

Each Gaussian splat Gi is defined by its:
- Center **μᵢ**
- Covariance **Σᵢ**
- Color **cᵢ**
- Opacity **αᵢ**

Formula:
**Gᵢ(x) = αᵢ · exp[ -½ (x − μᵢ)ᵀ Σᵢ⁻¹ (x − μᵢ) ]**

Total density field:
**ρ(x) = Σ Gᵢ(x)**

SuGaR introduces a **surface-alignment regularization** that encourages:
- The smallest Gaussian eigenvector to align with the **local surface normal**
- Gaussian centers to lie close to the **isosurface** of the density field

This transforms the volumetric 3DGS representation into a **surface-aligned distribution** suitable for mesh extraction.

---

### ⚙️ Pipeline

| Stage | Input | Output | Method |
|:------|:------|:--------|:-------|
| **1. Base 3DGS Training** | Multi-view images | Gaussian splats | Standard Gaussian Splatting |
| **2. Surface Alignment** | Raw Gaussians | Surface-aligned splats | Alignment + Distance losses |
| **3. Surface Sampling** | Aligned splats | Oriented point cloud | Isosurface sampling |
| **4. Mesh Reconstruction** | Point cloud | Triangular mesh | Poisson Surface Reconstruction |
| **5. Hybrid Refinement** | Mesh + Gaussians | Optimized hybrid model | Joint optimization |
| **6. Post-Processing** | Mesh | Textured / smoothed mesh | Cleaning, decimation, UV mapping |

---

### Loss Formulation
The surface-alignment objective combines **orientation** and **distance** consistency:

**𝓛ₛᵤʳ𝒻 = λₐₗᵢgₙ · ‖ (nᵢᵀ vᵢ)² − 1 ‖ + λ_dᵢₛₜ · ‖ dᵢ − d̂ᵢ ‖**

Where:
- **nᵢ** — local surface normal  
- **vᵢ** — Gaussian principal direction  
- **dᵢ** — distance of Gaussian center to the target surface  

These terms ensure that each Gaussian is **oriented and positioned** coherently along the visible surfaces.

---

### 🎯 Advantages
- **Fast mesh extraction:** Poisson reconstruction completes in minutes  
- **High-quality geometry:** Smooth, watertight, and accurate surfaces  
- **Hybrid representation:** Editable mesh + Gaussian rendering  
- **Physically consistent normals and albedo**

---

### ⚠️ Limitations
- Sensitive to sparse or thin structures  
- Requires a pre-optimized 3DGS model  
- Over-strong alignment regularization can cause surface flattening  

---

### ⏱️ Performance
| Metric | Typical Value |
|:-------|:--------------|
| Fine-tuning iterations | ~7000 |
| Mesh extraction time | 2–3 minutes |
| Rendering cost | ≈ same as 3DGS |
| Output size | Slightly higher (due to alignment terms) |

---



## 8. Performance & Tuning

Tips to balance quality vs resource usage:
- Image resolution: reduce resolution when memory is constrained. Try 720p if 1080p is too large.
- Sampling: subsample mask pixels during pointcloud conversion (e.g., keep 1/2 or 1/4 pixels).
- Poisson depth: lower values reduce memory and runtime.
- Splat radius: increase to smooth noisy pointclouds; decrease to capture fine detail.
- Run reconstruction on machines with high VRAM GPUs for best performance.

---

## 9. Troubleshooting

Problem: GPU not visible in container
- Verify host has drivers: run `nvidia-smi` on host.
- Ensure NVIDIA Container Toolkit installed and Docker restarted.
- Run container with `--gpus all` flag.

Problem: SAM2 GUI does not open from Docker
- Ensure X11 socket mounted: `-v /tmp/.X11-unix:/tmp/.X11-unix -e DISPLAY=$DISPLAY`.
- Allow container to access X server: on host run `xhost +local:root` (security tradeoff).
- Alternatively, use VNC or X forwarding.

Problem: Masks are noisy or leaking
- Add more negative points around suspected leak areas.
- Use contour refinement settings in SAM2 (if exposed) to tighten boundaries.

Problem: Output mesh has holes or disconnected components
- Increase sampling density or triangulation accuracy.
- Run gap-filling postprocessing or use Poisson reconstruction with higher depth.

---

## 10. Limitations and Known Issues

- Single-view reconstruction is inherently ambiguous; multi-view data yields significantly better results.
- Very complex backgrounds or low contrast between object and background reduces mask accuracy.
- Reconstruction quality depends on annotation accuracy and image coverage.
- Large-scale datasets require significant GPU memory and storage.

---

## 11. Roadmap & Future Work

Planned improvements:
- Semi-automated point suggestion: use object detectors to propose initial positive/negative points.
- Native multi-view integration: **IMPLEMENTED** — a COLMAP wrapper is included in the repository under the `colmap/` subfolder. Use this wrapper to compute camera poses, run sparse and dense reconstruction, and export camera parameters / depth maps for downstream mask→pointcloud conversion.
- Texture baking and PBR export workflow.
- Adaptive splat subsampling and streaming reconstruction for very large scenes.

---

## 12. References

- SAM projects and interactive segmentation literature.
- Gaussian splatting and surface reconstruction (Poisson, Marching Cubes).
- External repos used as references: https://github.com/peasant98/SAM2-Docker, https://github.com/Anttwo/SuGaR

---

If you want, I can also:
- add a runnable `scripts/run_pipeline.sh` and example `docker-compose.yml`,
- add default config files under `configs/`,
- or create a concise `README.md` from this document.

