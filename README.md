# Nephele

**Nephele** processes images to generate background-free 3D meshes by combining SAM2 (segmentation), COLMAP (structure-from-motion), and SuGaR (Gaussian splatting).

---

## Requirements

- Docker & Docker Compose
- NVIDIA GPU + CUDA drivers (for SAM2 / SuGaR pipeline)
- Python 3.11+ (for local dev only)

---

## 1. Clone

```bash
git clone --recurse-submodules https://github.com/TEXTaiLES/SAMplify_SuGaR
cd SAMplify_SuGaR
```

If you forgot `--recurse-submodules`:
```bash
git submodule update --init --recursive
```

---

## 2. Set environment variable

```bash
export nephele_PATH="/absolute/path/to/SAMplify_SuGaR"
```

---

## 3. Prepare your dataset

Place `.jpg` images inside:
```
SAM2/data/input/<your_dataset_name>/
```

---

## 4. Run the pipeline

```bash
bash run_pipeline.sh <your_dataset_name>
```

Outputs are written to `SAM2/data/output/` and `SUGAR/SuGaR/outputs/`.

---

## 5. Nefele UI

The UI lets you pick segmentation points and monitor pipeline progress.

### Setup

```bash
cd nefele_ui
cp .env.example .env
```

Edit `.env` — the key values:

| Variable | Description |
|---|---|
| `SAMPLIFY_ROOT` | Absolute path to this repo on the host |
| `WEB_PORT` | Port to expose the UI (default `8092`) |
| `HOST_UID` / `HOST_GID` | Match the owner of `SAM2/data` (run `id` to check) |
| `WORKER_URL` | SAM2 worker endpoint (default `http://sam2:5001`) |
| `HESTIA_API_KEY` | Required for scan browsing / upload features |

### Start

```bash
docker compose up --build -d
```

UI is available at `http://localhost:8092`.

### Stop

```bash
docker compose down
```

---

## Project structure

```
SAMplify_SuGaR/
├── SAM2/            # Segmentation pipeline (Docker)
│   └── data/
│       ├── input/   # Put your images here
│       └── output/  # Segmentation results
├── SUGAR/           # 3D mesh reconstruction (Docker)
├── PGSR/            # Alternative Gaussian renderer (Docker)
├── nefele_ui/       # Web UI (Flask + Docker)
├── colmap/          # COLMAP helpers
└── run_pipeline.sh  # End-to-end runner
```

---

## Citation

```bibtex
@software{Nephele_TEXTaiLES_2026,
  author  = {{Athena Research Center}},
  title   = {{Nephele: SAM2 + COLMAP + SuGaR pipeline for background-free 3D mesh reconstruction}},
  url     = {https://github.com/TEXTaiLES/SAMplify_SuGaR},
  version = {0.1.0},
  year    = {2025},
  license = {MIT}
}
```

## License

MIT
