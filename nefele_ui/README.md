# Nefele UI

Upload images, pick points, and download your 3D model.

**Open in your browser:** https://nephele.textailes.athenarc.gr

---

## Setup (first time only)

1. Clone the repository:
   ```bash
   git clone https://github.com/TEXTaiLES/SAMplify_SuGaR.git
   cd SAMplify_SuGaR/nefele_ui
   ```

2. Copy the config file:
   ```bash
   cp .env.example .env
   ```

3. Open `.env` and fill in:
   - `SAMPLIFY_ROOT` — path to the project folder on this machine
   - `HESTIA_API_KEY` — your API key

4. Start:
   ```bash
   docker compose up --build -d
   ```

---

## Stop

```bash
docker compose down
```
