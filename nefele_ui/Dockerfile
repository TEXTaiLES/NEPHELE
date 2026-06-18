# CPU-only image for the Nefele UI (Point Picker Flask app).
# Talks to the SAM2 worker / HESTIA API over HTTP — no torch/CUDA needed.
# Self-contained: the app code is COPYed in, so the image runs on any host.
FROM python:3.11-slim

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    IN_MNT=/data/in \
    OUT_MNT=/data/out

RUN apt-get update && apt-get install -y --no-install-recommends \
      tini ca-certificates curl procps \
      libjpeg62-turbo zlib1g \
      ffmpeg \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app/ ./app/

ARG UID=1000
ARG GID=1000
ARG UNAME=user
RUN groupadd -g ${GID} ${UNAME} || true \
 && useradd -m -u ${UID} -g ${GID} -s /bin/bash ${UNAME} \
 && mkdir -p /app ${IN_MNT} ${OUT_MNT} \
 && chown -R ${UID}:${GID} /app ${IN_MNT} ${OUT_MNT}

USER ${UID}:${GID}
WORKDIR /app

EXPOSE 5000
ENTRYPOINT ["tini", "--"]
CMD ["python3", "-u", "-m", "app.wsgi"]
