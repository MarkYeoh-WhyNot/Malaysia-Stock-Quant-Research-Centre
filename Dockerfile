# Single image shared by all OpenClaw services (api, daemon, telegram,
# event-watcher) — docker-compose.yml selects the entrypoint per service via
# `command:`. Using one image keeps the dependency set identical across
# services and avoids the "host-mounted venv" fragility of the old compose
# setup (bind-mounted /opt/openclaw/app + /opt/openclaw/venv).
FROM python:3.12-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
        curl sqlite3 \
    && rm -rf /var/lib/apt/lists/*

RUN useradd --create-home --uid 1000 openclaw
WORKDIR /app

COPY requirements.lock .
RUN pip install --no-cache-dir -r requirements.lock

COPY . .
RUN mkdir -p data logs backups && chown -R openclaw:openclaw /app

ENV PYTHONPATH=/app \
    PYTHONUNBUFFERED=1

USER openclaw
