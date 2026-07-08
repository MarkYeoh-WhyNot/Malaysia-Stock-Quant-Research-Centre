# OpenClaw — 24/7 VPS Deployment Runbook

Deploys all five services (api, daemon, telegram, event-watcher, caddy) via
Docker Compose to a fresh VPS. No host-mounted venv, no `/opt/openclaw`
dependency — the repo builds its own image (see `Dockerfile`).

## 1. Provision the VPS

Any small Linux VPS works (DigitalOcean, Hetzner, Vultr, etc.). Minimum spec:
2 GB RAM, Ubuntu 24.04 LTS. ~US$6–12/month.

```bash
# On the VPS, as root (first login):
adduser openclaw
usermod -aG sudo openclaw
mkdir -p /home/openclaw/.ssh
cp ~/.ssh/authorized_keys /home/openclaw/.ssh/    # your public key
chown -R openclaw:openclaw /home/openclaw/.ssh
chmod 700 /home/openclaw/.ssh && chmod 600 /home/openclaw/.ssh/authorized_keys
```

Then disable password auth in `/etc/ssh/sshd_config`
(`PasswordAuthentication no`) and `systemctl restart sshd`. From here on, SSH
in as `openclaw`, not root.

## 2. Install Docker

```bash
curl -fsSL https://get.docker.com | sudo sh
sudo usermod -aG docker $USER
# log out and back in for the group change to take effect
docker version
```

## 3. Firewall

```bash
sudo apt-get install -y ufw
sudo ufw allow OpenSSH
sudo ufw allow 80/tcp
sudo ufw allow 443/tcp
sudo ufw enable
```

Port 8001 (the API) is **not** opened — it's only reachable through Caddy on
80/443, as configured in `docker-compose.yml`.

## 4. Clone and configure

```bash
git clone https://github.com/MarkYeoh-WhyNot/Malaysia-Stock-Quant-Research-Centre.git
cd Malaysia-Stock-Quant-Research-Centre
cp .env.example .env
nano .env   # fill in every value — see checklist below
```

`.env` checklist (see `.env.example` for the full list):
- `ANTHROPIC_API_KEY` — required
- `TELEGRAM_BOT_TOKEN`, `TELEGRAM_CHAT_ID` — required for briefings/alerts
- `ALERT_TELEGRAM_CHAT_ID` — operational alerts (daemon crash/restart, budget
  exhausted); defaults to `TELEGRAM_CHAT_ID` if left blank
- `OPENCLAW_API_KEY` — generate with `openssl rand -hex 32`; without this the
  dashboard API has **no authentication**
- `DASHBOARD_ORIGIN` — set to `https://yourdomain` (or `http://localhost` for
  IP-only access) once you know it
- `AI_DAILY_BUDGET_USD` — cost cap, default $50
- Leave `OANDA_*` blank — legacy, unused for Bursa equities

If you have an existing `openclaw.db` from a prior deployment, copy it in
before first start:

```bash
docker volume create malaysia-stock-quant-research-centre_openclaw-data
docker run --rm -v malaysia-stock-quant-research-centre_openclaw-data:/data \
    -v /path/to/old/openclaw.db:/src/openclaw.db alpine \
    cp /src/openclaw.db /data/openclaw.db
```

## 5. (Optional) Point a domain at the VPS

If you have a domain, add an A record pointing to the VPS IP, then set
`DASHBOARD_DOMAIN=yourdomain.com` in `.env`. Caddy will automatically obtain
and renew a Let's Encrypt certificate. Without a domain, Caddy falls back to
a self-signed "internal" certificate (browsers will warn once).

## 6. Start the stack

```bash
docker compose up -d --build
docker compose ps       # all 5 should show Up (api, daemon healthy)
```

## 7. Verification checklist

```bash
# Dashboard reachable via Caddy, not directly on 8001:
curl -sk https://<vps-ip-or-domain>/api/health          # 200, no key needed
curl -so /dev/null -w '%{http_code}\n' http://<vps-ip>:8001/api/health  # should time out / refuse

# Auth is enforced:
curl -sk -o /dev/null -w '%{http_code}\n' https://<vps-ip>/api/mission-control          # 401
curl -sk -H "X-API-Key: $OPENCLAW_API_KEY" -o /dev/null -w '%{http_code}\n' \
    https://<vps-ip>/api/mission-control                                                # 200

# Daemon heartbeat / healthcheck:
docker compose ps daemon    # should show "healthy" within ~1 minute

# Logs:
docker compose logs -f daemon
docker compose logs -f telegram   # confirm it connects (not "InvalidToken")
```

Send yourself a Telegram message from the bot's `/status` command to confirm
end-to-end connectivity. You should also receive a "daemon started" alert on
first boot.

## 8. Day-2 operations

```bash
docker compose ps                       # status of all services
docker compose logs -f <service>        # tail logs
docker compose restart <service>        # restart one service
docker compose down                     # stop everything (data persists in named volumes)
docker compose up -d --build             # rebuild + restart after a `git pull`
docker compose exec daemon sqlite3 /app/data/openclaw.db "..."   # ad-hoc DB query
```

Backups land automatically in the `openclaw-backups` volume nightly
(03:00 UTC), 14-day retention. To pull the latest one to your local machine:

```bash
docker compose cp daemon:/app/backups/. ./local-backups/
```

## 9. What's already automated

- **Restarts**: every service has `restart: unless-stopped` — Docker restarts
  a crashed container automatically on a real Linux host. (This was verified
  working via `docker compose up -d` in local testing; a direct `docker kill`
  did not trigger it inside a sandboxed Docker Desktop test environment —
  re-confirm the crash → auto-restart path once on the actual VPS.)
- **Scheduler catch-up**: daily jobs (KB hunt, briefing, KLSE refresh,
  screener ideas, CPO signal, analyst monitor, DB maintenance) persist their
  last-run time and catch up on the next cycle if the daemon was down during
  their scheduled hour — no missed days from downtime.
- **Alerts**: Telegram notification on daemon start/crash and on daily budget
  exhaustion.
- **Backups + log pruning**: nightly, no manual intervention needed.
- **TLS**: automatic via Caddy (Let's Encrypt with a domain, self-signed
  without one).

## 10. Deferred / known gaps

See the project's deferred backlog for code-health items (dead OANDA code,
doc drift, async restructuring). Operationally still missing: CI/tests in
a pipeline, offsite backup replication (rclone to S3/B2 — the backup file is
just a gzip, easy to wire up later), and a real uptime monitor pinging
`/api/health` from outside the VPS.
