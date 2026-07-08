#!/bin/sh
# Docker healthcheck for the research daemon: it has no HTTP endpoint, so
# liveness is inferred from a heartbeat file touched every scan cycle
# (scripts/research_daemon.py::_touch_heartbeat, ~60s cadence). Fail if the
# file is missing or older than 5 minutes — long enough to tolerate a slow
# backtest cycle, short enough to catch a genuinely stuck/dead process.
set -e
HEARTBEAT="/app/runtime/daemon_heartbeat"

if [ ! -f "$HEARTBEAT" ]; then
    echo "no heartbeat file yet"
    exit 1
fi

AGE=$(( $(date +%s) - $(date -r "$HEARTBEAT" +%s) ))
if [ "$AGE" -gt 300 ]; then
    echo "heartbeat stale (${AGE}s old)"
    exit 1
fi

echo "heartbeat ok (${AGE}s old)"
