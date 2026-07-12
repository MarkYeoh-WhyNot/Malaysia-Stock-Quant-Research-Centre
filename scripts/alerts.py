"""Operational alerts (daemon crash/restart, budget exhausted) via Telegram.

Deliberately independent of scripts/telegram_bot.py — a plain Bot API POST
so an alert can still be sent when the bot process itself is the thing that
died, or before the daemon has fully initialized any agent classes.
"""
import logging

import requests

from config.settings import TELEGRAM_BOT_TOKEN, ALERT_TELEGRAM_CHAT_ID, MARKET

logger = logging.getLogger(__name__)


# Phase 6.4 (audit §5.6): severity levels so a glance at the chat tells you
# whether something needs attention now (CRITICAL) or can wait (INFO).
_LEVEL_EMOJI = {
    "INFO": "🔔", "WATCH": "👀", "WARNING": "⚠️", "CRITICAL": "🚨",
}
_VALID_LEVELS = tuple(_LEVEL_EMOJI)


def _log_delivery(level: str, message: str, delivered: bool) -> None:
    """Best-effort persistence of the delivery outcome to daemon_logs so the
    Alerting department card has a real record instead of no record at all.
    Isolated in its own try/except and imported lazily — must never be the
    reason an alert call raises, and must not create a hard import-time
    dependency for a module that has to work before the daemon/DB is up.
    """
    try:
        from data.database import db_session
        with db_session() as conn:
            conn.execute(
                "INSERT INTO daemon_logs (level, source, message) VALUES (?, 'alerts', ?)",
                (level, f"{'sent' if delivered else 'dropped'}: {message[:200]}"),
            )
    except Exception as e:
        logger.warning(f"[Alerts] Failed to log delivery outcome: {e}")


def send_alert(message: str, level: str = "INFO") -> bool:
    """Best-effort Telegram notification. Never raises — a failed alert must
    not crash the caller (often itself in an exception handler).

    level: INFO (pipeline update) | WATCH (approaching a threshold) |
    WARNING (data anomaly / missing data) | CRITICAL (drawdown breach /
    execution failure / cost-model error). Unknown levels fall back to INFO.
    """
    level = level if level in _VALID_LEVELS else "INFO"
    emoji = _LEVEL_EMOJI[level]
    if not TELEGRAM_BOT_TOKEN or not ALERT_TELEGRAM_CHAT_ID:
        logger.warning(f"[Alerts:{level}] Not configured, dropping alert: {message}")
        _log_delivery(level, message, delivered=False)
        return False
    try:
        resp = requests.post(
            f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage",
            json={"chat_id": ALERT_TELEGRAM_CHAT_ID,
                  "text": f"{emoji} [{level}][{MARKET}] Mark's Research Centre: {message}"},
            timeout=10,
        )
        if resp.status_code != 200:
            logger.warning(f"[Alerts] Telegram API returned {resp.status_code}: {resp.text[:200]}")
            _log_delivery(level, message, delivered=False)
            return False
        _log_delivery(level, message, delivered=True)
        return True
    except Exception as e:
        logger.warning(f"[Alerts] Failed to send alert: {e}")
        _log_delivery(level, message, delivered=False)
        return False
