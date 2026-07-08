"""Operational alerts (daemon crash/restart, budget exhausted) via Telegram.

Deliberately independent of scripts/telegram_bot.py — a plain Bot API POST
so an alert can still be sent when the bot process itself is the thing that
died, or before the daemon has fully initialized any agent classes.
"""
import logging

import requests

from config.settings import TELEGRAM_BOT_TOKEN, ALERT_TELEGRAM_CHAT_ID

logger = logging.getLogger(__name__)


def send_alert(message: str) -> bool:
    """Best-effort Telegram notification. Never raises — a failed alert must
    not crash the caller (often itself in an exception handler)."""
    if not TELEGRAM_BOT_TOKEN or not ALERT_TELEGRAM_CHAT_ID:
        logger.warning(f"[Alerts] Not configured, dropping alert: {message}")
        return False
    try:
        resp = requests.post(
            f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage",
            json={"chat_id": ALERT_TELEGRAM_CHAT_ID, "text": f"🔔 Mark's Research Centre: {message}"},
            timeout=10,
        )
        if resp.status_code != 200:
            logger.warning(f"[Alerts] Telegram API returned {resp.status_code}: {resp.text[:200]}")
            return False
        return True
    except Exception as e:
        logger.warning(f"[Alerts] Failed to send alert: {e}")
        return False
