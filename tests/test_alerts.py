"""Phase 6.4: Telegram alert severity levels."""
from unittest.mock import patch

import scripts.alerts as alerts


def test_unconfigured_alert_returns_false_and_does_not_raise():
    with patch.object(alerts, "TELEGRAM_BOT_TOKEN", ""), \
         patch.object(alerts, "ALERT_TELEGRAM_CHAT_ID", ""):
        assert alerts.send_alert("test message") is False
        assert alerts.send_alert("test message", level="CRITICAL") is False


def test_unknown_level_falls_back_to_info():
    with patch.object(alerts, "TELEGRAM_BOT_TOKEN", "tok"), \
         patch.object(alerts, "ALERT_TELEGRAM_CHAT_ID", "chat"), \
         patch("scripts.alerts.requests.post") as mock_post:
        mock_post.return_value.status_code = 200
        alerts.send_alert("m", level="NOT_A_LEVEL")
        sent_text = mock_post.call_args.kwargs["json"]["text"]
        assert "[INFO]" in sent_text


def test_critical_level_formats_message():
    with patch.object(alerts, "TELEGRAM_BOT_TOKEN", "tok"), \
         patch.object(alerts, "ALERT_TELEGRAM_CHAT_ID", "chat"), \
         patch("scripts.alerts.requests.post") as mock_post:
        mock_post.return_value.status_code = 200
        alerts.send_alert("kill switch tripped", level="CRITICAL")
        sent_text = mock_post.call_args.kwargs["json"]["text"]
        assert "[CRITICAL]" in sent_text
        assert "🚨" in sent_text
