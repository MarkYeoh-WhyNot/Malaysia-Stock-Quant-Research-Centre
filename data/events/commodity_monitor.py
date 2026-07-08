"""
CommodityMonitor — detects significant single-day commodity price moves.
Uses Yahoo Finance (yfinance) already integrated in the project.
Triggered commodity moves create events for plantation and O&G stocks.
"""
import logging
from datetime import datetime, timezone

import yfinance as yf

logger = logging.getLogger(__name__)

from config.settings import MARKET_MODE

# Watchlist is market-specific. Bursa watches commodities that drive KLSE
# sectors; crypto watches the two majors whose big moves drive the whole
# market (BTC/ETH beta). Both use yfinance tickers — BTC-USD/ETH-USD are
# available there, so this monitor needs no ccxt path.
if MARKET_MODE == "crypto":
    COMMODITY_WATCHLIST = {
        "BTC-USD": {
            "name": "Bitcoin",
            "threshold_pct": 5.0,
            "affected_sectors": ["crypto_majors"],
            "affected_tickers": ["ETH/USDT", "SOL/USDT", "AVAX/USDT", "LINK/USDT"],
            "lag_days": 2,
            "event_type": "btc_move",
        },
        "ETH-USD": {
            "name": "Ethereum",
            "threshold_pct": 6.0,
            "affected_sectors": ["smart_contract", "defi", "layer2"],
            "affected_tickers": ["ARB/USDT", "OP/USDT", "UNI/USDT", "AAVE/USDT"],
            "lag_days": 2,
            "event_type": "eth_move",
        },
    }
else:
    COMMODITY_WATCHLIST = {
        "CPO=F": {
            "name": "Crude Palm Oil",
            "threshold_pct": 2.0,
            "affected_sectors": ["plantation"],
            "affected_tickers": ["5285.KL", "2291.KL", "5182.KL", "1961.KL", "5069.KL", "2445.KL"],
            "lag_days": 3,
            "event_type": "cpo_move",
        },
        "BZ=F": {
            "name": "Brent Crude Oil",
            "threshold_pct": 3.0,
            "affected_sectors": ["oil_gas"],
            "affected_tickers": ["5398.KL", "5183.KL", "6033.KL", "7277.KL"],
            "lag_days": 3,
            "event_type": "crude_oil_move",
        },
        "GC=F": {
            "name": "Gold",
            "threshold_pct": 2.0,
            "affected_sectors": ["mining"],
            "affected_tickers": [],
            "lag_days": 1,
            "event_type": "gold_move",
        },
        "CL=F": {
            "name": "WTI Crude Oil",
            "threshold_pct": 3.0,
            "affected_sectors": ["oil_gas"],
            "affected_tickers": ["5398.KL", "5183.KL", "6033.KL", "7277.KL"],
            "lag_days": 3,
            "event_type": "crude_oil_move",
        },
    }


class CommodityMonitor:
    """Monitors commodity price moves and generates events when thresholds are breached."""

    def check_moves(self) -> list:
        """
        Check each commodity for significant single-day moves.
        Returns list of event dicts for moves exceeding threshold.
        """
        events = []
        today_str = datetime.utcnow().strftime("%Y-%m-%d")

        for symbol, config in COMMODITY_WATCHLIST.items():
            try:
                ticker = yf.Ticker(symbol)
                hist = ticker.history(period="3d", interval="1d")
                if hist.empty or len(hist) < 2:
                    logger.debug(f"CommodityMonitor: insufficient data for {symbol}")
                    continue

                prev_close = float(hist["Close"].iloc[-2])
                last_close = float(hist["Close"].iloc[-1])

                if prev_close <= 0:
                    continue

                pct_change = ((last_close - prev_close) / prev_close) * 100.0

                if abs(pct_change) < config["threshold_pct"]:
                    continue

                # Threshold breached — build event
                name = config["name"]
                lag = config["lag_days"]
                affected = config["affected_tickers"]
                sectors = config["affected_sectors"]
                sentiment = "positive" if pct_change > 0 else "negative"
                direction = "up" if pct_change > 0 else "down"
                magnitude = "high" if abs(pct_change) > (config["threshold_pct"] * 2) else "medium"

                event_id = f"commodity_{symbol.replace('=','_')}_{today_str}"

                headline = (
                    f"{name} moved {pct_change:+.1f}% today "
                    f"(prev: {prev_close:.2f} → {last_close:.2f})"
                )
                body = (
                    f"{name} {direction} {abs(pct_change):.1f}% today. "
                    f"Historically lags into affected stocks by {lag} trading days. "
                    f"Affected tickers: {', '.join(affected) if affected else 'none'}. "
                    f"Affected sectors: {', '.join(sectors)}."
                )

                events.append({
                    "event_id": event_id,
                    "source": "yahoo_finance",
                    "ticker": None,
                    "event_type": config["event_type"],
                    "headline": headline,
                    "body": body,
                    "affected_tickers": affected,
                    "affected_sectors": sectors,
                    "sentiment": sentiment,
                    "magnitude": magnitude,
                    "published_at": datetime.utcnow().isoformat(),
                    "confidence": 0.65,  # pre-set moderate confidence
                    "is_actionable": True,
                })

                logger.info(f"CommodityMonitor: {name} triggered ({pct_change:+.1f}%)")

            except Exception as exc:
                logger.warning(f"CommodityMonitor: error checking {symbol}: {exc}")

        return events
