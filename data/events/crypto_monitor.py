"""
CryptoMonitor — detects perp funding-rate extremes, open-interest jumps, and
mark/index basis dislocations on liquid crypto majors.

Mirrors CommodityMonitor.check_moves()'s contract (same event-dict shape) so
EventWatcher.run_cycle() can call it identically. Crypto-only — the Bursa
equity market has no perp funding/OI concept.
"""
import logging
from datetime import datetime

from data.binance.client import get_funding_rate, get_open_interest

logger = logging.getLogger(__name__)

# Majors only — funding/OI calls are heavier than a bulk ticker fetch, and
# funding-rate signal is only meaningful on the liquid contracts anyway.
FUNDING_WATCHLIST = ["BTC/USDT", "ETH/USDT", "SOL/USDT", "BNB/USDT"]

# Funding rate is typically settled every 8h; >0.05% (annualised ~55%) is a
# genuinely crowded/expensive side of the trade, not routine noise.
FUNDING_THRESHOLD_PCT = 0.05
# Basis (mark vs index) beyond 0.5% signals real dislocation, not quote jitter.
BASIS_THRESHOLD_PCT = 0.5
# OI jump vs the previous check (in-memory, per daemon process lifetime).
OI_JUMP_THRESHOLD_PCT = 15.0


class CryptoMonitor:
    """Monitors crypto perp funding/OI/basis and generates events on
    thresholds. OI deltas are tracked in-memory across check_moves() calls
    within one EventWatcher process run (resets on daemon restart — a missed
    baseline just means the first post-restart cycle can't detect an OI
    delta yet, which is a fine trade-off for a low-cost signal)."""

    def __init__(self):
        self._last_oi: dict[str, float] = {}

    def check_moves(self) -> list:
        events = []
        today_str = datetime.utcnow().strftime("%Y-%m-%d")

        for symbol in FUNDING_WATCHLIST:
            base = symbol.split("/")[0]

            # ── Funding rate extremes ────────────────────────────────────────
            try:
                fr = get_funding_rate(symbol)
            except Exception as exc:
                logger.warning(f"CryptoMonitor: funding fetch error for {symbol}: {exc}")
                fr = None

            if fr and abs(fr["funding_rate_pct"]) >= FUNDING_THRESHOLD_PCT:
                direction = "positive (longs pay shorts)" if fr["funding_rate_pct"] > 0 else "negative (shorts pay longs)"
                events.append({
                    "event_id": f"funding_{base}_{today_str}",
                    "source": "binance_perp",
                    "ticker": symbol,
                    "event_type": "funding_spike",
                    "headline": f"{base} perp funding rate {fr['funding_rate_pct']:+.3f}% ({direction})",
                    "body": (
                        f"{base} perpetual funding rate is {fr['funding_rate_pct']:+.3f}% this interval, "
                        f"{direction}. Extreme funding signals crowded positioning — historically prone to "
                        f"mean-reversion or a squeeze against the crowded side."
                    ),
                    "affected_tickers": [symbol],
                    "affected_sectors": ["crypto_majors"],
                    "sentiment": "negative" if fr["funding_rate_pct"] > 0 else "positive",
                    "magnitude": "high" if abs(fr["funding_rate_pct"]) >= FUNDING_THRESHOLD_PCT * 2 else "medium",
                    "published_at": datetime.utcnow().isoformat(),
                    "confidence": 0.55,
                    "is_actionable": True,
                })

                # ── Basis (mark vs index) dislocation, reuses the same fetch ──
                mark, index = fr.get("mark_price"), fr.get("index_price")
                if mark and index and index > 0:
                    basis_pct = (mark - index) / index * 100.0
                    if abs(basis_pct) >= BASIS_THRESHOLD_PCT:
                        events.append({
                            "event_id": f"basis_{base}_{today_str}",
                            "source": "binance_perp",
                            "ticker": symbol,
                            "event_type": "basis_dislocation",
                            "headline": f"{base} perp basis {basis_pct:+.2f}% (mark vs index)",
                            "body": (
                                f"{base} perp mark price is {basis_pct:+.2f}% away from index/spot. "
                                f"Large basis is a cash-and-carry opportunity or a liquidity-thin market signal."
                            ),
                            "affected_tickers": [symbol],
                            "affected_sectors": ["crypto_majors"],
                            "sentiment": "neutral",
                            "magnitude": "high" if abs(basis_pct) >= BASIS_THRESHOLD_PCT * 2 else "medium",
                            "published_at": datetime.utcnow().isoformat(),
                            "confidence": 0.50,
                            "is_actionable": True,
                        })

            # ── Open-interest jumps ──────────────────────────────────────────
            try:
                oi = get_open_interest(symbol)
            except Exception as exc:
                logger.warning(f"CryptoMonitor: OI fetch error for {symbol}: {exc}")
                oi = None

            if oi:
                current = float(oi["open_interest"])
                prev = self._last_oi.get(symbol)
                if prev and prev > 0:
                    change_pct = (current - prev) / prev * 100.0
                    if abs(change_pct) >= OI_JUMP_THRESHOLD_PCT:
                        events.append({
                            "event_id": f"oi_{base}_{today_str}_{int(current)}",
                            "source": "binance_perp",
                            "ticker": symbol,
                            "event_type": "oi_surge",
                            "headline": f"{base} open interest {change_pct:+.1f}% since last check",
                            "body": (
                                f"{base} perp open interest moved {change_pct:+.1f}% "
                                f"({prev:,.0f} -> {current:,.0f} contracts). Rapid OI buildup into a price move "
                                f"raises liquidation-cascade risk on a reversal."
                            ),
                            "affected_tickers": [symbol],
                            "affected_sectors": ["crypto_majors"],
                            "sentiment": "neutral",
                            "magnitude": "high" if abs(change_pct) >= OI_JUMP_THRESHOLD_PCT * 2 else "medium",
                            "published_at": datetime.utcnow().isoformat(),
                            "confidence": 0.50,
                            "is_actionable": True,
                        })
                        logger.info(f"CryptoMonitor: {base} OI jump {change_pct:+.1f}%")
                self._last_oi[symbol] = current

        return events
