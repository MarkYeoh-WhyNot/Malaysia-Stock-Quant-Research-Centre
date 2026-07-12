"""Source Health Inspector — validates data sources degrade gracefully.

Monitors configured data sources (DefiLlama, CoinGecko, Yahoo Finance, Binance,
KLSE Screener, etc.) for availability and graceful degradation on failure.
Each source should return empty/partial data rather than raising on network errors,
rate limits, or malformed responses.
"""

import logging
from typing import Optional, Dict, Any, List, Tuple
from governance.base import Inspector
from governance.schemas import Finding

logger = logging.getLogger(__name__)


class SourceHealthInspector(Inspector):
    """L0 inspector that validates all data sources degrade gracefully.

    For each configured data source, attempts a lightweight health check and
    emits a Finding. Sources that are unavailable or rate-limited degrade to
    return empty/partial results rather than raising exceptions.

    One Finding per source per invocation; scope is "data_source:<source_name>".
    """

    name = "SourceHealthInspector"
    level = "L0"

    # Configured sources to check — each is (source_name, check_callable, description)
    SOURCES: List[Tuple[str, str, str]] = [
        ("defillama", "defillama", "DeFi protocol metrics (TVL, fees, revenue)"),
        ("coingecko", "coingecko", "Cryptocurrency token supply data"),
        ("yahoo", "yahoo", "OHLCV price history and fundamentals"),
        ("binance", "binance", "Crypto spot/perp prices and funding rates"),
        ("klse_screener", "klse_screener", "KLSE fundamentals and screening"),
        ("finnhub", "finnhub", "Financial events and earnings calendars"),
    ]

    def inspect(self, scope: str, ctx: Dict[str, Any]) -> Optional[Finding]:
        """Check health of a single data source.

        Args:
            scope: Identifier like "data_source:defillama" or "data_source:coingecko"
            ctx: Dictionary with:
                - "source": source name to check (if not provided, extract from scope)
                - "market_mode": "bursa" or "crypto" (optional; all sources checked regardless)

        Returns:
            A Finding with status PASS/WARN/FAIL and severity INFO/WARNING/BLOCKER,
            or None if scope is malformed.
        """
        # Parse scope or get source from context
        source_name = None
        if scope and scope.startswith("data_source:"):
            source_name = scope.replace("data_source:", "").lower()
        else:
            source_name = ctx.get("source", "").lower()

        if not source_name:
            return None

        # Dispatch to the appropriate check
        if source_name == "defillama":
            return self._check_defillama(scope, ctx)
        elif source_name == "coingecko":
            return self._check_coingecko(scope, ctx)
        elif source_name == "yahoo":
            return self._check_yahoo(scope, ctx)
        elif source_name == "binance":
            return self._check_binance(scope, ctx)
        elif source_name == "klse_screener":
            return self._check_klse_screener(scope, ctx)
        elif source_name == "finnhub":
            return self._check_finnhub(scope, ctx)
        else:
            # Unknown source — not configured
            return None

    def _check_defillama(self, scope: str, ctx: Dict[str, Any]) -> Optional[Finding]:
        """Check DefiLlama API health via a lightweight fetch attempt."""
        try:
            from data.defillama.client import DefiLlamaClient

            client = DefiLlamaClient(timeout=5)
            # Try to fetch a small set of well-known protocols
            result = client.fetch_protocol_metrics(["lido", "aave"])

            # If we got results, the source is healthy
            if result:
                return Finding(
                    agent=self.name,
                    level=self.level,
                    scope=scope or "data_source:defillama",
                    status="PASS",
                    severity="INFO",
                    evidence={"protocols_fetched": len(result), "status": "healthy"},
                    local_recommendation="DefiLlama API is responsive.",
                )
            else:
                # No results returned — likely rate-limited or degraded
                return Finding(
                    agent=self.name,
                    level=self.level,
                    scope=scope or "data_source:defillama",
                    status="WARN",
                    severity="WARNING",
                    evidence={"protocols_fetched": 0, "status": "degraded"},
                    local_recommendation="DefiLlama API returned empty results (rate-limited or temporarily unavailable).",
                )
        except Exception as e:
            # Should not happen if client degrades gracefully, but guard against it
            logger.warning(f"SourceHealthInspector: DefiLlama check raised: {e}")
            return Finding(
                agent=self.name,
                level=self.level,
                scope=scope or "data_source:defillama",
                status="WARN",
                severity="WARNING",
                evidence={"error": str(e), "status": "exception_raised"},
                local_recommendation="DefiLlama client raised an exception; this should not happen (client should degrade gracefully).",
                escalate_to="DataEngineer",
            )

    def _check_coingecko(self, scope: str, ctx: Dict[str, Any]) -> Optional[Finding]:
        """Check CoinGecko API health via a lightweight fetch attempt."""
        try:
            from data.coingecko.client import CoinGeckoClient

            # Try to fetch a small set of well-known tokens
            result = CoinGeckoClient.fetch_token_supply(["BTC", "ETH"])

            # If we got results, the source is healthy
            if result:
                return Finding(
                    agent=self.name,
                    level=self.level,
                    scope=scope or "data_source:coingecko",
                    status="PASS",
                    severity="INFO",
                    evidence={"tokens_fetched": len(result), "status": "healthy"},
                    local_recommendation="CoinGecko API is responsive.",
                )
            else:
                # No results returned — likely rate-limited or degraded
                return Finding(
                    agent=self.name,
                    level=self.level,
                    scope=scope or "data_source:coingecko",
                    status="WARN",
                    severity="WARNING",
                    evidence={"tokens_fetched": 0, "status": "degraded"},
                    local_recommendation="CoinGecko API returned empty results (rate-limited or temporarily unavailable).",
                )
        except Exception as e:
            # Should not happen if client degrades gracefully, but guard against it
            logger.warning(f"SourceHealthInspector: CoinGecko check raised: {e}")
            return Finding(
                agent=self.name,
                level=self.level,
                scope=scope or "data_source:coingecko",
                status="WARN",
                severity="WARNING",
                evidence={"error": str(e), "status": "exception_raised"},
                local_recommendation="CoinGecko client raised an exception; this should not happen (client should degrade gracefully).",
                escalate_to="DataEngineer",
            )

    def _check_yahoo(self, scope: str, ctx: Dict[str, Any]) -> Optional[Finding]:
        """Check Yahoo Finance API health via a lightweight fetch attempt."""
        try:
            from data.yahoo.client import get_historical_data

            # Try to fetch a small recent window for a reliable ticker
            # (Use AAPL for a test — highly liquid, rarely delisted)
            market_mode = ctx.get("market_mode", "bursa")
            if market_mode == "crypto":
                test_ticker = "BTCUSDT"  # Binance spot in crypto mode
            else:
                test_ticker = "1155.KL"  # Maybank in Bursa mode

            result = get_historical_data(test_ticker, lookback_days=5)

            # If we got results, the source is healthy
            if not result.empty:
                return Finding(
                    agent=self.name,
                    level=self.level,
                    scope=scope or "data_source:yahoo",
                    status="PASS",
                    severity="INFO",
                    evidence={"bars_fetched": len(result), "status": "healthy"},
                    local_recommendation="Yahoo Finance API is responsive.",
                )
            else:
                # Empty DataFrame — likely degraded
                return Finding(
                    agent=self.name,
                    level=self.level,
                    scope=scope or "data_source:yahoo",
                    status="WARN",
                    severity="WARNING",
                    evidence={"bars_fetched": 0, "status": "degraded"},
                    local_recommendation="Yahoo Finance returned empty results (rate-limited or data unavailable).",
                )
        except Exception as e:
            logger.warning(f"SourceHealthInspector: Yahoo Finance check raised: {e}")
            return Finding(
                agent=self.name,
                level=self.level,
                scope=scope or "data_source:yahoo",
                status="WARN",
                severity="WARNING",
                evidence={"error": str(e), "status": "exception_raised"},
                local_recommendation="Yahoo Finance client raised an exception; this should not happen (client should degrade gracefully).",
                escalate_to="DataEngineer",
            )

    def _check_binance(self, scope: str, ctx: Dict[str, Any]) -> Optional[Finding]:
        """Check Binance API health via a lightweight fetch attempt."""
        try:
            from data.binance.client import get_historical_data

            # Try to fetch a small recent window for BTC/USDT
            result = get_historical_data("BTC/USDT", lookback_days=5)

            # If we got results, the source is healthy
            if not result.empty:
                return Finding(
                    agent=self.name,
                    level=self.level,
                    scope=scope or "data_source:binance",
                    status="PASS",
                    severity="INFO",
                    evidence={"bars_fetched": len(result), "status": "healthy"},
                    local_recommendation="Binance API is responsive.",
                )
            else:
                # Empty DataFrame — likely degraded
                return Finding(
                    agent=self.name,
                    level=self.level,
                    scope=scope or "data_source:binance",
                    status="WARN",
                    severity="WARNING",
                    evidence={"bars_fetched": 0, "status": "degraded"},
                    local_recommendation="Binance API returned empty results (rate-limited or data unavailable).",
                )
        except Exception as e:
            logger.warning(f"SourceHealthInspector: Binance check raised: {e}")
            return Finding(
                agent=self.name,
                level=self.level,
                scope=scope or "data_source:binance",
                status="WARN",
                severity="WARNING",
                evidence={"error": str(e), "status": "exception_raised"},
                local_recommendation="Binance client raised an exception; this should not happen (client should degrade gracefully).",
                escalate_to="DataEngineer",
            )

    def _check_klse_screener(self, scope: str, ctx: Dict[str, Any]) -> Optional[Finding]:
        """Check KLSE Screener health via a lightweight fetch attempt."""
        try:
            from data.klse_screener.client import KLSEScreenerClient

            client = KLSEScreenerClient(timeout=5)
            # Try to get fundamental data for a single well-known stock
            result = client.get_fundamentals("1155.KL")

            # If we got results, the source is healthy
            if result:
                return Finding(
                    agent=self.name,
                    level=self.level,
                    scope=scope or "data_source:klse_screener",
                    status="PASS",
                    severity="INFO",
                    evidence={"stock_fetched": True, "status": "healthy"},
                    local_recommendation="KLSE Screener API is responsive.",
                )
            else:
                # Empty result — likely degraded
                return Finding(
                    agent=self.name,
                    level=self.level,
                    scope=scope or "data_source:klse_screener",
                    status="WARN",
                    severity="WARNING",
                    evidence={"stock_fetched": False, "status": "degraded"},
                    local_recommendation="KLSE Screener returned empty results (rate-limited or data unavailable).",
                )
        except Exception as e:
            logger.warning(f"SourceHealthInspector: KLSE Screener check raised: {e}")
            return Finding(
                agent=self.name,
                level=self.level,
                scope=scope or "data_source:klse_screener",
                status="WARN",
                severity="WARNING",
                evidence={"error": str(e), "status": "exception_raised"},
                local_recommendation="KLSE Screener client raised an exception; this should not happen (client should degrade gracefully).",
                escalate_to="DataEngineer",
            )

    def _check_finnhub(self, scope: str, ctx: Dict[str, Any]) -> Optional[Finding]:
        """Check Finnhub API health via a lightweight fetch attempt."""
        try:
            from data.events.finnhub_client import FinnhubClient
            import config.settings as settings

            # Check if API key is configured
            finnhub_api_key = getattr(settings, "FINNHUB_API_KEY", None)

            if not finnhub_api_key:
                return Finding(
                    agent=self.name,
                    level=self.level,
                    scope=scope or "data_source:finnhub",
                    status="WARN",
                    severity="WARNING",
                    evidence={"api_key_configured": False, "status": "unconfigured"},
                    local_recommendation="FINNHUB_API_KEY is not configured; Finnhub data is unavailable.",
                )

            client = FinnhubClient(api_key=finnhub_api_key, timeout=5)
            # Try to get company info for a well-known symbol
            result = client.get_company_info("AAPL")

            # If we got results, the source is healthy
            if result:
                return Finding(
                    agent=self.name,
                    level=self.level,
                    scope=scope or "data_source:finnhub",
                    status="PASS",
                    severity="INFO",
                    evidence={"company_fetched": True, "status": "healthy"},
                    local_recommendation="Finnhub API is responsive.",
                )
            else:
                # Empty result — likely degraded
                return Finding(
                    agent=self.name,
                    level=self.level,
                    scope=scope or "data_source:finnhub",
                    status="WARN",
                    severity="WARNING",
                    evidence={"company_fetched": False, "status": "degraded"},
                    local_recommendation="Finnhub returned empty results (rate-limited or data unavailable).",
                )
        except Exception as e:
            logger.warning(f"SourceHealthInspector: Finnhub check raised: {e}")
            return Finding(
                agent=self.name,
                level=self.level,
                scope=scope or "data_source:finnhub",
                status="WARN",
                severity="WARNING",
                evidence={"error": str(e), "status": "exception_raised"},
                local_recommendation="Finnhub client raised an exception; this should not happen (client should degrade gracefully).",
                escalate_to="DataEngineer",
            )
