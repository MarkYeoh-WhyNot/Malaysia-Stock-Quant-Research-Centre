"""CoinGecko free API client for cryptocurrency token supply data.

Public API endpoints only (no API key needed). Resilient like the binance
client: never raises; returns empty dict on failure. Rate-limited free tier
(10–50 calls/min); gracefully degrades on 429 responses.
"""
from __future__ import annotations

import logging
import time
from typing import Dict, List

import requests

from data.database import db_session

logger = logging.getLogger(__name__)

# Rate limiting: free tier ~10–50 calls/min, so 100–600ms between requests
_RATE_LIMIT_MS = 200


class TokenSupply:
    """Simple data class for token supply information."""

    def __init__(
        self,
        symbol: str,
        circulating_supply: float | None = None,
        total_supply: float | None = None,
        max_supply: float | None = None,
        fully_diluted_valuation: float | None = None,
        market_cap: float | None = None,
    ):
        self.symbol = symbol
        self.circulating_supply = circulating_supply
        self.total_supply = total_supply
        self.max_supply = max_supply
        self.fully_diluted_valuation = fully_diluted_valuation
        self.market_cap = market_cap


class CoinGeckoClient:
    """CoinGecko free API client for token supply data.

    - fetch_token_supply(symbols): fetch supply metrics for a list of symbols
    - record_supply(supply): write TokenSupply dict to the token_supply table

    Symbols like "BTC" and "ETH" are converted to CoinGecko coin IDs on request.
    """

    BASE_URL = "https://api.coingecko.com/api/v3"

    # Mapping of crypto symbols to CoinGecko coin IDs
    # Expand as needed — these are the most common
    SYMBOL_TO_ID = {
        "BTC": "bitcoin",
        "ETH": "ethereum",
        "BNB": "binancecoin",
        "SOL": "solana",
        "ADA": "cardano",
        "DOT": "polkadot",
        "LINK": "chainlink",
        "USDT": "tether",
        "USDC": "usd-coin",
        "XRP": "ripple",
        "DOGE": "dogecoin",
        "LTC": "litecoin",
        "BCH": "bitcoin-cash",
        "AVAX": "avalanche-2",
        "MATIC": "matic-network",
        "ARB": "arbitrum",
        "OP": "optimism",
        "AAVE": "aave",
        "UNI": "uniswap",
        "SUSHI": "sushi",
    }

    @classmethod
    def _symbol_to_id(cls, symbol: str) -> str:
        """Convert symbol (e.g. "BTC") to CoinGecko coin ID (e.g. "bitcoin").

        Falls back to lowercase symbol if not in mapping — CoinGecko is
        forgiving and will try to match by symbol anyway.
        """
        return cls.SYMBOL_TO_ID.get(symbol.upper(), symbol.lower())

    @classmethod
    def fetch_token_supply(cls, symbols: List[str]) -> Dict[str, TokenSupply]:
        """Fetch token supply for a list of symbols.

        Args:
            symbols: List of coin symbols (e.g. ["BTC", "ETH", "BNB"])

        Returns:
            Dict mapping symbol -> TokenSupply with fetched data.
            On any failure or 429 rate limit, logs gracefully and returns
            whatever was successfully fetched so far (partial results OK).

        Note: Free tier is rate-limited (10–50 calls/min). Callers should
        batch requests and/or implement backoff.
        """
        result: Dict[str, TokenSupply] = {}

        if not symbols:
            return result

        # Batch by coin IDs: ids=bitcoin,ethereum,... (fetch all at once)
        ids = ",".join([cls._symbol_to_id(s) for s in symbols])

        params = {
            "ids": ids,
            "vs_currency": "usd",
            "order": "market_cap_desc",
            "per_page": 250,  # max per request
            "sparkline": False,
            "locale": "en",
        }

        try:
            url = f"{cls.BASE_URL}/coins/markets"
            resp = requests.get(url, params=params, timeout=10)

            if resp.status_code == 429:
                logger.warning("CoinGecko client: rate limited (429) — retry later")
                return result

            resp.raise_for_status()
            data = resp.json()

            if not isinstance(data, list):
                logger.warning(f"CoinGecko client: unexpected response shape: {type(data)}")
                return result

            # Build a lookup by symbol for reverse mapping
            symbol_index = {}
            for i, symbol in enumerate(symbols):
                symbol_index[cls._symbol_to_id(symbol).lower()] = symbol

            # Parse response
            for row in data:
                coin_id = row.get("id", "").lower()
                # Map back to the original input symbol
                original_symbol = symbol_index.get(coin_id)
                if not original_symbol:
                    continue  # CoinGecko returned a coin we didn't ask for

                supply = TokenSupply(
                    symbol=original_symbol,
                    circulating_supply=row.get("circulating_supply"),
                    total_supply=row.get("total_supply"),
                    max_supply=row.get("max_supply"),
                    fully_diluted_valuation=row.get("fully_diluted_valuation"),
                    market_cap=row.get("market_cap"),
                )
                result[original_symbol] = supply

            # Log what was fetched
            logger.info(f"CoinGecko client: fetched {len(result)}/{len(symbols)} symbols")

            # Respect rate limit: if we made a call, delay before next call
            time.sleep(_RATE_LIMIT_MS / 1000.0)

        except requests.exceptions.Timeout:
            logger.warning("CoinGecko client: request timeout")
        except requests.exceptions.ConnectionError:
            logger.warning("CoinGecko client: connection error (geo-block?)")
        except Exception as e:
            logger.warning(f"CoinGecko client: fetch failed: {e}")

        return result

    @classmethod
    def record_supply(cls, supply_dict: Dict[str, TokenSupply]) -> None:
        """Write token supply data to the database.

        Args:
            supply_dict: Dict mapping symbol -> TokenSupply (e.g. output
                         of fetch_token_supply).

        Uses UNIQUE(symbol, fetched_at) constraint to allow idempotent
        writes — same symbol on the same day upserts to the latest fetch.
        """
        if not supply_dict:
            return

        try:
            with db_session() as conn:
                for symbol, supply in supply_dict.items():
                    conn.execute("""
                        INSERT INTO token_supply
                        (symbol, circulating_supply, total_supply, max_supply,
                         fully_diluted_valuation, market_cap)
                        VALUES (?, ?, ?, ?, ?, ?)
                    """, [
                        supply.symbol,
                        supply.circulating_supply,
                        supply.total_supply,
                        supply.max_supply,
                        supply.fully_diluted_valuation,
                        supply.market_cap,
                    ])
            logger.info(f"CoinGecko client: recorded {len(supply_dict)} tokens to DB")
        except Exception as e:
            logger.warning(f"CoinGecko client: record_supply failed: {e}")


def fetch_and_record_supply(symbols: List[str] | None = None) -> Dict[str, TokenSupply]:
    """Convenience wrapper: fetch + record in one call.

    Args:
        symbols: List of symbols, or None to use defaults from settings.

    Returns:
        Dict of fetched supplies (same as fetch_token_supply).
    """
    if symbols is None:
        from config.settings import DEFAULT_SYMBOLS
        # Crypto mode: extract symbol part from "BTC/USDT" -> "BTC"
        symbols = [s.split("/")[0] for s in DEFAULT_SYMBOLS]

    supply_data = CoinGeckoClient.fetch_token_supply(symbols)
    CoinGeckoClient.record_supply(supply_data)
    return supply_data
