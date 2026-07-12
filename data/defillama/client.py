"""DefiLlama protocol metrics client — free data feed for DeFi protocol TVL/fees/revenue.

No API key required. Resilient: handles rate limiting (429), network errors, and
malformed responses gracefully (never raises to callers).
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Dict, List

import requests

from data.database import db_session

logger = logging.getLogger(__name__)

DEFILLAMA_API_BASE = "https://api.llama.fi"
RATE_LIMIT_RETRY_DELAY = 60  # seconds to wait on 429


@dataclass
class ProtocolMetrics:
    """One DeFi protocol's key metrics at a snapshot."""
    symbol: str
    protocol_name: str
    tvl: float | None = None
    tvl_rank: int | None = None
    fees_24h: float | None = None
    revenue_24h: float | None = None


class DefiLlamaClient:
    """Fetch protocol metrics from DefiLlama's free endpoint."""

    def __init__(self, timeout: int = 10):
        """Initialize the client.

        Args:
            timeout: HTTP request timeout in seconds.
        """
        self.timeout = timeout

    def fetch_protocol_metrics(self, symbols: List[str]) -> Dict[str, ProtocolMetrics]:
        """Fetch protocol metrics from DefiLlama for the given symbols.

        Calls https://api.llama.fi/protocols once to get all protocols, then
        filters to the requested symbols. Handles rate limiting and errors
        gracefully.

        Args:
            symbols: List of protocol symbols to fetch (e.g., ["lido", "aave"]).
                     Case-insensitive; matched against protocol names.

        Returns:
            Dict mapping symbol → ProtocolMetrics. Missing symbols are omitted
            (not errors).
        """
        url = f"{DEFILLAMA_API_BASE}/protocols"
        try:
            response = requests.get(url, timeout=self.timeout)
            response.raise_for_status()
        except requests.exceptions.Timeout:
            logger.warning(f"defillama client: timeout fetching {url}")
            return {}
        except requests.exceptions.HTTPError as e:
            if response.status_code == 429:
                logger.warning(f"defillama client: rate limited (429); consider retry after {RATE_LIMIT_RETRY_DELAY}s")
            else:
                logger.warning(f"defillama client: HTTP {response.status_code} from {url}")
            return {}
        except Exception as e:
            logger.warning(f"defillama client: fetch failed: {e}")
            return {}

        try:
            protocols = response.json()
            if not isinstance(protocols, list):
                logger.warning(f"defillama client: unexpected response type (expected list, got {type(protocols).__name__})")
                return {}
        except Exception as e:
            logger.warning(f"defillama client: JSON decode failed: {e}")
            return {}

        # Normalize requested symbols to lowercase for matching
        symbol_lower = [s.lower() for s in symbols]
        result: Dict[str, ProtocolMetrics] = {}

        for proto in protocols:
            if not isinstance(proto, dict):
                continue
            # Protocol entry has keys like: name, slug, tvl, rank, "24h" (fees), ...
            proto_name = proto.get("name", "").lower()
            proto_slug = proto.get("slug", "").lower()

            # Match against both name and slug
            matched_symbol = None
            for requested_sym in symbol_lower:
                if requested_sym in [proto_name, proto_slug]:
                    matched_symbol = requested_sym
                    break

            if matched_symbol is None:
                continue

            # Extract metrics from the protocol entry
            tvl = proto.get("tvl")
            if tvl is not None:
                try:
                    tvl = float(tvl)
                except (TypeError, ValueError):
                    tvl = None

            tvl_rank = proto.get("rank")
            if tvl_rank is not None:
                try:
                    tvl_rank = int(tvl_rank)
                except (TypeError, ValueError):
                    tvl_rank = None

            # DefiLlama returns fees and revenue in a nested "24h" object or top-level
            fees_24h = None
            revenue_24h = None

            # Try top-level keys first
            if "24h" in proto:
                fees_24h_data = proto.get("24h")
                if isinstance(fees_24h_data, dict):
                    fees_24h = fees_24h_data.get("fees")
                    revenue_24h = fees_24h_data.get("revenue")

            # Fall back to top-level keys if not found in 24h object
            if fees_24h is None:
                fees_24h = proto.get("fees_24h")
            if revenue_24h is None:
                revenue_24h = proto.get("revenue_24h")

            # Convert to float if present
            if fees_24h is not None:
                try:
                    fees_24h = float(fees_24h)
                except (TypeError, ValueError):
                    fees_24h = None

            if revenue_24h is not None:
                try:
                    revenue_24h = float(revenue_24h)
                except (TypeError, ValueError):
                    revenue_24h = None

            result[matched_symbol] = ProtocolMetrics(
                symbol=matched_symbol,
                protocol_name=proto.get("name", matched_symbol),
                tvl=tvl,
                tvl_rank=tvl_rank,
                fees_24h=fees_24h,
                revenue_24h=revenue_24h,
            )

        return result

    def record_metrics(self, metrics: Dict[str, ProtocolMetrics]) -> None:
        """Write protocol metrics to the database.

        Args:
            metrics: Dict mapping symbol → ProtocolMetrics (as returned from
                     fetch_protocol_metrics).
        """
        if not metrics:
            return

        try:
            with db_session() as conn:
                for symbol, m in metrics.items():
                    conn.execute(
                        """
                        INSERT INTO protocol_metrics
                        (symbol, protocol_name, tvl, tvl_rank, fees_24h, revenue_24h)
                        VALUES (?, ?, ?, ?, ?, ?)
                        """,
                        (m.symbol, m.protocol_name, m.tvl, m.tvl_rank, m.fees_24h, m.revenue_24h),
                    )
            logger.info(f"defillama client: recorded {len(metrics)} protocol metrics")
        except Exception as e:
            logger.error(f"defillama client: failed to record metrics: {e}")
