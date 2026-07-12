"""Tests for SourceHealthInspector — validates graceful degradation of data sources."""

import pytest
from unittest.mock import patch, MagicMock
from governance.inspectors.source_health import SourceHealthInspector
from governance.schemas import Finding


class TestSourceHealthInspector:
    """Test suite for SourceHealthInspector.

    Validates that:
    1. Healthy sources return PASS/INFO findings
    2. Degraded sources return WARN/WARNING findings
    3. Exceptions raised by clients are caught and reported as WARN findings
    4. Each source gets its own Finding object per invocation
    """

    def setup_method(self):
        """Initialize the inspector for each test."""
        self.inspector = SourceHealthInspector()

    # ── DefiLlama Tests ────────────────────────────────────────────────────

    def test_defillama_healthy(self):
        """Good case: DefiLlama returns protocol metrics — expect PASS."""
        with patch("data.defillama.client.DefiLlamaClient") as mock_client_class:
            mock_instance = MagicMock()
            mock_client_class.return_value = mock_instance
            # Mock successful fetch with data
            mock_instance.fetch_protocol_metrics.return_value = {
                "lido": MagicMock(symbol="lido", protocol_name="Lido", tvl=10e9),
                "aave": MagicMock(symbol="aave", protocol_name="Aave", tvl=5e9),
            }

            finding = self.inspector.inspect(
                "data_source:defillama", {"source": "defillama"}
            )

            assert finding is not None
            assert finding.status == "PASS"
            assert finding.severity == "INFO"
            assert "healthy" in str(finding.evidence).lower()

    def test_defillama_degraded(self):
        """Bad case: DefiLlama rate-limited, returns empty dict — expect WARN."""
        with patch("data.defillama.client.DefiLlamaClient") as mock_client_class:
            mock_instance = MagicMock()
            mock_client_class.return_value = mock_instance
            # Mock degraded response (empty dict)
            mock_instance.fetch_protocol_metrics.return_value = {}

            finding = self.inspector.inspect(
                "data_source:defillama", {"source": "defillama"}
            )

            assert finding is not None
            assert finding.status == "WARN"
            assert finding.severity == "WARNING"
            assert "degraded" in str(finding.evidence).lower()

    def test_defillama_exception(self):
        """Bad case: DefiLlama client raises exception — expect WARN (graceful handling)."""
        with patch("data.defillama.client.DefiLlamaClient") as mock_client_class:
            mock_instance = MagicMock()
            mock_client_class.return_value = mock_instance
            # Mock exception on fetch
            mock_instance.fetch_protocol_metrics.side_effect = RuntimeError("Network error")

            finding = self.inspector.inspect(
                "data_source:defillama", {"source": "defillama"}
            )

            assert finding is not None
            assert finding.status == "WARN"
            assert finding.severity == "WARNING"
            assert "exception_raised" in str(finding.evidence).lower()
            assert finding.escalate_to == "DataEngineer"

    # ── CoinGecko Tests ────────────────────────────────────────────────────

    def test_coingecko_healthy(self):
        """Good case: CoinGecko returns token supply — expect PASS."""
        with patch("data.coingecko.client.CoinGeckoClient") as mock_client_class:
            from data.coingecko.client import TokenSupply

            # Mock successful fetch with data
            mock_client_class.fetch_token_supply.return_value = {
                "BTC": TokenSupply(
                    symbol="BTC",
                    circulating_supply=21e6,
                    total_supply=21e6,
                    market_cap=1e12,
                ),
                "ETH": TokenSupply(
                    symbol="ETH",
                    circulating_supply=120e6,
                    total_supply=120e6,
                    market_cap=5e11,
                ),
            }

            finding = self.inspector.inspect(
                "data_source:coingecko", {"source": "coingecko"}
            )

            assert finding is not None
            assert finding.status == "PASS"
            assert finding.severity == "INFO"
            assert "healthy" in str(finding.evidence).lower()

    def test_coingecko_degraded(self):
        """Bad case: CoinGecko rate-limited, returns empty dict — expect WARN."""
        with patch("data.coingecko.client.CoinGeckoClient") as mock_client_class:
            # Mock degraded response (empty dict)
            mock_client_class.fetch_token_supply.return_value = {}

            finding = self.inspector.inspect(
                "data_source:coingecko", {"source": "coingecko"}
            )

            assert finding is not None
            assert finding.status == "WARN"
            assert finding.severity == "WARNING"
            assert "degraded" in str(finding.evidence).lower()

    def test_coingecko_exception(self):
        """Bad case: CoinGecko client raises exception — expect WARN."""
        with patch("data.coingecko.client.CoinGeckoClient") as mock_client_class:
            # Mock exception on fetch
            mock_client_class.fetch_token_supply.side_effect = ConnectionError(
                "Network unreachable"
            )

            finding = self.inspector.inspect(
                "data_source:coingecko", {"source": "coingecko"}
            )

            assert finding is not None
            assert finding.status == "WARN"
            assert finding.severity == "WARNING"
            assert "exception_raised" in str(finding.evidence).lower()
            assert finding.escalate_to == "DataEngineer"

    # ── Yahoo Finance Tests ────────────────────────────────────────────────

    def test_yahoo_healthy(self):
        """Good case: Yahoo Finance returns price data — expect PASS."""
        import pandas as pd

        with patch("data.yahoo.client.get_historical_data") as mock_get:
            # Mock successful fetch with OHLCV data
            mock_df = pd.DataFrame(
                {
                    "open": [100.0, 101.0, 102.0, 103.0, 104.0],
                    "high": [101.0, 102.0, 103.0, 104.0, 105.0],
                    "low": [99.0, 100.0, 101.0, 102.0, 103.0],
                    "close": [100.5, 101.5, 102.5, 103.5, 104.5],
                    "volume": [1e6, 1.1e6, 1.2e6, 1.3e6, 1.4e6],
                }
            )
            mock_get.return_value = mock_df

            finding = self.inspector.inspect(
                "data_source:yahoo", {"source": "yahoo", "market_mode": "bursa"}
            )

            assert finding is not None
            assert finding.status == "PASS"
            assert finding.severity == "INFO"
            assert "healthy" in str(finding.evidence).lower()

    def test_yahoo_degraded(self):
        """Bad case: Yahoo Finance returns empty DataFrame — expect WARN."""
        import pandas as pd

        with patch("data.yahoo.client.get_historical_data") as mock_get:
            # Mock degraded response (empty DataFrame)
            mock_get.return_value = pd.DataFrame()

            finding = self.inspector.inspect(
                "data_source:yahoo", {"source": "yahoo"}
            )

            assert finding is not None
            assert finding.status == "WARN"
            assert finding.severity == "WARNING"
            assert "degraded" in str(finding.evidence).lower()

    def test_yahoo_exception(self):
        """Bad case: Yahoo Finance client raises exception — expect WARN."""
        with patch("data.yahoo.client.get_historical_data") as mock_get:
            # Mock exception on fetch
            mock_get.side_effect = Exception("Data unavailable")

            finding = self.inspector.inspect(
                "data_source:yahoo", {"source": "yahoo"}
            )

            assert finding is not None
            assert finding.status == "WARN"
            assert finding.severity == "WARNING"
            assert "exception_raised" in str(finding.evidence).lower()

    # ── Binance Tests ──────────────────────────────────────────────────────

    def test_binance_healthy(self):
        """Good case: Binance returns price data — expect PASS."""
        import pandas as pd

        with patch("data.binance.client.get_historical_data") as mock_get:
            # Mock successful fetch (different patch target for binance)
            mock_df = pd.DataFrame(
                {
                    "open": [40000.0, 41000.0, 42000.0, 43000.0, 44000.0],
                    "high": [41000.0, 42000.0, 43000.0, 44000.0, 45000.0],
                    "low": [39000.0, 40000.0, 41000.0, 42000.0, 43000.0],
                    "close": [40500.0, 41500.0, 42500.0, 43500.0, 44500.0],
                    "volume": [100, 110, 120, 130, 140],
                }
            )
            mock_get.return_value = mock_df

            finding = self.inspector.inspect(
                "data_source:binance", {"source": "binance"}
            )

            assert finding is not None
            assert finding.status == "PASS"
            assert finding.severity == "INFO"

    def test_binance_degraded(self):
        """Bad case: Binance returns empty DataFrame — expect WARN."""
        import pandas as pd

        with patch("data.binance.client.get_historical_data") as mock_get:
            # Mock degraded response
            mock_get.return_value = pd.DataFrame()

            finding = self.inspector.inspect(
                "data_source:binance", {"source": "binance"}
            )

            assert finding is not None
            assert finding.status == "WARN"
            assert finding.severity == "WARNING"

    # ── KLSE Screener Tests ────────────────────────────────────────────────

    def test_klse_screener_healthy(self):
        """Good case: KLSE Screener returns fundamental data — expect PASS."""
        with patch(
            "data.klse_screener.client.KLSEScreenerClient"
        ) as mock_client_class:
            mock_instance = MagicMock()
            mock_client_class.return_value = mock_instance
            # Mock successful fetch with fundamentals
            mock_instance.get_fundamentals.return_value = {
                "ticker": "1155.KL",
                "pe_ratio": 15.5,
                "dividend_yield": 3.2,
                "market_cap": 1e12,
            }

            finding = self.inspector.inspect(
                "data_source:klse_screener", {"source": "klse_screener"}
            )

            assert finding is not None
            assert finding.status == "PASS"
            assert finding.severity == "INFO"

    def test_klse_screener_degraded(self):
        """Bad case: KLSE Screener returns empty dict — expect WARN."""
        with patch(
            "data.klse_screener.client.KLSEScreenerClient"
        ) as mock_client_class:
            mock_instance = MagicMock()
            mock_client_class.return_value = mock_instance
            # Mock degraded response
            mock_instance.get_fundamentals.return_value = {}

            finding = self.inspector.inspect(
                "data_source:klse_screener", {"source": "klse_screener"}
            )

            assert finding is not None
            assert finding.status == "WARN"
            assert finding.severity == "WARNING"

    # ── Finnhub Tests ──────────────────────────────────────────────────────

    def test_finnhub_healthy(self):
        """Good case: Finnhub returns company info — expect PASS."""
        # Patch the settings module to have FINNHUB_API_KEY set
        with patch("governance.inspectors.source_health.getattr") as mock_getattr:
            with patch("data.events.finnhub_client.FinnhubClient") as mock_client_class:
                # Make getattr return a fake API key for FINNHUB_API_KEY
                def getattr_side_effect(obj, attr, default=None):
                    if attr == "FINNHUB_API_KEY":
                        return "test_key_12345"
                    return getattr.__wrapped__(obj, attr, default)

                mock_getattr.side_effect = getattr_side_effect

                mock_instance = MagicMock()
                mock_client_class.return_value = mock_instance
                # Mock successful fetch
                mock_instance.get_company_info.return_value = {
                    "name": "Apple Inc.",
                    "ticker": "AAPL",
                    "country": "US",
                }

                finding = self.inspector.inspect(
                    "data_source:finnhub", {"source": "finnhub"}
                )

                assert finding is not None
                assert finding.status == "PASS"
                assert finding.severity == "INFO"

    def test_finnhub_unconfigured(self):
        """Bad case: Finnhub API key not configured — expect WARN."""
        # Since FINNHUB_API_KEY doesn't exist in config.settings by default,
        # it will always be unconfigured
        finding = self.inspector.inspect(
            "data_source:finnhub", {"source": "finnhub"}
        )

        assert finding is not None
        assert finding.status == "WARN"
        assert finding.severity == "WARNING"
        assert "unconfigured" in str(finding.evidence).lower()

    # ── Scope Parsing Tests ────────────────────────────────────────────────

    def test_scope_parsing_with_prefix(self):
        """Test that scope like 'data_source:defillama' is correctly parsed."""
        with patch("data.defillama.client.DefiLlamaClient") as mock_client_class:
            mock_instance = MagicMock()
            mock_client_class.return_value = mock_instance
            mock_instance.fetch_protocol_metrics.return_value = {"lido": MagicMock()}

            # Scope with prefix should extract 'defillama'
            finding = self.inspector.inspect("data_source:defillama", {})

            assert finding is not None
            assert "defillama" in finding.scope.lower()

    def test_scope_parsing_from_context(self):
        """Test that source from context is used when scope doesn't have prefix."""
        with patch("data.defillama.client.DefiLlamaClient") as mock_client_class:
            mock_instance = MagicMock()
            mock_client_class.return_value = mock_instance
            mock_instance.fetch_protocol_metrics.return_value = {"lido": MagicMock()}

            # Source from context
            finding = self.inspector.inspect("some_scope", {"source": "defillama"})

            assert finding is not None

    def test_unknown_source_returns_none(self):
        """Test that inspecting an unknown source returns None."""
        finding = self.inspector.inspect(
            "data_source:unknown_source", {"source": "unknown_source"}
        )

        assert finding is None

    def test_malformed_scope_returns_none(self):
        """Test that malformed scope (no source specified) returns None."""
        finding = self.inspector.inspect("", {})

        assert finding is None

    # ── Finding Schema Tests ───────────────────────────────────────────────

    def test_finding_has_required_fields(self):
        """Test that a Finding from the inspector has all required fields."""
        with patch("data.defillama.client.DefiLlamaClient") as mock_client_class:
            mock_instance = MagicMock()
            mock_client_class.return_value = mock_instance
            mock_instance.fetch_protocol_metrics.return_value = {}

            finding = self.inspector.inspect(
                "data_source:defillama", {"source": "defillama"}
            )

            assert finding is not None
            assert finding.agent == "SourceHealthInspector"
            assert finding.level == "L0"
            assert finding.scope is not None
            assert finding.status in ["PASS", "WARN", "FAIL"]
            assert finding.severity in ["INFO", "WARNING", "BLOCKER"]

    def test_finding_has_evidence_dict(self):
        """Test that Finding evidence is a dict with status and details."""
        with patch("data.defillama.client.DefiLlamaClient") as mock_client_class:
            mock_instance = MagicMock()
            mock_client_class.return_value = mock_instance
            mock_instance.fetch_protocol_metrics.return_value = {"lido": MagicMock()}

            finding = self.inspector.inspect(
                "data_source:defillama", {"source": "defillama"}
            )

            assert finding is not None
            assert isinstance(finding.evidence, dict)
            assert "status" in finding.evidence

    def test_finding_recommendation_text(self):
        """Test that Finding has a non-empty local_recommendation."""
        with patch("data.defillama.client.DefiLlamaClient") as mock_client_class:
            mock_instance = MagicMock()
            mock_client_class.return_value = mock_instance
            mock_instance.fetch_protocol_metrics.return_value = {}

            finding = self.inspector.inspect(
                "data_source:defillama", {"source": "defillama"}
            )

            assert finding is not None
            assert finding.local_recommendation is not None
            assert len(finding.local_recommendation) > 0
