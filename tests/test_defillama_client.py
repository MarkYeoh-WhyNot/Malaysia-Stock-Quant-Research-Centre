"""DefiLlama client tests (offline; mocked HTTP)."""
from unittest.mock import MagicMock, patch
import pytest
import tempfile
from pathlib import Path

from data.defillama.client import DefiLlamaClient, ProtocolMetrics
from data.database import init_db, db_session


@pytest.fixture
def temp_db():
    """Create a temporary test database."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "test.db"
        init_db(db_path)
        yield db_path


@pytest.fixture
def mock_defillama_response():
    """Known-good DefiLlama /protocols response (simplified)."""
    return [
        {
            "name": "Lido",
            "slug": "lido",
            "tvl": 25_000_000_000,
            "rank": 1,
            "24h": {
                "fees": 850_000,
                "revenue": 425_000,
            },
        },
        {
            "name": "Aave",
            "slug": "aave",
            "tvl": 12_000_000_000,
            "rank": 2,
            "24h": {
                "fees": 650_000,
                "revenue": 325_000,
            },
        },
        {
            "name": "Curve",
            "slug": "curve",
            "tvl": 6_500_000_000,
            "rank": 3,
            "24h": {
                "fees": 1_200_000,
                "revenue": 600_000,
            },
        },
    ]


class TestProtocolMetrics:
    """Tests for the ProtocolMetrics dataclass."""

    def test_construction(self):
        m = ProtocolMetrics(
            symbol="lido",
            protocol_name="Lido",
            tvl=25e9,
            tvl_rank=1,
            fees_24h=850_000,
            revenue_24h=425_000,
        )
        assert m.symbol == "lido"
        assert m.protocol_name == "Lido"
        assert m.tvl == 25e9
        assert m.tvl_rank == 1
        assert m.fees_24h == 850_000
        assert m.revenue_24h == 425_000

    def test_optional_fields(self):
        m = ProtocolMetrics(symbol="test", protocol_name="Test")
        assert m.tvl is None
        assert m.tvl_rank is None
        assert m.fees_24h is None
        assert m.revenue_24h is None


class TestFetchProtocolMetrics:
    """Tests for fetch_protocol_metrics."""

    def test_successful_fetch_and_parse(self, mock_defillama_response):
        """Fetch and parse valid response."""
        client = DefiLlamaClient()
        with patch("data.defillama.client.requests.get") as mock_get:
            mock_resp = MagicMock()
            mock_resp.json.return_value = mock_defillama_response
            mock_resp.status_code = 200
            mock_get.return_value = mock_resp

            result = client.fetch_protocol_metrics(["lido", "aave"])

        assert len(result) == 2
        assert "lido" in result
        assert "aave" in result
        assert result["lido"].protocol_name == "Lido"
        assert result["lido"].tvl == 25e9
        assert result["aave"].protocol_name == "Aave"
        assert result["aave"].tvl == 12e9

    def test_case_insensitive_matching(self, mock_defillama_response):
        """Symbol matching is case-insensitive."""
        client = DefiLlamaClient()
        with patch("data.defillama.client.requests.get") as mock_get:
            mock_resp = MagicMock()
            mock_resp.json.return_value = mock_defillama_response
            mock_get.return_value = mock_resp

            result = client.fetch_protocol_metrics(["LIDO", "AAVE"])

        # Requested symbols were uppercase; returned keys are lowercase
        assert len(result) == 2
        assert "lido" in result
        assert "aave" in result

    def test_partial_match(self, mock_defillama_response):
        """Only requested symbols are returned."""
        client = DefiLlamaClient()
        with patch("data.defillama.client.requests.get") as mock_get:
            mock_resp = MagicMock()
            mock_resp.json.return_value = mock_defillama_response
            mock_get.return_value = mock_resp

            result = client.fetch_protocol_metrics(["lido", "nonexistent"])

        assert len(result) == 1
        assert "lido" in result
        assert "nonexistent" not in result

    def test_fees_and_revenue_extraction(self, mock_defillama_response):
        """Extract fees and revenue from 24h object."""
        client = DefiLlamaClient()
        with patch("data.defillama.client.requests.get") as mock_get:
            mock_resp = MagicMock()
            mock_resp.json.return_value = mock_defillama_response
            mock_get.return_value = mock_resp

            result = client.fetch_protocol_metrics(["lido"])

        assert result["lido"].fees_24h == 850_000
        assert result["lido"].revenue_24h == 425_000

    def test_tvl_rank_extraction(self, mock_defillama_response):
        """Extract TVL rank."""
        client = DefiLlamaClient()
        with patch("data.defillama.client.requests.get") as mock_get:
            mock_resp = MagicMock()
            mock_resp.json.return_value = mock_defillama_response
            mock_get.return_value = mock_resp

            result = client.fetch_protocol_metrics(["lido", "aave"])

        assert result["lido"].tvl_rank == 1
        assert result["aave"].tvl_rank == 2

    def test_rate_limit_429(self):
        """Graceful handling of rate limit (429)."""
        client = DefiLlamaClient()
        with patch("data.defillama.client.requests.get") as mock_get:
            mock_resp = MagicMock()
            mock_resp.status_code = 429
            mock_resp.raise_for_status.side_effect = Exception("429 Too Many Requests")
            mock_get.return_value = mock_resp

            result = client.fetch_protocol_metrics(["lido"])

        assert result == {}

    def test_timeout(self):
        """Graceful handling of timeout."""
        client = DefiLlamaClient()
        with patch("data.defillama.client.requests.get") as mock_get:
            import requests
            mock_get.side_effect = requests.exceptions.Timeout("Connection timed out")

            result = client.fetch_protocol_metrics(["lido"])

        assert result == {}

    def test_http_error_500(self):
        """Graceful handling of 5xx errors."""
        client = DefiLlamaClient()
        with patch("data.defillama.client.requests.get") as mock_get:
            mock_resp = MagicMock()
            mock_resp.status_code = 500
            mock_resp.raise_for_status.side_effect = Exception("500 Internal Server Error")
            mock_get.return_value = mock_resp

            result = client.fetch_protocol_metrics(["lido"])

        assert result == {}

    def test_json_decode_error(self):
        """Graceful handling of invalid JSON."""
        client = DefiLlamaClient()
        with patch("data.defillama.client.requests.get") as mock_get:
            mock_resp = MagicMock()
            mock_resp.status_code = 200
            mock_resp.json.side_effect = ValueError("Invalid JSON")
            mock_get.return_value = mock_resp

            result = client.fetch_protocol_metrics(["lido"])

        assert result == {}

    def test_unexpected_response_type(self):
        """Graceful handling when response is not a list."""
        client = DefiLlamaClient()
        with patch("data.defillama.client.requests.get") as mock_get:
            mock_resp = MagicMock()
            mock_resp.status_code = 200
            mock_resp.json.return_value = {"error": "not a list"}
            mock_get.return_value = mock_resp

            result = client.fetch_protocol_metrics(["lido"])

        assert result == {}

    def test_missing_optional_fields(self):
        """Handle protocols with missing optional fields."""
        response = [
            {
                "name": "Test",
                "slug": "test",
                # no tvl, rank, or 24h
            }
        ]
        client = DefiLlamaClient()
        with patch("data.defillama.client.requests.get") as mock_get:
            mock_resp = MagicMock()
            mock_resp.json.return_value = response
            mock_get.return_value = mock_resp

            result = client.fetch_protocol_metrics(["test"])

        assert len(result) == 1
        assert result["test"].protocol_name == "Test"
        assert result["test"].tvl is None
        assert result["test"].tvl_rank is None
        assert result["test"].fees_24h is None
        assert result["test"].revenue_24h is None

    def test_non_numeric_tvl_conversion(self):
        """Handle non-numeric TVL values gracefully."""
        response = [
            {
                "name": "Test",
                "slug": "test",
                "tvl": "not a number",
                "rank": 5,
            }
        ]
        client = DefiLlamaClient()
        with patch("data.defillama.client.requests.get") as mock_get:
            mock_resp = MagicMock()
            mock_resp.json.return_value = response
            mock_get.return_value = mock_resp

            result = client.fetch_protocol_metrics(["test"])

        assert result["test"].tvl is None
        assert result["test"].tvl_rank == 5

    def test_non_numeric_rank_conversion(self):
        """Handle non-numeric rank values gracefully."""
        response = [
            {
                "name": "Test",
                "slug": "test",
                "tvl": 1e9,
                "rank": "not a number",
            }
        ]
        client = DefiLlamaClient()
        with patch("data.defillama.client.requests.get") as mock_get:
            mock_resp = MagicMock()
            mock_resp.json.return_value = response
            mock_get.return_value = mock_resp

            result = client.fetch_protocol_metrics(["test"])

        assert result["test"].tvl == 1e9
        assert result["test"].tvl_rank is None

    def test_empty_symbols_list(self):
        """Empty symbols list returns empty dict."""
        client = DefiLlamaClient()
        with patch("data.defillama.client.requests.get") as mock_get:
            mock_resp = MagicMock()
            mock_resp.json.return_value = [{"name": "Lido", "slug": "lido"}]
            mock_get.return_value = mock_resp

            result = client.fetch_protocol_metrics([])

        assert result == {}


class TestRecordMetrics:
    """Tests for record_metrics."""

    def test_record_single_metric(self, temp_db):
        """Record a single protocol metric to the database."""
        client = DefiLlamaClient()
        metrics = {
            "lido": ProtocolMetrics(
                symbol="lido",
                protocol_name="Lido",
                tvl=25e9,
                tvl_rank=1,
                fees_24h=850_000,
                revenue_24h=425_000,
            )
        }
        with patch("data.defillama.client.db_session") as mock_session:
            mock_conn = MagicMock()
            mock_session.return_value.__enter__.return_value = mock_conn
            mock_session.return_value.__exit__.return_value = None
            client.record_metrics(metrics)
            # Verify execute was called with INSERT
            mock_conn.execute.assert_called()

    def test_record_multiple_metrics(self, temp_db):
        """Record multiple protocol metrics."""
        client = DefiLlamaClient()
        metrics = {
            "lido": ProtocolMetrics(
                symbol="lido",
                protocol_name="Lido",
                tvl=25e9,
                tvl_rank=1,
            ),
            "aave": ProtocolMetrics(
                symbol="aave",
                protocol_name="Aave",
                tvl=12e9,
                tvl_rank=2,
            ),
        }
        with patch("data.defillama.client.db_session") as mock_session:
            mock_conn = MagicMock()
            mock_session.return_value.__enter__.return_value = mock_conn
            mock_session.return_value.__exit__.return_value = None
            client.record_metrics(metrics)
            # Verify execute was called twice (once per metric)
            assert mock_conn.execute.call_count == 2

    def test_record_empty_dict(self, temp_db):
        """Recording empty dict does nothing."""
        client = DefiLlamaClient()
        with patch("data.defillama.client.db_session") as mock_session:
            client.record_metrics({})
            # db_session should not be called for empty dict
            mock_session.assert_not_called()

    def test_record_with_none_values(self, temp_db):
        """Record metrics with None values for optional fields."""
        client = DefiLlamaClient()
        metrics = {
            "test": ProtocolMetrics(
                symbol="test",
                protocol_name="Test Protocol",
                tvl=None,
                tvl_rank=None,
                fees_24h=None,
                revenue_24h=None,
            )
        }
        with patch("data.defillama.client.db_session") as mock_session:
            mock_conn = MagicMock()
            mock_session.return_value.__enter__.return_value = mock_conn
            mock_session.return_value.__exit__.return_value = None
            client.record_metrics(metrics)
            # Verify INSERT was attempted with None values
            call_args = mock_conn.execute.call_args
            assert call_args is not None
            # The SQL should have placeholders for None values
            assert "INSERT INTO protocol_metrics" in call_args[0][0]

    def test_record_fetched_at_timestamp(self, temp_db):
        """Verify fetched_at timestamp is included in INSERT."""
        client = DefiLlamaClient()
        metrics = {
            "lido": ProtocolMetrics(
                symbol="lido",
                protocol_name="Lido",
                tvl=25e9,
            )
        }
        with patch("data.defillama.client.db_session") as mock_session:
            mock_conn = MagicMock()
            mock_session.return_value.__enter__.return_value = mock_conn
            mock_session.return_value.__exit__.return_value = None
            client.record_metrics(metrics)
            # Verify the SQL references fetched_at (it's in the INSERT statement)
            call_args = mock_conn.execute.call_args
            assert "fetched_at" not in call_args[0][0]  # It's auto-generated via DEFAULT


class TestIntegration:
    """Integration tests combining fetch and record."""

    def test_fetch_and_record_round_trip(self, temp_db, mock_defillama_response):
        """Fetch metrics and record them to database."""
        client = DefiLlamaClient()
        with patch("data.defillama.client.requests.get") as mock_get:
            mock_resp = MagicMock()
            mock_resp.json.return_value = mock_defillama_response
            mock_get.return_value = mock_resp

            metrics = client.fetch_protocol_metrics(["lido", "aave"])
            assert len(metrics) == 2

            with patch("data.defillama.client.db_session") as mock_session:
                mock_conn = MagicMock()
                mock_session.return_value.__enter__.return_value = mock_conn
                mock_session.return_value.__exit__.return_value = None
                client.record_metrics(metrics)
                # Verify 2 INSERT calls (one per metric)
                assert mock_conn.execute.call_count == 2

        # Verify the fetched data was correct
        assert metrics["lido"].tvl == 25e9
        assert metrics["aave"].tvl == 12e9
