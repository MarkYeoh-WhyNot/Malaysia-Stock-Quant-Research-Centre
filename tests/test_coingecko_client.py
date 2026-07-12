"""CoinGecko client contract tests. No network — requests.get is mocked."""
from unittest.mock import MagicMock, patch

import pytest

from data.coingecko.client import CoinGeckoClient, TokenSupply


# ── Fixtures ──────────────────────────────────────────────────────────────────

_KNOWN_GOOD_RESPONSE = [
    {
        "id": "bitcoin",
        "symbol": "btc",
        "name": "Bitcoin",
        "circulating_supply": 21_000_000.0,
        "total_supply": 21_000_000.0,
        "max_supply": 21_000_000.0,
        "market_cap": 1_200_000_000_000.0,
        "fully_diluted_valuation": 1_200_000_000_000.0,
    },
    {
        "id": "ethereum",
        "symbol": "eth",
        "name": "Ethereum",
        "circulating_supply": 120_000_000.0,
        "total_supply": 120_000_000.0,
        "max_supply": None,
        "market_cap": 480_000_000_000.0,
        "fully_diluted_valuation": 480_000_000_000.0,
    },
    {
        "id": "binancecoin",
        "symbol": "bnb",
        "name": "BNB",
        "circulating_supply": 600_000_000.0,
        "total_supply": 600_000_000.0,
        "max_supply": 600_000_000.0,
        "market_cap": 120_000_000_000.0,
        "fully_diluted_valuation": 120_000_000_000.0,
    },
]


@pytest.fixture
def mock_response():
    """Mock a successful CoinGecko markets response."""
    resp = MagicMock()
    resp.status_code = 200
    resp.json.return_value = _KNOWN_GOOD_RESPONSE
    return resp


# ── fetch_token_supply tests ──────────────────────────────────────────────────


def test_fetch_parses_supply_data(mock_response):
    with patch("data.coingecko.client.requests.get", return_value=mock_response):
        result = CoinGeckoClient.fetch_token_supply(["BTC", "ETH", "BNB"])
    assert len(result) == 3
    assert "BTC" in result
    assert "ETH" in result
    assert "BNB" in result


def test_fetch_extracts_circulating_supply(mock_response):
    with patch("data.coingecko.client.requests.get", return_value=mock_response):
        result = CoinGeckoClient.fetch_token_supply(["BTC"])
    btc = result["BTC"]
    assert btc.circulating_supply == 21_000_000.0


def test_fetch_extracts_total_supply(mock_response):
    with patch("data.coingecko.client.requests.get", return_value=mock_response):
        result = CoinGeckoClient.fetch_token_supply(["BTC"])
    btc = result["BTC"]
    assert btc.total_supply == 21_000_000.0


def test_fetch_extracts_max_supply(mock_response):
    with patch("data.coingecko.client.requests.get", return_value=mock_response):
        result = CoinGeckoClient.fetch_token_supply(["BTC"])
    btc = result["BTC"]
    assert btc.max_supply == 21_000_000.0


def test_fetch_handles_null_max_supply(mock_response):
    with patch("data.coingecko.client.requests.get", return_value=mock_response):
        result = CoinGeckoClient.fetch_token_supply(["ETH"])
    eth = result["ETH"]
    # ETH has no max supply
    assert eth.max_supply is None


def test_fetch_extracts_market_cap(mock_response):
    with patch("data.coingecko.client.requests.get", return_value=mock_response):
        result = CoinGeckoClient.fetch_token_supply(["BTC"])
    btc = result["BTC"]
    assert btc.market_cap == 1_200_000_000_000.0


def test_fetch_extracts_fully_diluted_valuation(mock_response):
    with patch("data.coingecko.client.requests.get", return_value=mock_response):
        result = CoinGeckoClient.fetch_token_supply(["BTC"])
    btc = result["BTC"]
    assert btc.fully_diluted_valuation == 1_200_000_000_000.0


def test_fetch_handles_empty_symbol_list():
    result = CoinGeckoClient.fetch_token_supply([])
    assert result == {}


def test_fetch_symbol_to_id_mapping():
    """Test that symbols are correctly mapped to CoinGecko IDs."""
    # BTC -> bitcoin, ETH -> ethereum, etc.
    assert CoinGeckoClient._symbol_to_id("BTC") == "bitcoin"
    assert CoinGeckoClient._symbol_to_id("ETH") == "ethereum"
    assert CoinGeckoClient._symbol_to_id("BNB") == "binancecoin"


def test_fetch_symbol_to_id_fallback():
    """Unmapped symbols fall back to lowercase."""
    assert CoinGeckoClient._symbol_to_id("UNKNOWN") == "unknown"


def test_fetch_rate_limit_429_graceful(mock_response):
    """Rate limit 429 is handled gracefully — returns empty dict."""
    mock_response.status_code = 429
    with patch("data.coingecko.client.requests.get", return_value=mock_response):
        result = CoinGeckoClient.fetch_token_supply(["BTC"])
    assert result == {}


def test_fetch_network_timeout_graceful():
    """Network timeout is handled gracefully — returns empty dict."""
    import requests
    with patch("data.coingecko.client.requests.get",
               side_effect=requests.exceptions.Timeout):
        result = CoinGeckoClient.fetch_token_supply(["BTC"])
    assert result == {}


def test_fetch_connection_error_graceful():
    """Connection error (geo-block?) is handled gracefully."""
    import requests
    with patch("data.coingecko.client.requests.get",
               side_effect=requests.exceptions.ConnectionError):
        result = CoinGeckoClient.fetch_token_supply(["BTC"])
    assert result == {}


def test_fetch_unexpected_response_shape():
    """If response is not a list, gracefully skip."""
    resp = MagicMock()
    resp.status_code = 200
    resp.json.return_value = {"unexpected": "shape"}  # Not a list
    with patch("data.coingecko.client.requests.get", return_value=resp):
        result = CoinGeckoClient.fetch_token_supply(["BTC"])
    assert result == {}


def test_fetch_respects_timeout():
    """fetch_token_supply passes timeout=10 to requests.get."""
    resp = MagicMock()
    resp.status_code = 200
    resp.json.return_value = _KNOWN_GOOD_RESPONSE
    with patch("data.coingecko.client.requests.get", return_value=resp) as mock_get:
        CoinGeckoClient.fetch_token_supply(["BTC"])
    # Check that timeout was passed
    assert mock_get.call_args.kwargs["timeout"] == 10


# ── record_supply tests ───────────────────────────────────────────────────────


def test_record_supply_writes_to_db():
    """record_supply writes TokenSupply objects to the database."""
    from data.database import db_session

    supply_dict = {
        "BTC": TokenSupply(
            symbol="BTC",
            circulating_supply=21_000_000.0,
            total_supply=21_000_000.0,
            max_supply=21_000_000.0,
            fully_diluted_valuation=1_200_000_000_000.0,
            market_cap=1_200_000_000_000.0,
        ),
        "ETH": TokenSupply(
            symbol="ETH",
            circulating_supply=120_000_000.0,
            total_supply=120_000_000.0,
            max_supply=None,
            fully_diluted_valuation=480_000_000_000.0,
            market_cap=480_000_000_000.0,
        ),
    }

    CoinGeckoClient.record_supply(supply_dict)

    # Verify written to DB
    with db_session() as conn:
        cursor = conn.execute(
            "SELECT symbol, circulating_supply, total_supply, max_supply, "
            "market_cap, fully_diluted_valuation FROM token_supply "
            "WHERE symbol IN ('BTC', 'ETH') ORDER BY symbol"
        )
        rows = cursor.fetchall()

    assert len(rows) >= 2
    # Find BTC row
    btc_row = [r for r in rows if r["symbol"] == "BTC"]
    if btc_row:
        assert btc_row[0]["circulating_supply"] == 21_000_000.0
        assert btc_row[0]["market_cap"] == 1_200_000_000_000.0

    # Find ETH row
    eth_row = [r for r in rows if r["symbol"] == "ETH"]
    if eth_row:
        assert eth_row[0]["total_supply"] == 120_000_000.0
        assert eth_row[0]["max_supply"] is None


def test_record_supply_empty_dict():
    """record_supply with empty dict is idempotent."""
    # Should not raise
    CoinGeckoClient.record_supply({})


def test_token_supply_data_class():
    """TokenSupply is a simple data class."""
    supply = TokenSupply(
        symbol="BTC",
        circulating_supply=21_000_000.0,
        total_supply=21_000_000.0,
    )
    assert supply.symbol == "BTC"
    assert supply.circulating_supply == 21_000_000.0
    assert supply.max_supply is None


# ── Integration tests ─────────────────────────────────────────────────────────


def test_fetch_and_record_integration(mock_response):
    """fetch_and_record_supply fetches and records in one call."""
    from data.coingecko.client import fetch_and_record_supply

    with patch("data.coingecko.client.requests.get", return_value=mock_response):
        result = fetch_and_record_supply(["BTC", "ETH"])

    assert "BTC" in result
    assert "ETH" in result


def test_fetch_url_construction(mock_response):
    """Verify the URL and params sent to CoinGecko."""
    with patch("data.coingecko.client.requests.get", return_value=mock_response) as mock_get:
        CoinGeckoClient.fetch_token_supply(["BTC", "ETH"])

    # Check the URL
    call_args = mock_get.call_args
    assert call_args.args[0] == f"{CoinGeckoClient.BASE_URL}/coins/markets"

    # Check params
    params = call_args.kwargs["params"]
    assert "bitcoin" in params["ids"]
    assert "ethereum" in params["ids"]
    assert params["vs_currency"] == "usd"
