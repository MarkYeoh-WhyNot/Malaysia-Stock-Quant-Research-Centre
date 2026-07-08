"""Market-profile regression pins (dual-market Phase A).

The Bursa profile must reproduce the pre-extraction constants EXACTLY — these
pins make it impossible for the profile refactor (or future profile edits) to
silently change Bursa behavior. Crypto-profile tests load the module directly
via load_market_profile('crypto'); no env-reload gymnastics.
"""
import pytest

from config.markets import load_market_profile, VALID_MODES
from config import settings


# ── Default mode is Bursa, bit-identical to pre-extraction values ────────────

def test_default_mode_is_bursa():
    assert settings.MARKET_MODE == "bursa"
    assert settings.MARKET == "KLSE"
    assert settings.MARKET_CURRENCY == "MYR"
    assert settings.UNIVERSE_NAME == "FBMKLCI"
    assert settings.DATA_BACKEND == "yahoo"
    assert settings.BENCHMARK_SYMBOL == "^KLSE"


def test_bursa_pins_cost_model():
    assert settings.BURSA_COMMISSION_RATE == 0.0008
    assert settings.BURSA_STAMP_DUTY_RATE == 0.0010
    assert settings.BURSA_STAMP_DUTY_CAP_MYR == 1000.0
    assert settings.BURSA_CLEARING_RATE == 0.0003
    assert settings.BURSA_CLEARING_CAP_MYR == 1000.0
    assert settings.BURSA_BOARD_LOT == 100
    assert settings.BURSA_MIN_DAILY_VALUE_MYR == 500_000.0
    assert settings.TRADING_DAYS_PER_YEAR == 252
    assert settings.SETTLEMENT_CYCLE == "T+2"
    # The canonical cost pin: RM100k buy, blue chip = 80+30+100+50
    assert settings.bursa_trade_cost(100_000, "buy", "BLUE_CHIP") == pytest.approx(260.0)
    assert settings.bursa_slippage_tier(25_000_000) == "BLUE_CHIP"


def test_bursa_pins_universe_and_gates():
    assert len(settings.KLCI_STOCKS) == 29
    assert "1155.KL" in settings.DEFAULT_SYMBOLS
    assert settings.KLCI_BY_SYMBOL["1155.KL"]["name"] == "Maybank"
    # Gate defaults ARE the Bursa values — no overrides applied
    assert settings.GATE_CONFIG.stage3_max_drawdown == 0.25
    assert settings.GATE_CONFIG.max_single_name_pct == 0.15
    assert settings.GATE_CONFIG.dq_min_confidence == 80.0


def test_bursa_ticker_rules():
    assert settings.TICKER_REGEX.search("1155.KL")
    assert not settings.TICKER_REGEX.search("BTC/USDT")
    assert "short sell" in settings.BLOCKED_MODES
    assert "intraday" in settings.BLOCKED_MODES


def test_bursa_sizing_matches_board_lots():
    # 95% of RM100k at RM10 → 9500 shares (same as PortfolioExecutor pin)
    assert settings.size_units(100_000, 10.0, 0.95) == 9500
    assert settings.size_units(1_000, 100.0, 0.95) == 0


# ── Crypto profile (loaded directly, no env switching) ───────────────────────

@pytest.fixture(scope="module")
def crypto():
    return load_market_profile("crypto")


def test_crypto_identity(crypto):
    assert crypto.MARKET == "CRYPTO"
    assert crypto.MARKET_CURRENCY == "USDT"
    assert crypto.UNIVERSE_NAME == "CRYPTO_MAJORS"
    assert crypto.DATA_BACKEND == "binance"
    assert crypto.BENCHMARK_SYMBOL == "BTC/USDT"
    assert crypto.TRADING_DAYS_PER_YEAR == 365
    assert crypto.CALENDAR == "daily"
    assert crypto.SETTLEMENT_CYCLE == "T+0"


def test_crypto_universe(crypto):
    symbols = [s["symbol"] for s in crypto.UNIVERSE]
    assert "BTC/USDT" in symbols and "ETH/USDT" in symbols
    assert len(symbols) == len(set(symbols)) == 20
    assert all(s.endswith("/USDT") for s in symbols)


def test_crypto_cost_model(crypto):
    # $100k taker on a major: 0.10% fee + 0.03% slippage = $130; symmetric sides
    assert crypto.trade_cost(100_000, "buy", "BLUE_CHIP") == pytest.approx(130.0)
    assert crypto.trade_cost(100_000, "sell", "BLUE_CHIP") == pytest.approx(130.0)
    assert crypto.slippage_tier(200_000_000) == "BLUE_CHIP"
    assert crypto.slippage_tier(50_000_000) == "MID_CAP"
    assert crypto.slippage_tier(1_000_000) == "SMALL_CAP"


def test_crypto_fractional_sizing(crypto):
    # $100k NAV at a $100k BTC price must buy ~0.95 BTC, not 0
    units = crypto.size_units(100_000, 100_000.0, 0.95)
    assert units == pytest.approx(0.95, abs=0.0001)
    assert crypto.size_units(100_000, 0, 0.95) == 0.0


def test_crypto_ticker_rules(crypto):
    assert crypto.TICKER_REGEX.search("BTC/USDT")
    assert not crypto.TICKER_REGEX.search("1155.KL")
    # derivatives / leverage are hard-blocked in addition to shorts/intraday
    for mode in ("perpetual", "margin", "leverage", "short sell", "intraday"):
        assert mode in crypto.BLOCKED_MODES


def test_crypto_gate_overrides_shape(crypto):
    # only known GateConfig fields may be overridden (typo protection)
    from config.settings import GateConfig
    valid = set(GateConfig.__dataclass_fields__)
    assert set(crypto.GATE_OVERRIDES) <= valid
    assert crypto.GATE_OVERRIDES["stage3_max_drawdown"] == 0.35


# ── Loader ────────────────────────────────────────────────────────────────────

def test_loader_rejects_unknown_mode():
    with pytest.raises(ValueError):
        load_market_profile("forex")


def test_loader_modes_complete():
    assert set(VALID_MODES) == {"bursa", "crypto"}
    for mode in VALID_MODES:
        p = load_market_profile(mode)
        # every profile exposes the full shared surface
        for attr in ("MARKET", "UNIVERSE", "UNIVERSE_NAME", "TRADING_DAYS_PER_YEAR",
                     "CALENDAR", "SETTLEMENT_CYCLE", "SLIPPAGE_TIERS", "MIN_DAILY_VALUE",
                     "TICKER_REGEX", "TICKER_EXAMPLE", "DATA_BACKEND", "BENCHMARK_SYMBOL",
                     "BLOCKED_MODES", "UNAVAILABLE_DATA_KEYWORDS", "MARKET_BRIEF",
                     "RED_TEAM_ATTACKS", "CONCENTRATION_SECTOR", "ENABLED_JOBS",
                     "GATE_OVERRIDES", "trade_cost", "slippage_tier", "size_units",
                     "RESEARCH_ANGLES", "ANGLE_KEYWORDS", "RESEARCH_QUERY_PERSONA",
                     "ALPHA_SEED_SYSTEM", "DATA_SOURCES_EXAMPLE", "RELEVANCE_TARGET",
                     "RELEVANCE_SCALE"):
            assert hasattr(p, attr), f"{mode} profile missing {attr}"
