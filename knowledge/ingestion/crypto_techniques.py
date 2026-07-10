"""Crypto technique library — the Crypto Mode arsenal.

Purpose-authored for liquid crypto perp/spot markets, NOT a rename of the Bursa
equity set. Same schema as BURSA_TECHNIQUE_LIBRARY (technique_library.py) so the
TechniqueLibrary class, indices, API, and dashboard consume it unchanged:
  name / angle / when_to_use / when_to_avoid / market_applicability /
  ic_improvement_vs_sma / stock_types / strategy_types / holding_periods /
  signal_types / implemented / complexity / overfitting_risk

`stock_types` reuses the equity field name but carries crypto tiers:
  "major" (BTC/ETH), "large_cap" (top-20), "alt" (smaller), "all".
`signal_types` adds crypto-native inputs: "funding", "basis", "onchain",
"dominance", "open_interest" alongside "price"/"volume"/"macro".

`implemented=True` marks tools already in the backtest codebase (methodology
gates); crypto-native data signals (funding/basis/onchain) are False until the
perp data layer + DSL leaves land (Workstream 3).
"""
from __future__ import annotations

CRYPTO_TECHNIQUE_LIBRARY: dict[str, dict] = {

    # ── Perp structure / carry ────────────────────────────────────────────────
    "funding_rate_carry": {
        "name": "Perpetual Funding-Rate Carry",
        "angle": "event_driven",
        "when_to_use": [
            "Persistently positive funding — short perp / long spot to harvest funding",
            "Persistently negative funding — long perp to be paid funding",
            "Range-bound majors where directional edge is weak but carry is steady",
        ],
        "when_to_avoid": [
            "Strong trends — funding carry is dwarfed by directional PnL and gap risk",
            "Thin alts where funding is noisy and liquidation risk is high",
            "Without a spot/short hedge — naked carry is just a directional bet",
        ],
        "market_applicability": "Very high — funding is a structural, recurring crypto-native edge absent in equities",
        "ic_improvement_vs_sma": "Carry Sharpe 1.0-1.8 on majors in calm regimes; decays hard in trends",
        "stock_types": ["major", "large_cap"],
        "strategy_types": ["carry", "market_neutral", "event"],
        "holding_periods": ["short_term", "medium_term"],
        "signal_types": ["funding", "basis", "price"],
        # Historical funding is now a first-class backtest input:
        # funding_level / funding_zscore DSL leaves + real per-bar funding
        # drag in the engine (funding-history integration, 2026-07-10).
        "implemented": True,
        "complexity": "medium",
        "overfitting_risk": "low",
    },

    "perp_basis_arb": {
        "name": "Perp / Spot Basis (Cash-and-Carry)",
        "angle": "statistical_modelling",
        "when_to_use": [
            "Perp trading at a large premium/discount to spot (basis dislocation)",
            "Delta-neutral: long the cheap leg, short the rich leg, collect convergence",
            "Volatility spikes that blow perp basis wide temporarily",
        ],
        "when_to_avoid": [
            "Basis within normal noise band — costs eat the edge",
            "Exchanges/pairs with poor spot liquidity for the hedge leg",
        ],
        "market_applicability": "High — a classic market-neutral crypto edge; capacity-limited but robust",
        "ic_improvement_vs_sma": "Near-market-neutral; Sharpe driven by convergence frequency",
        "stock_types": ["major", "large_cap"],
        "strategy_types": ["market_neutral", "carry"],
        "holding_periods": ["short_term", "medium_term"],
        "signal_types": ["basis", "price"],
        "implemented": False,
        "complexity": "high",
        "overfitting_risk": "low",
    },

    # ── Cross-sectional ───────────────────────────────────────────────────────
    "xs_momentum_majors": {
        "name": "Cross-Sectional Momentum (Top Liquid Pairs)",
        "angle": "price_action",
        "when_to_use": [
            "Rank the liquid universe by trailing return; long top / short bottom",
            "Trending regimes where winners keep winning across the alt complex",
            "Weekly-to-monthly rebalance to control turnover/funding drag",
        ],
        "when_to_avoid": [
            "Sharp mean-reverting chop — momentum whipsaws",
            "Only 2-3 liquid names — cross-section too thin to be alpha",
        ],
        "market_applicability": "High — cross-sectional momentum is well-documented in crypto; strong but crash-prone",
        "ic_improvement_vs_sma": "XS momentum IC 0.03-0.07 on a 20-30 pair universe",
        "stock_types": ["all"],
        "strategy_types": ["momentum", "cross_sectional"],
        "holding_periods": ["medium_term"],
        "signal_types": ["price"],
        "implemented": True,
        "complexity": "medium",
        "overfitting_risk": "medium",
    },

    "xs_reversal_short_term": {
        "name": "Short-Horizon Cross-Sectional Reversal",
        "angle": "behavioural",
        "when_to_use": [
            "1-3 day overreaction: long biggest losers / short biggest gainers",
            "Retail-driven pumps that overshoot then revert",
            "High-volatility, high-retail alt segments",
        ],
        "when_to_avoid": [
            "Genuine regime breaks / news-driven repricing (not noise)",
            "Illiquid alts where the 'reversal' is just the spread",
        ],
        "market_applicability": "High — crypto retail overreaction is strong and frequent",
        "ic_improvement_vs_sma": "Short-horizon reversal IC 0.04-0.08 gross; fragile to costs",
        "stock_types": ["large_cap", "alt"],
        "strategy_types": ["mean_reversion", "cross_sectional"],
        "holding_periods": ["short_term"],
        "signal_types": ["price", "volume"],
        "implemented": True,
        "complexity": "medium",
        "overfitting_risk": "high",
    },

    "btc_beta_neutralization": {
        "name": "BTC-Beta Neutralization",
        "angle": "statistical_modelling",
        "when_to_use": [
            "Isolate alt-specific alpha by hedging out BTC beta",
            "Any alt signal where raw returns are mostly levered BTC exposure",
            "Building market-neutral baskets that survive a BTC drawdown",
        ],
        "when_to_avoid": [
            "The thesis IS a BTC directional call — neutralizing removes the edge",
            "Unstable rolling beta on illiquid alts (estimation noise)",
        ],
        "market_applicability": "Critical — most alt 'alpha' is BTC beta in disguise; neutralize before trusting it",
        "ic_improvement_vs_sma": "Turns spurious alt signals honest; not an alpha source alone",
        "stock_types": ["alt", "large_cap"],
        "strategy_types": ["market_neutral", "cross_sectional"],
        "holding_periods": ["all"],
        "signal_types": ["price", "dominance"],
        "implemented": False,
        "complexity": "medium",
        "overfitting_risk": "low",
    },

    "pairs_cointegration": {
        "name": "Cointegration Pairs / Stat-Arb",
        "angle": "statistical_modelling",
        "when_to_use": [
            "Two economically-linked tokens (e.g. L2s, same-sector) that co-move",
            "Mean-revert the spread when it deviates > N sigma with an ADF-stationary pair",
            "Long/short the spread, dollar-neutral",
        ],
        "when_to_avoid": [
            "Spurious cointegration that breaks out-of-sample (very common in crypto)",
            "Forked/insider-driven pairs where regime shifts abruptly",
        ],
        "market_applicability": "Medium — real pairs exist but relationships break faster than in equities; strict OOS/embargo required",
        "ic_improvement_vs_sma": "Pair Sharpe 0.8-1.5 when the relationship holds; regime-fragile",
        "stock_types": ["large_cap", "alt"],
        "strategy_types": ["market_neutral", "mean_reversion"],
        "holding_periods": ["short_term", "medium_term"],
        "signal_types": ["price"],
        "implemented": False,
        "complexity": "high",
        "overfitting_risk": "high",
    },

    # ── Regime / volatility ───────────────────────────────────────────────────
    "garch_vol_regime": {
        "name": "GARCH Volatility Regime / Targeting",
        "angle": "statistical_modelling",
        "when_to_use": [
            "Scale position size inversely to forecast volatility (vol targeting)",
            "Detect vol-expansion regimes to de-risk before crypto crashes",
            "Overlay for momentum/carry to stabilize Sharpe",
        ],
        "when_to_avoid": [
            "As a standalone alpha — it is a risk overlay, not a signal",
            "Ultra-short horizons where the forecast is stale by fill",
        ],
        "market_applicability": "High — crypto vol clusters violently; vol targeting materially improves risk-adjusted returns",
        "ic_improvement_vs_sma": "Vol targeting typically +20-40% Sharpe vs unscaled",
        "stock_types": ["all"],
        "strategy_types": ["risk_overlay", "momentum"],
        "holding_periods": ["medium_term"],
        "signal_types": ["price"],
        "implemented": False,
        "complexity": "medium",
        "overfitting_risk": "low",
    },

    "hmm_regime": {
        "name": "Hidden Markov Regime Detector",
        "angle": "statistical_modelling",
        "when_to_use": [
            "Bull / bear / chop regime classification on BTC to gate the whole book",
            "Switching between momentum (trend) and reversal (chop) by regime",
            "Risk-on/risk-off detection combining price + funding + dominance",
        ],
        "when_to_avoid": [
            "Single-alt strategies — needs a market-level signal",
            "Very short holds — regime switches are slow relative to noise",
        ],
        "market_applicability": "High — crypto has sharp, persistent regimes; conditioning on them lifts Sharpe",
        "ic_improvement_vs_sma": "Regime-conditional strategies show 20-35% better Sharpe vs unconditional",
        "stock_types": ["all"],
        "strategy_types": ["momentum", "mean_reversion", "risk_overlay"],
        "holding_periods": ["medium_term"],
        "signal_types": ["price", "funding", "dominance"],
        "implemented": False,
        "complexity": "high",
        "overfitting_risk": "medium",
    },

    # ── Flow / on-chain (data-gated) ──────────────────────────────────────────
    "exchange_flow_signal": {
        "name": "Exchange In/Out-Flow Signal (on-chain)",
        "angle": "institutional",
        "when_to_use": [
            "Large exchange OUTflows (coins to cold storage) as accumulation signal",
            "Large exchange INflows as distribution/sell-pressure warning",
            "Stablecoin exchange inflows as dry-powder / risk-on proxy",
        ],
        "when_to_avoid": [
            "Without reliable labeled-wallet data — raw flows are very noisy",
            "Short horizons — flow signals play out over days/weeks",
        ],
        "market_applicability": "High potential, DATA-GATED — needs an on-chain feed (Glassnode/CryptoQuant); deferred until wired",
        "ic_improvement_vs_sma": "Reported edge in literature; unverified on this system's data",
        "stock_types": ["major", "large_cap"],
        "strategy_types": ["event", "institutional"],
        "holding_periods": ["medium_term", "long_term"],
        "signal_types": ["onchain", "volume"],
        "implemented": False,
        "complexity": "high",
        "overfitting_risk": "medium",
    },

    "oi_liquidation_cascade": {
        "name": "Open-Interest / Liquidation-Cascade Signal",
        "angle": "behavioural",
        "when_to_use": [
            "OI surging with price into overcrowded leverage — fade the crowd",
            "Post-liquidation-cascade snapbacks (forced sellers exhausted)",
            "Funding + OI extremes as contrarian timing",
        ],
        "when_to_avoid": [
            "Trend-continuation regimes where crowded ≠ wrong",
            "Without OI/liquidation data — pure price proxy is weak",
        ],
        "market_applicability": "Medium-high — leverage/liquidation dynamics are a genuine crypto-native edge; DATA-GATED on OI feed",
        "ic_improvement_vs_sma": "Event-conditional; strong around cascades, quiet otherwise",
        "stock_types": ["major", "large_cap"],
        "strategy_types": ["mean_reversion", "event"],
        "holding_periods": ["short_term"],
        "signal_types": ["open_interest", "funding", "price"],
        "implemented": False,
        "complexity": "high",
        "overfitting_risk": "high",
    },

    # ── Methodology gates (market-agnostic, already implemented) ───────────────
    "cross_sectional_ic": {
        "name": "Cross-Sectional IC Validation",
        "angle": "statistical_modelling",
        "when_to_use": [
            "Validating any signal across the liquid universe (not one lucky pair)",
            "Required before Stage 3 promotion — mean IC, IC t-stat, breadth",
            "Comparing candidate signals on an apples-to-apples basis",
        ],
        "when_to_avoid": [
            "Universes too small (<10 liquid pairs) for a meaningful cross-section",
        ],
        "market_applicability": "Critical — the core anti-single-name-luck gate; a signal that works on one pair is luck, not alpha",
        "ic_improvement_vs_sma": "Not a signal — a validation standard (IC>0.05, t-stat>1.5)",
        "stock_types": ["all"],
        "strategy_types": ["all"],
        "holding_periods": ["all"],
        "signal_types": ["all"],
        "implemented": True,
        "complexity": "medium",
        "overfitting_risk": "low",
    },

    "deflated_sharpe": {
        "name": "Deflated Sharpe / Multiple-Testing Control",
        "angle": "statistical_modelling",
        "when_to_use": [
            "Discounting backtest Sharpe for the number of trials run",
            "Guarding against data-mined 'edges' in a fast, noisy market",
            "Final gate before paper-trade promotion",
        ],
        "when_to_avoid": [
            "As a substitute for OOS/walk-forward — it complements, not replaces",
        ],
        "market_applicability": "Critical — crypto's low signal-to-noise makes multiple-testing control essential",
        "ic_improvement_vs_sma": "Not a signal — a hurdle that kills most lucky backtests",
        "stock_types": ["all"],
        "strategy_types": ["all"],
        "holding_periods": ["all"],
        "signal_types": ["all"],
        "implemented": True,
        "complexity": "medium",
        "overfitting_risk": "low",
    },
}


# ── Arsenal v2 fields (signature-DB slim adoption, 2026-07-11) ───────────────
# Same contract as _BURSA_ARSENAL_V2 in technique_library.py (see the comment
# there); validated against the live signal_dsl.LEAVES / factors.FACTORS
# registries by tests/test_arsenal_v2.py. Local _rep helper — importing it from
# technique_library would be circular (that module imports this one).

def _rep(representable, rep_type=None, leaves=(), factor=None, missing=()):
    return {"is_representable": representable, "representation_type": rep_type,
            "required_leaves": list(leaves), "required_factor": factor,
            "missing_leaves": list(missing)}


_CRYPTO_ARSENAL_V2: dict[str, dict] = {
    "funding_rate_carry": {
        "description": "Collects the periodic funding payment on perpetual "
                       "futures by holding the side that gets paid, when "
                       "funding is persistently one-sided.",
        "family_id": "carry_funding",
        "strategy_shape": "cross_sectional_factor",
        # Single-name funding conditions also exist as DSL leaves
        # (funding_level / funding_zscore); the canonical carry expression is
        # the cross-sectional basket.
        "representability": _rep(True, "cross_sectional_factor",
                                 leaves=["funding_level", "funding_zscore"],
                                 factor="funding_avg"),
        "example": {"factor_spec": {
            "factor": {"name": "funding_avg", "params": {"period": 21}},
            "top_n": 3, "bottom_n": 3, "rebalance_bars": 24}},
    },
    "perp_basis_arb": {
        "description": "Trades the price gap between a perpetual future and its "
                       "spot price, converging as the two prices re-align.",
        "family_id": "carry_funding",
        "strategy_shape": "unimplemented_concept",
        "representability": _rep(False, missing=["spot_perp_basis"]),
        "example": {"none": "no leaf or factor carries the perp-vs-spot basis; "
                            "the hedged two-leg structure is also outside the "
                            "single-instrument engine"},
    },
    "xs_momentum_majors": {
        "description": "Ranks the liquid pair universe by trailing return and "
                       "goes long winners / short losers.",
        "family_id": "cross_sectional_ranking",
        "strategy_shape": "cross_sectional_factor",
        "representability": _rep(True, "cross_sectional_factor",
                                 factor="momentum"),
        "example": {"factor_spec": {
            "factor": {"name": "momentum", "params": {"period": 30}},
            "top_n": 4, "bottom_n": 4, "rebalance_bars": 21}},
    },
    "xs_reversal_short_term": {
        "description": "Bets that the biggest 1-3 day movers in the universe "
                       "will revert, fading extreme short-term winners and "
                       "losers.",
        "family_id": "cross_sectional_ranking",
        "strategy_shape": "cross_sectional_factor",
        "representability": _rep(True, "cross_sectional_factor",
                                 factor="reversal"),
        "example": {"factor_spec": {
            "factor": {"name": "reversal", "params": {"period": 3}},
            "top_n": 4, "bottom_n": 4, "rebalance_bars": 3}},
    },
    "btc_beta_neutralization": {
        "description": "Hedges out an altcoin's BTC exposure so only its "
                       "BTC-independent (idiosyncratic) return remains.",
        "family_id": "stat_arb",
        "strategy_shape": "unimplemented_concept",
        "representability": _rep(False, missing=["btc_beta_residual"]),
        "example": {"none": "no factor computes rolling-beta residuals vs BTC; "
                            "raw signals cannot be beta-neutralized in the "
                            "current engine"},
    },
    "pairs_cointegration": {
        "description": "Trades the spread between two historically co-moving "
                       "tokens, betting on reversion when it diverges.",
        "family_id": "stat_arb",
        "strategy_shape": "unimplemented_concept",
        "representability": _rep(False, missing=["pair_spread_zscore"]),
        "example": {"none": "multi-leg spread trades are outside the "
                            "single-instrument DSL and basket engine"},
    },
    "garch_vol_regime": {
        "description": "Uses a GARCH volatility forecast to scale position "
                       "size, sizing down ahead of expected turbulence.",
        "family_id": "volatility_modeling",
        "strategy_shape": "unimplemented_concept",
        "representability": _rep(False, missing=["vol_forecast"]),
        "example": {"none": "a sizing/risk overlay, not a boolean entry/exit "
                            "condition; no vol-forecast leaf exists"},
    },
    "hmm_regime": {
        "description": "Classifies the market into bull/bear/chop regimes and "
                       "conditions strategy choice on the detected state.",
        "family_id": "regime_detection",
        "strategy_shape": "unimplemented_concept",
        "representability": _rep(False, missing=["regime_state"]),
        "example": {"none": "no leaf exposes a fitted latent regime state or "
                            "regime probability"},
    },
    "exchange_flow_signal": {
        "description": "Reads large token movements on/off exchanges as an "
                       "accumulation or distribution signal.",
        "family_id": "flow_institutional",
        "strategy_shape": "unimplemented_concept",
        "representability": _rep(False, missing=["exchange_netflow"]),
        "example": {"none": "data-gated: no on-chain exchange-flow feed is "
                            "wired, so no leaf can carry it"},
    },
    "oi_liquidation_cascade": {
        "description": "Fades crowded leveraged positioning, trading the "
                       "snapback after a forced-liquidation cascade.",
        "family_id": "positioning_leverage",
        "strategy_shape": "unimplemented_concept",
        # funding_level/funding_zscore capture only the funding fragment of the
        # thesis — OI/liquidation data itself is unavailable (Binance caps OI
        # history), so an honest example cannot exist yet.
        "representability": _rep(False, missing=["open_interest",
                                                 "liquidation_volume"]),
        "example": {"none": "open-interest and liquidation data are not wired "
                            "(no historical OI feed); the funding leaves alone "
                            "do not express the cascade thesis"},
    },
    "cross_sectional_ic": {
        "description": "Measures how well a signal's ranking predicts future "
                       "returns across the liquid pair universe — the core "
                       "validation gate.",
        "family_id": "validation_methodology",
        "strategy_shape": "methodology",
        "representability": _rep(False),
        "example": {"none": "a validation gate applied to other strategies, "
                            "not a tradable signal"},
    },
    "deflated_sharpe": {
        "description": "Discounts a backtested Sharpe ratio for the number of "
                       "trials run, guarding against data-mined 'edges'.",
        "family_id": "validation_methodology",
        "strategy_shape": "methodology",
        "representability": _rep(False),
        "example": {"none": "a multiple-testing hurdle applied to backtest "
                            "results, not a tradable signal"},
    },
}

for _key, _v2 in _CRYPTO_ARSENAL_V2.items():
    CRYPTO_TECHNIQUE_LIBRARY[_key].update(_v2)  # KeyError = typo'd overlay key
