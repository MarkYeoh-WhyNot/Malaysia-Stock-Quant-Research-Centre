"""
TechniqueLibrary — Quantitative Technique Selector for OpenClaw.

Provides structured metadata about quantitative techniques and a
get_relevant_techniques() method that builds prompt-injection strings
for generate_ideas() and research_idea() in strategy_researcher.py.

Each technique entry describes:
  - when_to_use / when_to_avoid (decision rules)
  - bursa_applicability          (KLSE-specific guidance)
  - ic_improvement_*             (quantitative benchmarks where known)
  - implemented                  (True = already in the backtest codebase)
  - complexity / overfitting_risk
"""

from __future__ import annotations
from typing import Optional

# ── Technique definitions ─────────────────────────────────────────────────────

TECHNIQUE_LIBRARY: dict[str, dict] = {

    # ── Statistical / Signal Processing ──────────────────────────────────────

    "kalman_filter": {
        "name":    "Kalman Filter Signal Smoother",
        "angle":   "statistical_modelling",
        "when_to_use": [
            "Mean reversion strategies on noisy price series",
            "Mid-cap stocks with irregular trading volume",
            "Any signal where noise-to-signal ratio is high",
            "Factor smoothing where SMA lags too badly",
        ],
        "when_to_avoid": [
            "Strong trending markets — filter lags momentum",
            "Very liquid blue-chips where price is informationally efficient",
            "Holding periods < 3 days — filter adds latency",
        ],
        "bursa_applicability":    "High — Bursa mid-caps have noisy price series due to low liquidity",
        "ic_improvement_vs_sma":  "15–25% on ASEAN mid-cap mean reversion strategies",
        "stock_types":            ["mid_cap", "small_cap"],
        "strategy_types":         ["mean_reversion", "momentum"],
        "holding_periods":        ["medium_term", "long_term"],
        "signal_types":           ["price", "volume"],
        "implemented":            False,
        "complexity":             "medium",
        "overfitting_risk":       "low",
    },

    "hidden_markov_model": {
        "name":    "Hidden Markov Regime Detector",
        "angle":   "statistical_modelling",
        "when_to_use": [
            "Detecting bull / bear / sideways market regimes",
            "Switching between momentum and mean reversion based on regime",
            "OPR cycle regime detection for banking sector rotation",
            "EPF flow regime detection (accumulation vs distribution)",
        ],
        "when_to_avoid": [
            "Single-stock strategies — needs portfolio-level or index signal",
            "Holding periods < 5 days — regime switches are slow",
            "Very illiquid stocks where price is discontinuous",
        ],
        "bursa_applicability":    "High — Bursa has distinct bull/bear regimes driven by EPF flows and OPR cycles",
        "ic_improvement_vs_sma":  "Regime-conditional strategies show 20–35% better Sharpe vs unconditional",
        "stock_types":            ["all"],
        "strategy_types":         ["momentum", "sector_rotation", "mean_reversion"],
        "holding_periods":        ["medium_term", "long_term"],
        "signal_types":           ["price", "macro"],
        "implemented":            False,
        "complexity":             "high",
        "overfitting_risk":       "medium",
    },

    "garch": {
        "name":    "GARCH Volatility Model",
        "angle":   "statistical_modelling",
        "when_to_use": [
            "Volatility-based position sizing (scale down in high-vol regimes)",
            "Detecting volatility clustering before earnings announcements",
            "Risk management overlay for existing price strategies",
            "Building volatility-adjusted entry thresholds",
        ],
        "when_to_avoid": [
            "Strategies that don't adjust position size — GARCH adds complexity for no benefit",
            "Very short holding periods < 2 days",
        ],
        "bursa_applicability":    "Medium — useful for earnings season volatility on KLCI stocks",
        "stock_types":            ["blue_chip", "mid_cap"],
        "strategy_types":         ["mean_reversion", "momentum", "event_driven"],
        "holding_periods":        ["short_term", "medium_term"],
        "signal_types":           ["price", "volatility"],
        "implemented":            False,
        "complexity":             "medium",
        "overfitting_risk":       "low",
    },

    "information_coefficient": {
        "name":    "Information Coefficient (IC) Cross-Sectional Validator",
        "angle":   "statistical_modelling",
        "when_to_use": [
            "Validating any factor's predictive power across KLCI 30 stocks",
            "Ranking competing factor definitions for the same strategy",
            "Measuring how fast a factor's predictive power decays",
            "Required gate for all Stage 3 cross-sectional validation",
        ],
        "when_to_avoid": [
            "Single-stock time series strategies — IC is a cross-sectional metric",
        ],
        "bursa_applicability":    "Critical — required for all cross-sectional validation at Stage 3",
        "stock_types":            ["all"],
        "strategy_types":         ["all"],
        "holding_periods":        ["all"],
        "signal_types":           ["all"],
        "implemented":            True,
        "complexity":             "low",
        "overfitting_risk":       "low",
    },

    # ── Technical / Price Signals ─────────────────────────────────────────────

    "sma_crossover": {
        "name":    "Simple Moving Average Crossover",
        "angle":   "price_action",
        "when_to_use": [
            "Trend following on liquid large-cap KLCI stocks",
            "Low-noise price environments (high average daily volume)",
            "Holding periods > 20 days where lag is acceptable",
        ],
        "when_to_avoid": [
            "Noisy mid/small-cap stocks — use Kalman filter instead",
            "Holding periods < 10 days — too many false signals",
            "Sideways or range-bound markets — whipsaws destroy returns",
        ],
        "bursa_applicability":    "Medium — works on KLCI blue-chips, poor on small caps",
        "stock_types":            ["blue_chip"],
        "strategy_types":         ["momentum", "trend_following"],
        "holding_periods":        ["medium_term", "long_term"],
        "signal_types":           ["price"],
        "implemented":            True,
        "complexity":             "low",
        "overfitting_risk":       "low",
    },

    "rsi_mean_reversion": {
        "name":    "RSI Mean Reversion",
        "angle":   "price_action",
        "when_to_use": [
            "Range-bound or oscillating markets",
            "Stocks with documented mean-reverting behaviour",
            "Post-earnings overreaction plays (RSI < 25 after bad quarter)",
            "Retail-driven stocks where sentiment overshoots",
        ],
        "when_to_avoid": [
            "Strong trending stocks — RSI oversold in a downtrend is a trap",
            "GLC stocks with thin retail participation",
            "Very illiquid stocks where RSI is noise",
        ],
        "bursa_applicability":    "High — retail-dominated Bursa creates persistent overreaction patterns",
        "stock_types":            ["blue_chip", "mid_cap"],
        "strategy_types":         ["mean_reversion", "event_driven"],
        "holding_periods":        ["short_term", "medium_term"],
        "signal_types":           ["price"],
        "implemented":            True,
        "complexity":             "low",
        "overfitting_risk":       "medium",
    },

    "bollinger_squeeze": {
        "name":    "Bollinger Band Squeeze Breakout",
        "angle":   "price_action",
        "when_to_use": [
            "Pre-announcement volatility compression plays",
            "Stocks approaching quarterly earnings releases",
            "Post-consolidation breakout confirmation",
        ],
        "when_to_avoid": [
            "Trending markets where bands never squeeze",
            "Very low-volume stocks — false breakouts are common",
        ],
        "bursa_applicability":    "Medium — works well around Bursa quarterly reporting seasons",
        "stock_types":            ["blue_chip", "mid_cap"],
        "strategy_types":         ["momentum", "event_driven"],
        "holding_periods":        ["short_term", "medium_term"],
        "signal_types":           ["price", "volatility"],
        "implemented":            True,
        "complexity":             "low",
        "overfitting_risk":       "medium",
    },

    # ── Fundamental / Event-Driven ────────────────────────────────────────────

    "event_study": {
        "name":    "Event Study / Abnormal Return Analysis",
        "angle":   "event_driven",
        "when_to_use": [
            "Post-earnings announcement drift (PEAD) on quarterly beats",
            "Dividend capture — abnormal return window around ex-date",
            "Index addition / deletion flow-driven price impact",
            "BNM OPR decision impact on banking stocks NIM",
        ],
        "when_to_avoid": [
            "Low-liquidity stocks where spreads exceed abnormal return",
            "Events with no clean announcement date (rolling fundamental signals)",
        ],
        "bursa_applicability":    "Very high — event-driven alpha is structurally under-exploited on Bursa",
        "ic_improvement_vs_sma":  "Event windows show 2–4× better IC than rolling price signals on KLCI",
        "stock_types":            ["all"],
        "strategy_types":         ["event_driven"],
        "holding_periods":        ["short_term", "medium_term"],
        "signal_types":           ["event", "fundamental"],
        "implemented":            False,
        "complexity":             "medium",
        "overfitting_risk":       "low",
    },

    "pead": {
        "name":    "Post-Earnings Announcement Drift (PEAD)",
        "angle":   "event_driven",
        "when_to_use": [
            "Quarterly earnings beats on KLCI growth stocks",
            "Stocks with analyst coverage gap (no instant repricing)",
            "Feb / May / Aug / Nov earnings seasons on Bursa",
        ],
        "when_to_avoid": [
            "GLCs and mature dividend payers — earnings rarely surprise",
            "Very large-cap stocks — institutions reprice instantly",
        ],
        "bursa_applicability":    "High — Bursa retail-dominated mid-caps show multi-week PEAD",
        "ic_improvement_vs_sma":  "3–5% abnormal return over 20 days post-positive-surprise on KLCI",
        "stock_types":            ["mid_cap", "growth"],
        "strategy_types":         ["event_driven"],
        "holding_periods":        ["short_term", "medium_term"],
        "signal_types":           ["fundamental", "event"],
        "implemented":            False,
        "complexity":             "medium",
        "overfitting_risk":       "low",
    },

    # ── Factor / Portfolio ────────────────────────────────────────────────────

    "pca_factor_model": {
        "name":    "Principal Component Analysis (PCA) Factor Model",
        "angle":   "statistical_modelling",
        "when_to_use": [
            "Decomposing KLCI returns into systematic factors beyond market beta",
            "Building a local Fama-French style model for Bursa",
            "Identifying hidden sector correlations (EPF, plantation, banking clusters)",
        ],
        "when_to_avoid": [
            "Strategies with < 15 stocks in the universe",
            "Short holding periods — PCA factors are slow-moving",
        ],
        "bursa_applicability":    "High — EPF and foreign flows create systematic factors not captured by global models",
        "stock_types":            ["all"],
        "strategy_types":         ["sector_rotation", "momentum"],
        "holding_periods":        ["medium_term", "long_term"],
        "signal_types":           ["price", "fundamental"],
        "implemented":            False,
        "complexity":             "high",
        "overfitting_risk":       "medium",
    },

    "fama_french_3factor": {
        "name":    "Fama-French 3-Factor Model (Localised for Bursa)",
        "angle":   "fundamental",
        "when_to_use": [
            "Value strategies (HML factor — high book-to-market vs low)",
            "Size premium strategies (SMB factor — small vs big)",
            "Benchmarking alpha after stripping out market, size, value loadings",
        ],
        "when_to_avoid": [
            "KLCI blue-chip-only universe — SMB factor has no variation",
            "Short-term technical strategies — fundamental factors are slow",
        ],
        "bursa_applicability":    "Medium — size and value premia exist on Bursa but are weaker than in US data",
        "stock_types":            ["all"],
        "strategy_types":         ["value", "fundamental"],
        "holding_periods":        ["medium_term", "long_term"],
        "signal_types":           ["fundamental"],
        "implemented":            False,
        "complexity":             "medium",
        "overfitting_risk":       "low",
    },

    "epf_flow_tracker": {
        "name":    "EPF / Institutional Flow Signal",
        "angle":   "institutional",
        "when_to_use": [
            "GLC stocks where EPF is top-3 shareholder",
            "Detecting pre-rebalancing accumulation in index heavyweights",
            "Dividend season — EPF known to reinvest dividends in same stocks",
        ],
        "when_to_avoid": [
            "Small caps not held by EPF",
            "Strategies requiring daily rebalance — EPF data lags",
        ],
        "bursa_applicability":    "Very high — EPF controls ~15% of Bursa market cap; predictable rebalancing creates alpha",
        "stock_types":            ["blue_chip", "GLC"],
        "strategy_types":         ["institutional", "event_driven"],
        "holding_periods":        ["medium_term", "long_term"],
        "signal_types":           ["fundamental", "institutional"],
        "implemented":            False,
        "complexity":             "medium",
        "overfitting_risk":       "low",
    },

    "cpo_correlation": {
        "name":    "Crude Palm Oil (CPO) Price Correlation Signal",
        "angle":   "commodity",
        "when_to_use": [
            "Plantation stocks: IOI (1961.KL), KLK (2445.KL), Sime Darby Plantation (5285.KL)",
            "Lagged CPO futures → plantation stock price with 3–5 day delay",
            "Refinery margin squeeze signals for downstream plantation",
        ],
        "when_to_avoid": [
            "Non-plantation sectors — CPO correlation is sector-specific",
            "Intraday strategies — CPO futures close before Bursa opens",
        ],
        "bursa_applicability":    "Very high — plantation stocks are ~15% of KLCI; CPO price is the primary driver",
        "ic_improvement_vs_sma":  "CPO-lagged signal IC 0.12–0.18 on plantation stocks vs 0.03–0.05 for generic momentum",
        "stock_types":            ["plantation"],
        "strategy_types":         ["commodity", "sector_rotation"],
        "holding_periods":        ["short_term", "medium_term"],
        "signal_types":           ["price", "commodity"],
        "implemented":            False,
        "complexity":             "low",
        "overfitting_risk":       "low",
    },

    "opr_banking_signal": {
        "name":    "BNM OPR Cycle Banking Sector Signal",
        "angle":   "macro",
        "when_to_use": [
            "Banking stocks: Maybank (1155.KL), Public Bank (1295.KL), CIMB (1023.KL)",
            "OPR hike cycle → NIM expansion → buy banking ahead of BNM meeting",
            "OPR cut cycle → NIM compression → reduce banking exposure",
        ],
        "when_to_avoid": [
            "Non-banking sectors — OPR sensitivity is sector-specific",
            "Very short-term trades < 5 days — OPR effects take weeks to flow through",
        ],
        "bursa_applicability":    "Very high — banking is ~30% of KLCI; OPR is the single biggest systematic driver",
        "ic_improvement_vs_sma":  "OPR-conditional banking signal IC ~0.15 vs ~0.04 for simple momentum",
        "stock_types":            ["banking"],
        "strategy_types":         ["macro", "sector_rotation"],
        "holding_periods":        ["medium_term", "long_term"],
        "signal_types":           ["macro", "fundamental"],
        "implemented":            False,
        "complexity":             "medium",
        "overfitting_risk":       "low",
    },
}

# ── Lookup helpers ────────────────────────────────────────────────────────────

_ANGLE_TO_KEYS: dict[str, list[str]] = {}
_STRATEGY_TYPE_TO_KEYS: dict[str, list[str]] = {}
_STOCK_TYPE_TO_KEYS: dict[str, list[str]] = {}

for _k, _v in TECHNIQUE_LIBRARY.items():
    _ANGLE_TO_KEYS.setdefault(_v.get("angle", ""), []).append(_k)
    for _st in _v.get("strategy_types", []):
        _STRATEGY_TYPE_TO_KEYS.setdefault(_st, []).append(_k)
    for _stype in _v.get("stock_types", []):
        _STOCK_TYPE_TO_KEYS.setdefault(_stype, []).append(_k)


# ── TechniqueLibrary class ────────────────────────────────────────────────────

class TechniqueLibrary:
    """Query and format quantitative techniques for prompt injection."""

    # ── Selection ─────────────────────────────────────────────────────────────

    def get_relevant_techniques(
        self,
        strategy_type: str = "",
        stock_type: str = "KLCI_blue_chip",
        holding_period: str = "medium_term",
        signal_type: str = "price",
        max_techniques: int = 3,
    ) -> str:
        """Return a concise prompt-injection string listing the most relevant techniques.

        Scoring weights:
          +3  strategy_type match
          +2  stock_type match (or 'all' wildcard)
          +2  holding_period match (or 'all' wildcard)
          +1  signal_type match (or 'all' wildcard)
          +1  implemented == True (prefer proven tools)
          -1  overfitting_risk == 'high'

        Returns a formatted string ready to inject into a Claude prompt.
        """
        scores: dict[str, float] = {}
        for key, tech in TECHNIQUE_LIBRARY.items():
            s = 0.0
            if strategy_type and strategy_type.lower() in [x.lower() for x in tech.get("strategy_types", [])]:
                s += 3
            elif "all" in tech.get("strategy_types", []):
                s += 1

            st_list = [x.lower() for x in tech.get("stock_types", [])]
            norm_stock = stock_type.lower().replace("klci_", "").replace("_", "")
            if "all" in st_list or norm_stock in st_list or any(norm_stock in x for x in st_list):
                s += 2

            hp_list = [x.lower() for x in tech.get("holding_periods", [])]
            if "all" in hp_list or holding_period.lower() in hp_list:
                s += 2

            sig_list = [x.lower() for x in tech.get("signal_types", [])]
            if "all" in sig_list or signal_type.lower() in sig_list:
                s += 1

            if tech.get("implemented"):
                s += 1
            if tech.get("overfitting_risk") == "high":
                s -= 1

            scores[key] = s

        top = sorted(scores, key=lambda k: scores[k], reverse=True)[:max_techniques]
        return self._format_for_prompt(top)

    def get_by_angle(self, angle: str) -> str:
        """Return prompt string for all techniques in a given research angle."""
        keys = _ANGLE_TO_KEYS.get(angle, [])
        return self._format_for_prompt(keys) if keys else ""

    def get_by_key(self, key: str) -> Optional[dict]:
        """Return a single technique dict by key, or None."""
        return TECHNIQUE_LIBRARY.get(key)

    def all_keys(self) -> list[str]:
        return list(TECHNIQUE_LIBRARY.keys())

    # ── Formatting ────────────────────────────────────────────────────────────

    def _format_for_prompt(self, keys: list[str]) -> str:
        """Format a list of technique keys as a concise prompt-injection block."""
        if not keys:
            return ""
        lines = []
        for key in keys:
            tech = TECHNIQUE_LIBRARY.get(key)
            if not tech:
                continue
            status = "✓ implemented" if tech.get("implemented") else "○ not yet implemented"
            complexity = tech.get("complexity", "?")
            applicability = tech.get("bursa_applicability", "")
            use_cases = "; ".join(tech.get("when_to_use", [])[:3])
            avoid = "; ".join(tech.get("when_to_avoid", [])[:2])
            ic_note = tech.get("ic_improvement_vs_sma", "")

            lines.append(f"  [{key}] {tech['name']} [{status}, complexity={complexity}]")
            lines.append(f"    Bursa applicability: {applicability}")
            lines.append(f"    Use when: {use_cases}")
            if avoid:
                lines.append(f"    Avoid when: {avoid}")
            if ic_note:
                lines.append(f"    IC benchmark: {ic_note}")
        return "\n".join(lines)

    def format_full_detail(self, key: str) -> str:
        """Return a full detailed description of one technique for Stage 1 research injection."""
        tech = TECHNIQUE_LIBRARY.get(key)
        if not tech:
            return f"[Technique '{key}' not found in library]"
        lines = [
            f"TECHNIQUE: {tech['name']}",
            f"Research angle: {tech.get('angle', '?')}",
            f"Bursa applicability: {tech.get('bursa_applicability', '?')}",
            f"Complexity: {tech.get('complexity', '?')} | "
            f"Overfitting risk: {tech.get('overfitting_risk', '?')} | "
            f"Implemented: {'Yes' if tech.get('implemented') else 'No'}",
            "",
            "WHEN TO USE:",
        ]
        for item in tech.get("when_to_use", []):
            lines.append(f"  • {item}")
        if tech.get("when_to_avoid"):
            lines.append("\nWHEN TO AVOID:")
            for item in tech["when_to_avoid"]:
                lines.append(f"  • {item}")
        if tech.get("ic_improvement_vs_sma"):
            lines.append(f"\nIC BENCHMARK: {tech['ic_improvement_vs_sma']}")
        return "\n".join(lines)

    def format_telegram_summary(self, key: Optional[str] = None) -> str:
        """Return a Telegram-formatted summary of one or all techniques."""
        if key:
            tech = TECHNIQUE_LIBRARY.get(key)
            if not tech:
                return f"Unknown technique: `{key}`\n\nAvailable: " + ", ".join(f"`{k}`" for k in TECHNIQUE_LIBRARY)
            status_icon = "✅" if tech.get("implemented") else "🔲"
            risk_icon   = {"low": "🟢", "medium": "🟡", "high": "🔴"}.get(tech.get("overfitting_risk", ""), "⬜")
            lines = [
                f"{status_icon} *{tech['name']}*",
                f"Key: `{key}` | Complexity: `{tech.get('complexity','?')}` | Overfit risk: {risk_icon}",
                f"\n*Bursa applicability:* _{tech.get('bursa_applicability','?')}_\n",
                "*Use when:*",
            ]
            for item in tech.get("when_to_use", []):
                lines.append(f"  • {item}")
            if tech.get("when_to_avoid"):
                lines.append("\n*Avoid when:*")
                for item in tech["when_to_avoid"][:3]:
                    lines.append(f"  • {item}")
            if tech.get("ic_improvement_vs_sma"):
                lines.append(f"\n*IC benchmark:* `{tech['ic_improvement_vs_sma']}`")
            return "\n".join(lines)

        # Full arsenal listing
        groups: dict[str, list] = {}
        for k, v in TECHNIQUE_LIBRARY.items():
            groups.setdefault(v.get("angle", "other"), []).append((k, v))

        lines = ["⚙️ *OpenClaw Technique Arsenal*\n"]
        angle_icons = {
            "statistical_modelling": "📐",
            "price_action": "📈",
            "event_driven": "📅",
            "fundamental": "📊",
            "institutional": "🏦",
            "commodity": "🌴",
            "macro": "🏛️",
        }
        for angle, techs in sorted(groups.items()):
            icon = angle_icons.get(angle, "🔬")
            lines.append(f"\n{icon} *{angle.replace('_', ' ').title()}*")
            for k, v in sorted(techs, key=lambda x: (not x[1].get("implemented"), x[0])):
                imp  = "✅" if v.get("implemented") else "🔲"
                cplx = v.get("complexity", "?")[0].upper()  # M/H/L
                lines.append(f"  {imp} `{k}` [{cplx}] — {v['name']}")
        lines.append("\n_Use /arsenal <key> for full details, e.g. /arsenal kalman\\_filter_")
        return "\n".join(lines)

    # ── API-friendly dict output ──────────────────────────────────────────────

    def to_api_list(self) -> list[dict]:
        """Return all techniques as a list of dicts for the /api/system/arsenal endpoint."""
        result = []
        for key, tech in TECHNIQUE_LIBRARY.items():
            result.append({
                "key":                  key,
                "name":                 tech["name"],
                "angle":                tech.get("angle", ""),
                "implemented":          tech.get("implemented", False),
                "complexity":           tech.get("complexity", ""),
                "overfitting_risk":     tech.get("overfitting_risk", ""),
                "bursa_applicability":  tech.get("bursa_applicability", ""),
                "ic_benchmark":         tech.get("ic_improvement_vs_sma", ""),
                "when_to_use":          tech.get("when_to_use", []),
                "when_to_avoid":        tech.get("when_to_avoid", []),
                "stock_types":          tech.get("stock_types", []),
                "strategy_types":       tech.get("strategy_types", []),
                "holding_periods":      tech.get("holding_periods", []),
                "signal_types":         tech.get("signal_types", []),
            })
        return result
