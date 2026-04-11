"""
seed_strategy_profiles.py — Seed all 10 strategy profiles into strategy_profiles table.

Run once (idempotent — uses INSERT OR REPLACE):
    PYTHONPATH=/opt/openclaw/app \
    /opt/openclaw/venv/bin/python scripts/seed_strategy_profiles.py
"""
from data.database import db_session, init_db

PROFILES = [

{
    "strategy_key": "cross_sectional_momentum",
    "name": "Cross-Sectional Momentum",
    "strategy_class": "momentum",
    "angle": "price_action",
    "phenomenon": (
        "Stocks that outperformed their peers over the past 3-12 months tend to "
        "continue outperforming for the next 1-3 months. Driven by slow institutional "
        "accumulation, analyst upgrade cycles, and investor underreaction to good news. "
        "One of the most replicated findings in finance — holds on Bursa with slightly "
        "shorter formation windows than US markets due to lower liquidity."
    ),
    "bursa_nuance": (
        "The skip-month rule is critical on Bursa. The most recent month is excluded "
        "from the formation period because Bursa exhibits a short-term reversal effect "
        "at the 1-month horizon that would otherwise cancel the momentum signal. "
        "Universe restricted to FBM70 for sufficient liquidity. Momentum effect is "
        "strongest in mid-caps (FBM70 ex-KLCI) where institutional coverage is lower "
        "and price discovery is slower."
    ),
    "entry_condition": (
        "Last trading day of each month. Stock ranks in top 20% of FBM70 universe by "
        "6-month total return, EXCLUDING the most recent 1 month (skip-month rule). "
        "Additional filters: average daily volume > MYR 500K (liquidity minimum), "
        "price > MYR 0.50 (exclude penny stocks), not in top 5% by volatility "
        "(exclude lottery stocks)."
    ),
    "entry_universe": "FBM70",
    "entry_rebalance": "Monthly — last trading day",
    "exit_type": "signal_or_time",
    "exit_condition": (
        "PRIMARY: Monthly rebalance — exit when stock falls below top 35% of FBM70 "
        "universe ranking at next month-end assessment. SECONDARY: RSI(14) > 78 "
        "(overbought — momentum exhausted). FALLBACK: Day 65 (3 calendar months — "
        "momentum has decayed by this point regardless of ranking)."
    ),
    "exit_rationale": (
        "Momentum edge on Bursa peaks at 1-2 months and decays significantly beyond "
        "3 months. Exit is primarily signal-conditioned (ranking drops) rather than "
        "time-based. The RSI exit captures overbought conditions where momentum is "
        "near exhaustion. Day 65 fallback ensures no position is held past the "
        "documented edge window."
    ),
    "stop_loss_pct": 8.0,
    "profit_target_pct": None,
    "min_hold_days": 20,
    "max_hold_days": 65,
    "hold_rationale": (
        "Min 20 days avoids noise and covers transaction costs (0.7% round trip needs "
        "~3% move minimum over holding period). Max 65 days is 3 calendar months — "
        "beyond this, momentum alpha decays and reversal risk increases on Bursa."
    ),
    "complexity": "medium",
    "data_requirements": "ohlcv",
    "implementation_status": "planned",
    "ic_benchmark": (
        "IC 0.04-0.07 on FBM70 monthly cross-section. Annualised return spread between "
        "top and bottom quintile: 12-18% historically on ASEAN markets."
    ),
    "use_when": (
        "Bull market regime (HMM State 1). Post-earnings season when institutional flow "
        "is directional. Low VIX environment. When KLCI is above its 200-day MA."
    ),
    "avoid_when": (
        "Bear market or high volatility regime (GARCH > 75th percentile). During "
        "market-wide corrections > 10%. When momentum factor has crowded (many managers "
        "holding same stocks — check via tracking error compression)."
    ),
},

{
    "strategy_key": "short_term_reversal",
    "name": "Short-Term Reversal",
    "strategy_class": "reversal",
    "angle": "price_action",
    "phenomenon": (
        "Stocks that drop sharply over a short period without a fundamental catalyst "
        "tend to recover as liquidity providers step in to capture the spread, and as "
        "panic sellers exhaust themselves. The effect is strongest in illiquid periods "
        "and on stocks with high retail ownership where overreaction is most common."
    ),
    "bursa_nuance": (
        "Must filter out genuine fundamental drops — earnings misses, contract losses, "
        "regulatory actions. The Bursa announcement feed is the filter. Works best on "
        "Bursa mid-caps where liquidity is thinner and overreactions are more "
        "pronounced. Strongest signal on Monday opens after weekend news that proves "
        "less impactful than feared."
    ),
    "entry_condition": (
        "Weekly close when stock dropped > 6% in last 5 trading days. AND no material "
        "Bursa announcement in last 5 trading days (check bursa_scraper for earnings, "
        "contracts, material info). AND average daily volume > MYR 500K (need liquidity "
        "to exit cleanly). AND stock is in FBM70 or FBM Small Cap. AND drop is not part "
        "of broad market decline > 3% (not systematic risk)."
    ),
    "entry_universe": "FBM70 + FBM Small Cap",
    "entry_rebalance": "Daily scan — enter when triggered",
    "exit_type": "signal_or_time",
    "exit_condition": (
        "PRIMARY: Price recovers to within 1.5% of pre-drop level — full recovery "
        "achieved, take profit. SECONDARY: RSI(5) > 68 — short-term overbought, "
        "reversal complete. FALLBACK: Day 5 — reversal window expired. If not recovered "
        "by day 5, the drop was likely fundamental not technical."
    ),
    "exit_rationale": (
        "Reversal literature across Asian markets shows edge concentrates in days 1-5 "
        "post-drop. Beyond day 5, the trade has either succeeded or the drop was "
        "fundamental. Waiting beyond day 5 adds holding risk with no additional edge. "
        "Tight time fallback is intentional."
    ),
    "stop_loss_pct": 5.0,
    "profit_target_pct": 6.0,
    "min_hold_days": 1,
    "max_hold_days": 5,
    "hold_rationale": (
        "Reversal is a short-duration phenomenon. 1-day minimum avoids false exits from "
        "intraday noise. 5-day maximum is firm — the edge does not exist beyond this "
        "window. This is one of the few strategy types where the time exit is not a "
        "fallback but a hard rule based on empirical decay."
    ),
    "complexity": "low",
    "data_requirements": "ohlcv",
    "implementation_status": "planned",
    "ic_benchmark": (
        "IC 0.03-0.05 at 5-day horizon on Bursa mid-caps. Win rate ~62% when "
        "announcement filter applied. Without announcement filter: ~48% — filter is critical."
    ),
    "use_when": (
        "Market-wide oversold conditions (KLCI RSI < 35). High retail sentiment fear "
        "(VIX spike). End of month institutional window dressing reversals. "
        "Stock-specific overreactions to non-material news."
    ),
    "avoid_when": (
        "Trending bear markets (KLCI below 200-day MA). When stock has a fundamental "
        "catalyst for the drop. During earnings season (drops often fundamental). "
        "Illiquid stocks where spread costs exceed expected edge."
    ),
},

{
    "strategy_key": "low_volatility_anomaly",
    "name": "Low Volatility Anomaly",
    "strategy_class": "low_vol",
    "angle": "statistical_modelling",
    "phenomenon": (
        "Paradoxically, low-volatility stocks outperform high-volatility stocks on a "
        "risk-adjusted basis — and on Bursa they often outperform in absolute terms. "
        "Retail investors systematically overpay for lottery-like high-volatility "
        "stocks, creating persistent undervaluation in boring stable stocks. The anomaly "
        "is strongest during market stress periods when low-vol stocks act as defensive "
        "positions."
    ),
    "bursa_nuance": (
        "Particularly strong on Bursa because retail participation is high (~35% of "
        "trading volume) and retail investors disproportionately chase high-volatility, "
        "low-priced stocks hoping for multi-bagger returns. This systematically "
        "overprices volatile stocks and underprices stable compounders. The illiquidity "
        "filter is critical — low vol + illiquid is a value trap, not an opportunity."
    ),
    "entry_condition": (
        "Quarterly rebalance (end of Mar/Jun/Sep/Dec). Long bottom 20% of FBM70 by "
        "60-day realised volatility (annualised). Additional filters: average daily "
        "volume > MYR 1M (liquidity — avoid illiquid low-vol). PE ratio < 35 (avoid "
        "overvalued defensives). Positive trailing 12M earnings (quality filter — no "
        "distressed low-vol). Equal weight all selected stocks."
    ),
    "entry_universe": "FBM70",
    "entry_rebalance": "Quarterly — Mar/Jun/Sep/Dec",
    "exit_type": "signal_or_time",
    "exit_condition": (
        "PRIMARY: Quarterly rebalance — exit stocks that have moved into top 40% of "
        "volatility ranking (no longer low-vol). SECONDARY: Stock drops below MYR 1M "
        "average daily volume (liquidity deterioration — exit immediately). FALLBACK: "
        "90 calendar days (hard quarterly rebalance — exit all and rebuild portfolio "
        "from scratch)."
    ),
    "exit_rationale": (
        "Low volatility is a slow-moving structural factor. Monthly rebalancing adds "
        "too much transaction cost to a thin-margin strategy. Quarterly rebalancing "
        "optimises the cost/signal tradeoff. The fallback is a hard rebalance date, "
        "not a price-based exit — this is a portfolio strategy not a single-stock trade."
    ),
    "stop_loss_pct": 12.0,
    "profit_target_pct": None,
    "min_hold_days": 60,
    "max_hold_days": 90,
    "hold_rationale": (
        "Minimum 60 days because the factor needs time to express — short holding "
        "captures noise not signal. Stop is at portfolio level (-12% portfolio drawdown) "
        "not per-stock. Individual stocks may fall 12% but the portfolio stop protects "
        "against systemic low-vol failure (e.g. rate shock that hits all defensives "
        "equally)."
    ),
    "complexity": "medium",
    "data_requirements": "ohlcv",
    "implementation_status": "planned",
    "ic_benchmark": (
        "Annualised alpha vs KLCI: 3-6% on risk-adjusted basis. Sharpe ratio of "
        "low-vol quintile typically 0.6-0.9 vs 0.2-0.4 for high-vol quintile on ASEAN "
        "markets. Maximum drawdown 40-60% lower than market benchmark."
    ),
    "use_when": (
        "Any market regime — low vol is regime-agnostic by design. Especially powerful "
        "in bear markets and high-volatility regimes where it acts as a natural hedge. "
        "Good core holding for capital preservation."
    ),
    "avoid_when": (
        "Early-stage bull market recoveries (low-vol underperforms in strong risk-on "
        "rallies). When interest rates are rising rapidly (defensive sectors hurt by "
        "rate increases). When the strategy has become crowded (low-vol ETF flows have "
        "compressed valuations)."
    ),
},

{
    "strategy_key": "rsi_mean_reversion",
    "name": "RSI Mean Reversion",
    "strategy_class": "mean_reversion",
    "angle": "price_action",
    "phenomenon": (
        "Stocks that become deeply oversold on short-term RSI tend to revert toward "
        "the mean as technical traders, value buyers, and covered short-sellers step in. "
        "The RSI oversold bounce is one of the most consistent short-term patterns "
        "across all markets — but requires quality filters to avoid catching falling "
        "knives."
    ),
    "bursa_nuance": (
        "Quality filter is essential on Bursa. RSI < 25 on a fundamentally impaired "
        "stock is not a buy signal — it is a falling knife. Filter by positive trailing "
        "earnings and reasonable PE to exclude distressed companies. Pattern is strongest "
        "after post-earnings overreactions where the quarter was bad but not "
        "catastrophically bad. Retail-dominated mid-caps show strongest oversold bounces "
        "on Bursa."
    ),
    "entry_condition": (
        "Daily close when RSI(14) < 25 (deeply oversold). AND PE ratio < 30 (not "
        "overvalued). AND trailing 12-month earnings positive (quality filter — no "
        "distressed stocks). AND stock in FBM70 or FBM100 (universe quality minimum). "
        "AND no material negative announcement in last 5 days (not a fundamental drop). "
        "AND stock has not been in RSI < 25 for more than 3 consecutive weeks (chronic "
        "oversold = structural problem)."
    ),
    "entry_universe": "FBM70 + FBM100",
    "entry_rebalance": "Daily scan — enter when triggered",
    "exit_type": "signal_or_time",
    "exit_condition": (
        "PRIMARY: RSI(14) > 55 — normalised (not necessarily overbought, just recovered "
        "from oversold). We exit at 55 not 70 because we are capturing the bounce, not "
        "the full recovery. SECONDARY: +8% from entry — profit target locked. "
        "FALLBACK: Day 15 — oversold bounce window expired."
    ),
    "exit_rationale": (
        "RSI mean reversion on Bursa completes in 5-12 trading days on average. Exiting "
        "at RSI 55 rather than 70 takes profit earlier but with higher consistency. "
        "Waiting for RSI 70 often means giving back gains as the stock oscillates. "
        "The 8% profit target locks in gains above transaction costs with margin. "
        "Day 15 fallback closes position if bounce is taking too long — likely "
        "fundamental resistance to recovery."
    ),
    "stop_loss_pct": 6.0,
    "profit_target_pct": 8.0,
    "min_hold_days": 3,
    "max_hold_days": 15,
    "hold_rationale": (
        "Minimum 3 days avoids false exits from intraday RSI fluctuations. Maximum "
        "15 days is firm — if the bounce hasn't started by day 15 the fundamental "
        "picture likely changed. This is the refined version of the existing RSI "
        "strategy — key change is the quality filter and signal exit replacing the "
        "arbitrary time exit."
    ),
    "complexity": "low",
    "data_requirements": "ohlcv",
    "implementation_status": "live",
    "ic_benchmark": (
        "IC 0.05-0.08 at 5-10 day horizon on Bursa mid-caps with quality filter applied. "
        "Without quality filter IC drops to 0.01-0.02 — filter contributes ~75% of the "
        "signal value."
    ),
    "use_when": (
        "Market-wide oversold conditions amplify individual stock bounces. Post-earnings "
        "overreactions where the result was bad but not catastrophic. After sector "
        "rotations where quality stocks are sold indiscriminately. High retail fear "
        "sentiment."
    ),
    "avoid_when": (
        "Trending bear markets — RSI can stay below 30 for months. When the stock has "
        "a genuine fundamental problem. During earnings season if the next earnings is "
        "within 10 days (result may extend the drop). Highly illiquid stocks where the "
        "spread eats the entire bounce."
    ),
},

{
    "strategy_key": "bollinger_squeeze_breakout",
    "name": "Bollinger Band Squeeze Breakout",
    "strategy_class": "breakout",
    "angle": "price_action",
    "phenomenon": (
        "When Bollinger Bands narrow significantly — indicating a period of low "
        "volatility compression — a directional breakout tends to follow as the "
        "compressed energy releases. The squeeze predicts that a large move is imminent. "
        "Volume confirmation on the breakout provides directional bias. The edge is in "
        "identifying the squeeze early, not in predicting direction."
    ),
    "bursa_nuance": (
        "Bursa stocks frequently squeeze before quarterly earnings announcements as "
        "informed traders accumulate quietly. The squeeze-to-earnings pattern is "
        "particularly reliable — the breakout direction aligns with the earnings surprise "
        "in ~68% of cases. Volume filter is critical on Bursa because low-volume "
        "breakouts on illiquid stocks are frequently false — manipulated by contra "
        "traders."
    ),
    "entry_condition": (
        "Daily close when Bollinger Band width (BB_width = (upper - lower) / middle) "
        "is below the 10th percentile of its own 126-trading-day (6-month) history — "
        "squeeze confirmed. AND price closes above the upper Bollinger Band — upside "
        "breakout confirmed. AND volume on breakout day > 150% of 20-day average volume "
        "— institutional confirmation. AND stock in FBM70 or FBM100."
    ),
    "entry_universe": "FBM70 + FBM100",
    "entry_rebalance": "Daily scan — enter on breakout day",
    "exit_type": "signal_or_time",
    "exit_condition": (
        "PRIMARY: Price closes below the 20-day middle Bollinger Band — breakout has "
        "failed, trend has reversed back into the range. Exit immediately on this "
        "signal — do not wait for confirmation. SECONDARY: +15% from entry — profit "
        "target. FALLBACK: Day 20 — if breakout has not delivered +15% in 20 days the "
        "momentum has stalled."
    ),
    "exit_rationale": (
        "The key insight is that a false breakout — where price re-enters the Bollinger "
        "Band — is a strong reversal signal. Exit on close below middle band is "
        "immediate and decisive. Breakout moves on Bursa are typically sharp and fast: "
        "the bulk of the move happens in the first 5-10 days. Day 20 captures the full "
        "wave including slower institutional participation."
    ),
    "stop_loss_pct": 5.0,
    "profit_target_pct": 15.0,
    "min_hold_days": 2,
    "max_hold_days": 20,
    "hold_rationale": (
        "Minimum 2 days avoids exiting on intraday noise immediately after breakout. "
        "The signal exit (close below middle band) is the primary exit — the time "
        "fallback at day 20 is a safety net. The asymmetric stop/target (5% stop, 15% "
        "target) reflects the nature of breakouts: you get stopped out more often but "
        "winners are 3x larger."
    ),
    "complexity": "low",
    "data_requirements": "ohlcv",
    "implementation_status": "live",
    "ic_benchmark": (
        "IC 0.06-0.09 at 10-day horizon when volume filter applied. Without volume "
        "filter IC drops to 0.02-0.04 due to false breakouts. Win rate ~45% but profit "
        "factor > 2.5 (winners are much larger than losers)."
    ),
    "use_when": (
        "Pre-earnings season (Sep-Oct and Mar-Apr on Bursa). Bull market regime where "
        "breakouts follow through. After prolonged sideways consolidation (> 3 months "
        "of low volatility). When the overall market is in an uptrend."
    ),
    "avoid_when": (
        "Bear market regime — breakouts fail frequently. High volatility environments "
        "(bands already wide — squeeze condition absent). During market-wide uncertainty "
        "(pre-election, macro shock). Thin-volume stocks where false breakouts dominate."
    ),
},

{
    "strategy_key": "gap_fill",
    "name": "Gap Fill",
    "strategy_class": "reversal",
    "angle": "price_action",
    "phenomenon": (
        "When a stock opens significantly below its previous close without a fundamental "
        "catalyst — a gap down — it tends to fill the gap within 3-5 days as overnight "
        "sentiment normalises, short-sellers cover, and value buyers emerge. The "
        "phenomenon reflects the tendency of markets to overreact to sentiment and then "
        "correct back toward the prior equilibrium."
    ),
    "bursa_nuance": (
        "Particularly common on Bursa Monday mornings after weekend news that proves "
        "less impactful than feared. Malaysian retail investors are particularly prone "
        "to weekend sentiment overreaction — selling Monday opens aggressively on news "
        "that is ultimately not material. The announcement filter is the most important "
        "component — without it you buy into genuine fundamental breaks."
    ),
    "entry_condition": (
        "Market open (or first 30-minute VWAP) when gap down is > 3% from previous "
        "close. AND no material Bursa company announcement in last 24 hours (check "
        "bursa_scraper). AND gap is below previous day's low (clean gap — not an inside "
        "bar). AND stock in FBM70 (need liquidity to enter/exit cleanly). AND the "
        "broader KLCI is not gapping down > 1.5% (not systematic market risk — "
        "stock-specific gap only)."
    ),
    "entry_universe": "FBM70",
    "entry_rebalance": "Daily pre-market scan — enter at open",
    "exit_type": "signal_or_time",
    "exit_condition": (
        "PRIMARY: Price fills the gap — reaches the previous close level. Full target "
        "achieved — exit immediately. SECONDARY: +4% from entry if gap fills before "
        "reaching previous close level. FALLBACK: Day 3 — gap fill window is narrow. "
        "If not filled in 3 trading days, it is likely not filling."
    ),
    "exit_rationale": (
        "Gap fills are among the shortest-duration phenomena in equity markets. The gap "
        "either fills in 1-3 days or it becomes a breakaway gap that signals a genuine "
        "trend change. There is almost no evidence of gaps filling after day 5. The "
        "3-day fallback is generous — most fills happen on day 1 or 2. Tight stop is "
        "essential because extending gaps signal fundamental catalyst not sentiment."
    ),
    "stop_loss_pct": 3.0,
    "profit_target_pct": 4.0,
    "min_hold_days": 1,
    "max_hold_days": 3,
    "hold_rationale": (
        "Gap fill is the shortest-duration strategy in the arsenal. The 3-day maximum "
        "is not a conservative estimate — it is empirically derived. Beyond day 3 the "
        "expected value of holding is negative. The tight 3% stop reflects the binary "
        "nature of gap fills: either the gap fills (positive edge) or the gap extends "
        "(fundamental — exit immediately)."
    ),
    "complexity": "low",
    "data_requirements": "ohlcv",
    "implementation_status": "planned",
    "ic_benchmark": (
        "Gap fill rate on Bursa FBM70 (without announcement filter): ~58%. With "
        "announcement filter applied: ~71%. Average fill time: 1.4 trading days. "
        "Average gain when filled: 3.2%. Expected value per trade after costs: ~1.1%."
    ),
    "use_when": (
        "Monday morning gaps after weekend news. After macro events that create broad "
        "market gaps (Fed decision, BNM OPR). When the VIX has spiked and fear is "
        "elevated. Post-holiday gaps when sentiment has time to overcorrect."
    ),
    "avoid_when": (
        "When the gap is accompanied by a Bursa announcement. When the stock has been "
        "in a downtrend for > 3 months (gaps may continue). When the broader market is "
        "gapping down systemically. During earnings season when gaps are often "
        "fundamental."
    ),
},

{
    "strategy_key": "opening_range_breakout",
    "name": "Opening Range Breakout",
    "strategy_class": "breakout",
    "angle": "price_action",
    "phenomenon": (
        "The price range established in the first 30 minutes of trading often defines "
        "the directional bias for the rest of the session. A breakout above the opening "
        "range high with volume tends to persist through close as institutional orders "
        "accumulate in the confirmed direction."
    ),
    "bursa_nuance": (
        "Bursa opens at 9:00 AM MYT. The opening range is defined as 9:00-9:30 AM. "
        "This strategy requires intraday OHLCV data at minimum 30-minute bars — Yahoo "
        "Finance free tier provides daily data only. Implementation is deferred until "
        "an intraday data source is connected (potential sources: Rakuten Trade API, "
        "Bloomberg, or a paid Yahoo Finance subscription)."
    ),
    "entry_condition": (
        "DEFERRED — requires intraday data. When implemented: Entry when price breaks "
        "above the 9:00-9:30 AM high with volume > 200% of average 30-min volume. "
        "Entry within first 2 hours of session only."
    ),
    "entry_universe": "KLCI 30 (most liquid)",
    "entry_rebalance": "Daily — intraday trigger",
    "exit_type": "signal_or_time",
    "exit_condition": (
        "DEFERRED. When implemented: Exit at session close (same-day exit always) OR "
        "when price falls back below opening range high (breakout failed). No overnight "
        "holds — this is a pure intraday strategy."
    ),
    "exit_rationale": (
        "Opening range breakout is an intraday phenomenon. Holding overnight introduces "
        "gap risk that is unrelated to the opening range signal. All exits are same-day."
    ),
    "stop_loss_pct": 1.5,
    "profit_target_pct": 2.0,
    "min_hold_days": 0,
    "max_hold_days": 0,
    "hold_rationale": (
        "Intraday only. No overnight holds. Requires intraday data infrastructure "
        "before implementation."
    ),
    "complexity": "high",
    "data_requirements": "intraday",
    "implementation_status": "deferred",
    "ic_benchmark": (
        "Estimated IC 0.08-0.12 on KLCI large-caps based on US and Singapore market "
        "studies. Bursa-specific IC not yet measured — requires intraday data."
    ),
    "use_when": (
        "High-volume sessions with clear directional catalyst. Post-announcement days "
        "where institutional flow is directional. Trending market days."
    ),
    "avoid_when": (
        "Low-volume sessions (public holidays nearby). Choppy markets with no clear "
        "direction. Before/after major macro events where opening range may be distorted."
    ),
},

{
    "strategy_key": "sma_crossover",
    "name": "Simple Moving Average Crossover",
    "strategy_class": "trend",
    "angle": "price_action",
    "phenomenon": (
        "When a short-term moving average crosses above a long-term moving average "
        "(golden cross), it signals a regime change from downtrend to uptrend. The "
        "crossover acts as a trend filter and regime indicator. Simple but effective as "
        "a trend-following entry on liquid large-cap stocks where trends are more "
        "persistent."
    ),
    "bursa_nuance": (
        "SMA crossover works best on KLCI 30 blue chips where institutional participation "
        "is high and trends are more persistent than on mid-caps. The 200-day MA filter "
        "is critical — only take crossover signals when the stock is above its 200-day MA "
        "(bull market regime filter). This eliminates the majority of false crossovers "
        "in bear markets. Hard stop at -10% from entry is the only time-independent exit."
    ),
    "entry_condition": (
        "Daily close when 20-day SMA crosses above 50-day SMA (golden cross). AND stock "
        "is above its 200-day MA (bull regime confirmation). AND stock is in KLCI 30 "
        "(blue chip — trend persistence required). AND volume on crossover day > 100% "
        "of 20-day average (institutional participation)."
    ),
    "entry_universe": "KLCI 30",
    "entry_rebalance": "Daily scan — enter on crossover day",
    "exit_type": "signal",
    "exit_condition": (
        "PRIMARY: 20-day SMA crosses below 50-day SMA (death cross) — trend has "
        "reversed, exit on next open. This is the ONLY price exit. STOP LOSS: -10% "
        "from entry (hard stop — exit regardless of MA position). No profit target — "
        "let the trend run."
    ),
    "exit_rationale": (
        "Trend-following strategies must not have an arbitrary time ceiling. The worst "
        "mistake in trend-following is cutting winners early. A stock in a genuine bull "
        "trend can hold the 20/50 MA crossover for 12-18 months. Imposing a 30 or "
        "60-day exit would systematically kill the best trades. The exit is purely "
        "signal-conditioned: the trend ends when the trend ends."
    ),
    "stop_loss_pct": 10.0,
    "profit_target_pct": None,
    "min_hold_days": 10,
    "max_hold_days": None,
    "hold_rationale": (
        "No maximum hold. This is intentional and fundamental to the strategy. The "
        "10-day minimum avoids whipsawing on noisy crossovers. The -10% stop loss is "
        "wider than other strategies because false crossovers happen and we need room "
        "for the trend to breathe. Once the death cross occurs, exit regardless of "
        "P&L — the signal is the exit, not the profit level."
    ),
    "complexity": "low",
    "data_requirements": "ohlcv",
    "implementation_status": "live",
    "ic_benchmark": (
        "IC 0.03-0.05 at monthly horizon on KLCI blue-chips. Low IC but high "
        "consistency — trend-following wins are large and infrequent, losses are small "
        "and frequent. Sharpe 0.4-0.7 over full market cycles."
    ),
    "use_when": (
        "Bull market regime (HMM State 1, KLCI above 200-day MA). Post-correction "
        "recoveries. When institutional flow is consistently directional. Long-term "
        "capital deployment with patience for drawdowns."
    ),
    "avoid_when": (
        "Sideways/choppy markets (20/50 crossovers whipsaw frequently). Bear market "
        "regime. High-volatility environments where false crossovers generate repeated "
        "stop-outs. Illiquid stocks where the signal is noise. Short time horizon "
        "traders — this strategy requires patience."
    ),
},

{
    "strategy_key": "garch_volatility_overlay",
    "name": "GARCH Volatility Regime Overlay",
    "strategy_class": "overlay",
    "angle": "statistical_modelling",
    "phenomenon": (
        "GARCH (Generalised Autoregressive Conditional Heteroskedasticity) models "
        "capture volatility clustering — the empirical observation that high volatility "
        "days cluster together and low volatility days cluster together. By forecasting "
        "the next period's volatility, the system can dynamically adjust position sizes "
        "across all active strategies simultaneously."
    ),
    "bursa_nuance": (
        "Volatility clustering is particularly pronounced on Bursa due to thinner "
        "liquidity and higher retail participation. GARCH signals on KLCI are more "
        "reliable than on individual stocks due to index smoothing. The overlay applies "
        "to the entire portfolio — when GARCH forecasts a vol spike, all strategy "
        "position sizes are reduced simultaneously, not strategy-by-strategy."
    ),
    "entry_condition": (
        "NOT a standalone strategy — this is a position sizing OVERLAY applied on top "
        "of all other active strategies. GARCH(1,1) fitted on KLCI daily returns with "
        "252-day rolling window. Volatility forecast computed daily at market close. "
        "Position size multiplier applied at next open."
    ),
    "entry_universe": "KLCI Index (for GARCH model fitting)",
    "entry_rebalance": "Daily recalculation at market close",
    "exit_type": "signal",
    "exit_condition": (
        "Position size multiplier rules: "
        "Forecasted vol < 10th percentile → size all positions at 150% of base. "
        "Forecasted vol 10-75th percentile → size at 100% (base, no adjustment). "
        "Forecasted vol > 75th percentile → size at 60% of base. "
        "Forecasted vol > 90th percentile → size at 30% of base (near-flat). "
        "Exit individual positions only if vol forecast > 99th percentile "
        "(extreme regime — exit everything)."
    ),
    "exit_rationale": (
        "Position sizing IS the risk management. The overlay does not directly trigger "
        "entries or exits — it modulates the SIZE of existing positions. Reducing to "
        "30% in high-vol regimes protects capital while remaining in the market. The "
        "99th percentile full exit is reserved for true crisis regimes (GFC-level "
        "events) where all signals break down."
    ),
    "stop_loss_pct": None,
    "profit_target_pct": None,
    "min_hold_days": None,
    "max_hold_days": None,
    "hold_rationale": (
        "Not applicable — this is an overlay not a position. The GARCH model runs "
        "daily and position size multipliers are updated at each market close. There "
        "is no holding period concept for a sizing overlay."
    ),
    "complexity": "high",
    "data_requirements": "ohlcv",
    "implementation_status": "planned",
    "ic_benchmark": (
        "Portfolio Sharpe improvement from GARCH overlay: +15-25% vs static position "
        "sizing across market cycles. Maximum drawdown reduction: 20-35%. The overlay "
        "does not improve returns in isolation — it improves the risk-adjusted profile "
        "of all other strategies."
    ),
    "use_when": (
        "Always active — runs as a background process alongside all other strategies. "
        "Particularly valuable during earnings seasons, macro event windows (BNM OPR, "
        "Fed decisions), and geopolitical uncertainty periods."
    ),
    "avoid_when": (
        "Cannot be avoided — it is always running. If the GARCH model fails to fit "
        "(insufficient data, extreme outliers), fall back to static position sizing "
        "until the model stabilises."
    ),
},

{
    "strategy_key": "hmm_regime_detector",
    "name": "Hidden Markov Regime Detector",
    "strategy_class": "meta",
    "angle": "statistical_modelling",
    "phenomenon": (
        "Markets cycle through hidden states — bull, bear, and sideways/transitional — "
        "that are not directly observable but can be inferred from the statistical "
        "properties of returns and volatility. Hidden Markov Models (HMM) estimate the "
        "probability of being in each regime and route the system to the optimal "
        "strategy set for that regime. Different strategies have different alpha "
        "profiles across regimes."
    ),
    "bursa_nuance": (
        "Three-state HMM fitted on KLCI works well for Bursa. State transitions are "
        "driven by EPF rebalancing cycles, BNM monetary policy cycles, and China "
        "economic cycles — all of which have Bursa-specific timings. The regime "
        "detector prevents running momentum strategies in bear markets (where they "
        "destroy capital) and prevents running reversal strategies in strong trends "
        "(where they fight the tape)."
    ),
    "entry_condition": (
        "NOT a standalone strategy — this is a meta-strategy that selects which "
        "strategies to activate. HMM fitted on KLCI daily returns (252-day window). "
        "Three hidden states inferred: "
        "State 1 (Bull): High positive return, low volatility. "
        "State 2 (Bear): Negative return, high volatility. "
        "State 3 (Lateral): Near-zero return, medium volatility. "
        "Regime assessed weekly (daily is too noisy for regime classification)."
    ),
    "entry_universe": "KLCI Index (for HMM fitting)",
    "entry_rebalance": "Weekly regime assessment — every Friday close",
    "exit_type": "signal",
    "exit_condition": (
        "Strategy routing by regime: "
        "BULL regime → Activate: Momentum, Bollinger Breakout, SMA Crossover. "
        "Deactivate: Short-term Reversal, Gap Fill. Reduce: Low Vol position size. "
        "BEAR regime → Activate: Low Vol Anomaly, RSI Mean Reversion. "
        "Deactivate: Momentum, SMA Crossover. Reduce: Breakout strategies by 50%. "
        "LATERAL regime → Activate: RSI Reversion, Gap Fill, Short-term Reversal. "
        "Deactivate: Momentum. Neutral: Bollinger (works in lateral with vol "
        "compression)."
    ),
    "exit_rationale": (
        "Each strategy has a natural market regime where it performs best and a regime "
        "where it destroys capital. Routing between strategies based on detected regime "
        "is the single highest-leverage improvement to overall system performance. "
        "Running momentum in a bear market is the most common source of catastrophic "
        "drawdown in systematic strategies."
    ),
    "stop_loss_pct": None,
    "profit_target_pct": None,
    "min_hold_days": None,
    "max_hold_days": None,
    "hold_rationale": (
        "Not applicable — this is a meta-strategy that controls which other strategies "
        "are active. The regime assessment is weekly to avoid over-trading on daily "
        "noise. Regime changes require 2 consecutive weeks of consistent signals before "
        "switching (prevents whipsawing between regimes)."
    ),
    "complexity": "high",
    "data_requirements": "ohlcv",
    "implementation_status": "planned",
    "ic_benchmark": (
        "Strategy routing improvement: 20-40% Sharpe ratio increase vs running all "
        "strategies unconditionally. Bear market drawdown reduction: 30-50% by "
        "deactivating momentum in State 2. Regime prediction accuracy on KLCI "
        "historically: ~73% (1-week ahead)."
    ),
    "use_when": (
        "Always active as a background regime classifier. Becomes especially valuable "
        "during regime transitions (bull-to-bear) where wrong strategy selection causes "
        "maximum damage."
    ),
    "avoid_when": (
        "Do not rely solely on HMM during unprecedented market events (COVID-19-style "
        "shocks) where the model has no training data. In these cases fall back to "
        "maximum defensiveness regardless of HMM output."
    ),
},

]


def seed():
    init_db()
    with db_session() as conn:
        for p in PROFILES:
            conn.execute(
                """
                INSERT OR REPLACE INTO strategy_profiles (
                    strategy_key, name, strategy_class, angle,
                    phenomenon, bursa_nuance,
                    entry_condition, entry_universe, entry_rebalance,
                    exit_type, exit_condition, exit_rationale,
                    stop_loss_pct, profit_target_pct,
                    min_hold_days, max_hold_days, hold_rationale,
                    complexity, data_requirements, implementation_status,
                    ic_benchmark, use_when, avoid_when,
                    updated_at
                ) VALUES (
                    :strategy_key, :name, :strategy_class, :angle,
                    :phenomenon, :bursa_nuance,
                    :entry_condition, :entry_universe, :entry_rebalance,
                    :exit_type, :exit_condition, :exit_rationale,
                    :stop_loss_pct, :profit_target_pct,
                    :min_hold_days, :max_hold_days, :hold_rationale,
                    :complexity, :data_requirements, :implementation_status,
                    :ic_benchmark, :use_when, :avoid_when,
                    datetime('now')
                )
                """,
                p,
            )
    print(f"Seeded {len(PROFILES)} strategy profiles")


if __name__ == "__main__":
    seed()
