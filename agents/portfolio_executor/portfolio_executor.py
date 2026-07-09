import asyncio
import json
import logging
from datetime import datetime

import numpy as np

from agents.base_agent import BaseAgent
from agents.data_engineer.data_engineer import DataEngineer
from config.settings import (
    MODEL_FAST, GATE_CONFIG, TRADING_DAYS_PER_YEAR,
    BURSA_BOARD_LOT, PAPER_CAPITAL_MYR, PAPER_ALLOC_PCT,
    bursa_trade_cost, bursa_slippage_tier,
    ALLOW_SHORT, FUNDING_INTERVAL_HOURS, AVG_FUNDING_RATE_PER_INTERVAL, funding_cost,
    bars_per_day,
)
from data.database import db_session

logger = logging.getLogger(__name__)

# Minutes per sub-daily interval, for equity-slot alignment.
_INTERVAL_MINUTES = {"15m": 15, "1h": 60, "4h": 240}


def equity_slot(interval: str = "1d", now: datetime | None = None) -> str:
    """The paper_equity `date` key for a mark at `now` on this bar interval.

    Daily/weekly ideas keep the historical plain-date key (YYYY-MM-DD —
    byte-identical rows, Bursa parity). Sub-daily ideas get an interval-aligned
    datetime slot (YYYY-MM-DDTHH:MM) so UNIQUE(idea_id, date) dedupes one mark
    per BAR instead of one per day, with no schema rewrite.
    """
    now = now or datetime.utcnow()
    mins = _INTERVAL_MINUTES.get(interval)
    if mins is None:
        return now.strftime("%Y-%m-%d")
    floored = (now.hour * 60 + now.minute) // mins * mins
    return f"{now.strftime('%Y-%m-%d')}T{floored // 60:02d}:{floored % 60:02d}"


class PortfolioExecutor(BaseAgent):
    name = "PortfolioExecutor"
    description = "Bursa equity paper trading: signal-driven fills, board-lot sizing, NAV tracking"
    default_model = MODEL_FAST

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._data_engineer = DataEngineer()

    # ------------------------------------------------------------------
    # Market data helpers
    # ------------------------------------------------------------------

    def _latest_bar(self, ticker: str, interval: str = "1d") -> dict | None:
        """Last completed bar from the parquet cache: close, date, ADV value."""
        df = self._data_engineer.fetch_prices(ticker, interval=interval, days=90, use_cache=True)
        if df is None or df.empty or "close" not in df:
            return None
        close = float(df["close"].iloc[-1])
        date = str(df.index[-1])[:10]
        adv_value = 0.0
        if "volume" in df:
            tail = df.tail(20)
            adv_value = float((tail["close"] * tail["volume"]).mean())
        return {"close": close, "date": date, "adv_value": adv_value}

    # ------------------------------------------------------------------
    # Position sizing (board lots)
    # ------------------------------------------------------------------

    @staticmethod
    def size_shares(nav: float, price: float,
                    alloc_pct: float = PAPER_ALLOC_PCT):
        """Units to buy — delegates to the market profile's sizing rule
        (Bursa: whole 100-share board lots; crypto: fractional 0.0001 steps)."""
        from config.settings import size_units
        return size_units(nav, price, alloc_pct)

    # ------------------------------------------------------------------
    # Idea NAV (cash accounting per idea)
    # ------------------------------------------------------------------

    def _idea_cash(self, idea_id: int) -> float:
        """Current cash for an idea: starting capital ± all realized flows.

        Sign-agnostic for long/short: `units` is stored signed (negative for a
        short), so a short's entry correctly credits proceeds and its exit
        correctly debits the buy-to-cover cost with no direction branching.
        `funding_paid` (WS3) is a real cash flow — subtracted like any other
        cost (positive = paid, negative = received, so subtracting nets both).
        """
        with db_session() as conn:
            rows = conn.execute(
                "SELECT entry_price, exit_price, units, status, "
                "       COALESCE(entry_cost,0) AS entry_cost, "
                "       COALESCE(exit_cost,0)  AS exit_cost, "
                "       COALESCE(funding_paid,0) AS funding_paid "
                "FROM paper_trades WHERE idea_id=?", (idea_id,)
            ).fetchall()
        cash = PAPER_CAPITAL_MYR
        for r in rows:
            cash -= (r["entry_price"] or 0) * (r["units"] or 0) + r["entry_cost"]
            cash -= r["funding_paid"]
            if r["status"] == "closed":
                cash += (r["exit_price"] or 0) * (r["units"] or 0) - r["exit_cost"]
        return cash

    def _current_funding_rate(self, symbol: str) -> float:
        """Best-available per-8h funding rate for accrual (crypto only).

        Resolution order — real before modeled, cheap before networked:
          1. cached historical series (last settlement, if the parquet is
             fresh within ~2 settlement periods);
          2. live snapshot from the exchange;
          3. the modeled AVG_FUNDING_RATE_PER_INTERVAL constant (last resort,
             same fallback the backtester uses).
        """
        try:
            from agents.data_engineer.data_engineer import DataEngineer
            de = DataEngineer()
            path = de._cache_path(symbol, "8h_funding")
            if path.exists() and not de._is_stale(path, max_age_hours=16.0):
                cached = de._load_cache(path)
                if not cached.empty and "funding_rate" in cached:
                    return float(cached["funding_rate"].iloc[-1])
        except Exception:
            pass
        try:
            from data.binance.client import get_funding_rate
            live = get_funding_rate(symbol)
            if live and live.get("funding_rate") is not None:
                return float(live["funding_rate"])
        except Exception:
            pass
        return AVG_FUNDING_RATE_PER_INTERVAL

    def _accrue_funding(self, trade_id: int, units: float, price: float,
                        hours: float = 24.0, symbol: str = "") -> None:
        """Accrue `hours` worth of funding to a trade's running funding_paid
        (WS3, crypto only — a no-op call site elsewhere since
        FUNDING_INTERVAL_HOURS is None on Bursa). Daily ideas keep the
        historical fixed 24h/cycle; sub-daily ideas pass their elapsed time so
        marking every bar doesn't multiply the funding drag.

        Uses the REAL current funding rate when a symbol is provided
        (cached-series → live → modeled constant), so paper accrual tracks
        actual market funding instead of the disclosed average."""
        position = 1.0 if units > 0 else -1.0
        notional = abs(units) * price
        settlements = hours / FUNDING_INTERVAL_HOURS
        rate = (self._current_funding_rate(symbol) if symbol
                else AVG_FUNDING_RATE_PER_INTERVAL)
        day_funding = funding_cost(position, rate, notional) * settlements
        with db_session() as conn:
            conn.execute(
                "UPDATE paper_trades SET funding_paid = COALESCE(funding_paid,0) + ? WHERE id=?",
                (round(day_funding, 4), trade_id),
            )

    def _open_trade(self, idea_id: int):
        with db_session() as conn:
            return conn.execute(
                "SELECT * FROM paper_trades WHERE idea_id=? AND status='open' "
                "ORDER BY id DESC LIMIT 1", (idea_id,)
            ).fetchone()

    # ------------------------------------------------------------------
    # Paper trading
    # ------------------------------------------------------------------

    async def paper_entry(self, idea_id: int, ticker: str, direction: str = "long",
                          signal: str = "", **_legacy) -> dict:
        """Open a paper position at the latest cached close.

        WS3: `direction` may be "short" where ALLOW_SHORT (crypto perps) — the
        entire cash/PnL model is UNCHANGED by storing SIGNED units (negative
        for a short): _idea_cash()'s arithmetic already nets out correctly for
        either sign (a short's entry credits proceeds, exit debits the
        buy-to-cover cost), so no separate short-side accounting branch is
        needed. Bursa (ALLOW_SHORT=False) still rejects anything but "long".
        Still unleveraged (1x) in paper trading — the backtester carries the
        leverage/liquidation model (see BacktestEngineer._compute_performance);
        paper trading validates day-to-day signal execution, not margin math.
        """
        if direction == "short" and not ALLOW_SHORT:
            return {"error": "This market is long-only paper trading"}
        if direction not in ("long", "short"):
            return {"error": f"Unknown direction '{direction}' — use 'long' or 'short'"}
        if self._open_trade(idea_id):
            return {"error": f"Idea {idea_id} already has an open paper trade"}

        bar = self._latest_bar(ticker)
        nav = self._idea_cash(idea_id)

        # Phase 6.2: pre-trade risk checks (audit §11.2) — liquidity, data
        # confidence, unresolved corporate actions, board-lot affordability.
        from agents.portfolio_executor.execution_simulator import (
            pre_trade_check, simulate_fill)
        with db_session() as conn:
            dq_row = conn.execute(
                "SELECT confidence_score FROM data_quality_checks WHERE idea_id=? "
                "ORDER BY created_at DESC LIMIT 1", (idea_id,)
            ).fetchone()
            unresolved = conn.execute(
                "SELECT COUNT(*) n FROM corporate_actions WHERE ticker=? "
                "AND validation_status='suspected'", (ticker,)
            ).fetchone()["n"]
        dq_confidence = dq_row["confidence_score"] if dq_row else None
        check = pre_trade_check(ticker, nav, bar, dq_confidence, unresolved)
        if not check["passed"]:
            return {"error": "Pre-trade check failed: " + "; ".join(check["reasons"])}

        fill = simulate_fill(nav, bar["close"], bar["adv_value"], PAPER_ALLOC_PCT)
        if fill["status"] == "FAILED":
            return {"error": f"NAV RM{nav:,.0f} too small for one board lot of "
                             f"{ticker} @ RM{bar['close']:.2f} ({fill['reason']})"}
        sign = -1 if direction == "short" else 1
        units = sign * fill["units"]

        # A short OPENS with a sell (proceeds credited); a long opens with a buy.
        entry_side = "sell" if direction == "short" else "buy"
        tier = bursa_slippage_tier(bar["adv_value"])
        value = abs(units) * bar["close"]
        entry_cost = bursa_trade_cost(value, entry_side, tier)

        with db_session() as conn:
            conn.execute("""
                INSERT INTO paper_trades
                (idea_id, pair, direction, entry_price, units, signal,
                 entry_cost, opened_at, status)
                VALUES (?, ?, ?, ?, ?, ?, ?, datetime('now'), 'open')
            """, (idea_id, ticker, direction, bar["close"], units, signal, round(entry_cost, 2)))
            trade_id = conn.execute("SELECT last_insert_rowid() as id").fetchone()["id"]
            # Phase 6.3: reconciliation trail. Paper mode has no independent fill
            # source yet, so expected == actual by construction — the row exists
            # so the schema/trail is ready once real execution can diverge.
            conn.execute("""
                INSERT INTO paper_trade_reconciliation
                  (trade_id, side, expected_price, actual_price,
                   expected_cost, actual_cost, price_diff, cost_diff, clean)
                VALUES (?, ?, ?, ?, ?, ?, 0, 0, 1)
            """, (trade_id, entry_side, bar["close"], bar["close"],
                  round(entry_cost, 2), round(entry_cost, 2)))

        fill_note = f" [{fill['status']}: {fill['units']}/{fill['requested_units']} sh]" \
            if fill["status"] == "PARTIAL_FILL" else ""
        self.log_daemon("INFO",
                        f"Paper entry [{trade_id}] {direction} {ticker} {fill['units']} sh @ "
                        f"RM{bar['close']:.3f} (cost RM{entry_cost:.2f}, tier {tier}){fill_note}")
        return {"trade_id": trade_id, "ticker": ticker, "direction": direction,
                "entry_price": bar["close"], "units": units,
                "entry_cost": round(entry_cost, 2), "fill_date": bar["date"],
                "fill_status": fill["status"]}

    async def paper_exit(self, trade_id: int) -> dict:
        """Close a paper position at the latest cached close."""
        with db_session() as conn:
            row = conn.execute("SELECT * FROM paper_trades WHERE id=?", (trade_id,)).fetchone()
        if not row or row["status"] == "closed":
            return {"error": f"Trade {trade_id} not found or already closed"}

        ticker = row["pair"]
        bar = self._latest_bar(ticker)
        if not bar:
            return {"error": f"No price data for {ticker}"}

        units = row["units"]  # signed: negative for a short
        # A short CLOSES with a buy (to cover); a long closes with a sell.
        exit_side = "buy" if units < 0 else "sell"
        value = abs(units) * bar["close"]
        tier = bursa_slippage_tier(bar["adv_value"])
        exit_cost = bursa_trade_cost(value, exit_side, tier)
        # Signed units make long/short PnL fall out of the SAME formula: a
        # short's negative units correctly turn a price rise into a loss.
        # funding_paid (WS3) is subtracted like any other real cost/income —
        # 0 on Bursa, so this is a no-op there.
        pnl = ((bar["close"] - row["entry_price"]) * units
               - (row["entry_cost"] or 0) - exit_cost - (row["funding_paid"] or 0))

        with db_session() as conn:
            conn.execute("""
                UPDATE paper_trades
                SET exit_price=?, pnl=?, exit_cost=?, closed_at=datetime('now'), status='closed'
                WHERE id=?
            """, (bar["close"], round(pnl, 2), round(exit_cost, 2), trade_id))
            conn.execute("""
                INSERT INTO paper_trade_reconciliation
                  (trade_id, side, expected_price, actual_price,
                   expected_cost, actual_cost, price_diff, cost_diff, clean)
                VALUES (?, ?, ?, ?, ?, ?, 0, 0, 1)
            """, (trade_id, exit_side, bar["close"], bar["close"],
                  round(exit_cost, 2), round(exit_cost, 2)))

        self.log_daemon("INFO",
                        f"Paper exit [{trade_id}] {ticker} @ RM{bar['close']:.3f} "
                        f"PnL=RM{pnl:,.2f} (net of RM{(row['entry_cost'] or 0)+exit_cost:.2f} costs)")
        return {"trade_id": trade_id, "exit_price": bar["close"],
                "pnl": round(pnl, 2), "exit_cost": round(exit_cost, 2)}

    # ------------------------------------------------------------------
    # Daily mark-to-market and signal-driven position management
    # ------------------------------------------------------------------

    def mark_to_market(self, idea_id: int, interval: str = "1d") -> dict:
        """Record the current slot's NAV (cash + open position at latest close)
        into paper_equity — one row per calendar day for daily/weekly ideas
        (historical behavior), one per bar for sub-daily ideas."""
        open_trade = self._open_trade(idea_id)
        cash = self._idea_cash(idea_id)
        units, mark_price = 0.0, None
        nav = cash
        if open_trade:
            bar = self._latest_bar(open_trade["pair"], interval)
            if not bar:
                return {"error": f"No price data for {open_trade['pair']}"}
            units = open_trade["units"]
            mark_price = bar["close"]
            nav = cash + units * mark_price

        slot = equity_slot(interval)
        with db_session() as conn:
            conn.execute("""
                INSERT INTO paper_equity (idea_id, date, nav, cash, position_units, mark_price, marked_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(idea_id, date) DO UPDATE SET
                    nav=excluded.nav, cash=excluded.cash,
                    position_units=excluded.position_units, mark_price=excluded.mark_price,
                    marked_at=excluded.marked_at
            """, (idea_id, slot, round(nav, 2), round(cash, 2), units, mark_price,
                  datetime.utcnow().isoformat()))
        return {"idea_id": idea_id, "date": slot, "nav": round(nav, 2),
                "position_units": units, "mark_price": mark_price}

    async def daily_update(self, idea_id: int, ticker: str, params: dict,
                           interval: str = "1d") -> dict:
        """Run once per BAR per stage-4a idea (once per trading day for
        daily/weekly ideas — historical behavior; every 15m/1h/4h bar for
        sub-daily crypto ideas).

        Recomputes the strategy signal from the stored backtest params on fresh
        data, enters/exits accordingly, then marks the idea's NAV to market.

        WS3: current_signal may be -1/0/1 (DSL long/short trees). A held
        position whose direction no longer matches the signal (including a
        long<->short flip) is exited THIS cycle; a fresh entry in the new
        direction happens on the NEXT cycle if the signal still holds — avoids
        same-bar entry+exit and keeps cost/fill realism (mirrors the
        backtester's 1-bar signal delay).
        """
        from agents.backtest_engineer.backtest_engineer import BacktestEngineer

        df = self._data_engineer.fetch_prices(ticker, interval=interval, days=365,
                                              use_cache=True)
        if df is None or df.empty:
            return {"error": f"No price data for {ticker}"}

        # symbol threading lets funding_* leaves resolve their real-rate column
        signals = BacktestEngineer()._compute_signals(df, params, symbol=ticker)
        current_signal = int(signals.iloc[-1]) if len(signals) else 0

        action = "hold"
        open_trade = self._open_trade(idea_id)
        held_sign = 0
        if open_trade:
            held_sign = 1 if open_trade["units"] > 0 else -1

        if open_trade and current_signal != held_sign:
            result = await self.paper_exit(open_trade["id"])
            action = "exit" if "error" not in result else f"exit_failed: {result['error']}"
        elif not open_trade and current_signal != 0:
            direction = "long" if current_signal > 0 else "short"
            result = await self.paper_entry(idea_id, ticker, direction,
                                            signal=params.get("signal_type", ""))
            action = "entry" if "error" not in result else f"entry_failed: {result['error']}"

        # WS3: accrue funding once per cycle on whatever position is open now
        # (including one opened this same cycle — funding settlement doesn't
        # care how recently the position opened).
        still_open = self._open_trade(idea_id)
        if still_open and FUNDING_INTERVAL_HOURS:
            # Daily ideas keep the historical fixed 24h accrual per cycle;
            # sub-daily ideas accrue one bar's worth per mark.
            hours = 24.0 if bars_per_day(interval) <= 1.0 else 24.0 / bars_per_day(interval)
            self._accrue_funding(still_open["id"], still_open["units"],
                                 still_open["entry_price"], hours=hours,
                                 symbol=ticker)

        mtm = self.mark_to_market(idea_id, interval)
        return {"idea_id": idea_id, "ticker": ticker, "signal": current_signal,
                "action": action, "nav": mtm.get("nav")}

    # ------------------------------------------------------------------
    # Gate 4a evaluation — from the daily NAV series
    # ------------------------------------------------------------------

    async def evaluate_paper_performance(self, idea_id: int) -> dict:
        with db_session() as conn:
            series = conn.execute(
                "SELECT date, nav FROM paper_equity WHERE idea_id=? ORDER BY date",
                (idea_id,)
            ).fetchall()

        if len(series) < 2:
            return {"idea_id": idea_id, "status": "insufficient_equity_history",
                    "days_tracked": len(series)}

        navs = np.array([r["nav"] for r in series], dtype=float)
        first = datetime.fromisoformat(series[0]["date"])
        last = datetime.fromisoformat(series[-1]["date"])
        total_days = (last - first).days

        # Per-mark returns: one mark/day for daily ideas (annualize by trading
        # days — historical behavior), one mark/bar for sub-daily ideas
        # (annualize by that interval's bars/year).
        with db_session() as conn:
            _tf_row = conn.execute(
                "SELECT timeframe FROM alpha_ideas WHERE id=?", (idea_id,)).fetchone()
        _tf = (_tf_row["timeframe"] if _tf_row else None) or "1d"
        if bars_per_day(_tf) > 1.0:
            ann_factor = TRADING_DAYS_PER_YEAR * bars_per_day(_tf)
        else:
            ann_factor = TRADING_DAYS_PER_YEAR

        daily_returns = np.diff(navs) / navs[:-1]
        if len(daily_returns) > 1 and np.std(daily_returns) > 0:
            sharpe = (np.mean(daily_returns) / np.std(daily_returns)) * np.sqrt(ann_factor)
        else:
            sharpe = 0.0

        peak = np.maximum.accumulate(navs)
        dd = float(((peak - navs) / peak).max())

        with db_session() as conn:
            trades = conn.execute(
                "SELECT pnl FROM paper_trades WHERE idea_id=? AND status='closed'",
                (idea_id,)
            ).fetchall()
        pnls = [t["pnl"] for t in trades if t["pnl"] is not None]
        win_rate = (len([p for p in pnls if p > 0]) / len(pnls)) if pnls else 0.0
        completed_trades = len(pnls)

        # Phase 3.5: holding-cycle-aware duration. A flat 30-day floor is too
        # short for low-turnover (MEDIUM/LONG_TERM) strategies. The strategy has
        # "run long enough" once it clears its class-specific day floor OR has
        # accumulated enough completed round-trips — whichever comes first.
        with db_session() as conn:
            _bt = conn.execute(
                "SELECT holding_period_class FROM backtest_runs WHERE idea_id=? "
                "ORDER BY created_at DESC LIMIT 1", (idea_id,)
            ).fetchone()
        hp_class = (_bt["holding_period_class"] if _bt and _bt["holding_period_class"]
                    else "MEDIUM_TERM")
        min_days = GATE_CONFIG.stage4a_min_days_by_class.get(
            hp_class, GATE_CONFIG.stage4a_min_days)
        duration_pass = (
            total_days >= min_days
            or completed_trades >= GATE_CONFIG.stage4a_min_trades
        )

        gate4a_pass = (
            duration_pass
            and sharpe >= GATE_CONFIG.stage4a_min_sharpe
            and dd <= GATE_CONFIG.stage4a_max_drawdown
        )

        with db_session() as conn:
            conn.execute("""
                INSERT INTO pipeline_events (idea_id, stage, event_type, agent, notes)
                VALUES (?, 'stage4a', ?, 'PortfolioExecutor', ?)
            """, (idea_id, "advanced" if gate4a_pass else "monitoring",
                  f"class={hp_class} days={total_days}/{min_days} trades={completed_trades} "
                  f"sharpe={sharpe:.2f} dd={dd:.1%} win_rate={win_rate:.1%}"))

            if gate4a_pass:
                conn.execute("""
                    INSERT INTO gate_decisions (idea_id, gate, decision, decided_by, rationale)
                    VALUES (?, 'gate4a', 'approve', 'PortfolioExecutor', ?)
                """, (idea_id,
                      f"Paper trading ({hp_class}): {total_days} days / {completed_trades} trades, "
                      f"Sharpe={sharpe:.2f}, DD={dd:.1%}"))
                conn.execute("""
                    UPDATE alpha_ideas SET stage='stage4b', status='active', updated_at=datetime('now')
                    WHERE id=?
                """, (idea_id,))

        self.log_daemon(
            "INFO",
            f"Gate 4a [{idea_id}]: {'PASS' if gate4a_pass else 'monitoring'} "
            f"days={total_days} sharpe={sharpe:.2f} dd={dd:.1%}"
        )
        return {
            "idea_id": idea_id,
            "total_trades": len(pnls),
            "total_days": total_days,
            "days_tracked": len(series),
            "sharpe": round(float(sharpe), 3),
            "max_drawdown": round(dd, 4),
            "win_rate": round(win_rate, 4),
            "nav": round(float(navs[-1]), 2),
            "total_pnl": round(float(navs[-1] - PAPER_CAPITAL_MYR), 2),
            "holding_period_class": hp_class,
            "min_days_required": min_days,
            "duration_pass": duration_pass,
            "gate4a_pass": gate4a_pass,
        }

    # ------------------------------------------------------------------
    # Portfolio overview
    # ------------------------------------------------------------------

    async def portfolio_summary(self) -> dict:
        with db_session() as conn:
            paper_open = conn.execute(
                "SELECT COUNT(*) as n FROM paper_trades WHERE status='open'"
            ).fetchone()["n"]
            latest_navs = conn.execute("""
                SELECT pe.idea_id, pe.nav FROM paper_equity pe
                JOIN (SELECT idea_id, MAX(date) AS d FROM paper_equity GROUP BY idea_id) m
                  ON pe.idea_id = m.idea_id AND pe.date = m.d
            """).fetchall()
        total_nav = sum(r["nav"] for r in latest_navs)
        return {
            "paper_open_trades": paper_open,
            "ideas_tracked": len(latest_navs),
            "total_paper_nav": round(total_nav, 2),
            "capital_per_idea": PAPER_CAPITAL_MYR,
        }

    # ------------------------------------------------------------------
    # Live trading — Bursa broker integration pending (stage 4b)
    # ------------------------------------------------------------------

    async def live_entry(self, *args, **kwargs) -> dict:
        return {"error": "Live Bursa trading is not implemented (stage 4b pending broker integration)"}

    async def live_exit(self, *args, **kwargs) -> dict:
        return {"error": "Live Bursa trading is not implemented (stage 4b pending broker integration)"}

    # ------------------------------------------------------------------
    # run()
    # ------------------------------------------------------------------

    def run(self, task: dict) -> dict:
        action = task.get("action", "summary")
        if action == "summary":
            return asyncio.run(self.portfolio_summary())
        elif action == "paper_entry":
            return asyncio.run(self.paper_entry(
                task["idea_id"], task.get("ticker") or task.get("pair"),
                task.get("direction", "long"), task.get("signal", "")
            ))
        elif action == "paper_exit":
            return asyncio.run(self.paper_exit(task["trade_id"]))
        elif action == "daily_update":
            params = task.get("params") or {}
            if isinstance(params, str):
                params = json.loads(params)
            return asyncio.run(self.daily_update(
                task["idea_id"], task.get("ticker") or task.get("pair"), params
            ))
        elif action == "evaluate_paper":
            return asyncio.run(self.evaluate_paper_performance(task["idea_id"]))
        elif action in ("live_entry", "live_exit"):
            return {"error": "Live Bursa trading is not implemented (stage 4b pending broker integration)"}
        return {"error": f"Unknown action: {action}"}
