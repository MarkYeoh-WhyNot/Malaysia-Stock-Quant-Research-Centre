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
)
from data.database import db_session

logger = logging.getLogger(__name__)


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

    def _latest_bar(self, ticker: str) -> dict | None:
        """Last completed daily bar from the parquet cache: close, date, ADV value."""
        df = self._data_engineer.fetch_prices(ticker, days=90, use_cache=True)
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
                    alloc_pct: float = PAPER_ALLOC_PCT) -> int:
        """Shares to buy: alloc_pct of NAV, rounded down to a 100-share board lot."""
        if price <= 0:
            return 0
        lots = int((nav * alloc_pct) / price / BURSA_BOARD_LOT)
        return lots * BURSA_BOARD_LOT

    # ------------------------------------------------------------------
    # Idea NAV (cash accounting per idea)
    # ------------------------------------------------------------------

    def _idea_cash(self, idea_id: int) -> float:
        """Current cash for an idea: starting capital ± all realized flows."""
        with db_session() as conn:
            rows = conn.execute(
                "SELECT entry_price, exit_price, units, status, "
                "       COALESCE(entry_cost,0) AS entry_cost, "
                "       COALESCE(exit_cost,0)  AS exit_cost "
                "FROM paper_trades WHERE idea_id=?", (idea_id,)
            ).fetchall()
        cash = PAPER_CAPITAL_MYR
        for r in rows:
            cash -= (r["entry_price"] or 0) * (r["units"] or 0) + r["entry_cost"]
            if r["status"] == "closed":
                cash += (r["exit_price"] or 0) * (r["units"] or 0) - r["exit_cost"]
        return cash

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
        """Open a paper position at the latest cached KLSE close (long-only)."""
        if direction != "long":
            return {"error": "Bursa paper trading is long-only"}
        if self._open_trade(idea_id):
            return {"error": f"Idea {idea_id} already has an open paper trade"}

        bar = self._latest_bar(ticker)
        if not bar:
            return {"error": f"No price data for {ticker}"}

        nav = self._idea_cash(idea_id)
        units = self.size_shares(nav, bar["close"])
        if units < BURSA_BOARD_LOT:
            return {"error": f"NAV RM{nav:,.0f} too small for one board lot of "
                             f"{ticker} @ RM{bar['close']:.2f}"}

        tier = bursa_slippage_tier(bar["adv_value"])
        value = units * bar["close"]
        entry_cost = bursa_trade_cost(value, "buy", tier)

        with db_session() as conn:
            conn.execute("""
                INSERT INTO paper_trades
                (idea_id, pair, direction, entry_price, units, signal,
                 entry_cost, opened_at, status)
                VALUES (?, ?, 'long', ?, ?, ?, ?, datetime('now'), 'open')
            """, (idea_id, ticker, bar["close"], units, signal, round(entry_cost, 2)))
            trade_id = conn.execute("SELECT last_insert_rowid() as id").fetchone()["id"]

        self.log_daemon("INFO",
                        f"Paper entry [{trade_id}] long {ticker} {units} sh @ RM{bar['close']:.3f} "
                        f"(cost RM{entry_cost:.2f}, tier {tier})")
        return {"trade_id": trade_id, "ticker": ticker, "direction": "long",
                "entry_price": bar["close"], "units": units,
                "entry_cost": round(entry_cost, 2), "fill_date": bar["date"]}

    async def paper_exit(self, trade_id: int) -> dict:
        """Close a paper position at the latest cached KLSE close."""
        with db_session() as conn:
            row = conn.execute("SELECT * FROM paper_trades WHERE id=?", (trade_id,)).fetchone()
        if not row or row["status"] == "closed":
            return {"error": f"Trade {trade_id} not found or already closed"}

        ticker = row["pair"]
        bar = self._latest_bar(ticker)
        if not bar:
            return {"error": f"No price data for {ticker}"}

        value = row["units"] * bar["close"]
        tier = bursa_slippage_tier(bar["adv_value"])
        exit_cost = bursa_trade_cost(value, "sell", tier)
        pnl = ((bar["close"] - row["entry_price"]) * row["units"]
               - (row["entry_cost"] or 0) - exit_cost)

        with db_session() as conn:
            conn.execute("""
                UPDATE paper_trades
                SET exit_price=?, pnl=?, exit_cost=?, closed_at=datetime('now'), status='closed'
                WHERE id=?
            """, (bar["close"], round(pnl, 2), round(exit_cost, 2), trade_id))

        self.log_daemon("INFO",
                        f"Paper exit [{trade_id}] {ticker} @ RM{bar['close']:.3f} "
                        f"PnL=RM{pnl:,.2f} (net of RM{(row['entry_cost'] or 0)+exit_cost:.2f} costs)")
        return {"trade_id": trade_id, "exit_price": bar["close"],
                "pnl": round(pnl, 2), "exit_cost": round(exit_cost, 2)}

    # ------------------------------------------------------------------
    # Daily mark-to-market and signal-driven position management
    # ------------------------------------------------------------------

    def mark_to_market(self, idea_id: int) -> dict:
        """Record today's NAV (cash + open position at latest close) into paper_equity."""
        open_trade = self._open_trade(idea_id)
        cash = self._idea_cash(idea_id)
        units, mark_price = 0.0, None
        nav = cash
        if open_trade:
            bar = self._latest_bar(open_trade["pair"])
            if not bar:
                return {"error": f"No price data for {open_trade['pair']}"}
            units = open_trade["units"]
            mark_price = bar["close"]
            nav = cash + units * mark_price

        today = datetime.utcnow().strftime("%Y-%m-%d")
        with db_session() as conn:
            conn.execute("""
                INSERT INTO paper_equity (idea_id, date, nav, cash, position_units, mark_price)
                VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(idea_id, date) DO UPDATE SET
                    nav=excluded.nav, cash=excluded.cash,
                    position_units=excluded.position_units, mark_price=excluded.mark_price
            """, (idea_id, today, round(nav, 2), round(cash, 2), units, mark_price))
        return {"idea_id": idea_id, "date": today, "nav": round(nav, 2),
                "position_units": units, "mark_price": mark_price}

    async def daily_update(self, idea_id: int, ticker: str, params: dict) -> dict:
        """Run once per trading day per stage-4a idea.

        Recomputes the strategy signal from the stored backtest params on fresh
        data, enters/exits accordingly, then marks the idea's NAV to market.
        """
        from agents.backtest_engineer.backtest_engineer import BacktestEngineer

        df = self._data_engineer.fetch_prices(ticker, days=365, use_cache=True)
        if df is None or df.empty:
            return {"error": f"No price data for {ticker}"}

        signals = BacktestEngineer()._compute_signals(df, params)
        current_signal = int(signals.iloc[-1]) if len(signals) else 0

        action = "hold"
        open_trade = self._open_trade(idea_id)
        if current_signal == 1 and not open_trade:
            result = await self.paper_entry(idea_id, ticker, "long",
                                            signal=params.get("signal_type", ""))
            action = "entry" if "error" not in result else f"entry_failed: {result['error']}"
        elif current_signal == 0 and open_trade:
            result = await self.paper_exit(open_trade["id"])
            action = "exit" if "error" not in result else f"exit_failed: {result['error']}"

        mtm = self.mark_to_market(idea_id)
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

        daily_returns = np.diff(navs) / navs[:-1]
        if len(daily_returns) > 1 and np.std(daily_returns) > 0:
            sharpe = (np.mean(daily_returns) / np.std(daily_returns)) * np.sqrt(TRADING_DAYS_PER_YEAR)
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
