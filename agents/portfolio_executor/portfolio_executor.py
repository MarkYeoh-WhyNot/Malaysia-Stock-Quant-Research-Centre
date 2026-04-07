import asyncio
import json
import logging
from datetime import datetime
from agents.base_agent import BaseAgent
from config.settings import MODEL_FAST, GATE_CONFIG, OANDA_ENVIRONMENT
from data.database import db_session
from data.oanda.client import OANDAClient

logger = logging.getLogger(__name__)

# Maximum single-position risk as % of NAV
MAX_RISK_PER_TRADE_PCT = 0.01   # 1%
DEFAULT_STOP_PIPS       = 25


class PortfolioExecutor(BaseAgent):
    name = "PortfolioExecutor"
    description = "Live and paper trade execution, position sizing, and exit management"
    default_model = MODEL_FAST

    # ------------------------------------------------------------------
    # Position sizing
    # ------------------------------------------------------------------

    @staticmethod
    def _pip_size(pair: str) -> float:
        return 0.01 if "JPY" in pair else 0.0001

    def size_units(self, nav: float, stop_pips: int, pair: str,
                   risk_pct: float = MAX_RISK_PER_TRADE_PCT) -> int:
        pip = self._pip_size(pair)
        risk_amount = nav * risk_pct
        units = int(risk_amount / (stop_pips * pip))
        return max(units, 1000)

    # ------------------------------------------------------------------
    # Paper trading
    # ------------------------------------------------------------------

    async def paper_entry(self, idea_id: int, pair: str, direction: str,
                          signal: str = "", stop_pips: int = DEFAULT_STOP_PIPS) -> dict:
        async with OANDAClient() as client:
            price_data = await client.get_latest_price(pair)
            account = await client.get_account_summary()

        entry_price = price_data["ask"] if direction == "long" else price_data["bid"]
        nav = account["nav"]
        units = self.size_units(nav, stop_pips, pair)

        with db_session() as conn:
            conn.execute("""
                INSERT INTO paper_trades
                (idea_id, pair, direction, entry_price, units, signal, opened_at, status)
                VALUES (?, ?, ?, ?, ?, ?, datetime('now'), 'open')
            """, (idea_id, pair, direction, entry_price, units, signal))
            trade_id = conn.execute("SELECT last_insert_rowid() as id").fetchone()["id"]

        self.log_daemon("INFO", f"Paper entry [{trade_id}] {direction} {pair} @ {entry_price} ({units} units)")
        return {"trade_id": trade_id, "pair": pair, "direction": direction,
                "entry_price": entry_price, "units": units}

    async def paper_exit(self, trade_id: int) -> dict:
        with db_session() as conn:
            row = conn.execute("SELECT * FROM paper_trades WHERE id=?", (trade_id,)).fetchone()
        if not row or row["status"] == "closed":
            return {"error": f"Trade {trade_id} not found or already closed"}

        async with OANDAClient() as client:
            price_data = await client.get_latest_price(row["pair"])

        exit_price = price_data["bid"] if row["direction"] == "long" else price_data["ask"]
        pip = self._pip_size(row["pair"])
        direction_mult = 1 if row["direction"] == "long" else -1
        pnl = direction_mult * (exit_price - row["entry_price"]) * row["units"]

        with db_session() as conn:
            conn.execute("""
                UPDATE paper_trades
                SET exit_price=?, pnl=?, closed_at=datetime('now'), status='closed'
                WHERE id=?
            """, (exit_price, round(pnl, 4), trade_id))

        self.log_daemon("INFO", f"Paper exit [{trade_id}] @ {exit_price} PnL={pnl:.4f}")
        return {"trade_id": trade_id, "exit_price": exit_price, "pnl": round(pnl, 4)}

    # ------------------------------------------------------------------
    # Live trading (OANDA)
    # ------------------------------------------------------------------

    async def live_entry(self, idea_id: int, pair: str, direction: str,
                         stop_pips: int = DEFAULT_STOP_PIPS) -> dict:
        if OANDA_ENVIRONMENT != "live":
            self.log_daemon("WARN", "live_entry called but OANDA_ENVIRONMENT is not 'live' — aborting")
            return {"error": "OANDA_ENVIRONMENT must be 'live' for live orders"}

        async with OANDAClient() as client:
            account = await client.get_account_summary()
            nav = account["nav"]
            units = self.size_units(nav, stop_pips, pair)
            signed_units = units if direction == "long" else -units
            order_result = await client.place_market_order(
                pair, signed_units, stop_loss_pips=stop_pips
            )

        oanda_order_id = order_result.get("orderFillTransaction", {}).get("id", "")
        oanda_trade_id = order_result.get("orderFillTransaction", {}).get("tradeOpened", {}).get("tradeID", "")
        fill_price = float(order_result.get("orderFillTransaction", {}).get("price", 0))

        with db_session() as conn:
            conn.execute("""
                INSERT INTO live_trades
                (idea_id, oanda_order_id, oanda_trade_id, pair, direction, entry_price, units, opened_at, status)
                VALUES (?, ?, ?, ?, ?, ?, ?, datetime('now'), 'open')
            """, (idea_id, oanda_order_id, oanda_trade_id, pair, direction, fill_price, units))

        self.log_daemon("INFO", f"Live order [{oanda_trade_id}] {direction} {pair} @ {fill_price} ({units} units)")
        return {"oanda_trade_id": oanda_trade_id, "pair": pair, "direction": direction,
                "fill_price": fill_price, "units": units}

    async def live_exit(self, trade_id: int) -> dict:
        with db_session() as conn:
            row = conn.execute("SELECT * FROM live_trades WHERE id=?", (trade_id,)).fetchone()
        if not row:
            return {"error": f"Live trade {trade_id} not found"}
        if row["status"] == "closed":
            return {"error": "Already closed"}

        async with OANDAClient() as client:
            result = await client.close_trade(row["oanda_trade_id"])

        close_tx = result.get("orderFillTransaction", {})
        exit_price = float(close_tx.get("price", 0))
        realized_pnl = float(close_tx.get("pl", 0))

        with db_session() as conn:
            conn.execute("""
                UPDATE live_trades
                SET exit_price=?, pnl=?, closed_at=datetime('now'), status='closed'
                WHERE id=?
            """, (exit_price, realized_pnl, trade_id))

        self.log_daemon("INFO", f"Live exit [{trade_id}] @ {exit_price} PnL={realized_pnl}")
        return {"trade_id": trade_id, "exit_price": exit_price, "pnl": realized_pnl}

    # ------------------------------------------------------------------
    # Paper trade evaluation for Gate 4a
    # ------------------------------------------------------------------

    async def evaluate_paper_performance(self, idea_id: int) -> dict:
        with db_session() as conn:
            trades = conn.execute("""
                SELECT * FROM paper_trades WHERE idea_id=? AND status='closed'
                ORDER BY closed_at
            """, (idea_id,)).fetchall()
            idea = conn.execute("SELECT * FROM alpha_ideas WHERE id=?", (idea_id,)).fetchone()

        if not trades:
            return {"idea_id": idea_id, "status": "no_closed_trades"}

        import numpy as np
        pnls = [t["pnl"] for t in trades]
        cum_pnl = np.cumsum(pnls)
        peak = np.maximum.accumulate(cum_pnl)
        dd = ((peak - cum_pnl) / (np.abs(peak) + 1e-9)).max()

        total_days = 0
        if trades[0]["opened_at"] and trades[-1]["closed_at"]:
            try:
                start = datetime.fromisoformat(trades[0]["opened_at"])
                end = datetime.fromisoformat(trades[-1]["closed_at"])
                total_days = (end - start).days
            except Exception:
                pass

        positive = [p for p in pnls if p > 0]
        win_rate = len(positive) / max(len(pnls), 1)

        daily_returns = [sum(pnls[i:i+1]) for i in range(len(pnls))]
        if len(daily_returns) > 1 and np.std(daily_returns) > 0:
            sharpe = (np.mean(daily_returns) / np.std(daily_returns)) * np.sqrt(252)
        else:
            sharpe = 0.0

        gate4a_pass = (
            total_days >= GATE_CONFIG.stage4a_min_days
            and sharpe >= GATE_CONFIG.stage4a_min_sharpe
            and dd <= GATE_CONFIG.stage4a_max_drawdown
        )

        with db_session() as conn:
            conn.execute("""
                INSERT INTO pipeline_events (idea_id, stage, event_type, agent, notes)
                VALUES (?, 'stage4a', ?, 'PortfolioExecutor', ?)
            """, (idea_id, "advanced" if gate4a_pass else "rejected",
                  f"days={total_days} sharpe={sharpe:.2f} dd={dd:.1%} win_rate={win_rate:.1%}"))

            conn.execute("""
                INSERT INTO gate_decisions (idea_id, gate, decision, decided_by, rationale)
                VALUES (?, 'gate4a', ?, 'PortfolioExecutor', ?)
            """, (idea_id, "approve" if gate4a_pass else "reject",
                  f"Paper trading: {total_days} days, Sharpe={sharpe:.2f}, DD={dd:.1%}"))

            if gate4a_pass:
                conn.execute("""
                    UPDATE alpha_ideas SET stage='stage4b', status='active', updated_at=datetime('now')
                    WHERE id=?
                """, (idea_id,))

        self.log_daemon(
            "INFO" if gate4a_pass else "WARN",
            f"Gate 4a [{idea_id}]: {'PASS' if gate4a_pass else 'FAIL'} "
            f"days={total_days} sharpe={sharpe:.2f} dd={dd:.1%}"
        )
        return {
            "idea_id": idea_id,
            "total_trades": len(pnls),
            "total_days": total_days,
            "sharpe": round(float(sharpe), 3),
            "max_drawdown": round(float(dd), 4),
            "win_rate": round(win_rate, 4),
            "total_pnl": round(float(sum(pnls)), 4),
            "gate4a_pass": gate4a_pass,
        }

    # ------------------------------------------------------------------
    # Portfolio overview
    # ------------------------------------------------------------------

    async def portfolio_summary(self) -> dict:
        async with OANDAClient() as client:
            account = await client.get_account_summary()
            open_trades = await client.get_open_trades()

        with db_session() as conn:
            paper_open = conn.execute(
                "SELECT COUNT(*) as n FROM paper_trades WHERE status='open'"
            ).fetchone()["n"]
            live_open = conn.execute(
                "SELECT COUNT(*) as n FROM live_trades WHERE status='open'"
            ).fetchone()["n"]

        return {
            "account": account,
            "oanda_open_trades": len(open_trades),
            "paper_open_trades": paper_open,
            "live_open_trades": live_open,
        }

    # ------------------------------------------------------------------
    # run()
    # ------------------------------------------------------------------

    def run(self, task: dict) -> dict:
        action = task.get("action", "summary")
        if action == "summary":
            return asyncio.run(self.portfolio_summary())
        elif action == "paper_entry":
            return asyncio.run(self.paper_entry(
                task["idea_id"], task["pair"], task["direction"],
                task.get("signal", ""), int(task.get("stop_pips", DEFAULT_STOP_PIPS))
            ))
        elif action == "paper_exit":
            return asyncio.run(self.paper_exit(task["trade_id"]))
        elif action == "live_entry":
            return asyncio.run(self.live_entry(
                task["idea_id"], task["pair"], task["direction"],
                int(task.get("stop_pips", DEFAULT_STOP_PIPS))
            ))
        elif action == "live_exit":
            return asyncio.run(self.live_exit(task["trade_id"]))
        elif action == "evaluate_paper":
            return asyncio.run(self.evaluate_paper_performance(task["idea_id"]))
        return {"error": f"Unknown action: {action}"}
