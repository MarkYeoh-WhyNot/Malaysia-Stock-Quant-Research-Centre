import json
import logging
import numpy as np
import pandas as pd
from agents.base_agent import BaseAgent
from config.settings import MODEL_FAST, MODEL_MAIN, GATE_CONFIG, KLCI_BY_SYMBOL
from data.database import db_session
from data.yahoo.client import get_historical_data, BARS_PER_YEAR

logger = logging.getLogger(__name__)

SYSTEM = """You are a quantitative backtesting engineer specialising in Bursa Malaysia equities.
Parse equity strategy descriptions into structured signal parameters for vectorised backtesting.
Output only valid JSON."""


class BacktestEngineer(BaseAgent):
    name = "BacktestEngineer"
    description = "Vectorised KLSE equity backtesting, Gate 2/3 evaluation (Stage 2-3)"
    default_model = MODEL_MAIN

    # ── Data fetch ────────────────────────────────────────────────────────────

    def _fetch_prices(self, symbol: str, interval: str = "1d", days: int = 1825) -> pd.DataFrame:
        """Fetch via Yahoo Finance; use data cache if available."""
        try:
            from agents.data_engineer.data_engineer import DataEngineer
            de = DataEngineer()
            return de.fetch_prices(symbol, interval, days, use_cache=True)
        except Exception:
            return get_historical_data(symbol, interval=interval, days=days)

    # ── Factor parsing ────────────────────────────────────────────────────────

    def _parse_factor(self, factor_formula: str, title: str, hypothesis: str) -> dict:
        prompt = f"""Parse this Bursa Malaysia equity strategy into structured signal parameters.

Factor formula: {factor_formula}
Strategy title: {title}
Hypothesis: {hypothesis}

Return JSON:
{{
  "signal_type": "sma_crossover|ema_crossover|rsi|momentum|bollinger|macd|value|quality|volume_breakout",
  "fast_period": 20,
  "slow_period": 50,
  "rsi_period": 14,
  "rsi_oversold": 35,
  "rsi_overbought": 65,
  "bb_period": 20,
  "bb_std": 2.0,
  "momentum_period": 20,
  "macd_signal_period": 9,
  "volume_ma_period": 20,
  "volume_threshold": 1.5,
  "stop_loss_pct": 0.08,
  "take_profit_pct": 0.15,
  "long_only": true,
  "notes": "brief signal description"
}}"""
        result = self.call_claude_json(
            SYSTEM,
            [{"role": "user", "content": prompt}],
            model=MODEL_FAST,
            task_label="parse_factor",
        )
        return result if isinstance(result, dict) else {}

    # ── Signal computation ────────────────────────────────────────────────────

    @staticmethod
    def _rsi(series: pd.Series, period: int = 14) -> pd.Series:
        delta = series.diff()
        gain  = delta.clip(lower=0).rolling(period).mean()
        loss  = (-delta.clip(upper=0)).rolling(period).mean()
        rs    = gain / loss.replace(0, 1e-9)
        return 100 - (100 / (1 + rs))

    def _compute_signals(self, df: pd.DataFrame, params: dict) -> pd.Series:
        close  = df["close"]
        stype  = params.get("signal_type", "momentum")
        # KLSE equities: long-only by default (short-selling is restricted to
        # designated securities only)
        long_only = bool(params.get("long_only", True))

        if stype in ("sma_crossover", "ema_crossover"):
            fp = int(params.get("fast_period", 20))
            sp = int(params.get("slow_period", 50))
            if stype == "sma_crossover":
                fast_ma, slow_ma = close.rolling(fp).mean(), close.rolling(sp).mean()
            else:
                fast_ma = close.ewm(span=fp, adjust=False).mean()
                slow_ma = close.ewm(span=sp, adjust=False).mean()
            raw = np.where(fast_ma > slow_ma, 1, 0 if long_only else -1)

        elif stype == "rsi":
            rsi      = self._rsi(close, int(params.get("rsi_period", 14)))
            oversold = params.get("rsi_oversold", 35)
            overbought = params.get("rsi_overbought", 65)
            # Long when oversold, exit (or short) when overbought
            long_sig  = (rsi < oversold).astype(int)
            short_sig = (rsi > overbought).astype(int)
            raw = np.where(long_sig, 1, np.where(short_sig, 0 if long_only else -1, np.nan))
            # Forward-fill to hold position
            raw = pd.Series(raw, index=df.index).ffill().fillna(0).values

        elif stype == "bollinger":
            period   = int(params.get("bb_period", 20))
            std_mult = float(params.get("bb_std", 2.0))
            mid   = close.rolling(period).mean()
            std   = close.rolling(period).std()
            upper = mid + std_mult * std
            lower = mid - std_mult * std
            raw   = np.where(close < lower, 1, np.where(close > upper, 0 if long_only else -1, np.nan))
            raw   = pd.Series(raw, index=df.index).ffill().fillna(0).values

        elif stype == "macd":
            fp    = int(params.get("fast_period", 12))
            sp    = int(params.get("slow_period", 26))
            sig_p = int(params.get("macd_signal_period", 9))
            macd_line   = close.ewm(span=fp, adjust=False).mean() - close.ewm(span=sp, adjust=False).mean()
            signal_line = macd_line.ewm(span=sig_p, adjust=False).mean()
            raw = np.where(macd_line > signal_line, 1, 0 if long_only else -1)

        elif stype == "volume_breakout":
            vol_ma    = df["volume"].rolling(int(params.get("volume_ma_period", 20))).mean()
            vol_thresh = float(params.get("volume_threshold", 1.5))
            ret_1d    = close.pct_change()
            # Long on high-volume up days; exit on high-volume down days
            raw = np.where(
                (df["volume"] > vol_thresh * vol_ma) & (ret_1d > 0), 1,
                np.where((df["volume"] > vol_thresh * vol_ma) & (ret_1d < 0), 0 if long_only else -1, np.nan)
            )
            raw = pd.Series(raw, index=df.index).ffill().fillna(0).values

        else:  # momentum / default
            period = int(params.get("momentum_period", 20))
            ret    = close.pct_change(period)
            raw    = np.where(ret > 0, 1, 0 if long_only else -1)

        return pd.Series(raw, index=df.index, dtype=float)

    # ── Performance metrics ───────────────────────────────────────────────────

    def _compute_performance(self, df: pd.DataFrame, signals: pd.Series, interval: str) -> dict:
        close = df["close"].values
        sig   = signals.fillna(0).values

        # Execute at next-bar open (shift signal by 1)
        sig_shifted     = np.roll(sig, 1)
        sig_shifted[0]  = 0.0

        bar_returns     = np.diff(close) / np.where(close[:-1] != 0, close[:-1], 1e-9)
        # Bursa transaction cost: ~0.1% brokerage + 0.03% stamp duty each way
        TRANSACTION_COST = 0.0013
        signal_changes  = np.abs(np.diff(sig_shifted))
        net_returns     = sig_shifted[1:] * bar_returns - signal_changes * TRANSACTION_COST

        n = len(net_returns)
        if n < 20 or np.std(net_returns) < 1e-10:
            return {"sharpe": 0.0, "max_dd": 1.0, "win_rate": 0.0, "profit_factor": 0.0, "total_trades": 0}

        # Annualised Sharpe
        ann   = BARS_PER_YEAR.get(interval, 252)
        sharpe = (np.mean(net_returns) / np.std(net_returns)) * np.sqrt(ann)

        # Max drawdown
        cum   = np.cumprod(1 + np.clip(net_returns, -0.5, 0.5))
        peak  = np.maximum.accumulate(cum)
        dd    = (peak - cum) / np.where(peak != 0, peak, 1e-9)
        max_dd = float(dd.max())

        # Win rate and profit factor
        pos  = net_returns[net_returns > 0]
        neg  = net_returns[net_returns < 0]
        nz   = net_returns[net_returns != 0]
        win_rate = len(pos) / max(len(nz), 1)
        gross_win  = float(pos.sum()) if len(pos) > 0 else 0.0
        gross_loss = float(abs(neg.sum())) if len(neg) > 0 else 1e-9
        profit_factor = gross_win / gross_loss

        total_trades = int(np.sum(np.abs(np.diff(sig_shifted)) > 0))

        return {
            "sharpe":        round(float(sharpe), 3),
            "max_dd":        round(max_dd, 4),
            "win_rate":      round(win_rate, 4),
            "profit_factor": round(float(profit_factor), 3),
            "total_trades":  total_trades,
            "ann_return":    round(float(np.mean(net_returns)) * ann, 4),
        }

    # ── Train / val / test split ─────────────────────────────────────────────

    @staticmethod
    def _split(df: pd.DataFrame) -> tuple:
        n = len(df)
        t = int(n * GATE_CONFIG.stage3_data_split_train)
        v = int(n * GATE_CONFIG.stage3_data_split_val)
        return df.iloc[:t], df.iloc[t:t + v], df.iloc[t + v:]

    # ── Main backtest pipeline ────────────────────────────────────────────────

    def backtest_idea(self, idea_id: int) -> dict:
        with db_session() as conn:
            row = conn.execute("SELECT * FROM alpha_ideas WHERE id=?", (idea_id,)).fetchone()
        if not row:
            return {"error": f"Idea {idea_id} not found"}

        symbol   = row["pair"] or "1155.KL"
        interval = row["timeframe"] or "1d"
        stock    = KLCI_BY_SYMBOL.get(symbol, {})

        self.log_daemon("INFO", f"Backtesting [{idea_id}] {row['title']} — {symbol} {interval}")

        # Parse factor formula
        params = self._parse_factor(
            row["factor_formula"] or "",
            row["title"],
            row["hypothesis"] or "",
        )
        if not params or "error" in params:
            params = {"signal_type": "momentum", "momentum_period": 20, "long_only": True}

        # Fetch 5 years of daily data for robust train/val/test split
        df = self._fetch_prices(symbol, interval, days=1825)
        if df.empty or len(df) < 150:
            self.log_daemon("WARN", f"Insufficient data [{idea_id}]: {len(df)} bars for {symbol}")
            return {"error": "Insufficient historical data", "idea_id": idea_id, "symbol": symbol}

        train_df, val_df, test_df = self._split(df)

        results = {}
        for split_name, split_df in (("train", train_df), ("val", val_df), ("test", test_df)):
            if len(split_df) < 40:
                results[split_name] = {
                    "sharpe": 0.0, "max_dd": 1.0, "win_rate": 0.0,
                    "profit_factor": 0.0, "total_trades": 0, "ann_return": 0.0,
                }
                continue
            sig = self._compute_signals(split_df, params)
            results[split_name] = self._compute_performance(split_df, sig, interval)

        train_r = results["train"]
        val_r   = results["val"]
        test_r  = results["test"]
        train_val_gap = abs(train_r["sharpe"] - val_r["sharpe"])

        # Gate 2 — in-sample quality
        gate2_pass = (
            train_r["sharpe"] >= GATE_CONFIG.stage3_min_sharpe
            and val_r["sharpe"]   >= GATE_CONFIG.stage3_min_sharpe
            and train_r["max_dd"] <= GATE_CONFIG.stage3_max_drawdown
            and val_r["max_dd"]   <= GATE_CONFIG.stage3_max_drawdown
            and train_val_gap     <= GATE_CONFIG.stage3_max_train_val_gap
        )

        # Gate 3 — out-of-sample test
        gate3_pass = (
            gate2_pass
            and test_r["sharpe"]  >= GATE_CONFIG.stage3_min_test_sharpe
            and test_r["max_dd"]  <= GATE_CONFIG.stage3_max_drawdown
        )

        overall_pass = gate3_pass

        with db_session() as conn:
            conn.execute("""
                INSERT INTO backtest_runs
                  (idea_id, run_type, pair, timeframe, factor_formula,
                   train_sharpe, val_sharpe, test_sharpe,
                   train_dd, val_dd, test_dd,
                   train_val_gap, total_trades, win_rate, profit_factor,
                   params, result_data, passed)
                VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
            """, (
                idea_id, "klse_daily", symbol, interval, row["factor_formula"],
                train_r["sharpe"], val_r["sharpe"], test_r["sharpe"],
                train_r["max_dd"], val_r["max_dd"],  test_r["max_dd"],
                round(train_val_gap, 3), test_r["total_trades"],
                test_r["win_rate"], test_r["profit_factor"],
                json.dumps(params), json.dumps(results),
                1 if overall_pass else 0,
            ))

            new_stage  = "stage3" if overall_pass else "stage2"
            new_status = "active"  if overall_pass else "rejected"
            conn.execute("""
                UPDATE alpha_ideas
                SET backtest_sharpe=?, backtest_dd=?, stage=?, status=?, updated_at=datetime('now')
                WHERE id=?
            """, (test_r["sharpe"], test_r["max_dd"], new_stage, new_status, idea_id))

            conn.execute("""
                INSERT INTO pipeline_events (idea_id, stage, event_type, agent, notes)
                VALUES (?, 'stage2', ?, 'BacktestEngineer', ?)
            """, (idea_id,
                  "advanced" if overall_pass else "rejected",
                  f"Train={train_r['sharpe']:.2f} Val={val_r['sharpe']:.2f} "
                  f"Test={test_r['sharpe']:.2f} DD={test_r['max_dd']:.1%} "
                  f"AnnRet={test_r.get('ann_return',0):.1%}"))

            conn.execute("""
                INSERT INTO gate_decisions (idea_id, gate, decision, decided_by, rationale)
                VALUES (?, 'gate2_3', ?, 'BacktestEngineer', ?)
            """, (idea_id,
                  "approve" if overall_pass else "reject",
                  f"G2={'PASS' if gate2_pass else 'FAIL'} G3={'PASS' if gate3_pass else 'FAIL'} "
                  f"gap={train_val_gap:.2f}"))

        self.log_daemon(
            "INFO" if overall_pass else "WARN",
            f"Backtest [{idea_id}] {symbol} {'PASSED' if overall_pass else 'FAILED'} "
            f"train={train_r['sharpe']} val={val_r['sharpe']} test={test_r['sharpe']}",
        )
        return {
            "idea_id":      idea_id,
            "symbol":       symbol,
            "company":      stock.get("name", symbol),
            "interval":     interval,
            "gate2_pass":   gate2_pass,
            "gate3_pass":   gate3_pass,
            "train":        train_r,
            "val":          val_r,
            "test":         test_r,
            "train_val_gap": round(train_val_gap, 3),
            "params":       params,
            "bars_total":   len(df),
        }

    # ── run() ─────────────────────────────────────────────────────────────────

    def run(self, task: dict) -> dict:
        action = task.get("action", "backtest")
        if action == "backtest":
            idea_id = task.get("idea_id")
            if not idea_id:
                return {"error": "idea_id required"}
            return self.backtest_idea(idea_id)
        return {"error": f"Unknown action: {action}"}
