import sqlite3
import logging
from pathlib import Path
from contextlib import contextmanager
from datetime import datetime
from config.settings import DB_PATH

logger = logging.getLogger(__name__)

def get_connection(db_path: Path = DB_PATH) -> sqlite3.Connection:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path), timeout=30)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.execute("PRAGMA cache_size=10000")
    conn.execute("PRAGMA temp_store=MEMORY")
    conn.execute("PRAGMA foreign_keys=ON")
    return conn

@contextmanager
def db_session(db_path: Path = DB_PATH):
    conn = get_connection(db_path)
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()

def init_db(db_path: Path = DB_PATH):
    with db_session(db_path) as conn:
        conn.executescript("""
        CREATE TABLE IF NOT EXISTS kb_documents (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            slug TEXT UNIQUE NOT NULL,
            title TEXT,
            domain TEXT NOT NULL,
            content TEXT,
            summary TEXT,
            source_url TEXT,
            status TEXT DEFAULT 'unread',
            tags TEXT,
            created_at TEXT DEFAULT (datetime('now')),
            updated_at TEXT DEFAULT (datetime('now'))
        );
        CREATE TABLE IF NOT EXISTS kb_concepts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT UNIQUE NOT NULL,
            description TEXT,
            domain TEXT,
            count INTEGER DEFAULT 1,
            created_at TEXT DEFAULT (datetime('now'))
        );
        CREATE TABLE IF NOT EXISTS kb_links (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            source_id INTEGER REFERENCES kb_documents(id),
            target_id INTEGER REFERENCES kb_documents(id),
            relation TEXT,
            weight REAL DEFAULT 1.0
        );
        CREATE TABLE IF NOT EXISTS alpha_ideas (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            slug TEXT UNIQUE NOT NULL,
            title TEXT NOT NULL,
            hypothesis TEXT,
            ticker TEXT,
            timeframe TEXT,
            factor_formula TEXT,
            data_sources TEXT,
            stage TEXT DEFAULT 'gate0',
            status TEXT DEFAULT 'pending',
            novelty_score REAL,
            logic_score REAL,
            research_score REAL,
            backtest_sharpe REAL,
            backtest_dd REAL,
            created_at TEXT DEFAULT (datetime('now')),
            updated_at TEXT DEFAULT (datetime('now'))
        );
        CREATE TABLE IF NOT EXISTS pipeline_events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            idea_id INTEGER REFERENCES alpha_ideas(id),
            stage TEXT NOT NULL,
            event_type TEXT NOT NULL,
            agent TEXT,
            notes TEXT,
            created_at TEXT DEFAULT (datetime('now'))
        );
        CREATE TABLE IF NOT EXISTS gate_decisions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            idea_id INTEGER REFERENCES alpha_ideas(id),
            gate TEXT NOT NULL,
            decision TEXT NOT NULL,
            decided_by TEXT,
            rationale TEXT,
            created_at TEXT DEFAULT (datetime('now'))
        );
        CREATE TABLE IF NOT EXISTS backtest_runs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            idea_id INTEGER REFERENCES alpha_ideas(id),
            run_type TEXT,
            pair TEXT,
            timeframe TEXT,
            factor_formula TEXT,
            train_sharpe REAL,
            val_sharpe REAL,
            test_sharpe REAL,
            train_dd REAL,
            val_dd REAL,
            test_dd REAL,
            train_val_gap REAL,
            total_trades INTEGER,
            win_rate REAL,
            profit_factor REAL,
            params TEXT,
            result_data TEXT,
            passed INTEGER DEFAULT 0,
            created_at TEXT DEFAULT (datetime('now'))
        );
        CREATE TABLE IF NOT EXISTS paper_trades (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            idea_id INTEGER REFERENCES alpha_ideas(id),
            pair TEXT,
            direction TEXT,
            entry_price REAL,
            exit_price REAL,
            units REAL,
            pnl REAL,
            signal TEXT,
            opened_at TEXT,
            closed_at TEXT,
            status TEXT DEFAULT 'open'
        );
        CREATE TABLE IF NOT EXISTS live_trades (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            idea_id INTEGER REFERENCES alpha_ideas(id),
            oanda_order_id TEXT,
            oanda_trade_id TEXT,
            pair TEXT,
            direction TEXT,
            entry_price REAL,
            exit_price REAL,
            units REAL,
            pnl REAL,
            opened_at TEXT,
            closed_at TEXT,
            status TEXT DEFAULT 'open'
        );
        CREATE TABLE IF NOT EXISTS ai_usage (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            model TEXT NOT NULL,
            agent TEXT,
            input_tokens INTEGER,
            output_tokens INTEGER,
            cost_usd REAL,
            task TEXT,
            created_at TEXT DEFAULT (datetime('now'))
        );
        CREATE TABLE IF NOT EXISTS daemon_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            level TEXT NOT NULL,
            source TEXT,
            message TEXT,
            created_at TEXT DEFAULT (datetime('now'))
        );
        CREATE INDEX IF NOT EXISTS idx_kb_domain   ON kb_documents(domain);
        CREATE INDEX IF NOT EXISTS idx_ideas_stage ON alpha_ideas(stage);
        CREATE INDEX IF NOT EXISTS idx_logs_level  ON daemon_logs(level);
        """)
        # ── Safe post-schema migrations ──────────────────────────────────────
        # seeded=1 means AlphaSeedGenerator has already processed this document.
        # Guards against re-processing the same KB doc every daemon cycle.
        try:
            conn.execute("ALTER TABLE kb_documents ADD COLUMN seeded INTEGER DEFAULT 0")
            logger.info("Migration applied: kb_documents.seeded column added")
        except Exception:
            pass  # column already exists — safe to ignore

        # Minor fix 1: rename alpha_ideas.pair → ticker
        try:
            conn.execute("ALTER TABLE alpha_ideas RENAME COLUMN pair TO ticker")
            logger.info("Migration applied: alpha_ideas.pair renamed to ticker")
        except Exception:
            pass  # column already renamed — safe to ignore

        # Fix 2: rejection_reason on alpha_ideas
        try:
            conn.execute("ALTER TABLE alpha_ideas ADD COLUMN rejection_reason TEXT")
            logger.info("Migration applied: alpha_ideas.rejection_reason column added")
        except Exception:
            pass

        # Fix 2: rejection_patterns table (accumulates per-pattern counts)
        conn.execute("""
        CREATE TABLE IF NOT EXISTS rejection_patterns (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            factor_type TEXT,
            sector TEXT,
            reason_category TEXT,
            count INTEGER DEFAULT 1,
            last_seen TEXT,
            example_title TEXT,
            UNIQUE(factor_type, sector, reason_category)
        )
        """)

        # Fix 3: feasibility_score on alpha_ideas
        try:
            conn.execute("ALTER TABLE alpha_ideas ADD COLUMN feasibility_score REAL")
            logger.info("Migration applied: alpha_ideas.feasibility_score column added")
        except Exception:
            pass

        # price_based_proxy: OHLCV redirect for data-unavailability rejections
        try:
            conn.execute("ALTER TABLE alpha_ideas ADD COLUMN price_based_proxy TEXT")
            logger.info("Migration applied: alpha_ideas.price_based_proxy column added")
        except Exception:
            pass

        # Fix 5: needs_review + verification_note on backtest_runs
        try:
            conn.execute("ALTER TABLE backtest_runs ADD COLUMN needs_review INTEGER DEFAULT 0")
            logger.info("Migration applied: backtest_runs.needs_review column added")
        except Exception:
            pass
        try:
            conn.execute("ALTER TABLE backtest_runs ADD COLUMN verification_note TEXT")
            logger.info("Migration applied: backtest_runs.verification_note column added")
        except Exception:
            pass

        # Fix 4: holding_period_class on backtest_runs
        try:
            conn.execute("ALTER TABLE backtest_runs ADD COLUMN holding_period_class TEXT")
            logger.info("Migration applied: backtest_runs.holding_period_class column added")
        except Exception:
            pass

        # Fix 6: trade_count on backtest_runs
        try:
            conn.execute("ALTER TABLE backtest_runs ADD COLUMN trade_count INTEGER")
            logger.info("Migration applied: backtest_runs.trade_count column added")
        except Exception:
            pass

        # QC3: gross/net sharpe split
        for _col in ("sharpe_gross REAL", "sharpe_net REAL"):
            try:
                conn.execute(f"ALTER TABLE backtest_runs ADD COLUMN {_col}")
                logger.info(f"Migration applied: backtest_runs.{_col.split()[0]} added")
            except Exception:
                pass

        # QC2: walk-forward IS/OOS columns
        for _col in ("sharpe_is REAL", "sharpe_oos REAL", "oos_degradation REAL"):
            try:
                conn.execute(f"ALTER TABLE backtest_runs ADD COLUMN {_col}")
                logger.info(f"Migration applied: backtest_runs.{_col.split()[0]} added")
            except Exception:
                pass

        # QC5: regime stress test columns
        for _col in ("sharpe_low_vol REAL", "sharpe_mid_vol REAL",
                     "sharpe_high_vol REAL", "regimes_positive INTEGER"):
            try:
                conn.execute(f"ALTER TABLE backtest_runs ADD COLUMN {_col}")
                logger.info(f"Migration applied: backtest_runs.{_col.split()[0]} added")
            except Exception:
                pass

        # Sanity flags
        try:
            conn.execute("ALTER TABLE backtest_runs ADD COLUMN sanity_flags TEXT")
            logger.info("Migration applied: backtest_runs.sanity_flags added")
        except Exception:
            pass

        # Backtest Lab: verdict + verdict_reason on backtest_runs
        for _col in ("verdict TEXT", "verdict_reason TEXT"):
            try:
                conn.execute(f"ALTER TABLE backtest_runs ADD COLUMN {_col}")
                logger.info(f"Migration applied: backtest_runs.{_col.split()[0]} added")
            except Exception:
                pass

        # Part 1 schema fix: convenience alias columns + signal_type
        # net_sharpe / gross_sharpe / oos_sharpe are shorter aliases alongside
        # the existing sharpe_net / sharpe_gross / sharpe_oos columns.
        for _col in (
            "net_sharpe REAL",
            "gross_sharpe REAL",
            "oos_sharpe REAL",
            "max_dd REAL",
            "trades INTEGER",
            "hp_class TEXT",
            "signal_type TEXT",
            "n_trials INTEGER",
            "deflated_hurdle REAL",
            "benchmark_sharpe REAL",
            "excess_ann_return REAL",
            "ic_tstat_iid REAL",
        ):
            try:
                conn.execute(f"ALTER TABLE backtest_runs ADD COLUMN {_col}")
                logger.info(f"Migration applied: backtest_runs.{_col.split()[0]} added")
            except Exception:
                pass

        # Backtest Lab: equity curve / drawdown series cache
        conn.execute("""
            CREATE TABLE IF NOT EXISTS backtest_series (
                id           INTEGER PRIMARY KEY AUTOINCREMENT,
                idea_id      INTEGER NOT NULL,
                date         TEXT NOT NULL,
                strategy_pct REAL,
                benchmark_pct REAL,
                drawdown_pct  REAL,
                is_oos       INTEGER DEFAULT 0,
                created_at   TEXT DEFAULT (datetime('now')),
                UNIQUE(idea_id, date)
            )
        """)
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_bs_idea ON backtest_series(idea_id)"
        )

        # Daemon scheduler: persisted last-run timestamps so daily jobs catch up
        # after downtime instead of being silently skipped
        conn.execute("""
            CREATE TABLE IF NOT EXISTS job_state (
                job_name     TEXT PRIMARY KEY,
                last_run_utc TEXT NOT NULL
            )
        """)

        # Paper trading: daily NAV series per idea (mark-to-market)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS paper_equity (
                id             INTEGER PRIMARY KEY AUTOINCREMENT,
                idea_id        INTEGER NOT NULL REFERENCES alpha_ideas(id),
                date           TEXT NOT NULL,
                nav            REAL NOT NULL,
                cash           REAL,
                position_units REAL DEFAULT 0,
                mark_price     REAL,
                created_at     TEXT DEFAULT (datetime('now')),
                UNIQUE(idea_id, date)
            )
        """)
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_pe_idea ON paper_equity(idea_id)"
        )

        # Paper trades: per-side transaction costs (Bursa cost model)
        for _col in ("entry_cost REAL DEFAULT 0", "exit_cost REAL DEFAULT 0"):
            try:
                conn.execute(f"ALTER TABLE paper_trades ADD COLUMN {_col}")
                logger.info(f"Migration applied: paper_trades.{_col.split()[0]} added")
            except Exception:
                pass

        # CPO module: daily CPO spot price cache
        conn.execute("""
        CREATE TABLE IF NOT EXISTS cpo_prices (
            date       TEXT PRIMARY KEY,
            price      REAL NOT NULL,
            source     TEXT,
            created_at TEXT DEFAULT (datetime('now'))
        )
        """)
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_cpo_date ON cpo_prices(date)"
        )

        # EPF module: institutional ownership tracker
        conn.execute("""
        CREATE TABLE IF NOT EXISTS epf_holdings (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            ticker      TEXT NOT NULL,
            company     TEXT,
            date        TEXT NOT NULL,
            epf_pct     REAL NOT NULL,
            prev_pct    REAL,
            change_pct  REAL,
            direction   TEXT DEFAULT 'stable',
            source      TEXT,
            created_at  TEXT DEFAULT (datetime('now')),
            UNIQUE(ticker, date)
        )
        """)
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_epf_ticker ON epf_holdings(ticker)"
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_epf_date ON epf_holdings(date)"
        )

        # Analyst module: coverage initiation tracker
        conn.execute("""
        CREATE TABLE IF NOT EXISTS analyst_coverage_history (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            ticker          TEXT NOT NULL,
            company         TEXT,
            analyst_house   TEXT NOT NULL,
            report_type     TEXT NOT NULL,
            target_price    REAL,
            date            TEXT NOT NULL,
            is_first_coverage INTEGER DEFAULT 0,
            created_at      TEXT DEFAULT (datetime('now')),
            UNIQUE(ticker, analyst_house, date)
        )
        """)
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_cov_ticker "
            "ON analyst_coverage_history(ticker)"
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_cov_date "
            "ON analyst_coverage_history(date)"
        )
        conn.execute("""
        CREATE TABLE IF NOT EXISTS analyst_alerts (
            id            INTEGER PRIMARY KEY AUTOINCREMENT,
            ticker        TEXT NOT NULL,
            analyst_house TEXT,
            alert_type    TEXT NOT NULL,
            date          TEXT NOT NULL,
            idea_id       INTEGER,
            created_at    TEXT DEFAULT (datetime('now'))
        )
        """)
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_alert_ticker ON analyst_alerts(ticker)"
        )

        # KLSE Screener integration tables
        conn.execute("""
        CREATE TABLE IF NOT EXISTS fundamental_data (
            ticker        TEXT NOT NULL,
            name          TEXT,
            price         REAL,
            dy            REAL,
            dps_ttm       REAL,
            eps_ttm       REAL,
            pe            REAL,
            pb            REAL,
            roe           REAL,
            nta           REAL,
            rsi_14        REAL,
            stoch_14      REAL,
            market_cap_b  REAL,
            fetched_at    TEXT NOT NULL,
            PRIMARY KEY (ticker, fetched_at)
        )
        """)
        conn.execute("""
        CREATE TABLE IF NOT EXISTS quarterly_history (
            id             INTEGER PRIMARY KEY AUTOINCREMENT,
            ticker         TEXT NOT NULL,
            q_date         TEXT NOT NULL,
            quarter        INTEGER,
            financial_year TEXT,
            announced      TEXT,
            eps            REAL,
            dps            REAL,
            nta            REAL,
            revenue        TEXT,
            pl             TEXT,
            roe            REAL,
            qoq_pct        REAL,
            yoy_pct        REAL,
            UNIQUE(ticker, q_date)
        )
        """)
        conn.execute("""
        CREATE TABLE IF NOT EXISTS dividend_history (
            id             INTEGER PRIMARY KEY AUTOINCREMENT,
            ticker         TEXT NOT NULL,
            ex_date        TEXT NOT NULL,
            payment_date   TEXT,
            announced      TEXT,
            financial_year TEXT,
            subject        TEXT,
            dps_sen        REAL,
            dividend_type  TEXT,
            UNIQUE(ticker, ex_date)
        )
        """)
        conn.execute("""
        CREATE TABLE IF NOT EXISTS dividend_calendar (
            id           INTEGER PRIMARY KEY AUTOINCREMENT,
            ticker       TEXT,
            name         TEXT,
            ex_date      TEXT,
            payment_date TEXT,
            dps_sen      REAL,
            days_until   INTEGER,
            fetched_at   TEXT,
            UNIQUE(ticker, ex_date)
        )
        """)
        conn.execute("""
        CREATE TABLE IF NOT EXISTS screener_results (
            id               INTEGER PRIMARY KEY AUTOINCREMENT,
            screen_name      TEXT,
            ticker           TEXT,
            name             TEXT,
            dy               REAL,
            pe               REAL,
            pb               REAL,
            roe              REAL,
            rsi              REAL,
            price            REAL,
            matched_criteria TEXT,
            run_date         TEXT,
            idea_id          INTEGER
        )
        """)
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_fund_ticker ON fundamental_data(ticker)"
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_qh_ticker ON quarterly_history(ticker)"
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_dh_ticker ON dividend_history(ticker)"
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_dh_ex_date ON dividend_history(ex_date)"
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_dcal_ex_date ON dividend_calendar(ex_date)"
        )

        # Screener source tracking on alpha_ideas
        try:
            conn.execute("ALTER TABLE alpha_ideas ADD COLUMN screen_source TEXT")
            logger.info("Migration applied: alpha_ideas.screen_source column added")
        except Exception:
            pass  # already exists

        # Part 7: strategy_key on alpha_ideas
        try:
            conn.execute("ALTER TABLE alpha_ideas ADD COLUMN strategy_key TEXT DEFAULT 'other'")
            logger.info("Migration applied: alpha_ideas.strategy_key column added")
        except Exception:
            pass

        # ── Event-Driven Alpha Engine tables (2026-04-10) ─────────────────────
        conn.execute("""
        CREATE TABLE IF NOT EXISTS market_events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            event_id TEXT UNIQUE,
            source TEXT NOT NULL,
            ticker TEXT,
            company TEXT,
            event_type TEXT NOT NULL,
            headline TEXT NOT NULL,
            body TEXT,
            raw_url TEXT,
            published_at TEXT,
            detected_at TEXT DEFAULT (datetime('now')),
            confidence REAL,
            sentiment TEXT,
            magnitude TEXT,
            is_actionable INTEGER,
            historical_edge TEXT,
            action_taken TEXT,
            idea_id INTEGER,
            classified_at TEXT,
            affected_sectors TEXT,
            affected_tickers TEXT
        )
        """)
        conn.execute("""
        CREATE TABLE IF NOT EXISTS economic_calendar (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            event_name TEXT NOT NULL,
            event_type TEXT NOT NULL,
            scheduled_date TEXT NOT NULL,
            scheduled_time TEXT,
            country TEXT,
            importance TEXT,
            actual_value TEXT,
            forecast_value TEXT,
            previous_value TEXT,
            processed INTEGER DEFAULT 0,
            created_at TEXT DEFAULT (datetime('now')),
            UNIQUE(event_name, scheduled_date)
        )
        """)
        conn.execute("CREATE INDEX IF NOT EXISTS idx_events_ticker   ON market_events(ticker)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_events_type     ON market_events(event_type)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_events_detected ON market_events(detected_at)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_cal_date        ON economic_calendar(scheduled_date)")

        # ── Strategy Profiles (Part 3) ────────────────────────────────────────
        conn.execute("""
        CREATE TABLE IF NOT EXISTS strategy_profiles (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            strategy_key TEXT UNIQUE NOT NULL,
            name TEXT NOT NULL,
            strategy_class TEXT NOT NULL,
            angle TEXT NOT NULL,
            phenomenon TEXT NOT NULL,
            bursa_nuance TEXT NOT NULL,
            entry_condition TEXT NOT NULL,
            entry_universe TEXT NOT NULL,
            entry_rebalance TEXT NOT NULL,
            exit_type TEXT NOT NULL,
            exit_condition TEXT NOT NULL,
            exit_rationale TEXT NOT NULL,
            stop_loss_pct REAL,
            profit_target_pct REAL,
            min_hold_days INTEGER,
            max_hold_days INTEGER,
            hold_rationale TEXT NOT NULL,
            complexity TEXT NOT NULL,
            data_requirements TEXT NOT NULL,
            implementation_status TEXT NOT NULL,
            ic_benchmark TEXT,
            use_when TEXT,
            avoid_when TEXT,
            created_at TEXT DEFAULT (datetime('now')),
            updated_at TEXT DEFAULT (datetime('now'))
        )
        """)
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_sp_key ON strategy_profiles(strategy_key)"
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_sp_class ON strategy_profiles(strategy_class)"
        )

        # ── Stock Universe (Part 2) ───────────────────────────────────────────
        conn.execute("""
        CREATE TABLE IF NOT EXISTS stock_universe (
            ticker TEXT PRIMARY KEY,
            name TEXT,
            sector TEXT,
            index_member TEXT,
            market_cap_tier TEXT,
            added_at TEXT DEFAULT (datetime('now'))
        )
        """)
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_su_index ON stock_universe(index_member)"
        )

    logger.info(f"Database initialized at {db_path}")

if __name__ == "__main__":
    init_db()
    print(f"Database initialized at {DB_PATH}")
