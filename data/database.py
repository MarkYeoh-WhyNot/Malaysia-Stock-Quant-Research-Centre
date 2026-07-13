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
        CREATE TABLE IF NOT EXISTS protocol_metrics (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            symbol TEXT NOT NULL,
            protocol_name TEXT,
            tvl REAL,
            tvl_rank INTEGER,
            fees_24h REAL,
            revenue_24h REAL,
            fetched_at TEXT DEFAULT (datetime('now'))
        );
        CREATE INDEX IF NOT EXISTS idx_kb_domain   ON kb_documents(domain);
        CREATE INDEX IF NOT EXISTS idx_ideas_stage ON alpha_ideas(stage);
        CREATE INDEX IF NOT EXISTS idx_logs_level  ON daemon_logs(level);
        CREATE INDEX IF NOT EXISTS idx_protocol_symbol ON protocol_metrics(symbol);
        CREATE INDEX IF NOT EXISTS idx_protocol_fetched ON protocol_metrics(fetched_at);
        """)
        # ── Safe post-schema migrations ──────────────────────────────────────
        # seeded=1 means AlphaSeedGenerator has already processed this document.
        # Guards against re-processing the same KB doc every daemon cycle.
        try:
            conn.execute("ALTER TABLE kb_documents ADD COLUMN seeded INTEGER DEFAULT 0")
            logger.info("Migration applied: kb_documents.seeded column added")
        except Exception:
            pass  # column already exists — safe to ignore

        # content_hash: dedup key — same article re-ingested on a different
        # day gets a new date-prefixed slug but an identical hash.
        try:
            conn.execute("ALTER TABLE kb_documents ADD COLUMN content_hash TEXT")
            logger.info("Migration applied: kb_documents.content_hash column added")
        except Exception:
            pass

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

        # Semantic dedup + proxy-spawn cap
        # signal_signature: "txt:<hash>" at save time (normalized formula+ticker),
        # upgraded to "dsl:<hash>" at parse time (canonical condition tree).
        # parent_idea_id: which idea spawned this one (price proxies) — enforces
        # the one-proxy-per-parent cap.
        # kb_context: JSON list of kb_node slugs injected into the generation
        # prompt (NULL = KB-ungrounded idea) — lets the funnel compare pass
        # rates of grounded vs ungrounded ideas, i.e. MEASURE KB utility.
        for _col in ("signal_signature TEXT", "parent_idea_id INTEGER",
                     "kb_context TEXT"):
            try:
                conn.execute(f"ALTER TABLE alpha_ideas ADD COLUMN {_col}")
                logger.info(f"Migration applied: alpha_ideas.{_col.split()[0]} added")
            except Exception:
                pass
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_ai_signature ON alpha_ideas(signal_signature)"
        )

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

        # Sub-daily paper trading: exact mark timestamp alongside the slot key
        # (`date` holds YYYY-MM-DD for daily ideas — unchanged — and an
        # interval-aligned YYYY-MM-DDTHH:MM slot for sub-daily crypto ideas).
        try:
            conn.execute("ALTER TABLE paper_equity ADD COLUMN marked_at TEXT")
            logger.info("Migration applied: paper_equity.marked_at added")
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
            "robustness_score REAL",
            # Cross-sectional IC stats (cross_sectional_test persistence —
            # these existed only on long-lived DBs before; fresh DBs need them)
            "mean_ic REAL",
            "ic_tstat REAL",
            "stocks_positive_ic INTEGER",
            "best_stocks TEXT",
            # PSR principal rule (gate redesign 2026-07-10): P(true Sharpe >
            # deflated benchmark) per slice; n_trials is the WINDOWED count.
            "psr_test REAL",
            "psr_trainval REAL",
            # Concierge Pine Script generation (2026-07-10): deterministic
            # translation of the EXACT DSL tree this run backtested — NULL for
            # non-DSL routes (cross_sectional/fundamental_screen_portfolio) or
            # trees using a leaf Pine can't express (funding/dividends/CPO).
            "pinescript TEXT",
        ):
            try:
                conn.execute(f"ALTER TABLE backtest_runs ADD COLUMN {_col}")
                logger.info(f"Migration applied: backtest_runs.{_col.split()[0]} added")
            except Exception:
                pass

        # Phase 3.2 / 3.4: benchmark + capacity metrics on backtest_runs.
        for _col in ("equal_weight_sharpe REAL", "excess_vs_ew_ann_return REAL",
                     "benchmark_pass INTEGER",
                     "capacity_pct_adv REAL", "days_to_enter REAL",
                     "capacity_pass INTEGER"):
            try:
                conn.execute(f"ALTER TABLE backtest_runs ADD COLUMN {_col}")
                logger.info(f"Migration applied: backtest_runs.{_col.split()[0]} added")
            except Exception:
                pass

        # Phase 2.3: production-eligibility — current-constituent-only backtests
        # are research-grade, not production-eligible, until point-in-time
        # constituent history is materially complete (survivorship bias).
        for _col in ("production_eligible INTEGER", "universe_asof TEXT"):
            try:
                conn.execute(f"ALTER TABLE backtest_runs ADD COLUMN {_col}")
                logger.info(f"Migration applied: backtest_runs.{_col.split()[0]} added")
            except Exception:
                pass

        # Phase 0.6: market-rule / fee-model version stamp — traceability of every
        # run back to the cost assumptions in force when it ran.
        for _col in ("market_rules_version TEXT", "fee_model_version TEXT"):
            try:
                conn.execute(f"ALTER TABLE backtest_runs ADD COLUMN {_col}")
                logger.info(f"Migration applied: backtest_runs.{_col.split()[0]} added")
            except Exception:
                pass

        # WS3: long/short via perpetuals — leverage used and cumulative funding
        # drag (net of position sign, MODELED AVERAGE not real historical
        # funding — see AVG_FUNDING_RATE_PER_INTERVAL) on every run. 0/NULL on
        # Bursa (no leverage, no funding concept).
        for _col in ("leverage_used REAL", "funding_drag_pct REAL"):
            try:
                conn.execute(f"ALTER TABLE backtest_runs ADD COLUMN {_col}")
                logger.info(f"Migration applied: backtest_runs.{_col.split()[0]} added")
            except Exception:
                pass

        # Fidelity metrics (2026-07-11): compounded CAGR (vs arithmetic
        # ann_return in result_data), drawdown-quality (ulcer index + longest
        # underwater run), conservative-fill robustness (Sharpe under a 2-bar
        # delayed fill and its ratio to the research Sharpe), and a
        # capacity-adjusted Sharpe (size-aware impact haircut — a report, the
        # gated Sharpe is unchanged).
        for _col in ("cagr REAL", "ulcer_index REAL", "dd_duration_bars INTEGER",
                     "sharpe_net_conservative REAL", "fill_robustness REAL",
                     "capacity_adjusted_sharpe REAL"):
            try:
                conn.execute(f"ALTER TABLE backtest_runs ADD COLUMN {_col}")
                logger.info(f"Migration applied: backtest_runs.{_col.split()[0]} added")
            except Exception:
                pass

        # Phase 1.1: versioned transaction-cost schedules (audit §3.2). Costs are
        # date-dependent on Bursa (stamp-duty remission 0.15%→0.10% from
        # 2023-07-13). Store schedules by effective date so a backtest spanning
        # the boundary can apply the rate that was actually in force.
        conn.execute("""
            CREATE TABLE IF NOT EXISTS fee_schedules (
                id                INTEGER PRIMARY KEY AUTOINCREMENT,
                market            TEXT NOT NULL,
                instrument_type   TEXT NOT NULL,
                effective_from    TEXT NOT NULL,
                effective_to      TEXT,
                settlement_cycle  TEXT,
                board_lot         INTEGER,
                commission_rate   REAL,
                commission_min_fee REAL,
                clearing_rate     REAL,
                clearing_cap      REAL,
                stamp_duty_rate   REAL,
                stamp_duty_cap    REAL,
                notes             TEXT,
                created_at        TEXT DEFAULT (datetime('now')),
                UNIQUE(market, instrument_type, effective_from)
            )
        """)
        # Seed the two known Bursa listed-equity schedules (idempotent).
        conn.executemany("""
            INSERT OR IGNORE INTO fee_schedules
              (market, instrument_type, effective_from, effective_to,
               settlement_cycle, board_lot, commission_rate, clearing_rate,
               clearing_cap, stamp_duty_rate, stamp_duty_cap, notes)
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?)
        """, [
            ("KLSE", "listed_equity", "2000-01-01", "2023-07-12",
             "T+2", 100, 0.0008, 0.0003, 1000.0, 0.0015, 200.0,
             "Pre-remission statutory stamp duty 0.15%, cap RM200 "
             "(T+2 from 2019-04-29; T+3 before, not modelled separately)"),
            ("KLSE", "listed_equity", "2023-07-13", "2028-07-12",
             "T+2", 100, 0.0008, 0.0003, 1000.0, 0.0010, 1000.0,
             "Remitted stamp duty 0.10%, cap RM1,000 per contract note"),
            # Crypto spot (Binance base tier). Both markets' rows are seeded in
            # every DB — harmless and idempotent; the fee resolver filters by
            # the active settings.MARKET.
            ("CRYPTO", "spot", "2020-01-01", None,
             "T+0", 0, 0.0010, 0.0, 0.0, 0.0, 0.0,
             "Binance spot base tier: 0.10% taker per side, no BNB discount; "
             "no stamp/clearing/board lot on crypto spot"),
        ])

        # Phase 1.2: data-quality checks + Data Confidence Score (audit §6).
        # One row per (idea, ticker) evaluation; confidence_score gates promotion.
        conn.execute("""
            CREATE TABLE IF NOT EXISTS data_quality_checks (
                id                    INTEGER PRIMARY KEY AUTOINCREMENT,
                idea_id               INTEGER,
                ticker                TEXT NOT NULL,
                source                TEXT,
                bars                  INTEGER,
                price_completeness    REAL,
                volume_completeness   REAL,
                stale_price_frac      REAL,
                missing_day_frac      REAL,
                corporate_action_flag INTEGER DEFAULT 0,
                confidence_score      REAL,
                passed                INTEGER,
                notes                 TEXT,
                created_at            TEXT DEFAULT (datetime('now'))
            )
        """)
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_dq_idea ON data_quality_checks(idea_id)")

        # Phase 1.4: corporate actions (audit §7.2) — bonus/rights/splits/divs.
        conn.execute("""
            CREATE TABLE IF NOT EXISTS corporate_actions (
                id                INTEGER PRIMARY KEY AUTOINCREMENT,
                ticker            TEXT NOT NULL,
                event_date        TEXT NOT NULL,
                ex_date           TEXT,
                event_type        TEXT NOT NULL,
                cash_amount       REAL,
                ratio_numerator   REAL,
                ratio_denominator REAL,
                adjustment_factor REAL,
                source            TEXT,
                validation_status TEXT,
                notes             TEXT,
                created_at        TEXT DEFAULT (datetime('now')),
                UNIQUE(ticker, event_date, event_type)
            )
        """)
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_ca_ticker ON corporate_actions(ticker)")

        # Phase 2.1: point-in-time index membership (audit §7.1) — the fix for
        # survivorship bias. Seeded with the current KLCI as of UNIVERSE_ASOF;
        # historical entries/exits are backfilled incrementally (data-acquisition
        # bound). effective_to NULL = still a member.
        conn.execute("""
            CREATE TABLE IF NOT EXISTS universe_membership (
                id               INTEGER PRIMARY KEY AUTOINCREMENT,
                universe_name    TEXT NOT NULL,
                ticker           TEXT NOT NULL,
                company_name     TEXT,
                effective_from   TEXT NOT NULL,
                effective_to     TEXT,
                inclusion_reason TEXT,
                exclusion_reason TEXT,
                source           TEXT,
                confidence_score REAL,
                created_at       TEXT DEFAULT (datetime('now')),
                UNIQUE(universe_name, ticker, effective_from)
            )
        """)
        try:
            from config.settings import (
                KLCI_STOCKS as _UNIV, UNIVERSE_ASOF as _ASOF,
                UNIVERSE_NAME as _UNAME,
            )
            conn.executemany("""
                INSERT OR IGNORE INTO universe_membership
                  (universe_name, ticker, company_name, effective_from,
                   effective_to, inclusion_reason, source, confidence_score)
                VALUES (?, ?, ?, ?, NULL, 'current_constituent', 'market_profile', 1.0)
            """, [(_UNAME, s["symbol"], s["name"], _ASOF) for s in _UNIV])
        except Exception as _um_exc:
            logger.warning(f"universe_membership seed skipped: {_um_exc}")

        # Phase 5.4: strategy family classification (audit §9.3), reusing the
        # same keyword classifier RejectionMemory uses for rejection patterns —
        # one taxonomy for both what fails and what gets generated.
        try:
            conn.execute("ALTER TABLE alpha_ideas ADD COLUMN family TEXT")
            logger.info("Migration applied: alpha_ideas.family added")
        except Exception:
            pass

        # Phase 5.1: Bursa announcement ingestion + NLP labels (audit §7.6).
        conn.execute("""
            CREATE TABLE IF NOT EXISTS announcement_events (
                id                INTEGER PRIMARY KEY AUTOINCREMENT,
                ticker            TEXT,
                announcement_date TEXT NOT NULL,
                announcement_type TEXT,
                title             TEXT,
                source_url        TEXT,
                source            TEXT,
                nlp_labels        TEXT,
                sentiment_score   REAL,
                materiality_score REAL,
                is_actionable     INTEGER DEFAULT 0,
                created_at        TEXT DEFAULT (datetime('now')),
                UNIQUE(ticker, announcement_date, title)
            )
        """)
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_ann_ticker ON announcement_events(ticker)")

        # Phase 5.2: fundamental feature store (audit §7.4).
        conn.execute("""
            CREATE TABLE IF NOT EXISTS fundamental_features (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                ticker          TEXT NOT NULL,
                as_of_date      TEXT NOT NULL,
                revenue         REAL,
                net_profit      REAL,
                eps             REAL,
                roe             REAL,
                roa             REAL,
                gross_margin    REAL,
                net_debt_equity REAL,
                free_cash_flow  REAL,
                dividend_yield  REAL,
                payout_ratio    REAL,
                pe              REAL,
                pb              REAL,
                ev_ebitda       REAL,
                source          TEXT,
                created_at      TEXT DEFAULT (datetime('now')),
                UNIQUE(ticker, as_of_date)
            )
        """)
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_ff_ticker ON fundamental_features(ticker)")

        # Phase 5.3: macro + sector regime features (audit §7.5).
        conn.execute("""
            CREATE TABLE IF NOT EXISTS macro_features (
                id             INTEGER PRIMARY KEY AUTOINCREMENT,
                as_of_date     TEXT NOT NULL UNIQUE,
                opr            REAL,
                myr_usd        REAL,
                brent_crude    REAL,
                cpo_price      REAL,
                cpo_trend      TEXT,
                regime_label   TEXT,
                source         TEXT,
                created_at     TEXT DEFAULT (datetime('now'))
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS sector_features (
                id             INTEGER PRIMARY KEY AUTOINCREMENT,
                sector         TEXT NOT NULL,
                as_of_date     TEXT NOT NULL,
                mean_return_20d REAL,
                mean_return_60d REAL,
                breadth_pct    REAL,
                created_at     TEXT DEFAULT (datetime('now')),
                UNIQUE(sector, as_of_date)
            )
        """)

        # Phase 5.5: strategy cemetery (audit §5.4/§14.2) — one row per rejected
        # idea with revival conditions, alongside the existing aggregated
        # rejection_patterns table (counts by factor_type/sector/reason).
        conn.execute("""
            CREATE TABLE IF NOT EXISTS strategy_cemetery (
                id                    INTEGER PRIMARY KEY AUTOINCREMENT,
                idea_id               INTEGER,
                strategy_name         TEXT,
                factor_type           TEXT,
                sector                TEXT,
                rejected_at_stage     TEXT,
                rejection_reason      TEXT,
                revival_conditions    TEXT,
                created_at            TEXT DEFAULT (datetime('now'))
            )
        """)
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_cemetery_family "
            "ON strategy_cemetery(factor_type, sector)")

        # classified_by: how reason_category was determined — "explicit:..."
        # for a caller-supplied category (bypasses keyword guessing entirely),
        # the matched keyword string for a _classify() hit, or NULL for a
        # fall-through default. Lets future mis-classification be traced back
        # to its cause instead of being a black box (P2-5, 2026-07-13 audit —
        # found gate0's free-text keyword matching was mis-bucketing ~88% of
        # Bursa rejections as "overfitting" purely because Claude's rationale
        # mentions that dimension alongside others, even when it wasn't the
        # actual failure reason).
        try:
            conn.execute("ALTER TABLE strategy_cemetery ADD COLUMN classified_by TEXT")
            logger.info("Migration applied: strategy_cemetery.classified_by column added")
        except Exception:
            pass

        # Phase 6.3: post-trade reconciliation (audit §11.3/§14). Paper trading
        # has no independent fill source to reconcile against yet — expected and
        # actual are computed from the same cost model by construction — but the
        # table + writer establish the auditable trail ready for when execution
        # (Stage 4b) can diverge from the model.
        conn.execute("""
            CREATE TABLE IF NOT EXISTS paper_trade_reconciliation (
                id                  INTEGER PRIMARY KEY AUTOINCREMENT,
                trade_id            INTEGER NOT NULL,
                side                TEXT NOT NULL,
                expected_price      REAL,
                actual_price        REAL,
                expected_cost       REAL,
                actual_cost         REAL,
                price_diff          REAL,
                cost_diff           REAL,
                clean               INTEGER,
                created_at          TEXT DEFAULT (datetime('now'))
            )
        """)
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_recon_trade ON paper_trade_reconciliation(trade_id)")

        # Phase 4.1: liquidity features (audit §14.3) — historical tradability.
        conn.execute("""
            CREATE TABLE IF NOT EXISTS liquidity_features (
                id                  INTEGER PRIMARY KEY AUTOINCREMENT,
                ticker              TEXT NOT NULL,
                trade_date          TEXT NOT NULL,
                adv_20              REAL,
                adv_60              REAL,
                median_value_20     REAL,
                zero_volume_days_60 INTEGER,
                amihud_illiquidity  REAL,
                capacity_score      REAL,
                created_at          TEXT DEFAULT (datetime('now')),
                UNIQUE(ticker, trade_date)
            )
        """)

        # Phase 4.2: portfolio risk snapshots (audit §14) — concentration/exposure.
        conn.execute("""
            CREATE TABLE IF NOT EXISTS risk_snapshots (
                id                 INTEGER PRIMARY KEY AUTOINCREMENT,
                snapshot_at        TEXT DEFAULT (datetime('now')),
                open_positions     INTEGER,
                gross_exposure_myr REAL,
                max_single_pct     REAL,
                max_sector         TEXT,
                max_sector_pct     REAL,
                bank_pct           REAL,
                concentration_ok   INTEGER,
                kill_switch_active INTEGER,
                detail             TEXT
            )
        """)

        # Concierge chat agent: sessions, message history, and links from a
        # session to the ideas it submitted (powers list_session_ideas).
        conn.execute("""
            CREATE TABLE IF NOT EXISTS concierge_sessions (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                label       TEXT,
                created_at  TEXT DEFAULT (datetime('now')),
                last_active TEXT DEFAULT (datetime('now'))
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS concierge_messages (
                id             INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id     INTEGER NOT NULL,
                role           TEXT NOT NULL,
                content        TEXT,
                tool_calls_json TEXT,
                created_at     TEXT DEFAULT (datetime('now'))
            )
        """)
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_cm_session ON concierge_messages(session_id)")
        conn.execute("""
            CREATE TABLE IF NOT EXISTS concierge_idea_links (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id  INTEGER NOT NULL,
                idea_id     INTEGER NOT NULL,
                created_at  TEXT DEFAULT (datetime('now')),
                UNIQUE(session_id, idea_id)
            )
        """)

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

        # Backtest Lab: reconstructed trade blotter. The backtest is vectorized
        # (no discrete orders) — each row here is a trade RECONSTRUCTED from the
        # position series (a maximal run of constant non-zero position), priced
        # at the fill convention, with PnL attributed from the SAME net-return
        # series behind the gated Sharpe (so summed net_pct reconciles to the
        # backtest return). Faithful to the math, not an independent order log.
        conn.execute("""
            CREATE TABLE IF NOT EXISTS backtest_trades (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                idea_id     INTEGER NOT NULL,
                seq         INTEGER,
                direction   TEXT,
                entry_date  TEXT,
                exit_date   TEXT,
                entry_price REAL,
                exit_price  REAL,
                bars_held   INTEGER,
                gross_pct   REAL,
                cost_pct    REAL,
                net_pct     REAL,
                is_oos      INTEGER DEFAULT 0,
                created_at  TEXT DEFAULT (datetime('now'))
            )
        """)
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_bt_idea ON backtest_trades(idea_id)"
        )

        # GraphRAG knowledge layer: node registry over the existing kb_* tables.
        # kb_documents/kb_concepts stay untouched; kb_nodes unifies them (plus
        # techniques, ideas, rejection patterns) so kb_edges can link ANY node
        # kind — the old kb_links FKs allowed doc-to-doc only.
        # node_type is validated in knowledge/graph/store.py against
        # kb_node_type_registry (below), NOT a DB CHECK — so new node types are
        # one INSERT, never a table rebuild. Legacy DBs that still carry the old
        # CHECK are migrated by _migrate_kb_nodes_v2() after this block.
        conn.execute("""
            CREATE TABLE IF NOT EXISTS kb_nodes (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                node_type TEXT NOT NULL,
                ref_table TEXT,
                ref_id INTEGER,
                slug TEXT UNIQUE NOT NULL,
                title TEXT,
                domain TEXT,
                summary TEXT,
                tags TEXT,
                content_hash TEXT,
                extracted_at TEXT,
                confidence REAL,
                review_state TEXT,
                ingestion_version TEXT,
                created_at TEXT DEFAULT (datetime('now')),
                updated_at TEXT DEFAULT (datetime('now')),
                UNIQUE(ref_table, ref_id)
            )
        """)
        conn.execute("CREATE INDEX IF NOT EXISTS idx_kb_nodes_type ON kb_nodes(node_type)")

        conn.execute("""
            CREATE TABLE IF NOT EXISTS kb_edges (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                source_id INTEGER NOT NULL REFERENCES kb_nodes(id),
                target_id INTEGER NOT NULL REFERENCES kb_nodes(id),
                relation TEXT NOT NULL,
                weight REAL DEFAULT 1.0,
                origin TEXT DEFAULT 'llm',
                evidence_count INTEGER DEFAULT 1,
                last_seen_at TEXT,
                created_at TEXT DEFAULT (datetime('now')),
                UNIQUE(source_id, target_id, relation)
            )
        """)
        conn.execute("CREATE INDEX IF NOT EXISTS idx_kb_edges_src ON kb_edges(source_id)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_kb_edges_tgt ON kb_edges(target_id)")
        # Legacy DBs: backfill the evidence columns added 2026-07-12.
        for _col, _decl in (("evidence_count", "INTEGER DEFAULT 1"),
                            ("last_seen_at", "TEXT")):
            try:
                conn.execute(f"ALTER TABLE kb_edges ADD COLUMN {_col} {_decl}")
            except Exception:
                pass

        # Knowledge-graph node-type registry: the source of truth for valid
        # node_type values (validated in knowledge/graph/store.py). Adding a new
        # node type is one INSERT here — no kb_nodes rebuild.
        conn.execute("""
            CREATE TABLE IF NOT EXISTS kb_node_type_registry (
                node_type   TEXT PRIMARY KEY,
                description TEXT,
                status      TEXT DEFAULT 'active',
                created_at  TEXT DEFAULT (datetime('now'))
            )
        """)
        for _nt, _desc in (
            ("note", "Free-form note / human annotation"),
            ("concept", "Extracted concept from documents"),
            ("technique", "Technique from technique_library"),
            ("idea", "Raw / low-stage alpha_ideas candidate"),
            ("rejection_pattern", "Aggregated rejection pattern"),
            ("strategy", "Executable or evaluated idea (parsed/backtested/gated/promoted)"),
            ("signature", "signal_signature factor identity"),
            ("backtest_run", "A backtest_runs evidence row"),
            ("gate_decision", "A gate_decisions evidence row"),
            ("risk", "Named research risk (cost drag, parser approximation, …)"),
            ("finding", "Promoted governance_findings row"),
            ("leaf", "Executable DSL leaf from signal_dsl.py"),
            ("agent", "A governance/inspector agent that reports findings"),
        ):
            conn.execute(
                "INSERT OR IGNORE INTO kb_node_type_registry (node_type, description) VALUES (?, ?)",
                (_nt, _desc),
            )

        # Entity resolution: alias → canonical node (BTC/BTCUSDT/XBT, DPSR/…).
        conn.execute("""
            CREATE TABLE IF NOT EXISTS kb_aliases (
                alias       TEXT PRIMARY KEY,
                canonical   TEXT,
                node_id     INTEGER REFERENCES kb_nodes(id),
                alias_type  TEXT,
                confidence  REAL DEFAULT 1.0,
                origin      TEXT DEFAULT 'human',
                created_at  TEXT DEFAULT (datetime('now'))
            )
        """)
        try:
            conn.execute("ALTER TABLE kb_aliases ADD COLUMN canonical TEXT")
        except Exception:
            pass

        conn.execute("""
            CREATE TABLE IF NOT EXISTS kb_embeddings (
                node_id INTEGER PRIMARY KEY REFERENCES kb_nodes(id),
                model TEXT NOT NULL,
                dim INTEGER NOT NULL,
                vector BLOB NOT NULL,
                content_hash TEXT,
                updated_at TEXT DEFAULT (datetime('now'))
            )
        """)

        # FTS index over nodes. Synced by knowledge/graph/store.py (single
        # write path), NOT by triggers — legacy writers bypass the node layer
        # and a nightly reconcile job heals any gaps.
        conn.execute("""
            CREATE VIRTUAL TABLE IF NOT EXISTS kb_fts USING fts5(
                title, summary, content, tags, node_id UNINDEXED
            )
        """)

        # Human feedback captured in Obsidian (vault/feedback/) and ingested
        # back into the loop by scripts/ingest_obsidian_feedback.py. This is the
        # authoritative store; downstream effects (rejection memory, gate
        # decisions, FTS note nodes) are derived idempotently from these rows.
        # One active row per (target_slug, reviewer); content_hash makes
        # re-ingesting an unchanged file a no-op.
        conn.execute("""
            CREATE TABLE IF NOT EXISTS kb_feedback (
                id           INTEGER PRIMARY KEY AUTOINCREMENT,
                target_slug  TEXT NOT NULL,
                node_id      INTEGER REFERENCES kb_nodes(id),
                idea_id      INTEGER REFERENCES alpha_ideas(id),
                reviewer     TEXT DEFAULT 'human',
                verdict      TEXT,
                rating       INTEGER,
                tags         TEXT,
                note         TEXT,
                content_hash TEXT NOT NULL,
                source_path  TEXT,
                applied_at   TEXT,
                created_at   TEXT DEFAULT (datetime('now')),
                updated_at   TEXT DEFAULT (datetime('now')),
                UNIQUE(target_slug, reviewer)
            )
        """)

        # Daemon scheduler: persisted last-run timestamps so daily jobs catch up
        # after downtime instead of being silently skipped
        conn.execute("""
            CREATE TABLE IF NOT EXISTS job_state (
                job_name     TEXT PRIMARY KEY,
                last_run_utc TEXT NOT NULL
            )
        """)

        # Event-driven revisit (pipeline/revisit.py): last-observed snapshot
        # per trigger key (e.g. "vol_regime:BTC/USDT" -> "high_vol") so a
        # regime CHANGE, not just its current value, fires a revisit scan.
        conn.execute("""
            CREATE TABLE IF NOT EXISTS revisit_state (
                key          TEXT PRIMARY KEY,
                value        TEXT,
                updated_at   TEXT DEFAULT (datetime('now'))
            )
        """)
        # New-data-source arrivals worth re-examining old rejects for (e.g.
        # "funding_history_backfilled"). No reliable automatic detector for
        # this trigger exists yet — rows are a manual convention: insert one
        # when a data source lands, the revisit scanner consumes it once.
        conn.execute("""
            CREATE TABLE IF NOT EXISTS data_source_events (
                id            INTEGER PRIMARY KEY AUTOINCREMENT,
                source_name   TEXT NOT NULL,
                description   TEXT,
                consumed      INTEGER DEFAULT 0,
                created_at    TEXT DEFAULT (datetime('now'))
            )
        """)

        # LeafSynthesizer (agents/leaf_synthesizer/): every attempt to turn a
        # genuinely-unrepresentable formula into a new DSL leaf, approved or
        # not — full audit trail since this is the one place an LLM writes
        # code that runs unreviewed inside the backtest engine (2026-07-13).
        conn.execute("""
            CREATE TABLE IF NOT EXISTS leaf_synthesis_attempts (
                id                INTEGER PRIMARY KEY AUTOINCREMENT,
                idea_id           INTEGER REFERENCES alpha_ideas(id),
                hypothesis        TEXT,
                rejection_reason  TEXT,
                status            TEXT NOT NULL,
                leaf_name         TEXT,
                spec_json         TEXT,
                generated_file    TEXT,
                test_file         TEXT,
                review_notes      TEXT,
                cost_usd          REAL DEFAULT 0,
                git_commit_sha    TEXT,
                created_at        TEXT DEFAULT (datetime('now'))
            )
        """)

        # module_source: full generated module text, captured on approval —
        # the container has no git binary and /app is an ephemeral image
        # layer, so this audit row (plus the runtime-volume file copy) is the
        # only thing guaranteed to survive a rebuild (2026-07-13 self-audit).
        try:
            conn.execute("ALTER TABLE leaf_synthesis_attempts ADD COLUMN module_source TEXT")
            logger.info("Migration applied: leaf_synthesis_attempts.module_source column added")
        except Exception:
            pass

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
                marked_at      TEXT,
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

        # WS3: cumulative funding paid (positive) / received (negative) while a
        # perp position was open — accrued once per daily_update cycle. 0 on
        # Bursa (no perp funding concept).
        try:
            conn.execute("ALTER TABLE paper_trades ADD COLUMN funding_paid REAL DEFAULT 0")
            logger.info("Migration applied: paper_trades.funding_paid added")
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

        # CoinGecko client: token supply cache for crypto market
        conn.execute("""
        CREATE TABLE IF NOT EXISTS token_supply (
            id                    INTEGER PRIMARY KEY AUTOINCREMENT,
            symbol                TEXT NOT NULL,
            circulating_supply    REAL,
            total_supply          REAL,
            max_supply            REAL,
            fully_diluted_valuation REAL,
            market_cap            REAL,
            fetched_at            TEXT DEFAULT (datetime('now')),
            UNIQUE(symbol, fetched_at)
        )
        """)
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_token_symbol ON token_supply(symbol)"
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_token_fetched ON token_supply(fetched_at)"
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

        # Parameter-sweep optimizer queue + results. n_configs feeds the
        # deflated-Sharpe hurdle of the idea's post-sweep gated backtest.
        conn.execute("""
        CREATE TABLE IF NOT EXISTS optimizer_runs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            idea_id INTEGER NOT NULL,
            status TEXT DEFAULT 'queued',      -- queued | running | done | failed
            seed INTEGER,
            n_configs INTEGER,
            started_at TEXT,
            finished_at TEXT,
            summary_json TEXT,                 -- top configs by val score (no trees)
            winner_json TEXT,                  -- winning config incl. dsl tree
            error TEXT,
            created_at TEXT DEFAULT (datetime('now')),
            FOREIGN KEY (idea_id) REFERENCES alpha_ideas(id)
        )
        """)
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_optruns_status ON optimizer_runs(status)"
        )

        # Governance findings: audit trail of inspector reports at each level (L0–L3)
        conn.execute("""
        CREATE TABLE IF NOT EXISTS governance_findings (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            agent TEXT NOT NULL,
            level TEXT NOT NULL,
            scope TEXT,
            status TEXT NOT NULL,
            severity TEXT NOT NULL,
            evidence TEXT,
            local_recommendation TEXT,
            escalate_to TEXT,
            created_at TEXT DEFAULT (datetime('now'))
        )
        """)
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_gov_scope ON governance_findings(scope)"
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_gov_status ON governance_findings(status)"
        )

    # Runs in its own connection AFTER the main session commits, because the
    # kb_nodes CHECK-drop rebuild needs foreign_keys OFF outside a transaction.
    _migrate_kb_nodes_v2(db_path)

    logger.info(f"Database initialized at {db_path}")


def _migrate_kb_nodes_v2(db_path: Path = DB_PATH):
    """One-time, idempotent rebuild of legacy kb_nodes: drop the node_type CHECK
    (validation moved to store.py + kb_node_type_registry) and add the
    confidence / review_state / ingestion_version columns.

    Fresh DBs are already born with the new schema (see init_db), so this only
    fires on pre-2026-07-12 databases that still carry the CHECK. Ids are
    preserved, so every kb_edges / kb_embeddings / kb_feedback / kb_aliases
    reference stays valid.
    """
    conn = sqlite3.connect(str(db_path), timeout=30)
    conn.row_factory = sqlite3.Row
    try:
        row = conn.execute(
            "SELECT sql FROM sqlite_master WHERE type='table' AND name='kb_nodes'"
        ).fetchone()
        if row is None:
            return
        if "CHECK" not in (row["sql"] or ""):
            return  # already migrated — new columns are guaranteed by the create above

        logger.info("[kb_nodes v2] Legacy CHECK detected — rebuilding kb_nodes without it")
        # foreign_keys must be toggled OUTSIDE a transaction; sqlite3 autocommits
        # DDL only when not in a transaction, so drive the transaction explicitly.
        conn.isolation_level = None
        conn.execute("PRAGMA foreign_keys=OFF")
        conn.execute("BEGIN")
        # This DB legitimately carries pre-existing orphan refs (synthetic/pruned
        # rows), so the test is "the rebuild must not INCREASE orphans", not
        # "zero orphans". Ids are copied 1:1, so this should hold exactly.
        fk_before = len(conn.execute("PRAGMA foreign_key_check").fetchall())
        conn.execute("""
            CREATE TABLE kb_nodes_new (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                node_type TEXT NOT NULL,
                ref_table TEXT,
                ref_id INTEGER,
                slug TEXT UNIQUE NOT NULL,
                title TEXT,
                domain TEXT,
                summary TEXT,
                tags TEXT,
                content_hash TEXT,
                extracted_at TEXT,
                confidence REAL,
                review_state TEXT,
                ingestion_version TEXT,
                created_at TEXT DEFAULT (datetime('now')),
                updated_at TEXT DEFAULT (datetime('now')),
                UNIQUE(ref_table, ref_id)
            )
        """)
        conn.execute("""
            INSERT INTO kb_nodes_new
                (id, node_type, ref_table, ref_id, slug, title, domain, summary,
                 tags, content_hash, extracted_at, created_at, updated_at)
            SELECT id, node_type, ref_table, ref_id, slug, title, domain, summary,
                   tags, content_hash, extracted_at, created_at, updated_at
            FROM kb_nodes
        """)
        before = conn.execute("SELECT COUNT(*) AS n FROM kb_nodes").fetchone()["n"]
        after = conn.execute("SELECT COUNT(*) AS n FROM kb_nodes_new").fetchone()["n"]
        if before != after:
            conn.execute("ROLLBACK")
            raise RuntimeError(f"[kb_nodes v2] row mismatch {before} != {after}; aborted")
        conn.execute("DROP TABLE kb_nodes")
        conn.execute("ALTER TABLE kb_nodes_new RENAME TO kb_nodes")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_kb_nodes_type ON kb_nodes(node_type)")
        fk_after = len(conn.execute("PRAGMA foreign_key_check").fetchall())
        if fk_after > fk_before:
            conn.execute("ROLLBACK")
            raise RuntimeError(
                f"[kb_nodes v2] rebuild increased FK orphans {fk_before} -> {fk_after}; aborted")
        conn.execute("COMMIT")
        conn.execute("PRAGMA foreign_keys=ON")
        logger.info(f"[kb_nodes v2] Rebuilt kb_nodes ({after} rows), CHECK removed")
    finally:
        conn.close()


if __name__ == "__main__":
    init_db()
    print(f"Database initialized at {DB_PATH}")
