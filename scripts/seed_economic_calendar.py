"""
seed_economic_calendar.py — one-time script to seed the economic_calendar table
with known 2026 BNM MPC dates and approximate China PMI release dates.

Run once:
    PYTHONPATH=/opt/openclaw/app /opt/openclaw/venv/bin/python scripts/seed_economic_calendar.py
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env"))

from data.database import db_session, init_db

# BNM Monetary Policy Committee (MPC) meeting dates 2026
# Source: Bank Negara Malaysia official MPC calendar
BNM_MPC_2026 = [
    ("2026-01-22", "15:00"),
    ("2026-03-05", "15:00"),
    ("2026-05-07", "15:00"),
    ("2026-07-09", "15:00"),
    ("2026-09-10", "15:00"),
    ("2026-11-05", "15:00"),
]

# Approximate China Manufacturing PMI release dates (usually 1st business day of each month)
CHINA_PMI_2026 = [
    "2026-01-02",
    "2026-02-03",
    "2026-03-02",
    "2026-04-01",
    "2026-05-04",
    "2026-06-01",
    "2026-07-01",
    "2026-08-03",
    "2026-09-01",
    "2026-10-01",
    "2026-11-02",
    "2026-12-01",
]

# US Fed FOMC meeting dates 2026 (approximate — confirm at federalreserve.gov)
FED_FOMC_2026 = [
    ("2026-01-28", "19:00"),  # Statement release ~2pm ET = 3am +1 MYT next day
    ("2026-03-18", "19:00"),
    ("2026-05-06", "19:00"),
    ("2026-06-17", "19:00"),
    ("2026-07-29", "19:00"),
    ("2026-09-16", "19:00"),
    ("2026-11-04", "19:00"),
    ("2026-12-16", "19:00"),
]

# Malaysia CPI release dates 2026 (approximately 3rd week of following month)
MALAYSIA_CPI_2026 = [
    "2026-02-19",
    "2026-03-19",
    "2026-04-23",
    "2026-05-21",
    "2026-06-18",
    "2026-07-23",
    "2026-08-20",
    "2026-09-17",
    "2026-10-22",
    "2026-11-19",
    "2026-12-17",
]


def seed():
    init_db()
    inserted = 0
    skipped = 0

    with db_session() as conn:
        # BNM OPR decisions
        for date_str, time_str in BNM_MPC_2026:
            try:
                conn.execute("""
                    INSERT OR IGNORE INTO economic_calendar
                      (event_name, event_type, scheduled_date, scheduled_time,
                       country, importance)
                    VALUES (?, ?, ?, ?, ?, ?)
                """, ("BNM OPR Decision", "bnm_opr", date_str, time_str, "MY", "high"))
                inserted += 1
            except Exception:
                skipped += 1

        # China Manufacturing PMI
        for date_str in CHINA_PMI_2026:
            try:
                conn.execute("""
                    INSERT OR IGNORE INTO economic_calendar
                      (event_name, event_type, scheduled_date, scheduled_time,
                       country, importance)
                    VALUES (?, ?, ?, ?, ?, ?)
                """, ("China Manufacturing PMI", "china_pmi", date_str, "09:00", "CN", "high"))
                inserted += 1
            except Exception:
                skipped += 1

        # Fed FOMC decisions
        for date_str, time_str in FED_FOMC_2026:
            try:
                conn.execute("""
                    INSERT OR IGNORE INTO economic_calendar
                      (event_name, event_type, scheduled_date, scheduled_time,
                       country, importance)
                    VALUES (?, ?, ?, ?, ?, ?)
                """, ("Fed Interest Rate Decision", "fed_decision", date_str, time_str, "US", "high"))
                inserted += 1
            except Exception:
                skipped += 1

        # Malaysia CPI
        for date_str in MALAYSIA_CPI_2026:
            try:
                conn.execute("""
                    INSERT OR IGNORE INTO economic_calendar
                      (event_name, event_type, scheduled_date, scheduled_time,
                       country, importance)
                    VALUES (?, ?, ?, ?, ?, ?)
                """, ("Malaysia CPI", "malaysia_cpi", date_str, "12:00", "MY", "medium"))
                inserted += 1
            except Exception:
                skipped += 1

    print(f"Economic calendar seeded: {inserted} inserted, {skipped} skipped (already existed)")
    print("Verify with: sqlite3 data/openclaw.db 'SELECT * FROM economic_calendar ORDER BY scheduled_date'")


if __name__ == "__main__":
    seed()
