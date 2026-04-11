"""
seed_stock_universe.py — Seed the stock_universe table with FBM70 tickers.

Run once (idempotent — uses INSERT OR REPLACE):
    PYTHONPATH=/opt/openclaw/app \
    /opt/openclaw/venv/bin/python scripts/seed_stock_universe.py
"""
from data.database import db_session, init_db

# ── FBM70 universe ────────────────────────────────────────────────────────────
# ticker, name, sector, index_member, market_cap_tier
FBM70_UNIVERSE = [
    # ── KLCI 30 ──────────────────────────────────────────────────────────────
    ("1023.KL",  "CIMB Group Holdings",            "Finance",       "KLCI",  "large"),
    ("1155.KL",  "Malayan Banking (Maybank)",       "Finance",       "KLCI",  "large"),
    ("1295.KL",  "Public Bank",                     "Finance",       "KLCI",  "large"),
    ("5285.KL",  "Sime Darby Plantation",           "Plantation",    "KLCI",  "large"),
    ("5347.KL",  "Tenaga Nasional",                 "Utilities",     "KLCI",  "large"),
    ("4863.KL",  "Telekom Malaysia",                "Telco",         "KLCI",  "large"),
    ("6947.KL",  "CelcomDigi",                      "Telco",         "KLCI",  "large"),
    ("5225.KL",  "IHH Healthcare",                  "Healthcare",    "KLCI",  "large"),
    ("2291.KL",  "IOI Corporation",                 "Plantation",    "KLCI",  "large"),
    ("5182.KL",  "Kuala Lumpur Kepong (KLK)",       "Plantation",    "KLCI",  "large"),
    ("1066.KL",  "RHB Bank",                        "Finance",       "KLCI",  "large"),
    ("5819.KL",  "Hong Leong Bank",                 "Finance",       "KLCI",  "large"),
    ("1082.KL",  "Hong Leong Financial Group",      "Finance",       "KLCI",  "large"),
    ("4197.KL",  "Sime Darby",                      "Industrial",    "KLCI",  "large"),
    ("5398.KL",  "Petronas Gas",                    "Oil & Gas",     "KLCI",  "large"),
    ("5183.KL",  "Petronas Dagangan",               "Oil & Gas",     "KLCI",  "large"),
    ("6033.KL",  "MISC",                            "Transport",     "KLCI",  "large"),
    ("4715.KL",  "Genting",                         "Gaming",        "KLCI",  "large"),
    ("3182.KL",  "Genting Malaysia",                "Gaming",        "KLCI",  "large"),
    ("5681.KL",  "Maxis",                           "Telco",         "KLCI",  "large"),
    ("6888.KL",  "Axiata Group",                    "Telco",         "KLCI",  "large"),
    ("1961.KL",  "PPB Group",                       "Consumer",      "KLCI",  "large"),
    ("7277.KL",  "Dialog Group",                    "Oil & Gas",     "KLCI",  "large"),
    ("5168.KL",  "Hartalega Holdings",              "Healthcare",    "KLCI",  "large"),
    ("5069.KL",  "Hap Seng Plantations",            "Plantation",    "KLCI",  "large"),
    # ── FBM70 Additional ─────────────────────────────────────────────────────
    # Consumer
    ("6599.KL",  "AEON Co. (M)",                   "Consumer",      "FBM70", "mid"),
    ("5196.KL",  "Berjaya Food",                    "Consumer",      "FBM70", "mid"),
    ("3026.KL",  "Dutch Lady Milk Industries",      "Consumer",      "FBM70", "mid"),
    ("4707.KL",  "Nestle Malaysia",                 "Consumer",      "FBM70", "large"),
    ("7052.KL",  "Padini Holdings",                 "Consumer",      "FBM70", "mid"),
    ("7084.KL",  "QL Resources",                    "Consumer",      "FBM70", "large"),
    ("7103.KL",  "Spritzer",                        "Consumer",      "FBM70", "small"),
    # Healthcare
    ("5878.KL",  "KPJ Healthcare",                  "Healthcare",    "FBM70", "mid"),
    ("7081.KL",  "Pharmaniaga",                     "Healthcare",    "FBM70", "small"),
    ("7153.KL",  "Kossan Rubber Industries",        "Healthcare",    "FBM70", "mid"),
    # Technology / EMS
    ("0166.KL",  "Inari Amertron",                  "Technology",    "FBM70", "large"),
    ("3867.KL",  "Malaysian Pacific Industries",    "Technology",    "FBM70", "mid"),
    ("5005.KL",  "Unisem (M)",                      "Technology",    "FBM70", "mid"),
    ("0128.KL",  "Frontken Corporation",             "Technology",    "FBM70", "mid"),
    ("0097.KL",  "ViTrox Corporation",               "Technology",    "FBM70", "mid"),
    ("0208.KL",  "Greatech Technology",              "Technology",    "FBM70", "mid"),
    # Industrial / Building Materials
    ("7162.KL",  "Astino",                           "Industrial",    "FBM70", "small"),
    ("5026.KL",  "Engtex Group",                     "Industrial",    "FBM70", "small"),
    ("3794.KL",  "Lafarge Malaysia",                 "Industrial",    "FBM70", "mid"),
    ("8869.KL",  "Press Metal Aluminium",            "Industrial",    "FBM70", "large"),
    # REITs
    ("5106.KL",  "Axis REIT",                        "REIT",          "FBM70", "mid"),
    ("5227.KL",  "IGB REIT",                         "REIT",          "FBM70", "large"),
    ("5079.KL",  "MRCB-Quill REIT",                  "REIT",          "FBM70", "small"),
    ("5212.KL",  "Pavilion REIT",                    "REIT",          "FBM70", "large"),
    ("5176.KL",  "Sunway REIT",                      "REIT",          "FBM70", "large"),
    # Property
    ("1061.KL",  "IOI Properties Group",             "Property",      "FBM70", "large"),
    ("8583.KL",  "Mah Sing Group",                   "Property",      "FBM70", "mid"),
    ("5288.KL",  "Sime Darby Property",              "Property",      "FBM70", "large"),
    ("8664.KL",  "SP Setia",                         "Property",      "FBM70", "large"),
    ("5148.KL",  "UEM Sunrise",                      "Property",      "FBM70", "mid"),
    # Utilities
    ("5264.KL",  "Malakoff Corporation",             "Utilities",     "FBM70", "mid"),
    ("6742.KL",  "YTL Power International",          "Utilities",     "FBM70", "large"),
    # Media / Services
    ("6399.KL",  "Astro Malaysia Holdings",          "Media",         "FBM70", "mid"),
    ("6084.KL",  "The Star Media Group",             "Media",         "FBM70", "small"),
    # Transport / Logistics
    ("5014.KL",  "Malaysia Airports Holdings",       "Transport",     "FBM70", "large"),
    ("2194.KL",  "MMC Corporation",                  "Transport",     "FBM70", "mid"),
    ("5246.KL",  "Westports Holdings",               "Transport",     "FBM70", "large"),
    # Auto
    ("5248.KL",  "Bermaz Auto",                      "Auto",          "FBM70", "mid"),
    ("5983.KL",  "MBM Resources",                    "Auto",          "FBM70", "small"),
    # Gloves
    ("7113.KL",  "Top Glove Corporation",            "Healthcare",    "FBM70", "large"),
    ("7106.KL",  "Supermax Corporation",             "Healthcare",    "FBM70", "mid"),
    # Plantation (additional)
    ("5254.KL",  "Boustead Plantations",             "Plantation",    "FBM70", "mid"),
    ("5222.KL",  "FGV Holdings",                     "Plantation",    "FBM70", "large"),
    ("5012.KL",  "Ta Ann Holdings",                  "Plantation",    "FBM70", "mid"),
]


def seed():
    init_db()
    with db_session() as conn:
        for ticker, name, sector, index_member, tier in FBM70_UNIVERSE:
            conn.execute(
                """
                INSERT OR REPLACE INTO stock_universe
                  (ticker, name, sector, index_member, market_cap_tier)
                VALUES (?, ?, ?, ?, ?)
                """,
                (ticker, name, sector, index_member, tier),
            )
    print(f"Seeded {len(FBM70_UNIVERSE)} stocks into stock_universe")


if __name__ == "__main__":
    seed()
