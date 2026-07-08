"""Nightly SQLite backup: sqlite3 .backup + gzip, with retention pruning.

Run standalone (`python scripts/backup_db.py`) or via
ResearchDaemon._process_backup() on its daily schedule.
"""
import gzip
import logging
import shutil
import sqlite3
from datetime import datetime, timedelta
from pathlib import Path

from config.settings import DB_PATH, BASE_DIR

logger = logging.getLogger(__name__)

BACKUP_DIR = BASE_DIR / "backups"
RETENTION_DAYS = 14


def run_backup() -> dict:
    BACKUP_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    dest = BACKUP_DIR / f"openclaw_{stamp}.db"

    src_conn = sqlite3.connect(str(DB_PATH))
    dst_conn = sqlite3.connect(str(dest))
    try:
        src_conn.backup(dst_conn)
    finally:
        dst_conn.close()
        src_conn.close()

    gz_path = Path(str(dest) + ".gz")
    with open(dest, "rb") as f_in, gzip.open(gz_path, "wb") as f_out:
        shutil.copyfileobj(f_in, f_out)
    dest.unlink()

    pruned = _prune_old_backups()
    logger.info(f"[Backup] Wrote {gz_path.name} ({gz_path.stat().st_size} bytes), pruned {pruned} old backups")
    return {"file": str(gz_path), "size_bytes": gz_path.stat().st_size, "pruned": pruned}


def _prune_old_backups() -> int:
    cutoff = datetime.utcnow() - timedelta(days=RETENTION_DAYS)
    pruned = 0
    for f in BACKUP_DIR.glob("openclaw_*.db.gz"):
        if datetime.utcfromtimestamp(f.stat().st_mtime) < cutoff:
            f.unlink()
            pruned += 1
    return pruned


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    result = run_backup()
    print(result)


if __name__ == "__main__":
    main()
