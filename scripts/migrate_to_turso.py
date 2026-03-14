"""
One-time migration script: copies all data from local SQLite → Turso (libsql).

Usage:
    1. Make sure your .env file has TURSO_DATABASE_URL and TURSO_AUTH_TOKEN set.
    2. Run from the project root:
           python scripts/migrate_to_turso.py

The script is safe to re-run — it uses INSERT OR REPLACE / INSERT OR IGNORE
so existing rows won't be duplicated.
"""

import sqlite3
import os
import sys
from pathlib import Path

# ── make sure project root is on the path ────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv

load_dotenv(ROOT / ".env")

try:
    import libsql_experimental as libsql

    TURSO_DRIVER = "libsql_experimental"

    def connect_turso(url: str, token: str):
        return libsql.connect(database=url, auth_token=token)

    def turso_execute(conn, query: str, params=()):
        return conn.execute(query, params)

    def turso_commit(conn):
        conn.commit()

except ImportError:
    import libsql_client

    TURSO_DRIVER = "libsql_client"

    def _normalize_turso_url(url: str) -> str:
        if url.startswith("libsql://"):
            return "https://" + url[len("libsql://"):]
        return url

    def connect_turso(url: str, token: str):
        return libsql_client.create_client_sync(
            url=_normalize_turso_url(url),
            auth_token=token,
        )

    def turso_execute(conn, query: str, params=()):
        return conn.execute(query, params)

    def turso_commit(conn):
        # libsql_client executes each statement immediately.
        return None

# ── paths ─────────────────────────────────────────────────────────────────────
SQLITE_PATH = ROOT / "sentiment_history.db"

TURSO_URL   = os.environ.get("TURSO_DATABASE_URL", "")
TURSO_TOKEN = os.environ.get("TURSO_AUTH_TOKEN", "")


def check_env():
    if not SQLITE_PATH.exists():
        print(f"[ERROR] Local database not found at {SQLITE_PATH}")
        print("        Make sure you run this from the project root and the file exists.")
        sys.exit(1)
    if not TURSO_URL:
        print("[ERROR] TURSO_DATABASE_URL is not set. Add it to your .env file.")
        sys.exit(1)
    if not TURSO_TOKEN:
        print("[ERROR] TURSO_AUTH_TOKEN is not set. Add it to your .env file.")
        sys.exit(1)


def migrate_sentiment_history(src, dst):
    rows = src.execute(
        "SELECT date, daily_sentiment_decay, news_volume, log_news_volume, "
        "decayed_news_volume, high_news_regime, created_at FROM sentiment_history"
    ).fetchall()

    for row in rows:
        turso_execute(
            dst,
            """
            INSERT OR REPLACE INTO sentiment_history
                (date, daily_sentiment_decay, news_volume, log_news_volume,
                 decayed_news_volume, high_news_regime, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            tuple(row),
        )
    turso_commit(dst)
    print(f"  sentiment_history : {len(rows)} rows migrated")
    return len(rows)


def migrate_prices(src, dst):
    rows = src.execute(
        "SELECT date, price, source, created_at FROM prices"
    ).fetchall()

    for row in rows:
        turso_execute(
            dst,
            """
            INSERT OR REPLACE INTO prices (date, price, source, created_at)
            VALUES (?, ?, ?, ?)
            """,
            tuple(row),
        )
    turso_commit(dst)
    print(f"  prices            : {len(rows)} rows migrated")
    return len(rows)


def migrate_news_articles(src, dst):
    rows = src.execute(
        "SELECT article_date, title, description, url, source, "
        "published_at, sentiment_score, created_at FROM news_articles"
    ).fetchall()

    for row in rows:
        turso_execute(
            dst,
            """
            INSERT OR IGNORE INTO news_articles
                (article_date, title, description, url, source,
                 published_at, sentiment_score, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            tuple(row),
        )
    turso_commit(dst)
    print(f"  news_articles     : {len(rows)} rows migrated")
    return len(rows)


def migrate_predictions(src, dst):
    rows = src.execute(
        "SELECT generated_at, last_price_date, last_price, forecasts, created_at "
        "FROM predictions"
    ).fetchall()

    for row in rows:
        turso_execute(
            dst,
            """
            INSERT INTO predictions
                (generated_at, last_price_date, last_price, forecasts, created_at)
            VALUES (?, ?, ?, ?, ?)
            """,
            tuple(row),
        )
    turso_commit(dst)
    print(f"  predictions       : {len(rows)} rows migrated")
    return len(rows)


def ensure_turso_schema(dst):
    """Create tables in Turso if they don't exist yet."""
    turso_execute(dst, """
        CREATE TABLE IF NOT EXISTS sentiment_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            date TEXT UNIQUE NOT NULL,
            daily_sentiment_decay REAL NOT NULL,
            news_volume INTEGER NOT NULL,
            log_news_volume REAL NOT NULL,
            decayed_news_volume REAL NOT NULL,
            high_news_regime INTEGER NOT NULL,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP
        )
    """)
    turso_execute(dst, "CREATE INDEX IF NOT EXISTS idx_sentiment_date ON sentiment_history(date)")

    turso_execute(dst, """
        CREATE TABLE IF NOT EXISTS prices (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            date TEXT UNIQUE NOT NULL,
            price REAL NOT NULL,
            source TEXT DEFAULT 'yahoo_finance',
            created_at TEXT DEFAULT CURRENT_TIMESTAMP
        )
    """)
    turso_execute(dst, "CREATE INDEX IF NOT EXISTS idx_prices_date ON prices(date)")

    turso_execute(dst, """
        CREATE TABLE IF NOT EXISTS news_articles (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            article_date TEXT NOT NULL,
            title TEXT NOT NULL,
            description TEXT,
            url TEXT UNIQUE,
            source TEXT,
            published_at TEXT,
            sentiment_score REAL,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP
        )
    """)
    turso_execute(dst, "CREATE INDEX IF NOT EXISTS idx_articles_date ON news_articles(article_date)")

    turso_execute(dst, """
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            generated_at TEXT NOT NULL,
            last_price_date TEXT NOT NULL,
            last_price REAL NOT NULL,
            forecasts TEXT NOT NULL,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP
        )
    """)
    turso_commit(dst)
    print("  Turso schema ready.")


def table_exists(conn, name):
    row = conn.execute(
        "SELECT count(*) FROM sqlite_master WHERE type='table' AND name=?", (name,)
    ).fetchone()
    return row and row[0] > 0


def main():
    check_env()

    print(f"\nSource : {SQLITE_PATH}")
    print(f"Target : {TURSO_URL}\n")

    src = sqlite3.connect(str(SQLITE_PATH))
    dst = connect_turso(TURSO_URL, TURSO_TOKEN)
    print(f"Driver : {TURSO_DRIVER}")

    print("Ensuring Turso schema...")
    ensure_turso_schema(dst)

    print("\nMigrating data...")
    total = 0

    if table_exists(src, "sentiment_history"):
        total += migrate_sentiment_history(src, dst)
    else:
        print("  sentiment_history : table not found in source, skipping")

    if table_exists(src, "prices"):
        total += migrate_prices(src, dst)
    else:
        print("  prices            : table not found in source, skipping")

    if table_exists(src, "news_articles"):
        total += migrate_news_articles(src, dst)
    else:
        print("  news_articles     : table not found in source, skipping")

    if table_exists(src, "predictions"):
        total += migrate_predictions(src, dst)
    else:
        print("  predictions       : table not found in source, skipping")

    src.close()
    try:
        dst.close()
    except Exception:
        pass
    print(f"\nDone! {total} total rows copied to Turso.")


if __name__ == "__main__":
    main()
