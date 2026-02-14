from __future__ import annotations

import argparse
import os
import sqlite3
import sys
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple


def normalize_postgres_database_url(url: str) -> str:
    url = url.strip()
    if url.startswith("postgres://"):
        return f"postgresql://{url[len('postgres://') :]}"
    return url


def sqlite_table_exists(conn: sqlite3.Connection, table_name: str) -> bool:
    row = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = ?",
        (table_name,),
    ).fetchone()
    return row is not None


def batched_rows(cursor: sqlite3.Cursor, batch_size: int) -> Iterable[List[Tuple]]:
    while True:
        rows = cursor.fetchmany(batch_size)
        if not rows:
            return
        yield rows


def ensure_target_schema(pg_conn: object) -> None:
    with pg_conn.cursor() as cur:
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS daily_prices (
                ticker TEXT NOT NULL,
                date TEXT NOT NULL,
                close DOUBLE PRECISION NOT NULL,
                volume DOUBLE PRECISION,
                PRIMARY KEY (ticker, date)
            )
            """
        )
        cur.execute("ALTER TABLE daily_prices ADD COLUMN IF NOT EXISTS volume DOUBLE PRECISION")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_daily_prices_date ON daily_prices (date)")
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS fundamentals_snapshot (
                ticker TEXT PRIMARY KEY,
                asof_date TEXT NOT NULL,
                trailing_pe DOUBLE PRECISION,
                forward_pe DOUBLE PRECISION,
                price_to_book DOUBLE PRECISION,
                trailing_eps DOUBLE PRECISION,
                forward_eps DOUBLE PRECISION,
                quote_type TEXT
            )
            """
        )
        cur.execute("CREATE INDEX IF NOT EXISTS idx_fundamentals_asof_date ON fundamentals_snapshot (asof_date)")
    pg_conn.commit()


def migrate_daily_prices(sqlite_conn: sqlite3.Connection, pg_conn: object, batch_size: int) -> int:
    if not sqlite_table_exists(sqlite_conn, "daily_prices"):
        return 0

    select_sql = "SELECT ticker, date, close, volume FROM daily_prices ORDER BY ticker, date"
    upsert_sql = """
        INSERT INTO daily_prices (ticker, date, close, volume)
        VALUES (%s, %s, %s, %s)
        ON CONFLICT(ticker, date) DO UPDATE SET
            close = excluded.close,
            volume = COALESCE(excluded.volume, daily_prices.volume)
    """

    total_rows = 0
    src_cur = sqlite_conn.cursor()
    src_cur.execute(select_sql)
    for rows in batched_rows(src_cur, batch_size):
        with pg_conn.cursor() as dst_cur:
            dst_cur.executemany(upsert_sql, rows)
        pg_conn.commit()
        total_rows += len(rows)
        if total_rows % (batch_size * 10) == 0:
            print(f"[progress] migrated {total_rows} daily price rows")

    return total_rows


def migrate_fundamentals(sqlite_conn: sqlite3.Connection, pg_conn: object, batch_size: int) -> int:
    if not sqlite_table_exists(sqlite_conn, "fundamentals_snapshot"):
        return 0

    select_sql = (
        "SELECT ticker, asof_date, trailing_pe, forward_pe, price_to_book, trailing_eps, forward_eps, quote_type "
        "FROM fundamentals_snapshot ORDER BY ticker"
    )
    upsert_sql = """
        INSERT INTO fundamentals_snapshot (
            ticker, asof_date, trailing_pe, forward_pe, price_to_book, trailing_eps, forward_eps, quote_type
        )
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
        ON CONFLICT(ticker) DO UPDATE SET
            asof_date = excluded.asof_date,
            trailing_pe = excluded.trailing_pe,
            forward_pe = excluded.forward_pe,
            price_to_book = excluded.price_to_book,
            trailing_eps = excluded.trailing_eps,
            forward_eps = excluded.forward_eps,
            quote_type = excluded.quote_type
    """

    total_rows = 0
    src_cur = sqlite_conn.cursor()
    src_cur.execute(select_sql)
    for rows in batched_rows(src_cur, batch_size):
        with pg_conn.cursor() as dst_cur:
            dst_cur.executemany(upsert_sql, rows)
        pg_conn.commit()
        total_rows += len(rows)

    return total_rows


def truncate_target_tables(pg_conn: object) -> None:
    with pg_conn.cursor() as cur:
        cur.execute("TRUNCATE TABLE daily_prices, fundamentals_snapshot")
    pg_conn.commit()


def parse_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Migrate local SQLite data to a cloud Postgres database.")
    parser.add_argument(
        "--source",
        type=Path,
        default=Path("data") / "price_history.sqlite3",
        help="Path to the source SQLite database file.",
    )
    parser.add_argument(
        "--database-url",
        default=os.getenv("DATABASE_URL", ""),
        help="Target Postgres URL. Defaults to DATABASE_URL env var.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=5000,
        help="Number of rows per write batch.",
    )
    parser.add_argument(
        "--truncate-target",
        action="store_true",
        help="Clear target tables before migration.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)

    source_path = Path(args.source).expanduser().resolve()
    if not source_path.exists():
        print(f"[error] source SQLite file not found: {source_path}", file=sys.stderr)
        return 1

    db_url = normalize_postgres_database_url(str(args.database_url))
    if not db_url:
        print("[error] DATABASE_URL is required (or pass --database-url).", file=sys.stderr)
        return 1
    if not db_url.startswith("postgresql://"):
        print(
            "[error] database URL must start with postgresql:// (or postgres://).",
            file=sys.stderr,
        )
        return 1
    if int(args.batch_size) < 1:
        print("[error] --batch-size must be >= 1", file=sys.stderr)
        return 1

    try:
        import psycopg
    except ImportError:
        print("[error] Missing dependency 'psycopg'. Install with: pip install -r requirements.txt", file=sys.stderr)
        return 1

    print(f"[info] source: {source_path}")
    print("[info] connecting to cloud Postgres...")

    sqlite_conn = sqlite3.connect(source_path)
    pg_conn = psycopg.connect(db_url, connect_timeout=15)
    try:
        ensure_target_schema(pg_conn)
        if args.truncate_target:
            print("[info] truncating target tables...")
            truncate_target_tables(pg_conn)

        print("[info] migrating daily_prices...")
        daily_rows = migrate_daily_prices(sqlite_conn, pg_conn, int(args.batch_size))
        print("[info] migrating fundamentals_snapshot...")
        fundamental_rows = migrate_fundamentals(sqlite_conn, pg_conn, int(args.batch_size))

        print(f"[done] migrated {daily_rows} daily price rows")
        print(f"[done] migrated {fundamental_rows} fundamentals rows")
        return 0
    finally:
        sqlite_conn.close()
        pg_conn.close()


if __name__ == "__main__":
    raise SystemExit(main())
