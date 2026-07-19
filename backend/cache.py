"""SQLite key-value cache.

Closed chess.com monthly archives never change, so they cache forever
(max_age=None). Live data (current month, lichess exports, profiles) passes a
TTL in seconds.
"""

import json
import sqlite3
import time
from pathlib import Path

DB_PATH = Path(__file__).parent / "cache.db"


def _conn() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.execute(
        "CREATE TABLE IF NOT EXISTS kv ("
        "  key TEXT PRIMARY KEY,"
        "  value TEXT NOT NULL,"
        "  fetched_at INTEGER NOT NULL"
        ")"
    )
    return conn


def get(key: str, max_age: float | None = None):
    with _conn() as conn:
        row = conn.execute("SELECT value, fetched_at FROM kv WHERE key = ?", (key,)).fetchone()
    if row is None:
        return None
    value, fetched_at = row
    if max_age is not None and time.time() - fetched_at > max_age:
        return None
    return json.loads(value)


def put(key: str, value) -> None:
    with _conn() as conn:
        conn.execute(
            "INSERT OR REPLACE INTO kv (key, value, fetched_at) VALUES (?, ?, ?)",
            (key, json.dumps(value), int(time.time())),
        )


def delete_prefix(prefix: str) -> int:
    with _conn() as conn:
        cur = conn.execute("DELETE FROM kv WHERE key LIKE ?", (prefix + "%",))
        return cur.rowcount
