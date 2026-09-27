"""Persistent embedding cache, keyed by canonical position.

A position's embedding does not depend on who reached it: the encoder sees a
board and nothing else. So the table is keyed by position rather than by player.

What that buys is re-scouting. Reopening a dossier, or adding a week of new
games, encodes only what is genuinely new. It does *not* meaningfully help
across different players: two players measured at 0.35% shared positions, since
everything before move eight is skipped and after move eight games diverge.

Rows are (space, key). `space` is the encoder's name, which must change
whenever the vector space changes, otherwise a hit returns a vector from an
older model and every distance downstream is quietly wrong.
"""

import sqlite3

import numpy as np

from .. import cache

# SQLite caps host parameters per statement; stay well under it
_CHUNK = 500


def _conn() -> sqlite3.Connection:
    # cache.DB_PATH is read at call time so tests inherit the tmp database
    conn = sqlite3.connect(cache.DB_PATH)
    conn.execute(
        "CREATE TABLE IF NOT EXISTS embeddings ("
        "  space TEXT NOT NULL,"
        "  key INTEGER NOT NULL,"
        "  vec BLOB NOT NULL,"
        "  PRIMARY KEY (space, key)"
        ") WITHOUT ROWID"
    )
    return conn


def get_many(space: str, keys: list[int]) -> dict[int, np.ndarray]:
    """Cached vectors for whichever keys are present. float16 on disk: the
    precision is far below what clustering can notice and it halves the file."""
    found: dict[int, np.ndarray] = {}
    with _conn() as conn:
        for i in range(0, len(keys), _CHUNK):
            chunk = keys[i: i + _CHUNK]
            rows = conn.execute(
                f"SELECT key, vec FROM embeddings WHERE space = ? AND key IN ({','.join('?' * len(chunk))})",
                (space, *chunk),
            ).fetchall()
            for key, blob in rows:
                found[key] = np.frombuffer(blob, dtype=np.float16).astype(np.float32)
    return found


def put_many(space: str, vectors: dict[int, np.ndarray]) -> None:
    with _conn() as conn:
        conn.executemany(
            "INSERT OR REPLACE INTO embeddings (space, key, vec) VALUES (?, ?, ?)",
            [(space, key, v.astype(np.float16).tobytes()) for key, v in vectors.items()],
        )


def count(space: str) -> int:
    with _conn() as conn:
        return conn.execute("SELECT COUNT(*) FROM embeddings WHERE space = ?", (space,)).fetchone()[0]


def drop(space: str) -> int:
    """Forget a vector space. Needed when a model is retrained under a name
    that was already used."""
    with _conn() as conn:
        return conn.execute("DELETE FROM embeddings WHERE space = ?", (space,)).rowcount
