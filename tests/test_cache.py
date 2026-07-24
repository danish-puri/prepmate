"""The SQLite key-value cache.

The adapters lean on two subtleties here: a stored empty dict is a real cached
"not found" and must not read back as a miss, and max_age=None means the entry
never goes stale (closed chess.com archives).
"""

import pytest

from backend import cache


class Clock:
    def __init__(self, now=1_000_000):
        self.now = now

    def time(self):
        return self.now


@pytest.fixture
def clock(monkeypatch):
    c = Clock()
    monkeypatch.setattr(cache, "time", c)
    return c


def test_missing_key_is_none():
    assert cache.get("nothing:here") is None


@pytest.mark.parametrize("value", [
    {"a": 1},
    [1, 2, 3],
    "text",
    42,
    {"nested": {"list": [1, {"deep": True}]}},
])
def test_roundtrip_preserves_the_value(value):
    cache.put("k", value)
    assert cache.get("k") == value


def test_put_overwrites():
    cache.put("k", {"v": 1})
    cache.put("k", {"v": 2})
    assert cache.get("k") == {"v": 2}


def test_empty_dict_is_a_hit_not_a_miss():
    """Adapters store {} to remember a 404, then treat it as 'known absent'."""
    cache.put("fide:profile:1", {})
    assert cache.get("fide:profile:1") == {}
    assert cache.get("fide:profile:1") is not None


def test_no_max_age_never_expires(clock):
    cache.put("archive", {"games": []})
    clock.now += 86400 * 3650
    assert cache.get("archive") == {"games": []}


def test_value_survives_until_max_age(clock):
    cache.put("live", {"v": 1})
    clock.now += 60
    assert cache.get("live", max_age=60) == {"v": 1}


def test_value_is_dropped_past_max_age(clock):
    cache.put("live", {"v": 1})
    clock.now += 61
    assert cache.get("live", max_age=60) is None


def test_expiry_does_not_delete_the_row(clock):
    """A stale read returns None but leaves the row, so a later put replaces it."""
    cache.put("live", {"v": 1})
    clock.now += 100
    assert cache.get("live", max_age=60) is None
    assert cache.get("live") == {"v": 1}


def test_delete_prefix_removes_only_matching_keys():
    cache.put("lichess:games:alice:1", {"a": 1})
    cache.put("lichess:games:alice:2", {"a": 2})
    cache.put("lichess:games:bob:1", {"b": 1})
    cache.put("chesscom:profile:alice", {"c": 1})

    assert cache.delete_prefix("lichess:games:alice") == 2
    assert cache.get("lichess:games:alice:1") is None
    assert cache.get("lichess:games:bob:1") == {"b": 1}
    assert cache.get("chesscom:profile:alice") == {"c": 1}


def test_delete_prefix_with_no_matches():
    cache.put("k", {"v": 1})
    assert cache.delete_prefix("absent:") == 0
    assert cache.get("k") == {"v": 1}


def test_keys_are_case_sensitive():
    cache.put("lichess:user:alice", {"v": 1})
    assert cache.get("lichess:user:Alice") is None
