"""Settings that switch on when the app runs on Vercel."""

import importlib
from pathlib import Path

from backend import cache, main


def test_cache_moves_to_tmp_on_vercel(monkeypatch):
    monkeypatch.delenv("CACHE_DB", raising=False)
    monkeypatch.setenv("VERCEL", "1")
    assert cache._default_path() == Path("/tmp/prepmate-cache.db")
    monkeypatch.delenv("VERCEL")
    assert cache._default_path() == Path(cache.__file__).parent / "cache.db"


def test_rate_limit_keys_on_x_real_ip_on_vercel(monkeypatch):
    monkeypatch.delenv("CLIENT_IP_HEADER", raising=False)
    monkeypatch.setenv("VERCEL", "1")
    try:
        assert importlib.reload(main).CLIENT_IP_HEADER == "x-real-ip"
        monkeypatch.delenv("VERCEL")
        assert importlib.reload(main).CLIENT_IP_HEADER == ""
    finally:
        importlib.reload(main)


def test_root_entrypoint_is_the_real_app():
    import app as entry
    assert entry.app is main.app
