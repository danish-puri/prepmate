"""Shared fixtures: an isolated cache DB and small builders for fake data."""

import pytest

from backend import cache
from backend.models import Game


@pytest.fixture(autouse=True)
def tmp_cache(tmp_path, monkeypatch):
    monkeypatch.setattr(cache, "DB_PATH", tmp_path / "cache.db")


@pytest.fixture
def anyio_backend():
    return "asyncio"


def make_game(platform="lichess", time_class="blitz", color="white", result="win",
              eco="A00", end_time=1750000000, moves=("e4", "e5")):
    return Game(platform=platform, color=color, time_class=time_class, result=result,
                termination="resigned", eco=eco, opening="Test Opening",
                end_time=end_time, player_rating=1800, opponent_rating=1800,
                moves=list(moves))


class FakeResponse:
    def __init__(self, json_data=None, text="", status_code=200):
        self._json = json_data
        self.text = text
        self.status_code = status_code
        self.headers = {}

    def json(self):
        return self._json

    def raise_for_status(self):
        assert self.status_code < 400


class FakeClient:
    """Maps URL substrings to responses and records every request."""

    def __init__(self, routes):
        self.routes = routes
        self.calls = []

    async def get(self, url, **kwargs):
        self.calls.append((url, kwargs))
        for fragment, response in self.routes.items():
            if fragment in url:
                return response
        raise AssertionError(f"unexpected request: {url}")
