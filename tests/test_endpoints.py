"""/api/openings and /api/movetree: tc filtering, coverage, validation.

Adapters are stubbed out, so these run without network or real accounts.
"""

import pytest
from fastapi.testclient import TestClient

from backend import main
from backend.adapters import chesscom, lichess
from tests.conftest import make_game

CC_GAMES = [
    make_game("chesscom", "blitz", end_time=1_767_225_600),   # 2026-01-01
    make_game("chesscom", "bullet", end_time=1_769_904_000),  # 2026-02-01
    make_game("chesscom", "daily", end_time=1_772_323_200),   # 2026-03-01
]
LI_GAMES = [
    make_game("lichess", "rapid", end_time=1_769_904_000),
    make_game("lichess", "classical", end_time=1_772_323_200),
]


@pytest.fixture
def client(monkeypatch):
    captured = {}

    async def cc_profile(client, username):
        return {"username": username}

    async def cc_games(client, username, since):
        captured["cc_since"] = since
        return list(CC_GAMES)

    async def li_user(client, username):
        return {"username": username}

    async def li_games(client, username, max_games, since, time_classes):
        captured["li_kwargs"] = {"max_games": max_games, "since": since,
                                 "time_classes": time_classes}
        if time_classes is None:
            return list(LI_GAMES), True
        return [g for g in LI_GAMES if g.time_class in time_classes], True

    monkeypatch.setattr(chesscom, "get_profile", cc_profile)
    monkeypatch.setattr(chesscom, "get_games", cc_games)
    monkeypatch.setattr(lichess, "get_user", li_user)
    monkeypatch.setattr(lichess, "get_games", li_games)
    with TestClient(main.app) as tc:
        tc.captured = captured
        yield tc


def test_openings_default_tc_drops_bullet_and_daily(client):
    d = client.get("/api/openings?chesscom=alice&lichess=alice").json()
    # 1 chesscom blitz + 2 lichess survive; chesscom bullet and daily do not
    assert d["games_analysed"] == 3
    assert d["coverage"]["chesscom"]["games"] == 1
    assert d["coverage"]["lichess"]["games"] == 2
    assert d["params"]["tc"] == ["blitz", "classical", "rapid"]


def test_openings_tc_param_filters_both_platforms(client):
    d = client.get("/api/openings?chesscom=alice&lichess=alice&tc=bullet,daily").json()
    assert d["games_analysed"] == 2
    assert d["coverage"]["chesscom"]["games"] == 2
    assert d["coverage"]["lichess"]["games"] == 0
    # the requested classes reach the lichess fetch so ?max= is spent on them
    assert client.captured["li_kwargs"]["time_classes"] == {"bullet", "daily"}


def test_openings_same_window_reaches_both_adapters(client):
    client.get("/api/openings?chesscom=alice&lichess=alice&months=3")
    assert client.captured["cc_since"] == client.captured["li_kwargs"]["since"]
    assert client.captured["cc_since"] == main._window_start(3)


def test_openings_coverage_dates_and_truncation(client):
    d = client.get("/api/openings?chesscom=alice&lichess=alice").json()
    assert d["coverage"]["lichess"] == {"games": 2, "from": "2026-02-01",
                                        "to": "2026-03-01", "truncated": True}
    assert d["coverage"]["chesscom"]["truncated"] is False


def test_openings_rejects_unknown_tc(client):
    assert client.get("/api/openings?lichess=alice&tc=hyperbullet").status_code == 422


def test_movetree_takes_tc_and_reports_coverage(client):
    d = client.get("/api/movetree?chesscom=alice&lichess=alice&tc=rapid").json()
    assert d["games_analysed"] == 1
    assert d["params"]["tc"] == ["rapid"]
    assert d["coverage"]["lichess"]["games"] == 1


def test_movetree_payload_shape(client):
    d = client.get("/api/movetree?chesscom=alice&lichess=alice").json()
    w = d["white"]
    # 1 chesscom blitz + 2 lichess games survive the default tc, all as White
    assert w["n"] == 3 and w["score"] == 100.0
    e4 = w["moves"][0]
    assert (e4["san"], e4["games"], e4["freq_pct"]) == ("e4", 3, 100.0)
    assert e4["children"][0]["san"] == "e5"
    assert d["black"] == {"n": 0, "score": 0.0, "moves": []}


def test_movetree_depth_and_min_games_params(client):
    # every fixture game opens e4, so min_games above the game count empties the tree
    d = client.get("/api/movetree?lichess=alice&min_games=3").json()
    assert d["white"]["n"] == 2 and d["white"]["moves"] == []
    d = client.get("/api/movetree?lichess=alice&depth=2&min_games=1").json()
    e5 = d["white"]["moves"][0]["children"][0]
    assert e5["san"] == "e5" and e5["children"] == []
    assert client.get("/api/movetree?lichess=alice&depth=1").status_code == 422


def test_performance_keeps_every_time_class(client):
    d = client.get("/api/performance?chesscom=alice&lichess=alice").json()
    assert d["games_analysed"] == 5
    assert client.captured["li_kwargs"]["time_classes"] is None
