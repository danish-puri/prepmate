"""/api/patterns: the neural pattern engine as the website serves it."""

import random

import chess
import pytest
from fastapi.testclient import TestClient

from backend import main
from backend.adapters import chesscom, lichess
from tests.conftest import make_game


def _random_game(seed: int, color: str, result: str, end_time: int):
    """A legal game long enough to reach the plies the engine reads."""
    rng = random.Random(seed)
    board, sans = chess.Board(), []
    while len(sans) < 70 and not board.is_game_over():
        move = rng.choice(list(board.legal_moves))
        sans.append(board.san(move))
        board.push(move)
    return make_game(platform="lichess", color=color, result=result, end_time=end_time, moves=sans)


GAMES = [_random_game(i, "white" if i % 2 else "black", ("win", "draw", "loss")[i % 3], 1750000000 + i)
         for i in range(40)]


@pytest.fixture
def client(monkeypatch):
    async def li_user(client, username):
        return {"username": username}

    async def li_games(client, username, **kwargs):
        return list(GAMES), False

    monkeypatch.setattr(lichess, "get_user", li_user)
    monkeypatch.setattr(lichess, "get_games", li_games)
    monkeypatch.setattr(main, "_pattern_encoder", None)
    with TestClient(main.app) as c:
        yield c


def test_report_shape(client):
    r = client.get("/api/patterns?lichess=alice")
    assert r.status_code == 200
    d = r.json()
    assert d["games_analysed"] == 40
    assert d["space"] == "cnn64x4d128-cf79a4ca"
    assert d["stats"]["encoded"] > 0
    assert d["predictability"]["available"] is True
    assert d["weaknesses"] and d["strengths"]
    assert d["window"] == {"months": 6, "max_games": main.PATTERN_MAX_GAMES,
                           "tc": ["blitz", "classical", "rapid"]}


def test_exemplars_come_back_in_real_orientation(client):
    d = client.get("/api/patterns?lichess=alice").json()
    for a in d["weaknesses"] + d["strengths"]:
        board = chess.Board(a["exemplar_board_fen"])
        # the player is to move in the real game, whichever colour they had
        assert board.turn == (chess.WHITE if a["exemplar_color"] == "white" else chess.BLACK)
        assert a["exemplar_move_san"]
        board.push_san(a["exemplar_move_san"])


def test_second_request_is_served_from_the_cache(client, monkeypatch):
    first = client.get("/api/patterns?lichess=alice").json()

    def boom(games):
        raise AssertionError("scouted twice")

    monkeypatch.setattr(main, "_run_patterns", boom)
    assert client.get("/api/patterns?lichess=alice").json()["weaknesses"] == first["weaknesses"]


def test_only_the_most_recent_games_are_read(client, monkeypatch):
    monkeypatch.setattr(main, "PATTERN_MAX_GAMES", 10)
    d = client.get("/api/patterns?lichess=alice").json()
    assert d["games_analysed"] == 10


def test_rejects_unknown_tc(client):
    assert client.get("/api/patterns?lichess=alice&tc=hyperbullet").status_code == 422


def test_needs_a_game_source(client):
    assert client.get("/api/patterns").status_code == 422


def test_other_endpoints_never_load_the_engine(client, monkeypatch):
    async def cc_profile(client, username):
        return None

    monkeypatch.setattr(chesscom, "get_profile", cc_profile)
    client.get("/api/openings?lichess=alice")
    assert main._pattern_encoder is None
