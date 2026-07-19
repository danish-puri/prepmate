"""Both adapters honor the shared window and time-class vocabulary."""

import json
from datetime import datetime, timezone

import pytest

from backend.adapters import chesscom, lichess
from tests.conftest import FakeClient, FakeResponse

SINCE = datetime(2026, 2, 1, tzinfo=timezone.utc)


def li_line(speed="blitz", created_ms=1750000000000):
    return json.dumps({
        "players": {"white": {"user": {"id": "alice"}, "rating": 1900},
                    "black": {"user": {"id": "bob"}, "rating": 1850}},
        "status": "resign", "winner": "white", "speed": speed,
        "opening": {"eco": "B01", "name": "Scandinavian Defense"},
        "createdAt": created_ms, "moves": "e4 d5",
    })


def li_client(lines):
    return FakeClient({"/games/user/": FakeResponse(text="\n".join(lines))})


@pytest.mark.anyio
async def test_lichess_perftype_and_since_sent():
    client = li_client([li_line()])
    await lichess.get_games(client, "alice", max_games=50, since=SINCE,
                            time_classes={"blitz", "daily", "bullet"})
    params = client.calls[0][1]["params"]
    assert params["perfType"] == "blitz,bullet,ultraBullet,correspondence"
    assert params["since"] == int(SINCE.timestamp() * 1000)
    assert params["max"] == 50


@pytest.mark.anyio
async def test_lichess_no_filter_requests_every_speed():
    client = li_client([li_line()])
    await lichess.get_games(client, "alice", max_games=50, since=SINCE, time_classes=None)
    perfs = client.calls[0][1]["params"]["perfType"].split(",")
    assert set(perfs) == {"ultraBullet", "bullet", "blitz", "rapid", "classical", "correspondence"}


@pytest.mark.anyio
async def test_lichess_speed_normalized_to_shared_vocabulary():
    client = li_client([li_line("ultraBullet"), li_line("correspondence"), li_line("rapid")])
    games, _ = await lichess.get_games(client, "alice", max_games=50, since=SINCE)
    assert [g.time_class for g in games] == ["bullet", "daily", "rapid"]


@pytest.mark.anyio
async def test_lichess_truncation_flag():
    client = li_client([li_line(), li_line()])
    _, truncated = await lichess.get_games(client, "alice", max_games=2, since=SINCE)
    assert truncated
    # different username so the first call's cached lines are not reused
    _, truncated = await lichess.get_games(li_client([li_line()]), "carol", max_games=2, since=SINCE)
    assert not truncated


@pytest.mark.anyio
async def test_lichess_identical_parallel_requests_share_one_export():
    import asyncio
    client = li_client([li_line()])
    results = await asyncio.gather(
        lichess.get_games(client, "alice", max_games=50, since=SINCE),
        lichess.get_games(client, "alice", max_games=50, since=SINCE),
    )
    assert len(client.calls) == 1  # second call must be served from cache
    assert all(len(games) == 1 for games, _ in results)


def cc_game(time_class="blitz", end_time=1750000000):
    return {
        "rated": True, "rules": "chess", "time_class": time_class, "end_time": end_time,
        "white": {"username": "alice", "result": "win", "rating": 1900},
        "black": {"username": "bob", "result": "resigned", "rating": 1850},
        "pgn": '[ECO "B01"]\n\n1. e4 d5 1-0',
    }


@pytest.mark.anyio
async def test_chesscom_archives_selected_by_calendar_cutoff():
    base = "https://api.chess.com/pub/player/alice/games"
    client = FakeClient({
        f"{base}/archives": FakeResponse(json_data={"archives": [
            f"{base}/2025/11", f"{base}/2025/12", f"{base}/2026/02", f"{base}/2026/03",
        ]}),
        "/2026/02": FakeResponse(json_data={"games": [cc_game("bullet")]}),
        "/2026/03": FakeResponse(json_data={"games": [cc_game("blitz"), cc_game("rapid")]}),
    })
    games = await chesscom.get_games(client, "alice", since=SINCE)
    # pre-cutoff archives (2025/11, 2025/12) are never requested
    fetched = [url for url, _ in client.calls if "/20" in url]
    assert fetched == [f"{base}/2026/02", f"{base}/2026/03"]
    # bullet stays in the raw fetch; filtering is the endpoint's job
    assert sorted(g.time_class for g in games) == ["blitz", "bullet", "rapid"]
