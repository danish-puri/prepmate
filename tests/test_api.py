"""/api/lookup, /profile, /ratings, /recent and /refresh.

The adapters are stubbed so nothing here touches the network, but the response
shaping in main.py runs for real, since that is what the frontend reads.
"""

import httpx
import pytest
from fastapi.testclient import TestClient

from backend import cache, main
from backend.adapters import chesscom, fide, lichess
from tests.conftest import make_game

CC_PROFILE = {
    "username": "hikaru", "name": "Hikaru Nakamura", "title": "GM",
    "country": "https://api.chess.com/pub/country/US",
    "joined": 1234567890, "url": "https://www.chess.com/member/hikaru",
}
CC_STATS = {
    "chess_blitz": {"last": {"rating": 3200}},
    "chess_rapid": {"last": {"rating": 2900}},
    "chess_bullet": {},  # present but never played, so it must not appear
}
LI_USER = {
    "username": "penguingm1", "title": "GM", "createdAt": 1234567890,
    "count": {"rated": 40000}, "url": "https://lichess.org/@/penguingm1",
    "perfs": {"blitz": {"rating": 3000, "games": 500},
              "rapid": {"rating": 2800, "games": 0}},  # 0 games, so no rating shown
}
FIDE_PROFILE = {
    "fide_id": "2016192", "name": "Hikaru Nakamura", "federation": "United States",
    "title": "Grandmaster", "birth_year": 1987, "ratings": {"std": 2802},
    "history": {"std": [{"date": "2026-07-01", "rating": 2802, "games": 0}]},
    "url": "https://ratings.fide.com/profile/2016192",
}


@pytest.fixture
def stub(monkeypatch):
    """Registers adapter stubs. Each `known` name resolves, everything else 404s."""
    def configure(cc_users=(), li_users=(), fide_ids=(), **overrides):
        async def cc_profile(client, username):
            return {**CC_PROFILE, "username": username} if username in cc_users else None

        async def cc_stats(client, username):
            return CC_STATS if username in cc_users else None

        async def li_user(client, username):
            return {**LI_USER, "username": username} if username in li_users else None

        async def fide_profile(client, fide_id):
            return {**FIDE_PROFILE, "fide_id": fide_id} if fide_id in fide_ids else None

        defaults = {"get_profile": cc_profile, "get_stats": cc_stats}
        for name, fn in {**defaults, **overrides.get("chesscom", {})}.items():
            monkeypatch.setattr(chesscom, name, fn)
        for name, fn in {**{"get_user": li_user}, **overrides.get("lichess", {})}.items():
            monkeypatch.setattr(lichess, name, fn)
        for name, fn in {**{"get_profile": fide_profile}, **overrides.get("fide", {})}.items():
            monkeypatch.setattr(fide, name, fn)
        return TestClient(main.app)
    return configure


# --- /api/lookup -------------------------------------------------------

def test_lookup_found_on_both_platforms(stub):
    with stub(cc_users={"hikaru"}, li_users={"hikaru"}) as client:
        d = client.get("/api/lookup?q=hikaru").json()
    assert d["query"] == "hikaru"
    assert d["chesscom"]["found"] is True and d["lichess"]["found"] is True
    assert d["fide"] == {"found": False, "searchable": False}


def test_lookup_shapes_the_chesscom_block(stub):
    with stub(cc_users={"hikaru"}) as client:
        cc = client.get("/api/lookup?q=hikaru").json()["chesscom"]
    assert cc["name"] == "Hikaru Nakamura" and cc["title"] == "GM"
    assert cc["country"] == "US"  # taken off the end of the country URL
    assert cc["ratings"] == {"blitz": 3200, "rapid": 2900}  # bullet never played


def test_lookup_shapes_the_lichess_block(stub):
    with stub(li_users={"penguingm1"}) as client:
        li = client.get("/api/lookup?q=penguingm1").json()["lichess"]
    assert li["games_total"] == 40000 and li["title"] == "GM"
    assert li["ratings"] == {"blitz": 3000}  # rapid has 0 games


def test_lookup_reports_the_platform_that_missed(stub):
    with stub(cc_users={"hikaru"}) as client:
        d = client.get("/api/lookup?q=hikaru").json()
    assert d["chesscom"]["found"] is True
    assert d["lichess"] == {"found": False}


def test_lookup_404_when_nobody_matches(stub):
    with stub() as client:
        assert client.get("/api/lookup?q=nobodyatall").status_code == 404


def test_lookup_retries_a_full_name_without_the_space(stub):
    """'Magnus Carlsen' is not a username anywhere, so the stripped form is tried."""
    with stub(cc_users={"MagnusCarlsen"}) as client:
        d = client.get("/api/lookup?q=Magnus Carlsen").json()
    assert d["query"] == "MagnusCarlsen"
    assert d["chesscom"]["found"] is True


def test_lookup_does_not_retry_when_nothing_would_change(stub):
    seen = []

    async def cc_profile(client, username):
        seen.append(username)
        return None

    with stub(chesscom={"get_profile": cc_profile}) as client:
        client.get("/api/lookup?q=hikaru")
    assert seen == ["hikaru"]  # already a legal username, so one attempt only


def test_lookup_treats_an_all_digit_query_as_a_fide_id(stub):
    with stub(fide_ids={"2016192"}) as client:
        d = client.get("/api/lookup?q=2016192").json()
    assert d["fide"]["found"] is True
    assert d["fide"]["name"] == "Hikaru Nakamura"
    assert d["fide"]["history"] is None  # the lookup card does not chart it


def test_lookup_searchable_tells_a_miss_from_a_non_id(stub):
    """The card only reads this when found is false, to pick which message to show.

    A name can never match a FIDE id, so searchable stays false and the card
    says 'search by FIDE id'. All digits that match nothing is a real miss.
    """
    with stub(cc_users={"hikaru", "999999"}) as client:
        by_name = client.get("/api/lookup?q=hikaru").json()["fide"]
        by_id = client.get("/api/lookup?q=999999").json()["fide"]
    assert by_name == {"found": False, "searchable": False}
    assert by_id == {"found": False, "searchable": True}


def test_lookup_trims_whitespace(stub):
    with stub(cc_users={"hikaru"}) as client:
        assert client.get("/api/lookup?q=  hikaru  ").json()["query"] == "hikaru"


def test_lookup_requires_a_query(stub):
    with stub() as client:
        assert client.get("/api/lookup?q=").status_code == 422
        assert client.get("/api/lookup").status_code == 422


def test_lookup_turns_an_upstream_failure_into_502(stub):
    async def boom(client, username):
        raise httpx.ConnectError("upstream is down")

    with stub(chesscom={"get_profile": boom}) as client:
        response = client.get("/api/lookup?q=hikaru")
    assert response.status_code == 502
    assert "upstream error" in response.json()["detail"]


# --- /api/profile ------------------------------------------------------

def test_profile_needs_at_least_one_handle(stub):
    with stub() as client:
        assert client.get("/api/profile").status_code == 422


def test_profile_combines_the_platforms_asked_for(stub):
    with stub(cc_users={"hikaru"}, li_users={"penguingm1"}, fide_ids={"2016192"}) as client:
        d = client.get("/api/profile?chesscom=hikaru&lichess=penguingm1&fide=2016192").json()
    assert d["chesscom"]["joined"] == 1234567890
    assert d["lichess"]["created_at"] == 1234567890
    assert d["fide"]["federation"] == "United States"
    assert "history" not in d["fide"]  # the chart pulls that from /api/ratings


def test_profile_marks_a_missing_handle_without_failing(stub):
    with stub(cc_users={"hikaru"}) as client:
        d = client.get("/api/profile?chesscom=hikaru&lichess=ghost").json()
    assert d["chesscom"]["found"] is True
    assert d["lichess"] == {"found": False}


def test_profile_404_when_every_handle_misses(stub):
    with stub() as client:
        assert client.get("/api/profile?chesscom=ghost&lichess=ghost").status_code == 404


# --- /api/ratings ------------------------------------------------------

LI_HISTORY = [
    {"name": "Blitz", "points": [[2026, 0, 15, 2900], [2026, 6, 1, 3000]]},
    {"name": "Rapid", "points": []},          # no points, so no series
    {"name": "Puzzles", "points": [[2026, 0, 1, 2500]]},  # not a real time class
]


def test_ratings_needs_at_least_one_handle(stub):
    with stub() as client:
        assert client.get("/api/ratings").status_code == 422


def test_ratings_converts_lichess_zero_indexed_months(stub):
    async def history(client, username):
        return LI_HISTORY

    with stub(li_users={"a"}, lichess={"get_rating_history": history}) as client:
        d = client.get("/api/ratings?lichess=a").json()
    assert d["lichess"]["blitz"] == [
        {"date": "2026-01-15", "rating": 2900},   # month 0 is January
        {"date": "2026-07-01", "rating": 3000},
    ]


def test_ratings_drops_empty_and_unknown_perfs(stub):
    async def history(client, username):
        return LI_HISTORY

    with stub(li_users={"a"}, lichess={"get_rating_history": history}) as client:
        d = client.get("/api/ratings?lichess=a").json()
    assert list(d["lichess"]) == ["blitz"]


def test_ratings_rebuilds_chesscom_history_from_games(stub):
    """chess.com has no history endpoint, so it comes from post-game ratings."""
    async def games(client, username, since):
        return [make_game(time_class="blitz", end_time=1767225600, player_rating=3180),
                make_game(time_class="blitz", end_time=1767312000, player_rating=3200)]

    with stub(cc_users={"hikaru"}, chesscom={"get_games": games}) as client:
        d = client.get("/api/ratings?chesscom=hikaru").json()
    assert d["chesscom_current"] == {"blitz": 3200, "rapid": 2900}
    assert d["chesscom"]["blitz"] == [
        {"date": "2026-01-01", "rating": 3180},
        {"date": "2026-01-02", "rating": 3200},
    ]


def test_ratings_returns_the_fide_history(stub):
    with stub(fide_ids={"2016192"}) as client:
        d = client.get("/api/ratings?fide=2016192").json()
    assert d["fide"]["std"][0]["rating"] == 2802


def test_ratings_404_when_nothing_resolves(stub):
    async def history(client, username):
        return None

    with stub(lichess={"get_rating_history": history}) as client:
        assert client.get("/api/ratings?lichess=ghost&fide=999").status_code == 404


# --- /api/recent -------------------------------------------------------

@pytest.fixture
def recent_client(stub):
    async def cc_games(client, username, since):
        return [make_game("chesscom", end_time=1767225600 + i, result="win")
                for i in range(5)]

    async def li_games(client, username, max_games, since, time_classes):
        return [make_game("lichess", end_time=1767225600 + 100 + i, result="loss")
                for i in range(5)], False

    return stub(cc_users={"hikaru"}, li_users={"a"},
                chesscom={"get_games": cc_games}, lichess={"get_games": li_games})


def test_recent_merges_both_platforms_newest_last(recent_client):
    with recent_client as client:
        d = client.get("/api/recent?chesscom=hikaru&lichess=a").json()
    assert d["n"] == 10 and d["wins"] == 5 and d["losses"] == 5
    assert d["sequence"][0]["platform"] == "chesscom"   # oldest
    assert d["sequence"][-1]["platform"] == "lichess"   # newest


def test_recent_honours_n(recent_client):
    with recent_client as client:
        d = client.get("/api/recent?chesscom=hikaru&lichess=a&n=3").json()
    assert d["n"] == 3 and d["losses"] == 3  # the newest three are the lichess ones


def test_recent_rejects_an_out_of_range_n(recent_client):
    with recent_client as client:
        assert client.get("/api/recent?lichess=a&n=0").status_code == 422
        assert client.get("/api/recent?lichess=a&n=101").status_code == 422


def test_recent_404_for_an_unknown_player(stub):
    with stub() as client:
        assert client.get("/api/recent?chesscom=ghost").status_code == 404


# --- /api/refresh ------------------------------------------------------

def test_refresh_drops_only_that_players_keys(stub):
    for key in ("chesscom:profile:hikaru", "chesscom:stats:hikaru",
                "lichess:user:a", "lichess:games:a:1", "lichess:history:a",
                "chesscom:profile:someoneelse"):
        cache.put(key, {"v": 1})

    with stub() as client:
        d = client.post("/api/refresh?chesscom=hikaru&lichess=a").json()
    assert d["deleted_keys"] == 5
    assert cache.get("chesscom:profile:someoneelse") == {"v": 1}


def test_refresh_lowercases_the_handles(stub):
    cache.put("chesscom:profile:hikaru", {"v": 1})
    with stub() as client:
        assert client.post("/api/refresh?chesscom=HiKaRu").json()["deleted_keys"] == 1


def test_refresh_with_no_handles_is_a_no_op(stub):
    with stub() as client:
        assert client.post("/api/refresh").json() == {"deleted_keys": 0}
