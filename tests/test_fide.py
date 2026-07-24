"""The FIDE adapter.

FIDE has no API, so this module scrapes HTML and is the piece most likely to
break when the site changes. These tests pin the parse against a page shaped
like the real one, so a redesign fails here loudly instead of quietly feeding
wrong numbers into the report.
"""

import pytest

from backend import cache
from backend.adapters import fide
from tests.conftest import FakeClient, FakeResponse

pytestmark = pytest.mark.anyio

PROFILE_HTML = """
<html><head><title>Magnus Carlsen FIDE Profile</title></head>
<body>
<div class="profile-info-id ">1503014</div>
<div class="profile-info-country ">
  <img src="/svg/NOR.svg"> Norway
</div>
<div class="profile-info-byear ">1990</div>
<h5>FIDE title</h5>
<div class="profile-info-title ">
  <p>Grandmaster</p>
</div>
<div class="profile-standart profile-game"><span>standard</span><p> 2839 </p></div>
<div class="profile-rapid profile-game"><span>rapid</span><p> 2831 </p></div>
<div class="profile-blitz profile-game"><span>blitz</span><p> 2886 </p></div>
<table>
<tr><td>&nbsp;2026-Jul&nbsp;</td><td>&nbsp;2839&nbsp;</td><td>&nbsp;10&nbsp;</td><td>&nbsp;2831&nbsp;</td><td>&nbsp;5&nbsp;</td><td>&nbsp;2886&nbsp;</td><td>&nbsp;0&nbsp;</td></tr>
<tr><td>&nbsp;2026-Jun&nbsp;</td><td>&nbsp;2833&nbsp;</td><td>&nbsp;8&nbsp;</td><td>&nbsp;2825&nbsp;</td><td>&nbsp;0&nbsp;</td><td>&nbsp;2880&nbsp;</td><td>&nbsp;4&nbsp;</td></tr>
</table>
</body></html>
"""

STATS_JSON = [{
    "white_total": "100", "white_win_num": "60", "white_draw_num": "30",
    "black_total": "80", "black_win_num": "30", "black_draw_num": "40",
    "white_total_std": "50", "white_win_num_std": "30", "white_draw_num_std": "15",
    "black_total_std": "40", "black_win_num_std": "15", "black_draw_num_std": "20",
    "white_total_rpd": "30", "white_win_num_rpd": "20", "white_draw_num_rpd": "10",
    "black_total_rpd": "25", "black_win_num_rpd": "10", "black_draw_num_rpd": "12",
    "white_total_blz": "20", "white_win_num_blz": "10", "white_draw_num_blz": "5",
    "black_total_blz": "15", "black_win_num_blz": "5", "black_draw_num_blz": "8",
}]


def profile_client(html=PROFILE_HTML, status_code=200):
    return FakeClient({"/profile/": FakeResponse(text=html, status_code=status_code)})


# --- get_profile -------------------------------------------------------

async def test_profile_identity_fields():
    data = await fide.get_profile(profile_client(), "1503014")
    assert data["fide_id"] == "1503014"
    assert data["name"] == "Magnus Carlsen"
    assert data["federation"] == "Norway"
    assert data["title"] == "Grandmaster"
    assert data["birth_year"] == 1990
    assert data["url"] == "https://ratings.fide.com/profile/1503014"


async def test_profile_current_ratings():
    data = await fide.get_profile(profile_client(), "1503014")
    assert data["ratings"] == {"std": 2839, "rapid": 2831, "blitz": 2886}


async def test_history_is_oldest_first():
    """The page lists newest first; the chart needs the reverse."""
    history = (await fide.get_profile(profile_client(), "1503014"))["history"]
    assert history["std"] == [
        {"date": "2026-06-01", "rating": 2833, "games": 8},
        {"date": "2026-07-01", "rating": 2839, "games": 10},
    ]


async def test_history_reads_each_time_control_column():
    history = (await fide.get_profile(profile_client(), "1503014"))["history"]
    assert [p["rating"] for p in history["rapid"]] == [2825, 2831]
    assert [p["rating"] for p in history["blitz"]] == [2880, 2886]
    # a month with no games still carries the rating, with games at 0
    assert history["rapid"][0]["games"] == 0


async def test_unrated_time_control_is_absent_not_zero():
    html = PROFILE_HTML.replace(
        '<div class="profile-blitz profile-game"><span>blitz</span><p> 2886 </p></div>',
        '<div class="profile-blitz profile-game"><span>blitz</span><p> Not rated </p></div>')
    data = await fide.get_profile(profile_client(html), "1503014")
    assert "blitz" not in data["ratings"]
    assert data["ratings"]["std"] == 2839


async def test_missing_optional_fields_do_not_break_the_parse():
    html = "<html><head><title>Jane Doe FIDE Profile</title></head><body>" \
           '<div class="profile-info-id ">99</div></body></html>'
    data = await fide.get_profile(profile_client(html), "99")
    assert data["name"] == "Jane Doe"
    assert data["federation"] is None and data["title"] is None
    assert data["birth_year"] is None and data["ratings"] == {}
    assert data["history"] == {"std": [], "rapid": [], "blitz": []}


@pytest.mark.parametrize("fide_id", ["abc", "150301x", "", "  ", "12.5"])
async def test_non_numeric_id_never_hits_the_network(fide_id):
    client = profile_client()
    assert await fide.get_profile(client, fide_id) is None
    assert client.calls == []


async def test_id_is_stripped_before_use():
    client = profile_client()
    assert (await fide.get_profile(client, "  1503014  "))["fide_id"] == "1503014"


async def test_generic_page_for_an_unknown_id_is_none():
    html = "<html><body>Player not found</body></html>"
    assert await fide.get_profile(profile_client(html), "999999") is None


async def test_page_without_a_title_is_none():
    html = '<html><body><div class="profile-info-id ">1503014</div></body></html>'
    assert await fide.get_profile(profile_client(html), "1503014") is None


async def test_profile_is_cached_and_the_second_call_is_free():
    client = profile_client()
    first = await fide.get_profile(client, "1503014")
    second = await fide.get_profile(client, "1503014")
    assert first == second
    assert len(client.calls) == 1


async def test_a_miss_is_cached_too():
    """Remembering the 'not found' keeps a bad ID from re-scraping every time."""
    client = profile_client("<html><body>nothing</body></html>")
    assert await fide.get_profile(client, "999999") is None
    assert await fide.get_profile(client, "999999") is None
    assert len(client.calls) == 1
    assert cache.get("fide:profile:999999") == {}


async def test_profile_sends_a_browser_user_agent():
    """The profile pages 403 anything that looks like a script."""
    client = profile_client()
    await fide.get_profile(client, "1503014")
    _, kwargs = client.calls[0]
    assert "Mozilla/5.0" in kwargs["headers"]["User-Agent"]


# --- get_stats ---------------------------------------------------------

def stats_client(payload=STATS_JSON):
    return FakeClient({"a_data_stats.php": FakeResponse(json_data=payload)})


async def test_stats_counts_by_colour():
    data = await fide.get_stats(stats_client(), "1503014")
    assert data["all"]["white"] == {"n": 100, "wins": 60, "draws": 30, "losses": 10}
    assert data["all"]["black"] == {"n": 80, "wins": 30, "draws": 40, "losses": 10}


async def test_stats_total_is_the_sum_of_both_colours():
    data = await fide.get_stats(stats_client(), "1503014")
    assert data["all"]["total"] == {"n": 180, "wins": 90, "draws": 70, "losses": 20}


async def test_stats_split_by_time_control():
    data = await fide.get_stats(stats_client(), "1503014")
    assert data["std"]["white"]["n"] == 50
    assert data["rapid"]["black"]["wins"] == 10
    assert data["blitz"]["total"] == {"n": 35, "wins": 15, "draws": 13, "losses": 7}


async def test_losses_never_go_negative():
    """Inconsistent upstream counts must not produce a negative loss count."""
    payload = [{"white_total": "5", "white_win_num": "4", "white_draw_num": "4",
                "black_total": "0", "black_win_num": "0", "black_draw_num": "0"}]
    data = await fide.get_stats(stats_client(payload), "1503014")
    assert data["all"]["white"]["losses"] == 0


async def test_missing_fields_count_as_zero():
    data = await fide.get_stats(stats_client([{"white_total": "10", "white_win_num": "10"}]), "1")
    assert data["all"]["white"] == {"n": 10, "wins": 10, "draws": 0, "losses": 0}
    assert data["all"]["black"] == {"n": 0, "wins": 0, "draws": 0, "losses": 0}


async def test_a_player_with_no_rated_games_is_none():
    payload = [{"white_total": "0", "white_win_num": "0", "black_total": "0"}]
    assert await fide.get_stats(stats_client(payload), "1503014") is None


async def test_empty_stats_payload_is_none():
    assert await fide.get_stats(stats_client([]), "1503014") is None


@pytest.mark.parametrize("fide_id", ["abc", ""])
async def test_stats_skips_non_numeric_ids(fide_id):
    client = stats_client()
    assert await fide.get_stats(client, fide_id) is None
    assert client.calls == []


async def test_stats_are_cached():
    client = stats_client()
    await fide.get_stats(client, "1503014")
    await fide.get_stats(client, "1503014")
    assert len(client.calls) == 1


async def test_stats_request_looks_like_the_profile_page_asking():
    client = stats_client()
    await fide.get_stats(client, "1503014")
    _, kwargs = client.calls[0]
    assert kwargs["params"] == {"id1": "1503014", "id2": ""}
    assert kwargs["headers"]["X-Requested-With"] == "XMLHttpRequest"
    assert kwargs["headers"]["Referer"].endswith("/profile/1503014")
