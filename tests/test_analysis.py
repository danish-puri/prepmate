"""The stats in analysis.py.

Every number the scouting report shows comes from here, and the brief says
never invent a stat, so these check the arithmetic against counts worked out
by hand rather than against whatever the code happens to return.
"""

import pytest

from backend import analysis
from tests.conftest import make_game, make_games


# --- score_pct ---------------------------------------------------------

@pytest.mark.parametrize("wins,draws,n,expected", [
    (10, 0, 10, 100.0),
    (0, 0, 10, 0.0),
    (0, 10, 10, 50.0),   # a draw is half a point
    (5, 0, 10, 50.0),
    (1, 1, 3, 50.0),     # 1.5 / 3
    (0, 0, 0, 0.0),      # no games, no division by zero
])
def test_score_pct(wins, draws, n, expected):
    assert analysis.score_pct(wins, draws, n) == expected


# --- opening_tables ----------------------------------------------------

def test_opening_table_counts_and_frequency():
    games = (make_games(6, eco="B01", result="win")
             + make_games(4, eco="C50", result="loss"))
    white = analysis.opening_tables(games)["white"]

    assert white["n"] == 10
    assert white["avg_score"] == 60.0
    b01, c50 = white["rows"]
    assert (b01["eco"], b01["games"], b01["wins"], b01["score"], b01["freq_pct"]) == \
           ("B01", 6, 6, 100.0, 60.0)
    assert (c50["eco"], c50["games"], c50["losses"], c50["score"], c50["freq_pct"]) == \
           ("C50", 4, 4, 0.0, 40.0)


def test_rows_sorted_by_game_count():
    games = (make_games(2, eco="A10") + make_games(9, eco="B20")
             + make_games(5, eco="C30"))
    ecos = [r["eco"] for r in analysis.opening_tables(games)["white"]["rows"]]
    assert ecos == ["B20", "C30", "A10"]


def test_rows_capped_at_max_rows():
    games = []
    for i in range(analysis.MAX_ROWS + 5):
        games += make_games(i + 1, eco=f"A{i:02d}")
    assert len(analysis.opening_tables(games)["white"]["rows"]) == analysis.MAX_ROWS


def test_games_without_an_eco_are_excluded():
    games = make_games(3, eco="B01") + make_games(2, eco=None)
    white = analysis.opening_tables(games)["white"]
    assert white["n"] == 3
    assert [r["eco"] for r in white["rows"]] == ["B01"]


def test_colours_are_tallied_separately():
    games = (make_games(4, color="white", result="win")
             + make_games(4, color="black", result="loss"))
    tables = analysis.opening_tables(games)
    assert tables["white"]["avg_score"] == 100.0 and tables["white"]["n"] == 4
    assert tables["black"]["avg_score"] == 0.0 and tables["black"]["n"] == 4


def test_opening_name_is_the_most_common_for_the_eco():
    games = (make_games(3, eco="B01", opening="Scandinavian Defense")
             + make_games(1, eco="B01", opening="Scandinavian Defense: Mieses"))
    row = analysis.opening_tables(games)["white"]["rows"][0]
    assert row["name"] == "Scandinavian Defense"


def test_small_sample_below_min_games():
    games = make_games(analysis.MIN_GAMES - 1, eco="B01")
    assert analysis.opening_tables(games)["white"]["rows"][0]["small_sample"] is True
    games = make_games(analysis.MIN_GAMES, eco="B01")
    assert analysis.opening_tables(games)["white"]["rows"][0]["small_sample"] is False


def test_weakness_and_strength_flags():
    # 20 wins in C50 and 20 losses in B01: average 50, so each sits 50 points off
    games = make_games(20, eco="C50", result="win") + make_games(20, eco="B01", result="loss")
    rows = {r["eco"]: r for r in analysis.opening_tables(games)["white"]["rows"]}
    assert rows["C50"]["flag"] == "strength"
    assert rows["B01"]["flag"] == "weakness"


def test_no_flag_below_the_sample_floor():
    """A bad score on too few games is noise, so it must not be flagged."""
    games = (make_games(30, eco="C50", result="win")
             + make_games(analysis.MIN_GAMES - 1, eco="B01", result="loss"))
    rows = {r["eco"]: r for r in analysis.opening_tables(games)["white"]["rows"]}
    assert rows["B01"]["score"] == 0.0 and rows["B01"]["flag"] is None


def test_no_flag_inside_the_threshold():
    # B01 scores 50, the average lands at ~53.6, well inside THRESHOLD_PTS
    games = (make_games(20, eco="C50", result="win") + make_games(20, eco="C50", result="loss")
             + make_games(16, eco="B01", result="win") + make_games(16, eco="B01", result="loss"))
    rows = {r["eco"]: r for r in analysis.opening_tables(games)["white"]["rows"]}
    assert rows["B01"]["flag"] is None


def test_empty_input_gives_empty_tables():
    tables = analysis.opening_tables([])
    assert tables["white"] == {"n": 0, "avg_score": 0.0, "rows": []}
    assert tables["black"] == {"n": 0, "avg_score": 0.0, "rows": []}


# --- prep_target -------------------------------------------------------

def test_prep_target_picks_the_most_played_weakness():
    games = (make_games(40, eco="C50", result="win")
             + make_games(16, eco="B01", result="loss")
             + make_games(30, eco="D00", result="loss"))
    target = analysis.prep_target(analysis.opening_tables(games))
    assert target["eco"] == "D00"  # both are weaknesses, D00 is played more
    assert target["color"] == "white"
    assert target["delta"] < 0


def test_prep_target_none_when_nothing_is_flagged():
    assert analysis.prep_target(analysis.opening_tables(make_games(20, eco="B01"))) is None
    assert analysis.prep_target(analysis.opening_tables([])) is None


def test_prep_target_considers_both_colours():
    games = (make_games(40, color="black", eco="C50", result="win")
             + make_games(20, color="black", eco="B01", result="loss")
             + make_games(20, color="white", eco="D00", result="draw"))
    assert analysis.prep_target(analysis.opening_tables(games))["color"] == "black"


# --- colour_totals -----------------------------------------------------

def test_colour_totals_splits_and_combines():
    games = (make_games(3, color="white", result="win")
             + make_games(1, color="white", result="draw")
             + make_games(2, color="black", result="loss"))
    totals = analysis.colour_totals(games)
    assert totals["white"] == {"n": 4, "wins": 3, "draws": 1, "losses": 0, "score": 87.5}
    assert totals["black"] == {"n": 2, "wins": 0, "draws": 0, "losses": 2, "score": 0.0}
    assert totals["total"] == {"n": 6, "wins": 3, "draws": 1, "losses": 2, "score": 58.3}


def test_colour_totals_on_no_games():
    totals = analysis.colour_totals([])
    assert totals["total"] == {"n": 0, "wins": 0, "draws": 0, "losses": 0, "score": 0.0}


# --- colour_split ------------------------------------------------------

def test_colour_split_percentages_by_time_class():
    games = (make_games(2, color="white", time_class="blitz", result="win")
             + make_games(1, color="white", time_class="blitz", result="draw")
             + make_games(1, color="white", time_class="blitz", result="loss"))
    assert analysis.colour_split(games)["white"]["blitz"] == {
        "n": 4, "win_pct": 50, "draw_pct": 25, "loss_pct": 25}


def test_colour_split_omits_time_classes_with_no_games():
    games = make_games(2, color="white", time_class="rapid")
    split = analysis.colour_split(games)
    assert list(split["white"]) == ["rapid"]
    assert split["black"] == {}


# --- loss_terminations -------------------------------------------------

def test_loss_terminations_maps_both_platform_vocabularies():
    games = (make_games(2, result="loss", termination="resigned")
             + make_games(1, result="loss", termination="resign")
             + make_games(2, result="loss", termination="timeout")
             + make_games(1, result="loss", termination="outoftime")
             + make_games(1, result="loss", termination="checkmated")
             + make_games(1, result="loss", termination="mate")
             + make_games(1, result="loss", termination="abandoned"))
    out = analysis.loss_terminations(games)
    assert out["n"] == 9
    assert out["resigned"]["count"] == 3
    assert out["on_time"]["count"] == 3
    assert out["checkmated"]["count"] == 2
    assert out["other"]["count"] == 1


def test_loss_terminations_ignores_wins_and_draws():
    games = (make_games(1, result="loss", termination="resigned")
             + make_games(5, result="win", termination="checkmated")
             + make_games(5, result="draw", termination="agreed"))
    out = analysis.loss_terminations(games)
    assert out["n"] == 1 and out["resigned"] == {"count": 1, "pct": 100}


def test_loss_terminations_with_no_losses():
    assert analysis.loss_terminations(make_games(3, result="win")) == {"n": 0}


# --- vs_higher_rated ---------------------------------------------------

def test_vs_higher_rated_uses_the_margin():
    games = (make_games(2, player_rating=1800, opponent_rating=1900, result="win")
             + make_games(2, player_rating=1800, opponent_rating=1950, result="loss")
             + make_games(5, player_rating=1800, opponent_rating=1899, result="loss"))
    out = analysis.vs_higher_rated(games)
    assert out == {"n": 4, "score": 50.0, "margin": 100}  # the 1899s are below the margin


def test_vs_higher_rated_custom_margin():
    games = make_games(3, player_rating=1800, opponent_rating=2000, result="win")
    assert analysis.vs_higher_rated(games, margin=300)["n"] == 0
    assert analysis.vs_higher_rated(games, margin=200)["n"] == 3


def test_vs_higher_rated_skips_games_missing_a_rating():
    games = (make_games(2, player_rating=None, opponent_rating=2500)
             + make_games(2, player_rating=1800, opponent_rating=None))
    assert analysis.vs_higher_rated(games)["n"] == 0


# --- rating_history_from_games -----------------------------------------

DAY = 86400
JAN1 = 1767225600  # 2026-01-01 00:00 UTC


def test_rating_history_keeps_the_last_rating_of_each_day():
    games = [
        make_game(end_time=JAN1 + 3600, player_rating=1800, time_class="blitz"),
        make_game(end_time=JAN1 + 7200, player_rating=1815, time_class="blitz"),
        make_game(end_time=JAN1 + DAY, player_rating=1790, time_class="blitz"),
    ]
    assert analysis.rating_history_from_games(games)["blitz"] == [
        {"date": "2026-01-01", "rating": 1815},
        {"date": "2026-01-02", "rating": 1790},
    ]


def test_rating_history_is_ordered_regardless_of_input_order():
    games = [
        make_game(end_time=JAN1 + 2 * DAY, player_rating=1820, time_class="rapid"),
        make_game(end_time=JAN1, player_rating=1800, time_class="rapid"),
        make_game(end_time=JAN1 + DAY, player_rating=1810, time_class="rapid"),
    ]
    dates = [p["date"] for p in analysis.rating_history_from_games(games)["rapid"]]
    assert dates == ["2026-01-01", "2026-01-02", "2026-01-03"]


def test_rating_history_separates_time_classes():
    games = [
        make_game(end_time=JAN1, player_rating=1800, time_class="blitz"),
        make_game(end_time=JAN1, player_rating=2000, time_class="rapid"),
    ]
    history = analysis.rating_history_from_games(games)
    assert history["blitz"][0]["rating"] == 1800
    assert history["rapid"][0]["rating"] == 2000


def test_rating_history_skips_games_without_a_rating():
    games = [make_game(end_time=JAN1, player_rating=None, time_class="blitz")]
    assert analysis.rating_history_from_games(games) == {}


# --- recent_form -------------------------------------------------------

def test_recent_form_returns_the_newest_n_oldest_first():
    games = [make_game(end_time=JAN1 + i, eco=f"A{i:02d}") for i in range(10)]
    form = analysis.recent_form(games, n=3)
    assert form["n"] == 3
    # the three newest (A07, A08, A09), presented oldest first for the UI strip
    assert [s["eco"] for s in form["sequence"]] == ["A07", "A08", "A09"]


def test_recent_form_tallies_the_window_not_the_whole_history():
    games = (make_games(5, result="loss", end_time=JAN1)
             + make_games(2, result="win", end_time=JAN1 + 100))
    form = analysis.recent_form(games, n=2)
    assert (form["wins"], form["draws"], form["losses"]) == (2, 0, 0)


def test_recent_form_when_fewer_games_than_n():
    form = analysis.recent_form(make_games(2), n=20)
    assert form["n"] == 2 and len(form["sequence"]) == 2


def test_recent_form_on_no_games():
    assert analysis.recent_form([], n=5) == {
        "n": 0, "wins": 0, "draws": 0, "losses": 0, "sequence": []}


def test_recent_form_entry_carries_what_the_strip_shows():
    game = make_game(platform="chesscom", time_class="rapid", result="draw",
                     eco="B01", opening="Scandinavian", opponent_rating=2100,
                     end_time=JAN1)
    entry = analysis.recent_form([game])["sequence"][0]
    assert entry == {"result": "draw", "platform": "chesscom", "time_class": "rapid",
                     "eco": "B01", "opening": "Scandinavian", "opponent_rating": 2100,
                     "end_time": JAN1}
