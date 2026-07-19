"""analysis.move_tree: per-node counts, pruning, depth cap, colour parity."""

from backend.analysis import move_tree
from tests.conftest import make_game


def white_games():
    return [
        make_game(color="white", result="win", moves=["e4", "e5"]),
        make_game(color="white", result="loss", moves=["e4", "e5"]),
        make_game(color="white", result="draw", moves=["e4", "c5"]),
        make_game(color="white", result="win", moves=["d4"]),
    ]


def test_node_counts_score_and_freq():
    t = move_tree(white_games(), min_node_games=1)["white"]
    assert t["n"] == 4
    assert t["score"] == 62.5  # 2 wins + half a draw over 4

    e4, d4 = t["moves"][0], t["moves"][1]
    assert (e4["san"], e4["games"], e4["wins"], e4["draws"], e4["losses"]) == ("e4", 3, 1, 1, 1)
    assert e4["score"] == 50.0
    assert e4["freq_pct"] == 75.0
    assert (d4["san"], d4["games"], d4["freq_pct"]) == ("d4", 1, 25.0)

    # children sorted by games, freq relative to the parent position
    e5, c5 = e4["children"]
    assert (e5["san"], e5["games"], e5["score"], e5["freq_pct"]) == ("e5", 2, 50.0, 66.7)
    assert (c5["san"], c5["games"], c5["freq_pct"]) == ("c5", 1, 33.3)


def test_rare_branches_pruned_at_min_node_games():
    t = move_tree(white_games(), min_node_games=2)["white"]
    assert [m["san"] for m in t["moves"]] == ["e4"]          # d4 seen once
    assert [c["san"] for c in t["moves"][0]["children"]] == ["e5"]  # c5 seen once


def test_depth_capped_at_max_plies():
    games = [make_game(moves=["e4", "e5", "Nf3", "Nc6", "Bb5", "a6"])] * 2
    t = move_tree(games, max_plies=2, min_node_games=1)["white"]
    e5 = t["moves"][0]["children"][0]
    assert e5["san"] == "e5" and e5["children"] == []


def test_black_tree_parity_and_games_without_moves_excluded():
    games = [
        make_game(color="black", result="win", moves=["e4", "c5"]),
        make_game(color="black", result="loss", moves=["e4", "c5"]),
        make_game(color="black", result="win", moves=[]),  # source had no movetext
    ]
    t = move_tree(games, min_node_games=1)
    black = t["black"]
    assert black["n"] == 2  # the moveless game cannot join the tree
    # ply 0 is the opponent's White move, ply 1 is the scouted player's reply
    assert black["moves"][0]["san"] == "e4"
    assert black["moves"][0]["children"][0]["san"] == "c5"
    assert t["white"] == {"n": 0, "score": 0.0, "moves": []}
