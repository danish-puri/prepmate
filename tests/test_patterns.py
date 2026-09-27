"""Replay, embedding store, and the mining statistics.

Nothing here needs torch or a checkpoint: the encoder is behind an interface and
these run against the structural fallback, which is exactly what that fallback
is for.
"""

import chess
import numpy as np
import pytest

from backend.patterns import encoder, mining, pipeline, replay, store
from backend.patterns.replay import Decision
from tests.conftest import make_game

E4E5 = ["e4", "e5", "Nf3", "Nc6", "Bb5", "a6", "Ba4", "Nf6", "O-O", "Be7", "Re1", "b5"]


# --- replay ----------------------------------------------------------------

def test_only_the_players_own_decisions_are_kept():
    white, _ = replay.decisions([make_game(color="white", moves=E4E5)], first_ply=0)
    black, _ = replay.decisions([make_game(color="black", moves=E4E5)], first_ply=0)
    assert all(d.ply % 2 == 0 for d in white)
    assert all(d.ply % 2 == 1 for d in black)


def test_ply_window_is_respected():
    ds, _ = replay.decisions([make_game(color="white", moves=E4E5)], first_ply=4, last_ply=8)
    assert ds and all(4 <= d.ply <= 8 for d in ds)


def test_forced_replies_are_not_decisions():
    ds, _ = replay.decisions([make_game(color="white", moves=E4E5)], first_ply=0)
    assert ds and all(d.legal > 1 for d in ds)


def test_snapshots_are_the_position_at_the_time_not_the_end_of_the_game():
    # canonical() returns the same object for a white-to-move board, so a
    # missing copy here silently stores every game's final position instead
    game = make_game(color="white", moves=E4E5)
    ds, boards = replay.decisions([game], first_ply=0)
    first = next(d for d in ds if d.ply == 0)
    assert boards[first.key].fen() == chess.Board().fen()
    assert len({boards[d.key].fen() for d in ds}) == len({d.key for d in ds})


def test_snapshot_keys_match_their_boards():
    ds, boards = replay.decisions([make_game(color="black", moves=E4E5)], first_ply=0)
    for d in ds:
        assert replay.position_key(boards[d.key]) == d.key


def test_sole_recapture_is_skipped():
    # 1.d4 d5 2.c4 dxc4 -- white's only recapture path is examined, and after
    # 1...exd5 style trades the single legal retake carries no information
    moves = ["e4", "d5", "exd5", "Qxd5"]
    ds, _ = replay.decisions([make_game(color="black", moves=moves)], first_ply=0)
    # 3...Qxd5 is the sole recapture on d5, so black's ply 3 must not appear
    assert 3 not in {d.ply for d in ds}


def test_transpositions_fold_to_one_key():
    a = chess.Board()
    for san in ("d4", "Nf6", "c4", "e6"):
        a.push_san(san)
    b = chess.Board()
    for san in ("c4", "Nf6", "d4", "e6"):
        b.push_san(san)
    assert replay.position_key(a) == replay.position_key(b)


def test_colour_mirrored_positions_share_a_key():
    board = chess.Board("r1bqkbnr/pppp1ppp/2n5/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4")
    assert replay.position_key(board) == replay.position_key(board.mirror())


def test_canonical_move_follows_the_mirror():
    board = chess.Board()
    board.push_san("e4")
    move = board.parse_san("e5")  # black, e7e5
    canon = replay.canonical_move(move, chess.BLACK)
    assert move.uci() == "e7e5" and canon.uci() == "e2e4"


def test_unparseable_moves_stop_that_game_without_raising():
    ds, _ = replay.decisions([make_game(color="white", moves=["e4", "e5", "Qxz9", "Nc6"])],
                             first_ply=0)
    assert [d.ply for d in ds] == [0]


# --- store -----------------------------------------------------------------

def test_store_roundtrips_and_isolates_spaces():
    v = {1: np.array([0.5, -0.25], np.float32), 2: np.array([1.0, 0.0], np.float32)}
    store.put_many("space-a", v)
    got = store.get_many("space-a", [1, 2, 3])
    assert set(got) == {1, 2}
    assert np.allclose(got[1], v[1])
    assert store.get_many("space-b", [1, 2]) == {}


def test_store_survives_more_keys_than_sqlite_takes_at_once():
    keys = list(range(1200))
    store.put_many("bulk", {k: np.array([float(k)], np.float32) for k in keys})
    assert len(store.get_many("bulk", keys)) == 1200


def test_dropping_a_space_leaves_others_alone():
    store.put_many("gone", {1: np.array([1.0], np.float32)})
    store.put_many("kept", {1: np.array([1.0], np.float32)})
    store.drop("gone")
    assert store.get_many("gone", [1]) == {} and store.get_many("kept", [1])


# --- mining ----------------------------------------------------------------

def test_kmeans_separates_two_obvious_groups():
    a = np.tile([1.0, 0.0], (30, 1)) + np.random.default_rng(0).normal(0, 0.01, (30, 2))
    b = np.tile([0.0, 1.0], (30, 1)) + np.random.default_rng(1).normal(0, 0.01, (30, 2))
    x = np.vstack([a, b])
    x /= np.linalg.norm(x, axis=1, keepdims=True)
    labels, _ = mining.spherical_kmeans(x, 2, seed=0)
    assert len(set(labels[:30])) == 1 and len(set(labels[30:])) == 1
    assert labels[0] != labels[30]


def test_kmeans_handles_k_above_the_sample_count():
    x = np.eye(3, dtype=np.float32)
    labels, centers = mining.spherical_kmeans(x, 10, seed=0)
    assert len(labels) == 3 and len(centers) == 3


def _fake(n_games, per_game, results):
    """n_games games, each contributing per_game decisions to one cluster."""
    games = [make_game(result=r) for r in results]
    ds, vecs, fens = [], [], {}
    for gi in range(n_games):
        for j in range(per_game):
            key = gi * 100 + j
            ds.append(Decision(key=key, game=gi, ply=10 + j, move="e2e4",
                               canon_move="e2e4", legal=20))
            vecs.append([1.0, 0.0])
            fens[key] = chess.Board().fen()
    return ds, np.array(vecs, np.float32), games, fens


def test_a_game_counts_once_per_cluster():
    # one game passing through the same structure twenty times is one
    # observation, not twenty; counting plies would make noise look certain
    ds, vecs, games, fens = _fake(4, 20, ["win", "loss", "loss", "draw"])
    arche = mining.archetypes(ds, vecs, games, fens, k=1)
    assert arche[0].positions == 80 and arche[0].games == 4
    assert (arche[0].wins, arche[0].draws, arche[0].losses) == (1, 1, 2)


def _two_clusters(small_results, rest_results):
    """A small cluster at [1,0] and a larger one at [0,1], one decision each."""
    games = [make_game(result=r) for r in small_results + rest_results]
    ds, vecs, fens = [], [], {}
    for gi in range(len(games)):
        ds.append(Decision(key=gi, game=gi, ply=12, move="e2e4", canon_move="e2e4", legal=20))
        vecs.append([1.0, 0.0] if gi < len(small_results) else [0.0, 1.0])
        fens[gi] = chess.Board().fen()
    return ds, np.array(vecs, np.float32), games, fens


def test_shrinkage_pulls_small_samples_toward_the_baseline():
    # three losses inside a player who otherwise scores 50%
    ds, vecs, games, fens = _two_clusters(["loss"] * 3, ["win"] * 10 + ["loss"] * 7)
    small = next(a for a in mining.archetypes(ds, vecs, games, fens, k=2) if a.games == 3)
    assert small.baseline == pytest.approx(50.0)
    assert small.score == 0.0                       # raw says he never scores here
    assert 0.0 < small.shrunk < small.baseline      # but three games is not evidence
    # (0 wins + 12 prior games at 50%) / (3 + 12)
    assert small.shrunk == pytest.approx(40.0, abs=0.1)


def test_a_large_sample_keeps_its_own_number():
    ds, vecs, games, fens = _two_clusters(["loss"] * 60, ["win"] * 60)
    big = next(a for a in mining.archetypes(ds, vecs, games, fens, k=2) if a.games == 60)
    assert big.score == 0.0 and big.shrunk < 9.0  # barely moved off the raw number


def test_ranking_is_by_points_dropped_not_by_percentage():
    games = [make_game(result="win") for _ in range(40)]
    for i in (0, 1):
        games[i] = make_game(result="loss")          # tiny, catastrophic cluster
    for i in range(20, 32):
        games[i] = make_game(result="loss")          # big, moderately bad cluster
    ds, vecs, fens = [], [], {}
    for gi in range(40):
        cluster = 0 if gi < 2 else 1 if gi < 20 else 2
        ds.append(Decision(key=gi, game=gi, ply=12, move="e2e4", canon_move="e2e4", legal=20))
        vecs.append([[1.0, 0.0], [0.0, 1.0], [0.7, 0.7]][cluster])
        fens[gi] = chess.Board().fen()
    arche = mining.archetypes(ds, np.array(vecs, np.float32), games, fens, k=3)
    assert arche[0].games > 2  # the wide cluster outranks the 0% two-game one


def test_nearest_returns_one_row_per_game():
    ds, vecs, games, fens = _fake(3, 5, ["win", "loss", "draw"])
    hits = mining.nearest(np.array([1.0, 0.0], np.float32), vecs, ds, games, n=10)
    assert len(hits) == 3 and len({h["result"] for h in hits}) == 3


def test_predictability_reports_bits_and_hit_rates():
    certain = [np.log(np.array([0.9, 0.1], np.float32))] * 10
    out = mining.predictability(certain)
    assert out["top1_pct"] == 100.0 and out["surprisal_bits"] < 0.2


# --- pipeline --------------------------------------------------------------

def test_pipeline_stats_and_cache_reuse():
    games = [make_game(color="white", moves=E4E5), make_game(color="black", moves=E4E5)]
    enc = encoder.StructuralEncoder()
    first = pipeline.build(games, enc, first_ply=0)
    assert first.stats["decisions"] == len(first.decisions)
    assert first.stats["cache_hits"] == 0 and first.stats["encoded"] > 0
    again = pipeline.build(games, enc, first_ply=0)
    assert again.stats["encoded"] == 0
    assert again.stats["cache_hits"] == again.stats["distinct_positions"]


def test_pipeline_vectors_are_unit_length():
    scout = pipeline.build([make_game(color="white", moves=E4E5)],
                           encoder.StructuralEncoder(), first_ply=0)
    assert np.allclose(np.linalg.norm(scout.vectors, axis=1), 1.0, atol=1e-5)


def test_report_on_games_without_moves_is_empty_not_a_crash():
    scout = pipeline.build([make_game(moves=[])], encoder.StructuralEncoder())
    out = pipeline.report(scout)
    assert out["weaknesses"] == [] and out["stats"]["decisions"] == 0


def test_predictability_is_unavailable_without_a_policy_head():
    scout = pipeline.build([make_game(color="white", moves=E4E5)],
                           encoder.StructuralEncoder(), first_ply=0)
    assert pipeline.predictability(scout, encoder.StructuralEncoder())["available"] is False


def test_planes_encode_the_start_position():
    x = encoder.planes(chess.Board())
    assert x.shape == (encoder.N_PLANES, 8, 8)
    assert x[0].sum() == 8 and x[5].sum() == 1     # white pawns, white king
    assert x[11].sum() == 1 and x[17].sum() == 64  # black king, constant plane
