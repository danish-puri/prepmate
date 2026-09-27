"""The deep encoder path, skipped when torch is not installed.

These use an untrained network on purpose. The point is the contract between
train/pretrain.py and TorchEncoder, which is where a mistake would be silent:
a checkpoint that loads but keys the cache wrong, or a policy head whose
probabilities do not sum to one, produces plausible looking output either way.
"""

import chess
import numpy as np
import pytest

from backend.patterns import encoder, pipeline, store
from tests.conftest import make_game
from tests.test_patterns import E4E5

torch = pytest.importorskip("torch")


@pytest.fixture
def checkpoint(tmp_path):
    """A tiny untrained net saved in the exact shape pretrain.py writes."""
    from backend.patterns.net import PositionNet

    cfg = {"name": "cnn8x1d16", "channels": 8, "blocks": 1, "dim": 16}
    model = PositionNet(**cfg)
    path = tmp_path / "tiny.pt"
    torch.save({"model": model.state_dict(), "config": cfg, "fingerprint": "deadbeef"}, path)
    return path


def test_checkpoint_loads_and_names_its_own_vector_space(checkpoint):
    enc = encoder.TorchEncoder(str(checkpoint))
    assert enc.dim == 16 and enc.is_deep
    # the fingerprint has to reach the cache key, or a retrained model of the
    # same shape would silently reuse the previous model's vectors
    assert enc.name == "cnn8x1d16-deadbeef"


def test_load_prefers_the_checkpoint_and_falls_back_without_one(checkpoint):
    assert encoder.load(str(checkpoint)).is_deep is True
    assert encoder.load(None).is_deep is False


def test_embeddings_are_unit_length_and_deterministic(checkpoint):
    enc = encoder.TorchEncoder(str(checkpoint))
    boards = [chess.Board(), chess.Board("8/8/4k3/8/8/4K3/4P3/8 w - - 0 1")]
    a, b = enc.encode(boards), enc.encode(boards)
    assert a.shape == (2, 16)
    assert np.allclose(np.linalg.norm(a, axis=1), 1.0, atol=1e-5)
    assert np.allclose(a, b)


def test_batching_does_not_change_the_answer(checkpoint):
    enc = encoder.TorchEncoder(str(checkpoint), batch_size=3)
    boards = [chess.Board()] * 7
    out = enc.encode(boards)
    assert out.shape == (7, 16) and np.allclose(out, out[0])


def test_policy_logits_are_log_softmaxes(checkpoint):
    enc = encoder.TorchEncoder(str(checkpoint))
    logf, logt = enc.policy_logits([chess.Board(), chess.Board()])
    assert logf.shape == (2, 64) and logt.shape == (2, 64)
    assert np.allclose(np.exp(logf).sum(axis=1), 1.0, atol=1e-4)


def test_empty_input_returns_an_empty_matrix_of_the_right_width(checkpoint):
    enc = encoder.TorchEncoder(str(checkpoint))
    assert enc.encode([]).shape == (0, 16)


def test_predictability_runs_over_the_deep_path(checkpoint):
    enc = encoder.TorchEncoder(str(checkpoint))
    scout = pipeline.build([make_game(color="white", moves=E4E5)], enc, first_ply=0)
    out = pipeline.predictability(scout, enc)
    assert out["available"] and out["n"] == len(scout.decisions)
    assert out["surprisal_bits"] >= 0 and 0 <= out["top1_pct"] <= 100


def test_two_spaces_never_share_cached_vectors(checkpoint, tmp_path):
    """A cache hit from the wrong model would poison every distance downstream."""
    from backend.patterns.net import PositionNet

    cfg = {"name": "cnn8x1d16", "channels": 8, "blocks": 1, "dim": 16}
    other = tmp_path / "other.pt"
    torch.save({"model": PositionNet(**cfg).state_dict(), "config": cfg,
                "fingerprint": "feedface"}, other)

    games = [make_game(color="white", moves=E4E5)]
    a = pipeline.build(games, encoder.TorchEncoder(str(checkpoint)), first_ply=0)
    b = pipeline.build(games, encoder.TorchEncoder(str(other)), first_ply=0)
    assert b.stats["cache_hits"] == 0            # different fingerprint, no reuse
    assert not np.allclose(a.vectors, b.vectors)  # and genuinely different weights
    assert store.count(a.space) and store.count(b.space)
