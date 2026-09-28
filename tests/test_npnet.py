"""The NumPy trunk the website runs must give the torch model's answers.

The reference numbers below were taken from the torch checkpoint pos-v1.pt, so
this runs anywhere NumPy does. The last test compares the two directly when
torch and the checkpoint are both available.
"""

from pathlib import Path

import chess
import numpy as np
import pytest

from backend.patterns import encoder as encoders
from backend.patterns.encoder import NumpyEncoder, planes_batch
from backend.patterns.npnet import NumpyNet

MIDDLEGAME = "r1bq1rk1/pp2bppp/2n1pn2/3p4/2PP4/2N2N2/PP2BPPP/R2QKB1R w KQ - 0 9"

# first six embedding values, best from-square log-prob, and the argmax squares
REFERENCE = {
    chess.STARTING_FEN: ([-0.175913, -0.117021, 0.065697, 0.151304, 0.118214, -0.003164], -0.647084, 11, 27),
    MIDDLEGAME: ([-0.136053, -0.009847, -0.026264, -0.05084, 0.061925, 0.03626], -0.889187, 12, 20),
}


@pytest.fixture(scope="module")
def net():
    return NumpyNet()


def test_embeddings_match_torch_reference(net):
    fens = list(REFERENCE)
    e = net.embed(planes_batch([chess.Board(f) for f in fens]))
    for i, fen in enumerate(fens):
        np.testing.assert_allclose(e[i, :6], REFERENCE[fen][0], atol=1e-4)


def test_policy_matches_torch_reference(net):
    fens = list(REFERENCE)
    f, t = net.policy(planes_batch([chess.Board(x) for x in fens]))
    for i, fen in enumerate(fens):
        _, best, frm, to = REFERENCE[fen]
        assert f[i].max() == pytest.approx(best, abs=1e-3)
        assert (int(f[i].argmax()), int(t[i].argmax())) == (frm, to)


def test_start_position_top_guess_is_d4(net):
    f, t = net.policy(planes_batch([chess.Board()]))
    assert (chess.square_name(int(f[0].argmax())), chess.square_name(int(t[0].argmax()))) == ("d2", "d4")


def test_embeddings_are_unit_vectors_and_policies_are_distributions(net):
    boards = [chess.Board(), chess.Board(MIDDLEGAME)]
    e = net.embed(planes_batch(boards))
    np.testing.assert_allclose(np.linalg.norm(e, axis=1), 1.0, atol=1e-5)
    f, t = net.policy(planes_batch(boards))
    np.testing.assert_allclose(np.exp(f).sum(1), 1.0, atol=1e-4)
    np.testing.assert_allclose(np.exp(t).sum(1), 1.0, atol=1e-4)


def test_numpy_encoder_shares_the_torch_vector_space_name():
    # same name means the two share one embedding cache, which is only safe
    # because they produce the same vectors
    enc = NumpyEncoder()
    assert enc.name == "cnn64x4d128-cf79a4ca"
    assert enc.is_deep and enc.dim == 128
    assert enc.encode([]).shape == (0, 128)


def test_load_picks_numpy_for_npz():
    from backend.patterns.npnet import DEFAULT_WEIGHTS
    assert isinstance(encoders.load(str(DEFAULT_WEIGHTS)), NumpyEncoder)


CHECKPOINT = Path(__file__).parent.parent / "models" / "pos-v1.pt"


@pytest.mark.skipif(not CHECKPOINT.exists(), reason="needs models/pos-v1.pt from the pos-v1 release")
def test_matches_torch_on_many_positions():
    pytest.importorskip("torch")
    import random
    random.seed(3)
    boards = []
    while len(boards) < 300:
        b = chess.Board()
        for _ in range(random.randint(6, 50)):
            moves = list(b.legal_moves)
            if not moves:
                break
            b.push(random.choice(moves))
        if b.turn == chess.WHITE and not b.is_game_over():
            boards.append(b)
    torch_enc = encoders.TorchEncoder(str(CHECKPOINT))
    np.testing.assert_allclose(NumpyEncoder().encode(boards), torch_enc.encode(boards), atol=1e-4)
