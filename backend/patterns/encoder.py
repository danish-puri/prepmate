"""Position encoders: a board goes in, a fixed-length vector comes out.

The deep model here is *frozen and pretrained*, and that is the whole design.
One opponent gives a few hundred games, which is a few hundred samples. Nothing
with real capacity can be fitted to that without memorizing it, so the network
never sees the opponent during training. It is trained once, offline, on a
large corpus of other people's games (train/pretrain.py) to predict which move
a human plays, and the layer underneath that prediction becomes the embedding.
Predicting human moves is the right pretext task: it forces the trunk to encode
what a position *demands*, which is what we then want to cluster.

Per-opponent work stays non-parametric on top of those vectors: clustering,
retrieval, and counting. See mining.py.

Boards arriving here are already canonical (replay.canonical), so the side to
move is always White and no side-to-move plane is needed.
"""

from typing import Protocol

import chess
import numpy as np

# 12 piece planes + 4 castling rights + en passant + a constant plane, which
# gives the convolutions a way to feel the edge of the board through padding
N_PLANES = 18

_PIECE_PLANE = {
    (chess.PAWN, chess.WHITE): 0, (chess.KNIGHT, chess.WHITE): 1, (chess.BISHOP, chess.WHITE): 2,
    (chess.ROOK, chess.WHITE): 3, (chess.QUEEN, chess.WHITE): 4, (chess.KING, chess.WHITE): 5,
    (chess.PAWN, chess.BLACK): 6, (chess.KNIGHT, chess.BLACK): 7, (chess.BISHOP, chess.BLACK): 8,
    (chess.ROOK, chess.BLACK): 9, (chess.QUEEN, chess.BLACK): 10, (chess.KING, chess.BLACK): 11,
}


def planes(board: chess.Board) -> np.ndarray:
    """One position as (N_PLANES, 8, 8) float32."""
    x = np.zeros((N_PLANES, 8, 8), dtype=np.float32)
    for square, piece in board.piece_map().items():
        x[_PIECE_PLANE[(piece.piece_type, piece.color)], square >> 3, square & 7] = 1.0
    for i, right in enumerate((
        board.has_kingside_castling_rights(chess.WHITE), board.has_queenside_castling_rights(chess.WHITE),
        board.has_kingside_castling_rights(chess.BLACK), board.has_queenside_castling_rights(chess.BLACK),
    )):
        if right:
            x[12 + i] = 1.0
    if board.ep_square is not None:
        x[16, board.ep_square >> 3, board.ep_square & 7] = 1.0
    x[17] = 1.0
    return x


def planes_batch(boards: list[chess.Board]) -> np.ndarray:
    return np.stack([planes(b) for b in boards]) if boards else np.zeros((0, N_PLANES, 8, 8), np.float32)


class Encoder(Protocol):
    """Anything that turns boards into vectors.

    `name` keys the embedding cache. Two encoders that produce different vector
    spaces must never share a name, or a cache hit will hand back a vector from
    the wrong space and every distance downstream becomes nonsense.
    """

    name: str
    dim: int

    def encode(self, boards: list[chess.Board]) -> np.ndarray: ...


class StructuralEncoder:
    """Handcrafted fallback, not the intended path.

    This exists so the pipeline runs on a machine with no torch and no
    checkpoint, and so the mining layer can be tested without a model. The
    features are the usual textbook ones and they do capture pawn structure
    and king safety, but they are a floor, not the deep encoder. Anything that
    depends on learned structure should check `is_deep` first.
    """

    name = "structural-v1"
    is_deep = False
    dim = 40

    def encode(self, boards: list[chess.Board]) -> np.ndarray:
        if not boards:
            return np.zeros((0, self.dim), np.float32)
        return np.stack([self._one(b) for b in boards])

    def _one(self, board: chess.Board) -> np.ndarray:
        f: list[float] = []
        for color in (chess.WHITE, chess.BLACK):
            pawns = board.pieces(chess.PAWN, color)
            files = [len(pawns & chess.BB_FILES[i]) for i in range(8)]
            f += [
                len(pawns) / 8, len(board.pieces(chess.KNIGHT, color)) / 2,
                len(board.pieces(chess.BISHOP, color)) / 2, len(board.pieces(chess.ROOK, color)) / 2,
                len(board.pieces(chess.QUEEN, color)),
                sum(1 for c in files if c > 1) / 4,                      # doubled
                sum(1 for i, c in enumerate(files) if c and not          # isolated
                    (files[i - 1] if i else 0) and not (files[i + 1] if i < 7 else 0)) / 4,
                sum(1 for i, c in enumerate(files) if c) / 8,            # files occupied
                float(len(board.pieces(chess.BISHOP, color)) >= 2),      # bishop pair
                self._king_shield(board, color),
                self._center(board, color),
                len(board.attacks(board.king(color) or 0)) / 8,
            ]
        f += [
            len(board.piece_map()) / 32,                                  # phase
            float(board.is_check()),
            board.legal_moves.count() / 40,
            board.halfmove_clock / 50,
        ]
        v = np.asarray(f, dtype=np.float32)
        return np.pad(v, (0, self.dim - len(v))) if len(v) < self.dim else v[: self.dim]

    @staticmethod
    def _king_shield(board: chess.Board, color: chess.Color) -> float:
        king = board.king(color)
        if king is None:
            return 0.0
        shield = board.attacks(king) & board.pieces(chess.PAWN, color)
        return len(shield) / 3

    @staticmethod
    def _center(board: chess.Board, color: chess.Color) -> float:
        center = chess.SquareSet([chess.D4, chess.E4, chess.D5, chess.E5])
        return sum(1 for sq in center if board.is_attacked_by(color, sq)) / 4


class TorchEncoder:
    """The frozen pretrained trunk. Loads a checkpoint written by train/pretrain.py.

    torch is imported lazily so the rest of the app, which has no ML deps, keeps
    importing this module for free.
    """

    is_deep = True

    def __init__(self, checkpoint: str, device: str = "cpu", batch_size: int = 512):
        import torch

        from .net import PositionNet

        blob = torch.load(checkpoint, map_location=device, weights_only=True)
        cfg = blob["config"]
        self._torch = torch
        self.dim = cfg["dim"]
        self.name = f"{cfg['name']}-{blob['fingerprint']}"
        self.batch_size = batch_size
        self.device = device
        self.model = PositionNet(**cfg)
        self.model.load_state_dict(blob["model"])
        self.model.eval().to(device)

    def encode(self, boards: list[chess.Board]) -> np.ndarray:
        if not boards:
            return np.zeros((0, self.dim), np.float32)
        torch = self._torch
        out = []
        with torch.inference_mode():
            for i in range(0, len(boards), self.batch_size):
                x = torch.from_numpy(planes_batch(boards[i: i + self.batch_size])).to(self.device)
                out.append(self.model.embed(x).cpu().numpy())
        return np.concatenate(out)

    def policy_logits(self, boards: list[chess.Board]) -> tuple[np.ndarray, np.ndarray]:
        """From-square and to-square log-softmaxes, one row per board.

        Returned per position rather than per move so callers can deduplicate:
        the logits depend only on the board, and a repeated position in the
        opponent's repertoire should not be a second forward pass.
        """
        torch = self._torch
        fs, ts = [], []
        with torch.inference_mode():
            for i in range(0, len(boards), self.batch_size):
                x = torch.from_numpy(planes_batch(boards[i: i + self.batch_size])).to(self.device)
                lf, lt = self.model.policy(x)
                fs.append(lf.cpu().numpy())
                ts.append(lt.cpu().numpy())
        return np.concatenate(fs), np.concatenate(ts)


class NumpyEncoder:
    """The same frozen trunk, run in NumPy from the .npz that
    train/export_numpy.py writes. This is what the website uses, since torch
    does not fit on its server. It shares the torch encoder's name because the
    two produce the same vectors, so they can share one embedding cache.
    """

    is_deep = True

    def __init__(self, weights: str | None = None, batch_size: int = 256):
        from .npnet import DEFAULT_WEIGHTS, NumpyNet

        self.net = NumpyNet(weights or DEFAULT_WEIGHTS)
        self.dim = self.net.dim
        self.name = f"{self.net.name}-{self.net.fingerprint}"
        self.batch_size = batch_size

    def _batches(self, boards: list[chess.Board]):
        for i in range(0, len(boards), self.batch_size):
            yield planes_batch(boards[i: i + self.batch_size])

    def encode(self, boards: list[chess.Board]) -> np.ndarray:
        if not boards:
            return np.zeros((0, self.dim), np.float32)
        return np.concatenate([self.net.embed(x) for x in self._batches(boards)])

    def policy_logits(self, boards: list[chess.Board]) -> tuple[np.ndarray, np.ndarray]:
        fs, ts = zip(*(self.net.policy(x) for x in self._batches(boards)))
        return np.concatenate(fs), np.concatenate(ts)


def load(checkpoint: str | None = None, device: str = "cpu") -> Encoder:
    """The deep encoder when a checkpoint is available, the fallback otherwise.
    An .npz checkpoint runs in NumPy, anything else goes to torch."""
    if checkpoint and checkpoint.endswith(".npz"):
        return NumpyEncoder(checkpoint)
    if checkpoint:
        return TorchEncoder(checkpoint, device=device)
    return StructuralEncoder()
