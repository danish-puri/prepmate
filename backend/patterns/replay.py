"""SAN move lists into the positions where the scouted player actually chose.

Everything downstream works on *decisions*, not plies. Of the ~80 plies in a
game only about half belong to the player, and a good share of those are
forced or a single legal recapture, which say nothing about how they think.
Filtering here is what keeps the encoder cheap: the deep model is the only
expensive stage in the pipeline, so the cheapest way to speed it up is to
hand it fewer positions.

Positions are canonicalized to "the player to move sits at the bottom, in
white's colours", so a structure embeds to the same vector whether the
opponent met it as White or as Black.
"""

from dataclasses import dataclass

import chess
import chess.polyglot

from ..models import Game

# Book territory is already covered by the ECO tables in analysis.py, and the
# tail of a long game is mostly technique. Both ends are skipped.
FIRST_PLY = 8
LAST_PLY = 60


@dataclass(frozen=True)
class Decision:
    """One position the player had to make a real choice in."""
    key: int          # canonical zobrist, the cache key for the embedding
    game: int         # index into the games list this came from
    ply: int
    move: str         # what they played, UCI, as it appeared on their board
    canon_move: str   # the same move on the canonical board, for the model
    legal: int        # how many moves they could have played


def canonical(board: chess.Board) -> chess.Board:
    """Board seen from the side to move. mirror() swaps colours and flips the
    ranks, so the result always has White to play."""
    return board if board.turn == chess.WHITE else board.mirror()


def canonical_move(move: chess.Move, turn: chess.Color) -> chess.Move:
    """The same move on the canonical board. mirror() flips the ranks, so a
    black move has to be flipped with it or it points at the wrong squares."""
    if turn == chess.WHITE:
        return move
    return chess.Move(chess.square_mirror(move.from_square),
                      chess.square_mirror(move.to_square), promotion=move.promotion)


def position_key(board: chess.Board) -> int:
    """Zobrist of the canonical board, folded into SQLite's signed 64-bit int.

    The hash ignores move counters, so the same structure reached by different
    move orders collapses to one cache entry. That transposition folding is
    also why this finds patterns the ECO tables cannot: ECO is a label on a
    move order, this is a label on a position.
    """
    h = chess.polyglot.zobrist_hash(canonical(board))
    return h - (1 << 64) if h >= (1 << 63) else h


def _sole_recapture(board: chess.Board, move: chess.Move, target: int | None) -> bool:
    """True when the player is taking back on the square just captured on and
    had no other way to do it. A forced recapture is not a decision."""
    if target is None or move.to_square != target or not board.is_capture(move):
        return False
    return sum(1 for m in board.legal_moves
               if m.to_square == target and board.is_capture(m)) == 1


def decisions(games: list[Game], first_ply: int = FIRST_PLY,
              last_ply: int = LAST_PLY) -> tuple[list[Decision], dict[int, chess.Board]]:
    """Every real choice the player made, plus one board per distinct position.

    The board map is the deduplicated work list for the encoder: a player who
    repeats their repertoire reaches the same positions over and over, and
    each one only ever needs encoding once.

    Illegal or unparseable move lists are skipped rather than raised on. The
    SAN comes from a regex over chess.com PGN text (chesscom.py:47), so a
    malformed game is a data problem, not a reason to fail a whole scout.
    """
    out: list[Decision] = []
    boards: dict[int, chess.Board] = {}

    for i, g in enumerate(games):
        if not g.moves:
            continue
        board = chess.Board()
        ours = chess.WHITE if g.color == "white" else chess.BLACK
        target: int | None = None  # square the previous move captured on
        for ply, san in enumerate(g.moves):
            if ply > last_ply:
                break
            try:
                move = board.parse_san(san)
            except (chess.IllegalMoveError, chess.InvalidMoveError, chess.AmbiguousMoveError):
                break
            if board.turn == ours and ply >= first_ply:
                legal = board.legal_moves.count()
                if legal > 1 and not _sole_recapture(board, move, target):
                    key = position_key(board)
                    if key not in boards:
                        # canonical() hands back the same object when White is
                        # to move, and this loop keeps pushing onto it, so the
                        # snapshot has to be a copy or every white-to-move entry
                        # ends up holding the final position of its game
                        boards[key] = canonical(board).copy(stack=False)
                    out.append(Decision(key=key, game=i, ply=ply, move=move.uci(),
                                        canon_move=canonical_move(move, ours).uci(),
                                        legal=legal))
            target = move.to_square if board.is_capture(move) else None
            board.push(move)

    return out, boards
