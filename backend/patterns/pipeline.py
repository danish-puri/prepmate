"""Orchestration: games in, patterns out, with the caching that makes it cheap.

Measured on 400 rated games from an active titled player (36,764 plies):

  replay          36,764 plies              ~1.1s in python-chess, unavoidable
  select          -> 9,405 decisions        75% dropped: the opponent's half of
                                            the moves, the book, the long tail,
                                            forced replies and sole recaptures
  deduplicate     -> 8,769 positions        only 7%; a repertoire repeats in the
                                            opening, and the opening is skipped
  cache lookup    -> whatever is new        0% on a first scout, 100% on a
                                            re-scout of unchanged games
  encode          only the new ones         the only expensive stage
  mine            ~20ms                     a few matrix multiplies

Selection is the win, not deduplication. Cross-player cache overlap was
measured at 0.35% and is not worth counting on: by move eight two players have
almost nothing in common. What the cache does buy is re-scouting, where every
position is a hit and adding a week of new games costs only the new games,
because a position's embedding never changes.
"""

import time
from dataclasses import dataclass, field

import chess
import numpy as np

from ..models import Game
from . import mining, replay, store
from .replay import Decision


@dataclass
class Scout:
    games: list[Game]
    decisions: list[Decision]
    vectors: np.ndarray
    boards: dict[int, chess.Board]
    space: str
    stats: dict = field(default_factory=dict)

    @property
    def fens(self) -> dict[int, str]:
        return {k: b.fen() for k, b in self.boards.items()}


def _unit(x: np.ndarray) -> np.ndarray:
    """Rows to unit length so cosine similarity is a dot product. A no-op for
    the deep encoder, which already normalizes, and a necessity for any other."""
    if not len(x):
        return x
    n = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.where(n == 0, 1, n)


def build(games: list[Game], encoder, first_ply: int = replay.FIRST_PLY,
          last_ply: int = replay.LAST_PLY, use_cache: bool = True) -> Scout:
    t0 = time.perf_counter()
    decisions, boards = replay.decisions(games, first_ply=first_ply, last_ply=last_ply)
    t_replay = time.perf_counter() - t0

    keys = list(boards)
    cached = store.get_many(encoder.name, keys) if use_cache else {}
    missing = [k for k in keys if k not in cached]

    t1 = time.perf_counter()
    if missing:
        fresh = encoder.encode([boards[k] for k in missing])
        cached |= dict(zip(missing, fresh))
        if use_cache:
            store.put_many(encoder.name, {k: cached[k] for k in missing})
    t_encode = time.perf_counter() - t1

    vectors = _unit(np.stack([cached[d.key] for d in decisions])
                    if decisions else np.zeros((0, getattr(encoder, "dim", 1)), np.float32))

    return Scout(games=games, decisions=decisions, vectors=vectors, boards=boards,
                 space=encoder.name, stats={
                     "games": len(games),
                     "plies": sum(len(g.moves) for g in games),
                     "decisions": len(decisions),
                     "distinct_positions": len(keys),
                     "cache_hits": len(keys) - len(missing),
                     "encoded": len(missing),
                     "replay_ms": round(t_replay * 1000),
                     "encode_ms": round(t_encode * 1000),
                 })


def predictability(scout: Scout, encoder) -> dict:
    """Needs a policy head, so only the deep encoder can answer this."""
    if not hasattr(encoder, "policy_logits") or not scout.decisions:
        return {"n": 0, "available": False}

    keys = list(scout.boards)
    index = {k: i for i, k in enumerate(keys)}
    logf, logt = encoder.policy_logits([scout.boards[k] for k in keys])

    rows = []
    for d in scout.decisions:
        board = scout.boards[d.key]
        i = index[d.key]
        played = chess.Move.from_uci(d.canon_move)
        legal = [played] + [m for m in board.legal_moves if m != played]
        s = np.array([logf[i, m.from_square] + logt[i, m.to_square] for m in legal], np.float32)
        s -= s.max()
        rows.append(s - np.log(np.exp(s).sum()))
    return {**mining.predictability(rows), "available": True}


def report(scout: Scout, top: int = 5, k: int | None = None, seed: int = 0) -> dict:
    """The scouting payload: ranked archetypes plus the run's own cost."""
    arche = mining.archetypes(scout.decisions, scout.vectors, scout.games,
                              scout.fens, k=k, seed=seed)
    return {
        "stats": scout.stats,
        "space": scout.space,
        "baseline": arche[0].baseline if arche else None,
        "weaknesses": [vars(a) for a in arche[:top]],
        "strengths": [vars(a) for a in sorted(arche, key=lambda a: a.points_lost)[:top]],
        "params": {"prior_games": mining.PRIOR_GAMES, "clusters": len(arche),
                   "first_ply": replay.FIRST_PLY, "last_ply": replay.LAST_PLY},
    }
