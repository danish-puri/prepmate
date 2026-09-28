"""Pattern detection on top of the frozen embeddings. All non-parametric.

Nothing here is fitted to the opponent beyond cluster centroids and counts,
which is the point: a few hundred games cannot support anything with capacity.
The deep model supplies the geometry, this file does arithmetic in it.

Three things come out:

  archetypes()    recurring position types the player reaches, each scored by
                  how they actually did from there. This is what the ECO tables
                  in analysis.py cannot see, because ECO labels a move order and
                  stops around move ten, while a cluster labels a structure and
                  collects every transposition into it.

  predictability() how closely their moves track what the population model
                  expects. High means prep lands. Low means they improvise.

  nearest()       retrieval: given a position you expect on the board, the most
                  similar positions they have had and how those went.
"""

from collections import Counter, defaultdict
from dataclasses import dataclass, field

import chess
import numpy as np

from ..models import Game
from .replay import Decision

# Pseudo-games of shrinkage toward the player's own overall score. A cluster
# with 4 games barely moves off baseline; one with 40 is mostly its own number.
# This replaces the hard n >= 15 cutoff in analysis.py, which treats a 15-game
# bucket and a 60-game bucket as equally trustworthy.
PRIOR_GAMES = 12

# Roughly one cluster per 150 decisions, bounded so the output stays readable
MIN_K, MAX_K, PER_CLUSTER = 6, 28, 150


@dataclass
class Archetype:
    cluster: int
    positions: int
    games: int
    wins: int
    draws: int
    losses: int
    score: float
    shrunk: float
    baseline: float
    deficit: float
    points_lost: float
    z: float
    mean_ply: float
    mean_pieces: float
    exemplar_fen: str
    openings: list[tuple[str, int]] = field(default_factory=list)
    # the exemplar as it stood in the real game, for showing a person: the
    # player's actual colour, the board un-mirrored, and the move they chose
    exemplar_color: str = "white"
    exemplar_board_fen: str = ""
    exemplar_move_san: str = ""


def choose_k(n: int) -> int:
    return max(MIN_K, min(MAX_K, n // PER_CLUSTER))


def spherical_kmeans(x: np.ndarray, k: int, seed: int = 0,
                     iters: int = 30) -> tuple[np.ndarray, np.ndarray]:
    """k-means on unit vectors, where cosine similarity is just a dot product.

    Brute force on purpose. At a few thousand positions and 128 dimensions the
    whole thing is a handful of matrix multiplies and runs in milliseconds, so
    an approximate index would add a dependency and buy nothing. The expensive
    stage in this pipeline is the encoder, not the mining.
    """
    n = len(x)
    k = max(1, min(k, n))
    rng = np.random.default_rng(seed)

    # k-means++ seeding, which matters more than iteration count for stability
    centers = [x[rng.integers(n)]]
    for _ in range(k - 1):
        d = 1.0 - (x @ np.stack(centers).T).max(axis=1)
        d = np.clip(d, 0, None)
        total = d.sum()
        # searchsorted can land on n when the cumulative sum stops a hair short
        # of 1.0, so clamp rather than let a rare seed raise
        pick = (rng.integers(n) if total <= 0 else
                min(int(np.searchsorted(np.cumsum(d / total), rng.random())), n - 1))
        centers.append(x[pick])
    c = np.stack(centers)

    labels = np.zeros(n, dtype=np.int32)
    for _ in range(iters):
        sim = x @ c.T
        new = sim.argmax(axis=1).astype(np.int32)
        if np.array_equal(new, labels):
            break
        labels = new
        for j in range(k):
            members = x[labels == j]
            if len(members):
                v = members.sum(axis=0)
                norm = np.linalg.norm(v)
                c[j] = v / norm if norm else c[j]
            else:
                # empty cluster: hand it the point that fits its cluster worst
                c[j] = x[sim.max(axis=1).argmin()]
    return labels, c


def _score(wins: float, draws: float, n: float) -> float:
    return (wins + 0.5 * draws) / n * 100 if n else 0.0


def _piece_count(fen: str) -> int:
    """Men left on the board, which is the readable proxy for game phase."""
    return sum(c.isalpha() for c in fen.split(" ", 1)[0])


def archetypes(decisions: list[Decision], vectors: np.ndarray, games: list[Game],
               fens: dict[int, str], k: int | None = None, seed: int = 0) -> list[Archetype]:
    """Cluster the positions, then score each cluster by the player's results.

    `vectors[i]` is the embedding of `decisions[i]`. A game is counted at most
    once per cluster: a single game can pass through the same structure fifteen
    times, and treating those as fifteen observations would inflate every
    sample size and make noise look significant.
    """
    if not decisions:
        return []
    labels, centers = spherical_kmeans(vectors, k or choose_k(len(decisions)), seed=seed)

    played = sorted({d.game for d in decisions})
    w = sum(games[i].result == "win" for i in played)
    dr = sum(games[i].result == "draw" for i in played)
    baseline = _score(w, dr, len(played))

    per_cluster: dict[int, set[int]] = defaultdict(set)
    plies: dict[int, list[int]] = defaultdict(list)
    for d, lab in zip(decisions, labels):
        per_cluster[int(lab)].add(d.game)
        plies[int(lab)].append(d.ply)

    out = []
    for j, game_ids in per_cluster.items():
        n = len(game_ids)
        wins = sum(games[i].result == "win" for i in game_ids)
        draws = sum(games[i].result == "draw" for i in game_ids)
        raw = _score(wins, draws, n)
        shrunk = (wins + 0.5 * draws + PRIOR_GAMES * baseline / 100) / (n + PRIOR_GAMES) * 100
        deficit = baseline - shrunk
        p = baseline / 100
        se = np.sqrt(p * (1 - p) / n) * 100 if 0 < p < 1 and n else 0.0

        members = np.flatnonzero(labels == j)
        exemplar = members[(vectors[members] @ centers[j]).argmax()]
        counts = Counter(games[i].opening for i in game_ids if games[i].opening)
        ex = decisions[exemplar]
        color = games[ex.game].color
        real = chess.Board(fens[ex.key])
        if color == "black":
            real = real.mirror()
        try:
            san = real.san(chess.Move.from_uci(ex.move))
        except ValueError:
            san = ""

        out.append(Archetype(
            cluster=j, positions=len(members), games=n,
            wins=wins, draws=draws, losses=n - wins - draws,
            score=round(raw, 1), shrunk=round(shrunk, 1), baseline=round(baseline, 1),
            deficit=round(deficit, 1),
            # points below par actually dropped here, which ranks a wide, mildly
            # bad structure above a narrow, catastrophic one. Prep time is finite
            # and should go where the losses are
            points_lost=round(deficit / 100 * n, 2),
            z=round(deficit / se, 2) if se else 0.0,
            mean_ply=round(float(np.mean(plies[j])), 1),
            mean_pieces=round(float(np.mean(
                [_piece_count(fens[decisions[i].key]) for i in members])), 1),
            exemplar_fen=fens[decisions[exemplar].key],
            openings=counts.most_common(3),
            exemplar_color=color, exemplar_board_fen=real.fen(), exemplar_move_san=san,
        ))
    return sorted(out, key=lambda a: -a.points_lost)


def predictability(logps: list[np.ndarray]) -> dict:
    """How well the population model calls this player's moves.

    `logps[i]` is the log-probability the model assigned to each legal move in
    decision i, and index 0 is the move actually played. Reported in bits so the
    number means something: 0 bits is a move the model was certain of, 4 bits is
    one it gave a 1-in-16 chance.
    """
    if not logps:
        return {"n": 0}
    chosen = np.array([lp[0] for lp in logps])
    ranks = np.array([int((lp > lp[0]).sum()) + 1 for lp in logps])
    return {
        "n": len(logps),
        "surprisal_bits": round(float(-chosen.mean() / np.log(2)), 2),
        "top1_pct": round(float((ranks == 1).mean() * 100), 1),
        "top3_pct": round(float((ranks <= 3).mean() * 100), 1),
    }


def nearest(query: np.ndarray, vectors: np.ndarray, decisions: list[Decision],
            games: list[Game], n: int = 6) -> list[dict]:
    """The player's most similar positions to one you expect to reach.

    Brute force again: at this scale a full scan is a single matrix-vector
    product, and the exactness is worth more than the microseconds an index
    would save.
    """
    if not len(vectors):
        return []
    sims = vectors @ (query / (np.linalg.norm(query) or 1))
    seen, out = set(), []
    for i in np.argsort(-sims):
        d = decisions[int(i)]
        if d.game in seen:  # one row per game, not per repeated position
            continue
        seen.add(d.game)
        g = games[d.game]
        out.append({"similarity": round(float(sims[i]), 3), "ply": d.ply, "move": d.move,
                    "result": g.result, "platform": g.platform, "eco": g.eco,
                    "opening": g.opening, "end_time": g.end_time})
        if len(out) >= n:
            break
    return out
