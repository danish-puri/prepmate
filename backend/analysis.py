"""Aggregation and heuristics over normalized games. Pure functions, no I/O.

Thresholds follow DESIGN.md: an opening is flagged when it has at least
MIN_GAMES games and its score sits THRESHOLD_PTS or more off the player's
overall score for that color. Percentages always ship with their n.
"""

from collections import defaultdict
from datetime import datetime, timezone

from .models import Game

MIN_GAMES = 15
THRESHOLD_PTS = 10
MAX_ROWS = 12


def score_pct(wins: int, draws: int, n: int) -> float:
    return round((wins + 0.5 * draws) / n * 100, 1) if n else 0.0


def opening_tables(games: list[Game]) -> dict:
    out = {}
    for color in ("white", "black"):
        color_games = [g for g in games if g.color == color and g.eco]
        n_color = len(color_games)
        buckets: dict[str, dict] = defaultdict(lambda: {"games": 0, "wins": 0, "draws": 0, "losses": 0, "names": defaultdict(int)})
        wins = draws = 0
        for g in color_games:
            b = buckets[g.eco]
            b["games"] += 1
            b["wins"] += g.result == "win"
            b["draws"] += g.result == "draw"
            b["losses"] += g.result == "loss"
            if g.opening:
                b["names"][g.opening] += 1
            wins += g.result == "win"
            draws += g.result == "draw"

        avg = score_pct(wins, draws, n_color)
        rows = []
        for eco, b in sorted(buckets.items(), key=lambda kv: -kv[1]["games"])[:MAX_ROWS]:
            score = score_pct(b["wins"], b["draws"], b["games"])
            flag = None
            if b["games"] >= MIN_GAMES:
                if score <= avg - THRESHOLD_PTS:
                    flag = "weakness"
                elif score >= avg + THRESHOLD_PTS:
                    flag = "strength"
            rows.append({
                "eco": eco,
                "name": max(b["names"], key=b["names"].get) if b["names"] else None,
                "games": b["games"],
                "wins": b["wins"],
                "draws": b["draws"],
                "losses": b["losses"],
                "score": score,
                "freq_pct": round(b["games"] / n_color * 100, 1) if n_color else 0.0,
                "flag": flag,
                "small_sample": b["games"] < MIN_GAMES,
            })
        out[color] = {"n": n_color, "avg_score": avg, "rows": rows}
    return out


def prep_target(tables: dict) -> dict | None:
    """Worst-scoring opening the player still plays often: the weakness with
    the largest game count across both colors."""
    candidates = []
    for color in ("white", "black"):
        for row in tables[color]["rows"]:
            if row["flag"] == "weakness":
                candidates.append({**row, "color": color, "delta": round(row["score"] - tables[color]["avg_score"], 1)})
    if not candidates:
        return None
    return max(candidates, key=lambda r: r["games"])


TREE_MAX_PLIES = 12
TREE_MIN_GAMES = 2


def _tree_node() -> dict:
    return {"games": 0, "wins": 0, "draws": 0, "children": {}}


def _bump(node: dict, result: str) -> None:
    node["games"] += 1
    node["wins"] += result == "win"
    node["draws"] += result == "draw"


def _serialize_children(node: dict, min_games: int) -> list[dict]:
    rows = []
    for san, child in sorted(node["children"].items(), key=lambda kv: -kv[1]["games"]):
        if child["games"] < min_games:
            continue
        rows.append({
            "san": san,
            "games": child["games"],
            "wins": child["wins"],
            "draws": child["draws"],
            "losses": child["games"] - child["wins"] - child["draws"],
            "score": score_pct(child["wins"], child["draws"], child["games"]),
            "freq_pct": round(child["games"] / node["games"] * 100, 1),
            "children": _serialize_children(child, min_games),
        })
    return rows


def move_tree(games: list[Game], max_plies: int = TREE_MAX_PLIES, min_node_games: int = TREE_MIN_GAMES) -> dict:
    """Per-colour move tree over the first max_plies plies. Each node carries
    the player's W/D/L for games that reached that position; branches seen
    fewer than min_node_games times are pruned to keep the payload honest
    (and small). Ply 0 is always White's first move, so in the black tree the
    scouted player's own moves sit on the odd plies."""
    out = {}
    for color in ("white", "black"):
        root = _tree_node()
        for g in games:
            if g.color != color or not g.moves:
                continue
            _bump(root, g.result)
            node = root
            for san in g.moves[:max_plies]:
                node = node["children"].setdefault(san, _tree_node())
                _bump(node, g.result)
        out[color] = {
            "n": root["games"],
            "score": score_pct(root["wins"], root["draws"], root["games"]),
            "moves": _serialize_children(root, min_node_games),
        }
    return out


def colour_split(games: list[Game]) -> dict:
    out: dict = {"white": {}, "black": {}}
    for color in ("white", "black"):
        for tc in ("rapid", "classical", "blitz", "bullet", "daily"):
            subset = [g for g in games if g.color == color and g.time_class == tc]
            if not subset:
                continue
            n = len(subset)
            w = sum(g.result == "win" for g in subset)
            d = sum(g.result == "draw" for g in subset)
            out[color][tc] = {
                "n": n,
                "win_pct": round(w / n * 100),
                "draw_pct": round(d / n * 100),
                "loss_pct": round((n - w - d) / n * 100),
            }
    return out


def colour_totals(games: list[Game]) -> dict:
    """Overall W/D/L counts per colour plus combined, for the stats donuts."""
    out = {}
    for key, subset in (
        ("white", [g for g in games if g.color == "white"]),
        ("black", [g for g in games if g.color == "black"]),
        ("total", games),
    ):
        n = len(subset)
        w = sum(g.result == "win" for g in subset)
        d = sum(g.result == "draw" for g in subset)
        out[key] = {"n": n, "wins": w, "draws": d, "losses": n - w - d, "score": score_pct(w, d, n)}
    return out


def loss_terminations(games: list[Game]) -> dict:
    losses = [g for g in games if g.result == "loss"]
    n = len(losses)
    if not n:
        return {"n": 0}
    counts: dict[str, int] = defaultdict(int)
    for g in losses:
        if g.termination in ("resigned", "resign"):
            counts["resigned"] += 1
        elif g.termination in ("timeout", "outoftime"):
            counts["on_time"] += 1
        elif g.termination in ("checkmated", "mate"):
            counts["checkmated"] += 1
        else:
            counts["other"] += 1
    return {"n": n, **{k: {"count": v, "pct": round(v / n * 100)} for k, v in counts.items()}}


def vs_higher_rated(games: list[Game], margin: int = 100) -> dict:
    subset = [
        g for g in games
        if g.player_rating and g.opponent_rating and g.opponent_rating - g.player_rating >= margin
    ]
    n = len(subset)
    w = sum(g.result == "win" for g in subset)
    d = sum(g.result == "draw" for g in subset)
    return {"n": n, "score": score_pct(w, d, n), "margin": margin}


def rating_history_from_games(games: list[Game]) -> dict:
    """Daily rating series per time class, from the player's post-game ratings.
    chess.com has no rating-history endpoint, so this reconstructs one from
    the archives: the last game of each day carries that day's closing rating."""
    per_tc: dict[str, dict[str, int]] = defaultdict(dict)
    for g in sorted(games, key=lambda g: g.end_time):
        if g.player_rating and g.end_time:
            day = datetime.fromtimestamp(g.end_time, tz=timezone.utc).strftime("%Y-%m-%d")
            per_tc[g.time_class][day] = g.player_rating
    return {
        tc: [{"date": d, "rating": r} for d, r in sorted(days.items())]
        for tc, days in per_tc.items()
    }


def recent_form(games: list[Game], n: int = 20) -> dict:
    latest = sorted(games, key=lambda g: g.end_time, reverse=True)[:n]
    latest.reverse()  # oldest first, newest last, matching the UI strip
    seq = [
        {
            "result": g.result,
            "platform": g.platform,
            "time_class": g.time_class,
            "eco": g.eco,
            "opening": g.opening,
            "opponent_rating": g.opponent_rating,
            "end_time": g.end_time,
        }
        for g in latest
    ]
    return {
        "n": len(seq),
        "wins": sum(s["result"] == "win" for s in seq),
        "draws": sum(s["result"] == "draw" for s in seq),
        "losses": sum(s["result"] == "loss" for s in seq),
        "sequence": seq,
    }
