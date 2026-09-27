"""Command line scout.

    python -m backend.patterns.scout --lichess <user> --model models/pos-v1.pt

Deliberately not an API route. The web app deploys with three dependencies and
no ML stack, and putting torch behind /api would add a few hundred megabytes to
an image whose whole job is to serve static pages and proxy two public APIs.
Pattern mining is prep work done ahead of a tournament, not something anyone
needs to happen inside a page load, so it runs here and the app stays small.
"""

import argparse
import asyncio
import json
from datetime import datetime, timezone

import httpx

from ..adapters import chesscom, lichess
from . import encoder as encoders
from . import mining, pipeline

USER_AGENT = "PrepMate/0.1 (personal chess prep tool; contact: puridanish5@gmail.com)"


async def collect(chesscom_user: str | None, lichess_user: str | None, months: int,
                  max_games: int, time_classes: set[str]):
    since_month = datetime.now(timezone.utc)
    total = since_month.year * 12 + since_month.month - months
    since = datetime(total // 12, total % 12 + 1, 1, tzinfo=timezone.utc)

    games = []
    async with httpx.AsyncClient(timeout=60, headers={"User-Agent": USER_AGENT},
                                 follow_redirects=True) as client:
        if chesscom_user:
            games += [g for g in await chesscom.get_games(client, chesscom_user, since=since)
                      if g.time_class in time_classes]
        if lichess_user:
            more, _ = await lichess.get_games(client, lichess_user, max_games=max_games,
                                              since=since, time_classes=time_classes)
            games += more
    return games


def render(report: dict, predict: dict) -> str:
    s, out = report["stats"], []
    out.append(f"{s['games']} games, {s['plies']} plies -> {s['decisions']} decisions "
               f"-> {s['distinct_positions']} positions "
               f"({s['cache_hits']} cached, {s['encoded']} encoded)")
    out.append(f"replay {s['replay_ms']}ms, encode {s['encode_ms']}ms, space {report['space']}")
    if predict.get("available"):
        out.append(f"predictability: {predict['surprisal_bits']} bits/move, "
                   f"top-1 {predict['top1_pct']}%, top-3 {predict['top3_pct']}%")
    if report["baseline"] is None:
        out.append("no decisions found, nothing to mine")
        return "\n".join(out)

    out.append(f"\nbaseline score {report['baseline']}% over the analysed games")
    for title, rows in (("WEAKEST STRUCTURES", report["weaknesses"]),
                        ("STRONGEST STRUCTURES", report["strengths"])):
        out.append(f"\n{title}")
        for a in rows:
            named = ", ".join(f"{n} x{c}" for n, c in a["openings"]) or "mixed"
            out.append(
                f"  score {a['shrunk']:>5.1f}% (raw {a['score']:>5.1f}, n={a['games']:>3})"
                f"  {a['points_lost']:>+6.2f} pts vs par  z={a['z']:>5.2f}")
            out.append(f"      around move {a['mean_ply'] / 2:.0f}, {a['mean_pieces']:.0f} men, {named}")
            out.append(f"      {a['exemplar_fen']}")
    out.append("\nz is a ranking aid, not a p-value: the clusters were chosen by looking"
               "\nat the same games they are scored on, so treat anything under 2 as noise.")
    return "\n".join(out)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--chesscom")
    ap.add_argument("--lichess")
    ap.add_argument("--months", type=int, default=6)
    ap.add_argument("--max-games", type=int, default=300)
    ap.add_argument("--tc", default="blitz,rapid,classical")
    ap.add_argument("--model", help="checkpoint from train/pretrain.py; omitted uses the fallback")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--top", type=int, default=5)
    ap.add_argument("--clusters", type=int, default=None)
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args()

    if not a.chesscom and not a.lichess:
        ap.error("pass --chesscom or --lichess")

    games = asyncio.run(collect(a.chesscom, a.lichess, a.months, a.max_games,
                                {t.strip() for t in a.tc.split(",") if t.strip()}))
    enc = encoders.load(a.model, device=a.device)
    if not getattr(enc, "is_deep", False):
        print("warning: no checkpoint given, using the structural fallback. "
              "Train one with train/pretrain.py for the real thing.\n")

    scout = pipeline.build(games, enc)
    report = pipeline.report(scout, top=a.top, k=a.clusters)
    predict = pipeline.predictability(scout, enc)
    print(json.dumps({**report, "predictability": predict}, indent=2) if a.json
          else render(report, predict))


if __name__ == "__main__":
    main()
