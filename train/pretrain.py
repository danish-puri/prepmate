"""Pretrain the position encoder on other people's games.

    python -m train.pretrain fetch --out data/corpus --users train/users.txt --per-user 250
    python -m train.pretrain train --corpus data/corpus --out models/pos-v1.pt

The scouted opponent is never in this corpus. A few hundred games is a few
hundred samples and would be memorized by anything with capacity, so the
network learns from a population instead and is frozen before it ever sees the
player being prepared for. Keep the user list disjoint from who you scout.

Pretext task: predict the move a human played, factored into a from-square and
a to-square. The trunk underneath that prediction is the embedding, and the
task is what makes the embedding useful. A net trained to predict human choices
has to represent what a position is *asking*, which is exactly the axis we
later want positions to cluster along. Material counts alone would not do it.

Positions are canonicalized and turned into planes by the same functions
inference uses (backend.patterns.replay / .encoder), because a mismatch there
is silent: the model would simply be wrong at scout time with no error.
"""

import argparse
import asyncio
import hashlib
import json
import sys
from pathlib import Path

import chess
import httpx
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backend.patterns.encoder import N_PLANES, planes  # noqa: E402
from backend.patterns.replay import canonical, canonical_move  # noqa: E402

USER_AGENT = "PrepMate/0.1 (encoder pretraining; contact: puridanish5@gmail.com)"
FIRST_PLY, LAST_PLY = 6, 80


# --- corpus ----------------------------------------------------------------

async def _export(client: httpx.AsyncClient, user: str, per_user: int) -> list[dict]:
    r = await client.get(
        f"https://lichess.org/api/games/user/{user}",
        params={"max": per_user, "rated": "true", "perfType": "blitz,rapid,classical"},
        headers={"Accept": "application/x-ndjson"},
    )
    if r.status_code != 200:
        print(f"  {user}: HTTP {r.status_code}, skipped")
        return []
    return [json.loads(line) for line in r.text.splitlines() if line.strip()]


def _positions(game: dict):
    """Every in-window position of a game, from the mover's point of view.

    Both sides are used here, unlike scouting: the model is learning what chess
    players in general do, so every human decision in the corpus is a sample.
    """
    moves = (game.get("moves") or "").split()
    if len(moves) < FIRST_PLY + 2:
        return
    board = chess.Board()
    for ply, san in enumerate(moves[:LAST_PLY]):
        try:
            move = board.parse_san(san)
        except (chess.IllegalMoveError, chess.InvalidMoveError, chess.AmbiguousMoveError):
            return
        if ply >= FIRST_PLY and board.legal_moves.count() > 1:
            cm = canonical_move(move, board.turn)
            yield canonical(board), cm.from_square, cm.to_square
        board.push(move)


async def fetch(users: list[str], per_user: int, out: Path) -> None:
    """Write a memory-mapped plane array plus targets and game ids.

    Planes are precomputed once rather than replayed each epoch: replaying is
    ~30x slower than reading bytes off disk, and training reads the corpus many
    times. uint8 rather than bits keeps the reader trivial; the array is a few
    hundred MB, which memmaps fine.
    """
    out.mkdir(parents=True, exist_ok=True)
    xs, froms, tos, gids = [], [], [], []
    gid = 0
    async with httpx.AsyncClient(timeout=180, headers={"User-Agent": USER_AGENT},
                                 follow_redirects=True) as client:
        for i, user in enumerate(users, 1):
            games = await _export(client, user, per_user)
            before = len(xs)
            for g in games:
                for board, f, t in _positions(g):
                    xs.append(planes(board).astype(np.uint8))
                    froms.append(f)
                    tos.append(t)
                    gids.append(gid)
                gid += 1
            print(f"[{i}/{len(users)}] {user}: {len(games)} games, "
                  f"+{len(xs) - before} positions, {len(xs)} total", flush=True)

    n = len(xs)
    arr = np.lib.format.open_memmap(out / "planes.npy", mode="w+",
                                    dtype=np.uint8, shape=(n, N_PLANES, 8, 8))
    for i, x in enumerate(xs):
        arr[i] = x
    arr.flush()
    np.savez(out / "targets.npz", frm=np.array(froms, np.int16),
             to=np.array(tos, np.int16), game=np.array(gids, np.int32))
    print(f"wrote {n} positions from {gid} games to {out}")


# --- training --------------------------------------------------------------

def train(corpus: Path, out: Path, epochs: int, batch: int, lr: float,
          channels: int, blocks: int, dim: int, device: str, val_frac: float = 0.05) -> None:
    import torch
    import torch.nn.functional as F

    from backend.patterns.net import PositionNet

    x = np.load(corpus / "planes.npy", mmap_mode="r")
    t = np.load(corpus / "targets.npz")
    frm, to, game = t["frm"].astype(np.int64), t["to"].astype(np.int64), t["game"]

    # split by game, not by position. Positions inside one game are highly
    # correlated, so a positional split leaks and flatters validation
    games = np.unique(game)
    rng = np.random.default_rng(0)
    rng.shuffle(games)
    val_games = set(games[: max(1, int(len(games) * val_frac))].tolist())
    is_val = np.fromiter((g in val_games for g in game), bool, len(game))
    tr_idx, va_idx = np.flatnonzero(~is_val), np.flatnonzero(is_val)
    print(f"{len(tr_idx)} train / {len(va_idx)} val positions, {len(games)} games")

    model = PositionNet(channels=channels, blocks=blocks, dim=dim).to(device)
    params = sum(p.numel() for p in model.parameters())
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    steps = epochs * (len(tr_idx) // batch)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=max(1, steps))
    print(f"PositionNet c={channels} b={blocks} d={dim}: {params/1e3:.0f}k params, {steps} steps")

    def batches(idx, size, shuffle):
        if shuffle:
            idx = idx[rng.permutation(len(idx))]
        for i in range(0, len(idx) - size + 1, size):
            sel = np.sort(idx[i: i + size])  # sorted keeps the memmap reads sequential
            yield (torch.from_numpy(np.asarray(x[sel], np.float32)).to(device),
                   torch.from_numpy(frm[sel]).to(device),
                   torch.from_numpy(to[sel]).to(device))

    for epoch in range(1, epochs + 1):
        model.train()
        run, seen = 0.0, 0
        for bx, bf, bt in batches(tr_idx, batch, True):
            lf, lt, _ = model(bx)
            loss = F.cross_entropy(lf, bf) + F.cross_entropy(lt, bt)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            run += loss.item() * len(bx)
            seen += len(bx)
            if seen % (batch * 50) == 0:
                print(f"  epoch {epoch} {seen}/{len(tr_idx)} loss {run/seen:.4f}", flush=True)

        model.eval()
        hit_f = hit_t = hit_both = n = 0
        with torch.inference_mode():
            for bx, bf, bt in batches(va_idx, batch, False):
                lf, lt, _ = model(bx)
                pf, pt = lf.argmax(1), lt.argmax(1)
                hit_f += (pf == bf).sum().item()
                hit_t += (pt == bt).sum().item()
                hit_both += ((pf == bf) & (pt == bt)).sum().item()
                n += len(bx)
        print(f"epoch {epoch}: train loss {run/max(seen,1):.4f} | val from {hit_f/max(n,1):.3f} "
              f"to {hit_t/max(n,1):.3f} exact {hit_both/max(n,1):.3f}", flush=True)

    out.parent.mkdir(parents=True, exist_ok=True)
    cfg = {"name": f"cnn{channels}x{blocks}d{dim}", "channels": channels,
           "blocks": blocks, "dim": dim}
    state = {k: v.cpu() for k, v in model.state_dict().items()}
    # the fingerprint goes into the cache key, so retraining can never collide
    # with vectors left behind by an earlier model of the same shape
    h = hashlib.sha1(b"".join(v.numpy().tobytes() for v in state.values())).hexdigest()[:8]
    torch.save({"model": state, "config": cfg, "fingerprint": h}, out)
    print(f"saved {out} (space {cfg['name']}-{h})")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)

    f = sub.add_parser("fetch")
    f.add_argument("--users", type=Path, required=True, help="file with one lichess username per line")
    f.add_argument("--per-user", type=int, default=250)
    f.add_argument("--out", type=Path, default=Path("data/corpus"))

    t = sub.add_parser("train")
    t.add_argument("--corpus", type=Path, default=Path("data/corpus"))
    t.add_argument("--out", type=Path, default=Path("models/pos-v1.pt"))
    t.add_argument("--epochs", type=int, default=6)
    t.add_argument("--batch", type=int, default=1024)
    t.add_argument("--lr", type=float, default=2e-3)
    t.add_argument("--channels", type=int, default=64)
    t.add_argument("--blocks", type=int, default=4)
    t.add_argument("--dim", type=int, default=128)
    t.add_argument("--device", default="cpu")

    a = ap.parse_args()
    if a.cmd == "fetch":
        users = [u.strip() for u in a.users.read_text().splitlines() if u.strip()
                 and not u.startswith("#")]
        asyncio.run(fetch(users, a.per_user, a.out))
    else:
        train(a.corpus, a.out, a.epochs, a.batch, a.lr, a.channels, a.blocks, a.dim, a.device)


if __name__ == "__main__":
    main()
