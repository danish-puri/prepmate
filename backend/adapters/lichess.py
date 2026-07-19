"""lichess public API adapter (lichess.org/api, no auth for public data).

Game exports come pre-tagged with openings (ECO + name), which is why lichess
is the first-class source in the analysis layer.
"""

import asyncio
import json
from datetime import datetime

import httpx

from .. import cache
from ..models import Game

BASE = "https://lichess.org/api"

# 4xx statuses that mean "no such player" (404 unknown, 400/410 malformed)
NOT_FOUND = {400, 404, 410}

_EXPORT_LOCK = asyncio.Lock()

# lichess status values that mean the game was drawn regardless of `winner`
_DRAW_STATUSES = {"draw", "stalemate"}
_TERMINATION = {"mate": "checkmated", "resign": "resigned", "outoftime": "timeout", "timeout": "abandoned"}

# one game export per user at a time, per lichess API guidance
_EXPORT_LOCK = asyncio.Lock()

# lichess speeds folded into the shared time_class vocabulary (chess.com's)
_SPEED_TC = {"ultraBullet": "bullet", "correspondence": "daily"}
# and the reverse: which lichess perfTypes to request per canonical time class
_PERFS_FOR_TC = {"bullet": ("bullet", "ultraBullet"), "blitz": ("blitz",), "rapid": ("rapid",),
                 "classical": ("classical",), "daily": ("correspondence",)}


async def get_user(client: httpx.AsyncClient, username: str) -> dict | None:
    key = f"lichess:user:{username.lower()}"
    cached = cache.get(key, max_age=600)
    if cached is not None:
        return cached or None
    r = await client.get(f"{BASE}/user/{username}")
    if r.status_code in NOT_FOUND:
        cache.put(key, {})
        return None
    r.raise_for_status()
    data = r.json()
    # closed accounts come back as {"disabled": true} with no ratings or
    # games; showing them as found would link a dead account into the dossier
    if data.get("disabled"):
        cache.put(key, {})
        return None
    cache.put(key, data)
    return data


def ratings_from_user(user: dict | None) -> dict:
    out = {}
    for tc in ("bullet", "blitz", "rapid", "classical"):
        perf = (user or {}).get("perfs", {}).get(tc)
        if perf and perf.get("games"):
            out[tc] = perf.get("rating")
    return out


async def get_games(client: httpx.AsyncClient, username: str, max_games: int = 300,
                    since: datetime | None = None,
                    time_classes: set[str] | None = None) -> tuple[list[Game], bool]:
    """Rated games since `since`, newest first, filtered server-side to
    `time_classes` (canonical names; None = every speed).

    Returns (games, truncated): truncated means the count cap cut the time
    window short, so lichess coverage stops before `since`.
    """
    if time_classes is None:
        perfs = "ultraBullet,bullet,blitz,rapid,classical,correspondence"
    else:
        perfs = ",".join(p for tc in sorted(time_classes) for p in _PERFS_FOR_TC[tc])
    key = f"lichess:games:{username.lower()}:{max_games}:{since.date() if since else 'all'}:{perfs}"
    lines = cache.get(key, max_age=3600)
    if lines is None:
        # lichess allows one game export per user at a time, and several
        # endpoints hit this adapter in parallel on a cold cache: serialize,
        # and re-check the cache so identical requests coalesce on one export
        async with _EXPORT_LOCK:
            lines = cache.get(key, max_age=3600)
            if lines is None:
                params = {"max": max_games, "rated": "true", "opening": "true", "perfType": perfs}
                if since is not None:
                    params["since"] = int(since.timestamp() * 1000)
                r = await client.get(
                    f"{BASE}/games/user/{username}",
                    params=params,
                    headers={"Accept": "application/x-ndjson"},
                )
                if r.status_code in NOT_FOUND:
                    return [], False
                r.raise_for_status()
                lines = r.text.splitlines()
                cache.put(key, lines)

    games = []
    # over HTTP/1.1 the lichess export stream can overshoot ?max=, so
    # enforce the cap here (games arrive newest first)
    for line in lines[:max_games]:
        if not line.strip():
            continue
        parsed = _parse_game(json.loads(line), username.lower())
        if parsed:
            games.append(parsed)
    return games, len(lines) >= max_games


async def get_rating_history(client: httpx.AsyncClient, username: str) -> list | None:
    key = f"lichess:history:{username.lower()}"
    cached = cache.get(key, max_age=3600)
    if cached is not None:
        return cached or None
    r = await client.get(f"{BASE}/user/{username}/rating-history")
    if r.status_code in NOT_FOUND:
        return None
    r.raise_for_status()
    data = r.json()
    cache.put(key, data)
    return data


def _parse_game(g: dict, username: str) -> Game | None:
    players = g.get("players", {})
    white_id = (players.get("white", {}).get("user") or {}).get("id", "")
    black_id = (players.get("black", {}).get("user") or {}).get("id", "")
    if white_id == username:
        color, me, opp = "white", players.get("white", {}), players.get("black", {})
    elif black_id == username:
        color, me, opp = "black", players.get("black", {}), players.get("white", {})
    else:
        return None

    status = g.get("status", "")
    winner = g.get("winner")
    if status in _DRAW_STATUSES or winner is None:
        result = "draw"
    elif winner == color:
        result = "win"
    else:
        result = "loss"

    opening = g.get("opening") or {}
    return Game(
        platform="lichess",
        color=color,
        time_class=_SPEED_TC.get(g.get("speed", ""), g.get("speed", "blitz")),
        result=result,
        termination=_TERMINATION.get(status, status),
        eco=opening.get("eco"),
        opening=opening.get("name"),
        end_time=int(g.get("createdAt", 0) / 1000),
        player_rating=me.get("rating"),
        opponent_rating=opp.get("rating"),
        moves=(g.get("moves") or "").split(),
    )
