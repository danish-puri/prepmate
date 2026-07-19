"""chess.com public API adapter (api.chess.com/pub, no auth).

chess.com asks for sequential requests (no parallel bursts) and an identifying
User-Agent; both are honored here. Monthly archives for closed months are
immutable, so they cache forever.
"""

import asyncio
import re
from datetime import datetime, timezone

import httpx

from .. import cache
from ..models import DRAW_CODES, Game

BASE = "https://api.chess.com/pub"

# chess.com asks for sequential requests; several API endpoints hit this
# adapter concurrently on a cold cache, so serialize at the HTTP level and
# retry once on a throttle or transient upstream failure
_LOCK = asyncio.Lock()
_RETRY_STATUSES = {429, 502, 503}


async def _get(client: httpx.AsyncClient, url: str, **kwargs) -> httpx.Response:
    async with _LOCK:
        r = await client.get(url, **kwargs)
        if r.status_code in _RETRY_STATUSES:
            await asyncio.sleep(min(float(r.headers.get("Retry-After", 1)), 10))
            r = await client.get(url, **kwargs)
        return r

# chess.com answers 404 for unknown users but 410 for malformed usernames
# (e.g. containing a space); both mean "no such player"
NOT_FOUND = {400, 404, 410}

_ECO_RE = re.compile(r'\[ECO "([^"]+)"\]')
_ECO_URL_RE = re.compile(r'\[ECOUrl "https://www\.chess\.com/openings/([^"]+)"\]')

# a single SAN token: piece move, pawn move (with optional capture/promotion),
# or castling, each with an optional check/mate suffix
_SAN_RE = re.compile(r"^(?:[KQRBN][a-h]?[1-8]?x?[a-h][1-8]|[a-h](?:x[a-h])?[1-8](?:=[QRBN])?|O-O(?:-O)?)[+#]?$")
_COMMENT_RE = re.compile(r"\{[^}]*\}")


def _san_moves(pgn: str) -> list[str]:
    """Mainline SAN from a chess.com PGN. The movetext follows the blank line
    after the headers and is littered with move numbers and %clk comments;
    keep only the tokens that parse as SAN."""
    movetext = pgn.split("\n\n", 1)[1] if "\n\n" in pgn else pgn
    moves = []
    for token in _COMMENT_RE.sub(" ", movetext).split():
        token = token.split(".")[-1]  # drops "1.", "1...", and glued "1.e4" prefixes
        if _SAN_RE.match(token):
            moves.append(token)
    return moves


# connector words left dangling once the move continuation is cut off
# ("Modern-Defense-with-1-e4" would otherwise become "Modern Defense with")
_TRAILING_JUNK = {"with", "and", "against", "of", "the"}


def _opening_name(slug: str) -> str:
    # Slugs append move continuations ("...-3...Bf5"); keep tokens up to the
    # first one that starts with a digit.
    words = []
    for token in slug.split("-"):
        if token[:1].isdigit():
            break
        words.append(token)
    while words and words[-1].lower() in _TRAILING_JUNK:
        words.pop()
    return " ".join(words) if words else slug


async def get_profile(client: httpx.AsyncClient, username: str) -> dict | None:
    key = f"chesscom:profile:{username.lower()}"
    cached = cache.get(key, max_age=600)
    if cached is not None:
        return cached or None
    r = await _get(client, f"{BASE}/player/{username.lower()}")
    if r.status_code in NOT_FOUND:
        cache.put(key, {})
        return None
    r.raise_for_status()
    data = r.json()
    # closed accounts keep a profile page but should not be scouted
    if data.get("status", "").startswith("closed"):
        cache.put(key, {})
        return None
    cache.put(key, data)
    return data


async def get_stats(client: httpx.AsyncClient, username: str) -> dict | None:
    key = f"chesscom:stats:{username.lower()}"
    cached = cache.get(key, max_age=600)
    if cached is not None:
        return cached or None
    r = await _get(client, f"{BASE}/player/{username.lower()}/stats")
    if r.status_code in NOT_FOUND:
        cache.put(key, {})
        return None
    r.raise_for_status()
    data = r.json()
    cache.put(key, data)
    return data


def ratings_from_stats(stats: dict | None) -> dict:
    out = {}
    for tc in ("bullet", "blitz", "rapid", "daily"):
        block = (stats or {}).get(f"chess_{tc}")
        if block and block.get("last"):
            out[tc] = block["last"].get("rating")
    return out


async def get_games(client: httpx.AsyncClient, username: str, since: datetime) -> list[Game]:
    """All rated games from archive months >= `since`'s calendar month.

    Selecting by calendar cutoff (not the last N archives) keeps the window
    honest for players with inactive months: an archive list with gaps would
    otherwise stretch "6 months" over years.
    """
    username = username.lower()
    r = await _get(client, f"{BASE}/player/{username}/games/archives")
    if r.status_code in NOT_FOUND:
        return []
    r.raise_for_status()
    cutoff = (since.year, since.month)
    archives = [url for url in r.json().get("archives", [])
                if (int(url.split("/")[-2]), int(url.split("/")[-1])) >= cutoff]

    now = datetime.now(timezone.utc)
    games: list[Game] = []
    for url in archives:  # sequential on purpose, per chess.com API guidance
        year, month = int(url.split("/")[-2]), int(url.split("/")[-1])
        closed = (year, month) < (now.year, now.month)
        key = f"chesscom:archive:{username}:{year}-{month:02d}"
        data = cache.get(key, max_age=None if closed else 600)
        if data is None:
            resp = await _get(client, url)
            resp.raise_for_status()
            data = resp.json()
            cache.put(key, data)
        for g in data.get("games", []):
            parsed = _parse_game(g, username)
            if parsed:
                games.append(parsed)
    return games


def _parse_game(g: dict, username: str) -> Game | None:
    if not g.get("rated") or g.get("rules", "chess") != "chess":
        return None
    white, black = g.get("white", {}), g.get("black", {})
    if white.get("username", "").lower() == username:
        me, opp, color = white, black, "white"
    elif black.get("username", "").lower() == username:
        me, opp, color = black, white, "black"
    else:
        return None

    code = me.get("result", "")
    if code == "win":
        result = "win"
    elif code in DRAW_CODES:
        result = "draw"
    else:
        result = "loss"

    pgn = g.get("pgn", "")
    eco_m = _ECO_RE.search(pgn)
    url_m = _ECO_URL_RE.search(pgn)
    return Game(
        platform="chesscom",
        color=color,
        time_class=g.get("time_class", "blitz"),
        result=result,
        termination=code,
        eco=eco_m.group(1) if eco_m else None,
        opening=_opening_name(url_m.group(1)) if url_m else None,
        end_time=g.get("end_time", 0),
        player_rating=me.get("rating"),
        opponent_rating=opp.get("rating"),
        moves=_san_moves(pgn),
    )
