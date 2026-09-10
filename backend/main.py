"""PrepMate backend.

Run from the project root:
    uvicorn backend.main:app --reload

Also serves the static frontend (static/) at /.
"""

import os
import re
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path

import httpx
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from . import analysis, cache
from .adapters import chesscom, fide, lichess
from .limiter import RateLimiter

USER_AGENT = "PrepMate/0.1 (personal chess prep tool; contact: puridanish5@gmail.com)"
STATIC = Path(__file__).parent.parent / "static"

# The browser loads the frontend from this same app, so cross-origin access is
# never needed in production and the default allowlist only covers local dev.
# Set ALLOWED_ORIGINS (comma separated) to open it up, or leave it empty to
# send no CORS headers at all.
DEFAULT_ORIGINS = "http://localhost:8000,http://127.0.0.1:8000"

# Sustained requests per minute per IP, and how many may arrive back to back.
# A page load fires about five calls, so the burst covers a few impatient
# reloads before anything gets turned away.
RATE_LIMIT_PER_MINUTE = float(os.getenv("RATE_LIMIT_PER_MINUTE", "60"))
RATE_LIMIT_BURST = int(os.getenv("RATE_LIMIT_BURST", "20"))

# The frontend gets its own, looser budget. It reaches no upstream API, so the
# only thing it spends is outbound bytes, which is the one metered resource on
# the host. A page load is one HTML file plus a couple of images, so 60 a minute
# is roughly twenty page loads and nobody clicking around will ever see a 429,
# while a script looping over the frontend stops being unbounded.
STATIC_RATE_LIMIT_PER_MINUTE = float(os.getenv("STATIC_RATE_LIMIT_PER_MINUTE", "60"))
STATIC_RATE_LIMIT_BURST = int(os.getenv("STATIC_RATE_LIMIT_BURST", "30"))

# Repeat visits are the cheaper half of this: a browser that caches sends no
# request at all. The images are stable for as long as their filename is, so
# they get a week. The HTML changes on every deploy, so it revalidates instead,
# and StaticFiles already sends an ETag that turns the recheck into a bodyless
# 304 rather than another download.
CACHEABLE_ASSETS = (".jpg", ".jpeg", ".png", ".webp", ".svg", ".ico", ".woff2", ".css", ".js")
ASSET_CACHE_CONTROL = "public, max-age=604800"
PAGE_CACHE_CONTROL = "public, no-cache"

# Which header carries the real caller when a proxy sits in front. Leave unset
# and the socket peer is used, which is correct for local runs. Only name a
# header the proxy overwrites on every request: X-Forwarded-For is appended to,
# so its leftmost entry is whatever the caller decided to send.
CLIENT_IP_HEADER = os.getenv("CLIENT_IP_HEADER", "")


def _allowed_origins() -> list[str]:
    raw = os.getenv("ALLOWED_ORIGINS", DEFAULT_ORIGINS)
    return [origin.strip().rstrip("/") for origin in raw.split(",") if origin.strip()]


def _client_ip(request: Request) -> str:
    """Who to charge for this request in the rate limiter."""
    if CLIENT_IP_HEADER:
        value = request.headers.get(CLIENT_IP_HEADER)
        if value:
            return value.strip()
    return request.client.host if request.client else "unknown"


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.client = httpx.AsyncClient(timeout=30, headers={"User-Agent": USER_AGENT}, follow_redirects=True)
    app.state.limiter = (
        RateLimiter(RATE_LIMIT_PER_MINUTE, RATE_LIMIT_BURST)
        if RATE_LIMIT_PER_MINUTE > 0 and RATE_LIMIT_BURST > 0 else None
    )
    app.state.static_limiter = (
        RateLimiter(STATIC_RATE_LIMIT_PER_MINUTE, STATIC_RATE_LIMIT_BURST)
        if STATIC_RATE_LIMIT_PER_MINUTE > 0 and STATIC_RATE_LIMIT_BURST > 0 else None
    )
    yield
    await app.state.client.aclose()


app = FastAPI(title="PrepMate", lifespan=lifespan)


@app.middleware("http")
async def rate_limit(request: Request, call_next):
    """Throttle per client IP, on two separate budgets.

    /api/* fans out to chess.com and lichess under my User-Agent, so it gets the
    tight bucket. The frontend reaches no upstream but still costs outbound
    bytes, so it gets a loose one rather than none at all: served unthrottled it
    was the only path on the app with no ceiling on what a script could spend.

    /healthz is exempt because a throttled probe would fail a deploy, and CORS
    preflights are exempt because they stop at the middleware and would
    otherwise spend a caller's budget twice per real request.
    """
    path = request.url.path
    if request.method == "OPTIONS" or path == "/healthz":
        return await call_next(request)

    is_api = path.startswith("/api/")
    limiter = getattr(request.app.state, "limiter" if is_api else "static_limiter", None)
    if limiter is None:
        return await _with_cache_headers(request, call_next)

    allowed, remaining, retry_after = limiter.take(_client_ip(request))
    if not allowed:
        return JSONResponse(
            status_code=429,
            content={"detail": "too many requests, slow down for a moment"},
            headers={
                "Retry-After": str(max(1, round(retry_after))),
                "X-RateLimit-Limit": str(int(limiter.per_minute)),
                "X-RateLimit-Remaining": "0",
            },
        )

    response = await _with_cache_headers(request, call_next)
    if is_api:
        response.headers["X-RateLimit-Limit"] = str(int(limiter.per_minute))
        response.headers["X-RateLimit-Remaining"] = str(remaining)
    return response


async def _with_cache_headers(request: Request, call_next):
    """Tell the browser how long it may keep a frontend file.

    Set here rather than on the mount so it survives a StaticFiles swap, and so
    the rule lives next to the rate limit it works with. Only successful GETs
    get a header: caching an error, or a response to a request that changed
    something, is how a stale page outlives the deploy that fixed it.
    """
    response = await call_next(request)
    path = request.url.path
    if path.startswith("/api/") or request.method != "GET" or response.status_code >= 400:
        return response

    if path.endswith(CACHEABLE_ASSETS):
        response.headers["Cache-Control"] = ASSET_CACHE_CONTROL
    else:
        response.headers["Cache-Control"] = PAGE_CACHE_CONTROL
    return response


# added last so it sits outermost and a 429 still carries CORS headers,
# otherwise a cross-origin caller sees a network error instead of the status
_origins = _allowed_origins()
if _origins:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=_origins,
        allow_methods=["GET", "POST"],
        allow_headers=["Content-Type"],
        # no cookies or auth headers are involved, and credentialed requests
        # would turn any allowlist mistake into a real one
        allow_credentials=False,
        max_age=600,
    )


def _client(request: Request) -> httpx.AsyncClient:
    return request.app.state.client


async def _wrap_upstream(coro):
    try:
        return await coro
    except httpx.HTTPError as exc:
        raise HTTPException(status_code=502, detail=f"upstream error: {exc}") from exc


@app.get("/api/lookup")
async def lookup(request: Request, q: str = Query(min_length=1)):
    """Check a username against both platforms. 404 when found on neither."""
    client = _client(request)
    q = q.strip()
    cc = await _wrap_upstream(chesscom.get_profile(client, q))
    li = await _wrap_upstream(lichess.get_user(client, q))

    # "Magnus Carlsen" is not a valid username anywhere; retry a full name
    # with the illegal characters stripped ("magnuscarlsen")
    normalized = re.sub(r"[^A-Za-z0-9_-]", "", q)
    if cc is None and li is None and normalized and normalized.lower() != q.lower():
        q = normalized
        cc = await _wrap_upstream(chesscom.get_profile(client, q))
        li = await _wrap_upstream(lichess.get_user(client, q))

    # an all-digit query is treated as a FIDE ID as well
    fd = await _wrap_upstream(fide.get_profile(client, q)) if q.isdigit() else None

    if cc is None and li is None and fd is None:
        raise HTTPException(status_code=404, detail="player not found")

    result = {"query": q, "chesscom": {"found": False}, "lichess": {"found": False},
              "fide": {"found": False, "searchable": q.isdigit()}}
    if fd is not None:
        result["fide"] = {"found": True, **fd, "history": None}
    if cc is not None:
        stats = await _wrap_upstream(chesscom.get_stats(client, q))
        result["chesscom"] = {
            "found": True,
            "username": cc.get("username"),
            "name": cc.get("name"),
            "title": cc.get("title"),
            "country": cc.get("country", "").rsplit("/", 1)[-1] or None,
            "ratings": chesscom.ratings_from_stats(stats),
            "url": cc.get("url"),
        }
    if li is not None:
        result["lichess"] = {
            "found": True,
            "username": li.get("username"),
            "title": li.get("title"),
            "games_total": li.get("count", {}).get("rated"),
            "ratings": lichess.ratings_from_user(li),
            "url": li.get("url"),
        }
    return result


@app.get("/api/profile")
async def profile(request: Request, chesscom_user: str | None = Query(default=None, alias="chesscom"),
                  lichess_user: str | None = Query(default=None, alias="lichess"),
                  fide_id: str | None = Query(default=None, alias="fide")):
    if not chesscom_user and not lichess_user and not fide_id:
        raise HTTPException(status_code=422, detail="pass at least one of ?chesscom=, ?lichess= or ?fide=")
    client = _client(request)
    out: dict = {}

    if fide_id:
        fd = await _wrap_upstream(fide.get_profile(client, fide_id))
        if fd is None:
            out["fide"] = {"found": False}
        else:
            out["fide"] = {"found": True, **{k: v for k, v in fd.items() if k != "history"}}

    if chesscom_user:
        cc = await _wrap_upstream(chesscom.get_profile(client, chesscom_user))
        if cc is None:
            out["chesscom"] = {"found": False}
        else:
            stats = await _wrap_upstream(chesscom.get_stats(client, chesscom_user))
            out["chesscom"] = {
                "found": True,
                "username": cc.get("username"),
                "name": cc.get("name"),
                "title": cc.get("title"),
                "country": cc.get("country", "").rsplit("/", 1)[-1] or None,
                "joined": cc.get("joined"),
                "ratings": chesscom.ratings_from_stats(stats),
                "url": cc.get("url"),
            }
    if lichess_user:
        li = await _wrap_upstream(lichess.get_user(client, lichess_user))
        if li is None:
            out["lichess"] = {"found": False}
        else:
            out["lichess"] = {
                "found": True,
                "username": li.get("username"),
                "title": li.get("title"),
                "created_at": li.get("createdAt"),
                "games_total": li.get("count", {}).get("rated"),
                "ratings": lichess.ratings_from_user(li),
                "url": li.get("url"),
            }

    if not any(v.get("found") for v in out.values()):
        raise HTTPException(status_code=404, detail="player not found")
    return out


TIME_CLASSES = {"bullet", "blitz", "rapid", "classical", "daily"}
DEFAULT_TC = "blitz,rapid,classical"


def _parse_tc(tc: str) -> set[str]:
    classes = {t.strip().lower() for t in tc.split(",") if t.strip()}
    unknown = classes - TIME_CLASSES
    if unknown:
        raise HTTPException(status_code=422,
                            detail=f"unknown time class: {', '.join(sorted(unknown))} (valid: {', '.join(sorted(TIME_CLASSES))})")
    if not classes:
        raise HTTPException(status_code=422, detail="tc must name at least one time class")
    return classes


def _window_start(months: int) -> datetime:
    """First day (UTC) of the calendar month `months - 1` months back. Both
    platforms window from this same boundary: chess.com by archive month,
    lichess via ?since=, so the merged dataset covers one time range."""
    now = datetime.now(timezone.utc)
    total = now.year * 12 + now.month - months
    return datetime(total // 12, total % 12 + 1, 1, tzinfo=timezone.utc)


async def _fetch_games(request: Request, chesscom_user: str | None, lichess_user: str | None,
                       months: int, max_games: int, time_classes: set[str] | None = None):
    """Merged game list plus per-platform coverage of the analysed window.

    `time_classes` filters both platforms identically (None = keep all).
    lichess additionally filters server-side so its count cap spends on
    relevant games; chess.com archives arrive whole and are filtered here.
    """
    if not chesscom_user and not lichess_user:
        raise HTTPException(status_code=422, detail="pass at least one of ?chesscom= or ?lichess=")
    client = _client(request)
    since = _window_start(months)
    games, li_truncated, found = [], False, set()

    if chesscom_user:
        if await _wrap_upstream(chesscom.get_profile(client, chesscom_user)) is not None:
            found.add("chesscom")
            games += await _wrap_upstream(chesscom.get_games(client, chesscom_user, since=since))
    if lichess_user:
        if await _wrap_upstream(lichess.get_user(client, lichess_user)) is not None:
            found.add("lichess")
            li_games, li_truncated = await _wrap_upstream(
                lichess.get_games(client, lichess_user, max_games=max_games,
                                  since=since, time_classes=time_classes))
            games += li_games

    if not found:
        raise HTTPException(status_code=404, detail="player not found")

    if time_classes is not None:
        games = [g for g in games if g.time_class in time_classes]

    coverage = {}
    for platform in found:
        stamps = [g.end_time for g in games if g.platform == platform and g.end_time]
        coverage[platform] = {
            "games": sum(g.platform == platform for g in games),
            "from": datetime.fromtimestamp(min(stamps), timezone.utc).date().isoformat() if stamps else None,
            "to": datetime.fromtimestamp(max(stamps), timezone.utc).date().isoformat() if stamps else None,
            "truncated": platform == "lichess" and li_truncated,
        }
    return games, coverage


@app.get("/api/openings")
async def openings(request: Request, chesscom_user: str | None = Query(default=None, alias="chesscom"),
                   lichess_user: str | None = Query(default=None, alias="lichess"),
                   months: int = Query(default=6, ge=1, le=24),
                   max_games: int = Query(default=300, ge=10, le=1000),
                   tc: str = Query(default=DEFAULT_TC)):
    classes = _parse_tc(tc)
    games, coverage = await _fetch_games(request, chesscom_user, lichess_user, months, max_games,
                                         time_classes=classes)
    tables = analysis.opening_tables(games)
    return {
        "games_analysed": len(games),
        "white": tables["white"],
        "black": tables["black"],
        "prep_target": analysis.prep_target(tables),
        "coverage": coverage,
        "params": {"min_games": analysis.MIN_GAMES, "threshold_pts": analysis.THRESHOLD_PTS,
                   "months": months, "max_games": max_games, "tc": sorted(classes)},
    }


@app.get("/api/movetree")
async def movetree(request: Request, chesscom_user: str | None = Query(default=None, alias="chesscom"),
                   lichess_user: str | None = Query(default=None, alias="lichess"),
                   months: int = Query(default=6, ge=1, le=24),
                   max_games: int = Query(default=300, ge=10, le=1000),
                   depth: int = Query(default=analysis.TREE_MAX_PLIES, ge=2, le=30),
                   min_games: int = Query(default=analysis.TREE_MIN_GAMES, ge=1, le=50),
                   tc: str = Query(default=DEFAULT_TC)):
    """Per-colour move tree of the player's repertoire, W/D/L at every node."""
    classes = _parse_tc(tc)
    games, coverage = await _fetch_games(request, chesscom_user, lichess_user, months, max_games,
                                         time_classes=classes)
    tree = analysis.move_tree(games, max_plies=depth, min_node_games=min_games)
    return {
        "games_analysed": len(games),
        "white": tree["white"],
        "black": tree["black"],
        "coverage": coverage,
        "params": {"depth_plies": depth, "min_node_games": min_games,
                   "months": months, "max_games": max_games, "tc": sorted(classes)},
    }


@app.get("/api/performance")
async def performance(request: Request, chesscom_user: str | None = Query(default=None, alias="chesscom"),
                      lichess_user: str | None = Query(default=None, alias="lichess"),
                      fide_id: str | None = Query(default=None, alias="fide"),
                      months: int = Query(default=6, ge=1, le=24),
                      max_games: int = Query(default=300, ge=10, le=1000),
                      tc: str = Query(default=DEFAULT_TC)):
    # validated before any upstream call so a typo costs nobody a request
    classes = _parse_tc(tc)
    params = {"months": months, "max_games": max_games, "tc": sorted(classes)}
    fide_stats = None
    if fide_id:
        fide_stats = await _wrap_upstream(fide.get_stats(_client(request), fide_id))

    if not chesscom_user and not lichess_user:
        if fide_stats is None:
            raise HTTPException(status_code=404, detail="player not found")
        return {"games_analysed": 0, "fide_stats": fide_stats, "params": params}

    games, _ = await _fetch_games(request, chesscom_user, lichess_user, months, max_games,
                                  time_classes=classes)
    return {
        "games_analysed": len(games),
        "totals": analysis.colour_totals(games),
        "colour_split": analysis.colour_split(games),
        "loss_terminations": analysis.loss_terminations(games),
        "vs_higher_rated": analysis.vs_higher_rated(games),
        "fide_stats": fide_stats,
        "params": params,
    }


@app.get("/api/ratings")
async def ratings(request: Request, chesscom_user: str | None = Query(default=None, alias="chesscom"),
                  lichess_user: str | None = Query(default=None, alias="lichess"),
                  fide_id: str | None = Query(default=None, alias="fide"),
                  months: int = Query(default=6, ge=1, le=24)):
    if not chesscom_user and not lichess_user and not fide_id:
        raise HTTPException(status_code=422, detail="pass at least one of ?chesscom=, ?lichess= or ?fide=")
    client = _client(request)
    out: dict = {}
    found_any = False

    if fide_id:
        fd = await _wrap_upstream(fide.get_profile(client, fide_id))
        if fd is not None:
            found_any = True
            out["fide"] = fd["history"]

    if lichess_user:
        history = await _wrap_upstream(lichess.get_rating_history(client, lichess_user))
        if history is not None:
            found_any = True
            series = {}
            for perf in history:
                name = perf.get("name", "").lower()
                if name in ("bullet", "blitz", "rapid", "classical") and perf.get("points"):
                    # lichess months are 0-indexed
                    series[name] = [
                        {"date": f"{y}-{m + 1:02d}-{d:02d}", "rating": r}
                        for y, m, d, r in perf["points"]
                    ]
            out["lichess"] = series

    if chesscom_user:
        stats = await _wrap_upstream(chesscom.get_stats(client, chesscom_user))
        if stats is not None:
            found_any = True
            out["chesscom_current"] = chesscom.ratings_from_stats(stats)
            # chess.com has no rating-history endpoint; reconstruct one from
            # the player's post-game ratings in the monthly archives
            games = await _wrap_upstream(chesscom.get_games(client, chesscom_user, since=_window_start(months)))
            out["chesscom"] = analysis.rating_history_from_games(games)

    if not found_any:
        raise HTTPException(status_code=404, detail="player not found")
    return out


@app.get("/api/recent")
async def recent(request: Request, chesscom_user: str | None = Query(default=None, alias="chesscom"),
                 lichess_user: str | None = Query(default=None, alias="lichess"),
                 n: int = Query(default=20, ge=1, le=100),
                 months: int = Query(default=3, ge=1, le=24),
                 max_games: int = Query(default=100, ge=10, le=1000),
                 tc: str = Query(default=DEFAULT_TC)):
    classes = _parse_tc(tc)
    games, _ = await _fetch_games(request, chesscom_user, lichess_user, months, max_games,
                                  time_classes=classes)
    return {
        **analysis.recent_form(games, n=n),
        "params": {"months": months, "max_games": max_games, "tc": sorted(classes)},
    }


@app.post("/api/refresh")
async def refresh(chesscom_user: str | None = Query(default=None, alias="chesscom"),
                  lichess_user: str | None = Query(default=None, alias="lichess")):
    """Drop cached live data for a player (closed months stay cached)."""
    deleted = 0
    if chesscom_user:
        u = chesscom_user.lower()
        deleted += cache.delete_prefix(f"chesscom:profile:{u}")
        deleted += cache.delete_prefix(f"chesscom:stats:{u}")
    if lichess_user:
        u = lichess_user.lower()
        deleted += cache.delete_prefix(f"lichess:user:{u}")
        deleted += cache.delete_prefix(f"lichess:games:{u}")
        deleted += cache.delete_prefix(f"lichess:history:{u}")
    return {"deleted_keys": deleted}


@app.get("/healthz")
async def healthz():
    """Liveness probe. No upstream calls, so a throttled API never fails a deploy."""
    return {"status": "ok"}


# static frontend last, so /api/* and /healthz win. Only static/ is exposed,
# which keeps the source tree and the cache database off the public URL.
app.mount("/", StaticFiles(directory=STATIC, html=True), name="static")
