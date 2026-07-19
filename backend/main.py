"""PrepMate backend.

Run from the project root:
    uvicorn backend.main:app --reload

Also serves the static frontend (lookup.html / profile.html) at /.
"""

import re
from contextlib import asynccontextmanager
from pathlib import Path

import httpx
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from . import analysis, cache
from .adapters import chesscom, fide, lichess

USER_AGENT = "PrepMate/0.1 (personal chess prep tool; contact: puridanish5@gmail.com)"
ROOT = Path(__file__).parent.parent


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.client = httpx.AsyncClient(timeout=30, headers={"User-Agent": USER_AGENT}, follow_redirects=True)
    yield
    await app.state.client.aclose()


app = FastAPI(title="PrepMate", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


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


async def _fetch_games(request: Request, chesscom_user: str | None, lichess_user: str | None,
                       months: int, max_games: int):
    if not chesscom_user and not lichess_user:
        raise HTTPException(status_code=422, detail="pass at least one of ?chesscom= or ?lichess=")
    client = _client(request)
    games, found_any = [], False

    if chesscom_user:
        if await _wrap_upstream(chesscom.get_profile(client, chesscom_user)) is not None:
            found_any = True
            games += await _wrap_upstream(chesscom.get_games(client, chesscom_user, months=months))
    if lichess_user:
        if await _wrap_upstream(lichess.get_user(client, lichess_user)) is not None:
            found_any = True
            games += await _wrap_upstream(lichess.get_games(client, lichess_user, max_games=max_games))

    if not found_any:
        raise HTTPException(status_code=404, detail="player not found")
    return games


@app.get("/api/openings")
async def openings(request: Request, chesscom_user: str | None = Query(default=None, alias="chesscom"),
                   lichess_user: str | None = Query(default=None, alias="lichess"),
                   months: int = Query(default=6, ge=1, le=24),
                   max_games: int = Query(default=300, ge=10, le=1000)):
    games = await _fetch_games(request, chesscom_user, lichess_user, months, max_games)
    tables = analysis.opening_tables(games)
    return {
        "games_analysed": len(games),
        "white": tables["white"],
        "black": tables["black"],
        "prep_target": analysis.prep_target(tables),
        "readout": {
            "loss_terminations": analysis.loss_terminations(games),
            "vs_higher_rated": analysis.vs_higher_rated(games),
        },
        "params": {"min_games": analysis.MIN_GAMES, "threshold_pts": analysis.THRESHOLD_PTS,
                   "chesscom_months": months, "lichess_max_games": max_games},
    }


@app.get("/api/performance")
async def performance(request: Request, chesscom_user: str | None = Query(default=None, alias="chesscom"),
                      lichess_user: str | None = Query(default=None, alias="lichess"),
                      fide_id: str | None = Query(default=None, alias="fide"),
                      months: int = Query(default=6, ge=1, le=24),
                      max_games: int = Query(default=300, ge=10, le=1000)):
    fide_stats = None
    if fide_id:
        fide_stats = await _wrap_upstream(fide.get_stats(_client(request), fide_id))

    if not chesscom_user and not lichess_user:
        if fide_stats is None:
            raise HTTPException(status_code=404, detail="player not found")
        return {"games_analysed": 0, "fide_stats": fide_stats}

    games = await _fetch_games(request, chesscom_user, lichess_user, months, max_games)
    return {
        "games_analysed": len(games),
        "totals": analysis.colour_totals(games),
        "colour_split": analysis.colour_split(games),
        "loss_terminations": analysis.loss_terminations(games),
        "vs_higher_rated": analysis.vs_higher_rated(games),
        "fide_stats": fide_stats,
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
            games = await _wrap_upstream(chesscom.get_games(client, chesscom_user, months=months))
            out["chesscom"] = analysis.rating_history_from_games(games)

    if not found_any:
        raise HTTPException(status_code=404, detail="player not found")
    return out


@app.get("/api/recent")
async def recent(request: Request, chesscom_user: str | None = Query(default=None, alias="chesscom"),
                 lichess_user: str | None = Query(default=None, alias="lichess"),
                 n: int = Query(default=20, ge=1, le=100),
                 months: int = Query(default=3, ge=1, le=24),
                 max_games: int = Query(default=100, ge=10, le=1000)):
    games = await _fetch_games(request, chesscom_user, lichess_user, months, max_games)
    return analysis.recent_form(games, n=n)


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


# static frontend last, so /api/* wins
app.mount("/", StaticFiles(directory=ROOT, html=True), name="static")
