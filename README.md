# PrepMate

by Danish Puri

I built PrepMate to prepare for chess tournaments. I type an opponent's name and get a scouting report built from their public games.

## What it does

- Matches a name to chess.com, lichess, and FIDE profiles
- Pulls their recent rated games from the public APIs
- Computes win, draw, and loss rates as White and as Black
- Shows the openings they play most and how they score in each
- Filters the openings and move tree by time control. Blitz, rapid, and classical are on by default, and I can toggle any of them off to see just the serious repertoire. Bullet and daily games stay out since that's rarely how anyone plays over the board
- Picks one prep target, the weakest opening they still play often
- Charts rating trajectory, loss terminations, and recent form

## Run it

```
uv venv
uv pip install -r requirements.txt
.venv/bin/uvicorn backend.main:app --port 8000
```

Then open http://127.0.0.1:8000 in a browser. Plain venv and pip work too.

## Test it

```
uv pip install -r requirements-dev.txt
.venv/bin/pytest
```

The suite stubs both platform APIs, so it runs offline and never touches a real
account. It covers the stats in `backend/analysis.py`, the cache, the FIDE
scraper, every endpoint, and the rate limit and CORS rules.

## Settings

All optional, read from the environment at startup.

| Variable | Default | What it does |
| --- | --- | --- |
| `ALLOWED_ORIGINS` | `http://localhost:8000,http://127.0.0.1:8000` | Comma separated list of origins allowed to call the API from a browser. The frontend is served by this same app, so nothing cross-origin is needed in normal use. Set it to an empty string to send no CORS headers at all. |
| `RATE_LIMIT_PER_MINUTE` | `60` | Sustained requests per minute per IP on `/api/*`. Set to `0` to turn limiting off. |
| `RATE_LIMIT_BURST` | `20` | How many requests may arrive back to back before the sustained rate kicks in. |

Requests over the limit get a 429 and a `Retry-After` header. `/healthz` and the
static pages are never limited. Every `/api/*` call fans out to chess.com and
lichess under my User-Agent, so the point of the limit is to keep one impatient
client from making trouble for them.

Behind a proxy, start uvicorn with `--proxy-headers --forwarded-allow-ips='*'`
so the limiter keys on the real caller instead of the proxy. The Railway config
already does this.

## How it works

FastAPI backend with static HTML pages on top. Each platform has its own adapter in `backend/adapters`, and responses are cached locally in SQLite so repeat lookups are fast and the upstream APIs stay happy.

## Data sources

api.chess.com, lichess.org/api, and ratings.fide.com. All public data, fetched politely. FIDE has no official API, so that adapter scrapes the profile page and can break if the page changes. Stats come straight from real game data, nothing is invented.
