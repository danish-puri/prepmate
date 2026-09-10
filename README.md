# PrepMate

PrepMate is a chess tournament-preparation app by Danish Puri. It turns an
opponent's public Chess.com and Lichess games into a scouting report with opening
statistics, recurring move sequences, and recent performance.

## Features

- Look up Chess.com and Lichess usernames, with FIDE profile lookup by ID.
- Compare opening frequency and results as White and Black.
- Explore move trees and filter game reports by time control.
- Identify a potential preparation target when enough games support it.
- See sample sizes and available game coverage alongside the results.
- Use the interface on a phone or desktop.

FIDE supplies profile and rating information, not game moves. Platform accounts
remain separate unless the user selects them together. Statistics describe the
available games and do not guarantee an opponent's future play.

## Stack

Python, FastAPI, httpx, SQLite, and plain HTML/CSS/JavaScript. SQLite caches
upstream responses, and request limits help control external API traffic.

## Run locally

Use Python 3.14, matching the Docker image.

```sh
# Create and activate a local environment.
python3 -m venv .venv
source .venv/bin/activate

# Install the application and start its API and frontend.
python -m pip install -r requirements.txt
uvicorn backend.main:app --host 127.0.0.1 --port 8000
```

Open http://localhost:8000. Player lookups require access to the public chess APIs.

## Tests

```sh
# Install test dependencies and run the offline suite.
python -m pip install -r requirements-dev.txt
python -m pytest -q
```

Tests use stubbed upstream responses and cover game analysis, adapters, caching,
API endpoints, filtering, rate limits, and static-page behavior.

## Configuration

Set these environment variables before starting the server:

| Variable | Purpose |
| --- | --- |
| `CACHE_DB` | SQLite cache location. Defaults to `backend/cache.db`. |
| `ALLOWED_ORIGINS` | Comma-separated CORS origins. Set empty for a same-origin deployment. |
| `RATE_LIMIT_PER_MINUTE` | API request allowance per IP. Defaults to `60`. |
| `RATE_LIMIT_BURST` | API burst allowance. Defaults to `20`. |
| `STATIC_RATE_LIMIT_PER_MINUTE` | Frontend request allowance per IP. Defaults to `60`. |
| `STATIC_RATE_LIMIT_BURST` | Frontend burst allowance. Defaults to `30`. |

The `/healthz` endpoint is exempt from request limits. Keep generated caches,
credentials, and local environment files out of version control.

## Deployment

The Dockerfile serves the frontend and API together. Railway uses `railway.json`,
its supplied `PORT`, and `/healthz` for deployment health checks. For a persistent
cache, mount a volume at `/data` and set `CACHE_DB=/data/cache.db` as shown in
`.env.railway.example`. The example contains configuration only, not credentials.

## Project layout

- `backend/`: API, public-data adapters, analysis, cache, and request limits.
- `static/`: browser pages and image assets.
- `tests/`: offline regression tests.

## Data sources

Chess.com public API, Lichess public API, and public FIDE profile pages. FIDE
profiles are parsed from HTML, so changes to its pages may require adapter updates.
