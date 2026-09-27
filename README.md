# PrepMate

PrepMate is a chess tournament-preparation app by Danish Puri. It turns an
opponent's public Chess.com and Lichess games into a scouting report with opening
statistics, recurring move sequences, and recent performance.

**Live at [prepmate-chess.fly.dev](https://prepmate-chess.fly.dev)**

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

## Deep pattern detection

The opening tables group games by ECO code, which labels a move order and stops
around move ten. That misses anything structural. Two games can reach the same
pawn skeleton through different openings and get filed apart, and a weakness
that only shows up in the middlegame is invisible.

So I built a second engine in `backend/patterns` that works on positions instead
of labels. It replays the games, encodes every position where the opponent had a
real choice with a small convolutional network, clusters the vectors, and scores
each cluster by how the opponent actually did from there. Clusters are
structures rather than openings, so transpositions collapse into one row and the
middlegame is finally in scope.

The network is pretrained on other players' games and then frozen. It never sees
the opponent. A few hundred games is a few hundred samples, and anything with
real capacity would just memorise them, so the only per-opponent work is
clustering and counting. The reasoning and the measured costs are in
[docs/patterns.md](docs/patterns.md).

It runs from the command line and not the web app, because I didn't want to add
Torch to a deployed image that otherwise has three dependencies.

```sh
python -m pip install -r requirements-patterns.txt
python -m train.pretrain fetch --users train/users.txt --per-user 200 --out data/corpus
python -m train.pretrain train --corpus data/corpus --out models/pos-v1.pt --device mps
python -m backend.patterns.scout --lichess <username> --model models/pos-v1.pt
```

Without `--model` it falls back to handcrafted features, so the pipeline still
runs for testing, but the trained encoder is the real thing.

## Stack

Python, FastAPI, httpx, SQLite, and plain HTML/CSS/JavaScript for the app.
PyTorch, python-chess, and NumPy for the optional pattern engine. SQLite caches
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
API endpoints, filtering, rate limits, static-page behavior, and the pattern
pipeline. The Torch-specific tests run when `requirements-patterns.txt` is installed
and are skipped otherwise.

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
| `CLIENT_IP_HEADER` | Header holding the real client IP behind a proxy, such as `Fly-Client-IP`. Unset by default. |

The `/healthz` endpoint is exempt from request limits. Keep generated caches,
credentials, and local environment files out of version control.

## Deployment

The Dockerfile serves the frontend and API together. Railway uses `railway.json`,
its supplied `PORT`, and `/healthz` for deployment health checks. For a persistent
cache, mount a volume at `/data` and set `CACHE_DB=/data/cache.db` as shown in
`.env.railway.example`. The example contains configuration only, not credentials.

The live site runs on Fly.io from `fly.toml`, with one always-on machine, a 1 GB
volume at `/data` for the cache, and a `/healthz` check every 30 seconds.

## Project layout

- `backend/`: API, public-data adapters, analysis, cache, and request limits.
- `backend/patterns/`: position encoder, clustering, and the command-line scout.
- `train/`: encoder pretraining on a public corpus.
- `docs/`: design notes for the pattern engine.
- `static/`: browser pages and image assets.
- `tests/`: offline regression tests.

## Data sources

Chess.com public API, Lichess public API, and public FIDE profile pages. FIDE
profiles are parsed from HTML, so changes to its pages may require adapter updates.
