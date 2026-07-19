# PrepMate

by Danish Puri

I built PrepMate to prepare for chess tournaments. I type an opponent's name and get a scouting report built from their public games.

## What it does

- Matches a name to chess.com, lichess, and FIDE profiles
- Pulls their recent rated games from the public APIs
- Computes win, draw, and loss rates as White and as Black
- Shows the openings they play most and how they score in each
- Picks one prep target, the weakest opening they still play often
- Charts rating trajectory, loss terminations, and recent form

## Run it

```
uv venv
uv pip install -r requirements.txt
.venv/bin/uvicorn backend.main:app --port 8000
```

Then open http://127.0.0.1:8000 in a browser. Plain venv and pip work too.

## How it works

FastAPI backend with static HTML pages on top. Each platform has its own adapter in `backend/adapters`, and responses are cached locally in SQLite so repeat lookups are fast and the upstream APIs stay happy.

## Data sources

api.chess.com, lichess.org/api, and ratings.fide.com. All public data, fetched politely. FIDE has no official API, so that adapter scrapes the profile page and can break if the page changes. Stats come straight from real game data, nothing is invented.
