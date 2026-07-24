"""Rate limiting and CORS.

The limiter lives on app.state and is built in the lifespan, so every
TestClient context below starts with a full bucket.
"""

import importlib

import pytest
from fastapi.testclient import TestClient

from backend import main
from backend.limiter import RateLimiter


@pytest.fixture
def app_with(monkeypatch):
    """Rebuild the app under given env vars, since both features read env at import."""
    def build(**env):
        for key, value in env.items():
            monkeypatch.setenv(key, value)
        return importlib.reload(main)
    yield build
    # undo the env first: this finalizer runs before monkeypatch's own, so
    # reloading here would otherwise bake the test settings into the module
    # that every later test file shares
    monkeypatch.undo()
    importlib.reload(main)


# --- the bucket itself -------------------------------------------------

def test_burst_then_refusal():
    limiter = RateLimiter(per_minute=60, burst=3)
    assert [limiter.take("ip")[0] for _ in range(4)] == [True, True, True, False]


def test_remaining_counts_down():
    limiter = RateLimiter(per_minute=60, burst=3)
    assert [limiter.take("ip")[1] for _ in range(3)] == [2, 1, 0]


def test_buckets_are_per_key():
    limiter = RateLimiter(per_minute=60, burst=1)
    assert limiter.take("a")[0] and limiter.take("b")[0]
    assert not limiter.take("a")[0]


def test_refills_over_time(monkeypatch):
    clock = {"t": 1000.0}
    monkeypatch.setattr("backend.limiter.time.monotonic", lambda: clock["t"])
    limiter = RateLimiter(per_minute=60, burst=2)  # one token per second
    assert limiter.take("ip")[0] and limiter.take("ip")[0]

    allowed, _, retry_after = limiter.take("ip")
    assert not allowed and retry_after == pytest.approx(1.0)

    clock["t"] += 1.0
    assert limiter.take("ip")[0]


def test_refill_stops_at_burst(monkeypatch):
    clock = {"t": 1000.0}
    monkeypatch.setattr("backend.limiter.time.monotonic", lambda: clock["t"])
    limiter = RateLimiter(per_minute=60, burst=2)
    limiter.take("ip")
    clock["t"] += 3600  # idle for an hour
    assert [limiter.take("ip")[0] for _ in range(3)] == [True, True, False]


def test_prune_drops_only_refilled_buckets(monkeypatch):
    clock = {"t": 1000.0}
    monkeypatch.setattr("backend.limiter.time.monotonic", lambda: clock["t"])
    monkeypatch.setattr("backend.limiter.PRUNE_THRESHOLD", 3)
    limiter = RateLimiter(per_minute=60, burst=2)
    limiter.take("old")
    clock["t"] += 60
    limiter.take("recent")
    limiter.take("also-recent")
    limiter.take("triggers-prune")
    assert "old" not in limiter._buckets
    assert {"recent", "also-recent", "triggers-prune"} <= set(limiter._buckets)


def test_rejects_nonsense_settings():
    with pytest.raises(ValueError):
        RateLimiter(per_minute=0, burst=10)
    with pytest.raises(ValueError):
        RateLimiter(per_minute=60, burst=0)


# --- wired into the app ------------------------------------------------

def test_api_returns_429_past_the_burst(app_with):
    mod = app_with(RATE_LIMIT_PER_MINUTE="60", RATE_LIMIT_BURST="2")
    with TestClient(mod.app) as client:
        assert client.get("/api/openings").status_code == 422  # spends a token
        assert client.get("/api/openings").status_code == 422
        response = client.get("/api/openings")
    assert response.status_code == 429
    assert int(response.headers["Retry-After"]) >= 1
    assert response.headers["X-RateLimit-Remaining"] == "0"
    assert response.json()["detail"]


def test_remaining_header_counts_down(app_with):
    mod = app_with(RATE_LIMIT_PER_MINUTE="60", RATE_LIMIT_BURST="5")
    with TestClient(mod.app) as client:
        first = client.get("/api/openings").headers["X-RateLimit-Remaining"]
        second = client.get("/api/openings").headers["X-RateLimit-Remaining"]
    assert (first, second) == ("4", "3")


def test_healthz_and_static_are_exempt(app_with):
    mod = app_with(RATE_LIMIT_PER_MINUTE="60", RATE_LIMIT_BURST="1")
    with TestClient(mod.app) as client:
        assert client.get("/api/openings").status_code == 422  # empties the bucket
        for _ in range(5):
            assert client.get("/healthz").status_code == 200
            assert client.get("/").status_code == 200


def test_limiting_can_be_switched_off(app_with):
    mod = app_with(RATE_LIMIT_PER_MINUTE="0")
    with TestClient(mod.app) as client:
        codes = {client.get("/api/openings").status_code for _ in range(20)}
    assert codes == {422}


# --- CORS --------------------------------------------------------------

def test_unlisted_origin_gets_no_cors_header(app_with):
    mod = app_with(ALLOWED_ORIGINS="https://prepmate.example")
    with TestClient(mod.app) as client:
        response = client.get("/api/openings", headers={"Origin": "https://evil.example"})
    assert "access-control-allow-origin" not in response.headers


def test_listed_origin_is_echoed_back(app_with):
    mod = app_with(ALLOWED_ORIGINS="https://prepmate.example")
    with TestClient(mod.app) as client:
        response = client.get("/api/openings", headers={"Origin": "https://prepmate.example"})
    assert response.headers["access-control-allow-origin"] == "https://prepmate.example"
    assert "access-control-allow-credentials" not in response.headers


def test_preflight_allows_only_get_and_post(app_with):
    mod = app_with(ALLOWED_ORIGINS="https://prepmate.example")
    with TestClient(mod.app) as client:
        preflight = {"Origin": "https://prepmate.example"}
        ok = client.options("/api/openings", headers={**preflight,
                                                      "Access-Control-Request-Method": "GET"})
        denied = client.options("/api/openings", headers={**preflight,
                                                          "Access-Control-Request-Method": "DELETE"})
    assert ok.headers["access-control-allow-methods"] == "GET, POST"
    assert denied.status_code == 400  # starlette names the disallowed method
    assert "DELETE" not in denied.headers.get("access-control-allow-methods", "")


def test_preflight_does_not_spend_the_budget(app_with):
    mod = app_with(RATE_LIMIT_PER_MINUTE="60", RATE_LIMIT_BURST="5",
                   ALLOWED_ORIGINS="https://prepmate.example")
    with TestClient(mod.app) as client:
        for _ in range(10):
            client.options("/api/openings", headers={"Origin": "https://prepmate.example",
                                                     "Access-Control-Request-Method": "GET"})
        response = client.get("/api/openings")
    assert response.headers["X-RateLimit-Remaining"] == "4"


def test_429_still_carries_cors_headers(app_with):
    """Without them a cross-origin caller sees a network error, not the 429."""
    mod = app_with(RATE_LIMIT_PER_MINUTE="60", RATE_LIMIT_BURST="1",
                   ALLOWED_ORIGINS="https://prepmate.example")
    origin = {"Origin": "https://prepmate.example"}
    with TestClient(mod.app) as client:
        client.get("/api/openings", headers=origin)
        response = client.get("/api/openings", headers=origin)
    assert response.status_code == 429
    assert response.headers["access-control-allow-origin"] == "https://prepmate.example"


def test_empty_allowlist_sends_no_cors_at_all(app_with):
    mod = app_with(ALLOWED_ORIGINS="")
    with TestClient(mod.app) as client:
        response = client.get("/api/openings", headers={"Origin": "http://localhost:8000"})
    assert "access-control-allow-origin" not in response.headers


def test_localhost_is_the_default_allowlist(app_with):
    mod = app_with()
    with TestClient(mod.app) as client:
        response = client.get("/api/openings", headers={"Origin": "http://localhost:8000"})
    assert response.headers["access-control-allow-origin"] == "http://localhost:8000"
