"""The health probe and what the static mount is allowed to serve.

The mount used to sit on the project root, which put the backend source and
the cache database on the public URL. These pin it to static/.
"""

from fastapi.testclient import TestClient

from backend import main

client = TestClient(main.app)


def test_healthz_is_ok():
    r = client.get("/healthz")
    assert r.status_code == 200
    assert r.json() == {"status": "ok"}


def test_root_serves_the_lookup_page():
    r = client.get("/")
    assert r.status_code == 200
    assert "PrepMate" in r.text


def test_frontend_pages_and_assets_are_reachable():
    for path in ("/index.html", "/profile.html", "/performance.html",
                 "/assets/logov2_prepmate.jpeg"):
        assert client.get(path).status_code == 200, path


def test_source_and_cache_are_not_exposed():
    for path in ("/backend/main.py", "/backend/cache.db", "/README.md",
                 "/railway.json", "/requirements.txt"):
        assert client.get(path).status_code == 404, path
