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


# --- cache headers -----------------------------------------------------
#
# Outbound bytes are the only metered resource on the host, so a repeat visitor
# should re-fetch as little as possible.

def test_images_are_cached_for_a_week():
    r = client.get("/assets/logov2_prepmate.jpeg")
    assert r.headers["Cache-Control"] == "public, max-age=604800"


def test_pages_revalidate_instead_of_being_held():
    """HTML changes on every deploy, so a held copy would outlive the fix."""
    for path in ("/", "/index.html", "/profile.html"):
        assert client.get(path).headers["Cache-Control"] == "public, no-cache", path


def test_a_page_recheck_costs_no_body():
    """StaticFiles sends an ETag, so revalidating is a 304 rather than a download."""
    first = client.get("/index.html")
    again = client.get("/index.html", headers={"If-None-Match": first.headers["etag"]})
    assert again.status_code == 304
    assert again.content == b""


def test_the_api_is_never_cached():
    assert "cache-control" not in client.get("/api/openings").headers


def test_a_missing_file_is_not_cached():
    assert "cache-control" not in client.get("/nope.html").headers
