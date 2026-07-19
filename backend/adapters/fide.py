"""FIDE adapter. No official API, so this scrapes ratings.fide.com — the
fragile piece, isolated here per DESIGN.md so a page redesign only breaks
this module.

Two sources, both from the public profile:
- profile page HTML: name, federation, title, birth year, current ratings,
  and the embedded monthly rating-history table (rating + games per TC)
- /a_data_stats.php: FIDE's own JSON stats endpoint (used by the profile's
  Statistics button) with W/D/L counts by colour and time control

FIDE publishes NO game moves; this is identity, strength, and aggregates.
"""

import re

import httpx

from .. import cache

BASE = "https://ratings.fide.com"
# the profile pages 403 on obviously non-browser agents; identify as a browser
# with the app appended
UA = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) PrepMate/0.1"

_MONTHS = {m: i + 1 for i, m in enumerate(
    ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"])}

_NAME_RE = re.compile(r"<title>\s*(.*?)\s+FIDE Profile\s*</title>", re.S)
_FED_RE = re.compile(r'profile-info-country[^>]*>.*?<img[^>]*>\s*([^<]+?)\s*<', re.S)
_BYEAR_RE = re.compile(r'profile-info-byear[^>]*>\s*(\d{4})')
_TITLE_RE = re.compile(r'FIDE title</h5>\s*<div class="profile-info-title[^>]*>\s*<p>([^<]+)</p>', re.S)
_RATING_RES = {
    "std": re.compile(r'profile-standart profile-game[^>]*>.*?<p>\s*([^<]+?)\s*</p>', re.S),
    "rapid": re.compile(r'profile-rapid profile-game[^>]*>.*?<p>\s*([^<]+?)\s*</p>', re.S),
    "blitz": re.compile(r'profile-blitz profile-game[^>]*>.*?<p>\s*([^<]+?)\s*</p>', re.S),
}
_HISTORY_ROW_RE = re.compile(
    r'<tr>\s*<td[^>]*>&nbsp;(\d{4})-([A-Z][a-z]{2})&nbsp;</td>(.*?)</tr>', re.S)
_TD_RE = re.compile(r'<td[^>]*>\s*(?:&nbsp;)?\s*([^<&\s][^<&]*?)?\s*(?:&nbsp;)?\s*</td>')


def _int(s):
    s = (s or "").strip()
    return int(s) if s.isdigit() else None


async def get_profile(client: httpx.AsyncClient, fide_id: str) -> dict | None:
    fide_id = fide_id.strip()
    if not fide_id.isdigit():
        return None
    key = f"fide:profile:{fide_id}"
    cached = cache.get(key, max_age=86400)
    if cached is not None:
        return cached or None

    r = await client.get(f"{BASE}/profile/{fide_id}", headers={"User-Agent": UA})
    r.raise_for_status()
    html = r.text
    if f'profile-info-id ">{fide_id}<' not in html.replace("\n", "").replace("\t", ""):
        # invalid IDs still return 200 with a generic page; the ID block is
        # the reliable presence signal
        if "profile-info-id" not in html:
            cache.put(key, {})
            return None

    name_m = _NAME_RE.search(html)
    if not name_m:
        cache.put(key, {})
        return None

    ratings = {}
    for tc, rx in _RATING_RES.items():
        m = rx.search(html)
        if m:
            v = _int(m.group(1))
            if v:
                ratings[tc] = v

    history: dict[str, list] = {"std": [], "rapid": [], "blitz": []}
    for year, mon, rest in _HISTORY_ROW_RE.findall(html):
        tds = _TD_RE.findall(rest)
        # layout: std rating, std games, rapid rating, rapid games, blitz rating, blitz games
        vals = [(_int(t) if t else None) for t in tds[:6]] + [None] * (6 - len(tds))
        date = f"{year}-{_MONTHS.get(mon, 1):02d}-01"
        for tc, idx in (("std", 0), ("rapid", 2), ("blitz", 4)):
            if vals[idx]:
                history[tc].append({"date": date, "rating": vals[idx], "games": vals[idx + 1] or 0})
    for tc in history:
        history[tc].reverse()  # page lists newest first; chart wants oldest first

    fed_m = _FED_RE.search(html)
    title_m = _TITLE_RE.search(html)
    byear_m = _BYEAR_RE.search(html)
    data = {
        "fide_id": fide_id,
        "name": name_m.group(1).strip(),
        "federation": fed_m.group(1).strip() if fed_m else None,
        "title": title_m.group(1).strip() if title_m else None,
        "birth_year": int(byear_m.group(1)) if byear_m else None,
        "ratings": ratings,
        "history": history,
        "url": f"{BASE}/profile/{fide_id}",
    }
    cache.put(key, data)
    return data


async def get_stats(client: httpx.AsyncClient, fide_id: str) -> dict | None:
    """W/D/L counts by colour and time control from FIDE's stats endpoint."""
    fide_id = fide_id.strip()
    if not fide_id.isdigit():
        return None
    key = f"fide:stats:{fide_id}"
    cached = cache.get(key, max_age=86400)
    if cached is not None:
        return cached or None

    r = await client.post(
        f"{BASE}/a_data_stats.php",
        params={"id1": fide_id, "id2": ""},
        headers={"User-Agent": UA, "X-Requested-With": "XMLHttpRequest",
                 "Referer": f"{BASE}/profile/{fide_id}"},
    )
    r.raise_for_status()
    try:
        raw = r.json()[0]
    except (ValueError, IndexError, KeyError):
        cache.put(key, {})
        return None

    def block(prefix: str) -> dict:
        out = {}
        for colour in ("white", "black"):
            n = _int(raw.get(f"{colour}_total{prefix}")) or 0
            w = _int(raw.get(f"{colour}_win_num{prefix}")) or 0
            d = _int(raw.get(f"{colour}_draw_num{prefix}")) or 0
            out[colour] = {"n": n, "wins": w, "draws": d, "losses": max(0, n - w - d)}
        n = out["white"]["n"] + out["black"]["n"]
        w = out["white"]["wins"] + out["black"]["wins"]
        d = out["white"]["draws"] + out["black"]["draws"]
        out["total"] = {"n": n, "wins": w, "draws": d, "losses": max(0, n - w - d)}
        return out

    data = {
        "all": block(""),
        "std": block("_std"),
        "rapid": block("_rpd"),
        "blitz": block("_blz"),
    }
    if data["all"]["total"]["n"] == 0:
        cache.put(key, {})
        return None
    cache.put(key, data)
    return data
