"""The shared calendar window and tc parsing in backend.main."""

from datetime import datetime, timezone
from unittest.mock import patch

import pytest
from fastapi import HTTPException

from backend.main import _parse_tc, _window_start


def _frozen(year, month, day=19):
    p = patch("backend.main.datetime", wraps=datetime)
    mocked = p.start()
    mocked.now.return_value = datetime(year, month, day, tzinfo=timezone.utc)
    return p


def test_window_start_within_year():
    p = _frozen(2026, 7)
    try:
        assert _window_start(1) == datetime(2026, 7, 1, tzinfo=timezone.utc)
        assert _window_start(6) == datetime(2026, 2, 1, tzinfo=timezone.utc)
    finally:
        p.stop()


def test_window_start_crosses_year_boundary():
    p = _frozen(2026, 7)
    try:
        assert _window_start(7) == datetime(2026, 1, 1, tzinfo=timezone.utc)
        assert _window_start(8) == datetime(2025, 12, 1, tzinfo=timezone.utc)
        assert _window_start(24) == datetime(2024, 8, 1, tzinfo=timezone.utc)
    finally:
        p.stop()


def test_window_start_in_january():
    p = _frozen(2026, 1)
    try:
        assert _window_start(1) == datetime(2026, 1, 1, tzinfo=timezone.utc)
        assert _window_start(2) == datetime(2025, 12, 1, tzinfo=timezone.utc)
    finally:
        p.stop()


def test_parse_tc_normalizes():
    assert _parse_tc("blitz,rapid,classical") == {"blitz", "rapid", "classical"}
    assert _parse_tc(" RAPID , Classical ") == {"rapid", "classical"}
    assert _parse_tc("daily") == {"daily"}


@pytest.mark.parametrize("bad", ["foo", "", " , ", "blitz,foo"])
def test_parse_tc_rejects_unknown(bad):
    with pytest.raises(HTTPException) as exc:
        _parse_tc(bad)
    assert exc.value.status_code == 422
