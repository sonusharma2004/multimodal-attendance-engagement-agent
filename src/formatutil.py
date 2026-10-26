"""Clock and duration formatting for educator-facing tables."""

from __future__ import annotations

import math
from datetime import datetime, timedelta, timezone

IST = timezone(timedelta(hours=5, minutes=30))


def now_ist() -> datetime:
    return datetime.now(IST)


def format_stamp(moment: datetime | None = None) -> str:
    moment = moment or now_ist()
    if moment.tzinfo is None:
        moment = moment.replace(tzinfo=IST)
    local = moment.astimezone(IST)
    hour = local.hour % 12 or 12
    ampm = "AM" if local.hour < 12 else "PM"
    return f"{local.day} {local.strftime('%b %Y')}, {hour}:{local.minute:02d} {ampm} IST"


def _finite(seconds: float | None) -> float | None:
    if seconds is None:
        return None
    try:
        value = float(seconds)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(value):
        return None
    return value


def format_clock(seconds: float | None) -> str:
    value = _finite(seconds)
    if value is None:
        return "—"
    total = int(round(max(0.0, value)))
    hours, rem = divmod(total, 3600)
    minutes, secs = divmod(rem, 60)
    if hours:
        return f"{hours}:{minutes:02d}:{secs:02d}"
    return f"{minutes}:{secs:02d}"


def format_duration(seconds: float | None) -> str:
    value = _finite(seconds)
    if value is None:
        return "—"
    total = int(round(max(0.0, value)))
    hours, rem = divmod(total, 3600)
    minutes, secs = divmod(rem, 60)
    if hours and minutes:
        return f"{hours}h {minutes}m"
    if hours:
        return f"{hours}h"
    if minutes and secs:
        return f"{minutes} min {secs} s"
    if minutes:
        return f"{minutes} min"
    return f"{secs} s"


def format_number(value: float | None, digits: int = 0) -> str:
    number = _finite(value)
    if number is None:
        return "—"
    return f"{number:.{digits}f}"
