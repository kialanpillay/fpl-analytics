"""Official FPL / Premier League CDN URL helpers.

Construct URLs from bootstrap ``photo`` and ``team_code``. Do not vendor images.
"""

from __future__ import annotations

from typing import Any

PHOTO_BASE = "https://resources.premierleague.com/premierleague/photos/players"
SHIRT_BASE = "https://fantasy.premierleague.com/dist/img/shirts/standard"
BADGE_BASE = "https://resources.premierleague.com/premierleague/badges"


def photo_code(photo: str | None) -> str | None:
    """Turn ``223340.jpg`` into ``223340``."""
    if not photo:
        return None
    stem = str(photo).strip()
    if not stem or stem.lower() in {"nan", "none", "null"}:
        return None
    for suffix in (".jpg", ".jpeg", ".png", ".webp"):
        if stem.lower().endswith(suffix):
            stem = stem[: -len(suffix)]
            break
    stem = stem.lstrip("p")
    return stem or None


def photo_url(photo: str | None, size: str = "250x250") -> str | None:
    code = photo_code(photo)
    if code is None:
        return None
    return f"{PHOTO_BASE}/{size}/p{code}.png"


def shirt_url(team_code: int | None, *, gk: bool = False) -> str | None:
    if team_code in (None, 0):
        return None
    suffix = f"{int(team_code)}-1-66" if gk else f"{int(team_code)}-66"
    return f"{SHIRT_BASE}/shirt_{suffix}.webp"


def badge_url(team_code: int | None, size: int = 70) -> str | None:
    if team_code in (None, 0):
        return None
    return f"{BADGE_BASE}/{size}/t{int(team_code)}.png"


def attach_assets(record: dict[str, Any], *, gk: bool | None = None) -> dict[str, Any]:
    """Add ``photo_url``, ``shirt_url``, ``badge_url`` to a player dict."""
    position = record.get("position")
    is_gk = gk if gk is not None else position == "GKP"
    team_code = record.get("team_code")
    out = dict(record)
    out["photo_url"] = photo_url(record.get("photo"))
    out["shirt_url"] = shirt_url(team_code, gk=is_gk)
    out["badge_url"] = badge_url(team_code)
    return out
