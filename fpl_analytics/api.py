"""Official Fantasy Premier League API client with on-disk caching.

Public, unauthenticated endpoints used throughout the season:

* ``GET /bootstrap-static/`` — players, teams, events, scoring
* ``GET /fixtures/`` — full fixture list with FDR
* ``GET /element-summary/{id}/`` — per-gameweek history
* ``GET /event/{gw}/live/`` — live gameweek points
* ``GET /entry/{id}/`` — public manager profile
* ``GET /entry/{id}/event/{gw}/picks/`` — public gameweek picks
* ``GET /event-status/`` — bonus / league processing
* ``GET /team/set-piece-notes/`` — set-piece taker notes
"""

from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urljoin

import requests

BASE_URL = "https://fantasy.premierleague.com/api/"
DEFAULT_CACHE_DIR = Path(__file__).resolve().parent.parent / "data" / "cache"
USER_AGENT = "fpl-analytics/2.0 (+https://github.com/kialanpillay/fpl-analytics)"


class FPLClient:
    def __init__(
        self,
        cache_dir: Path | str = DEFAULT_CACHE_DIR,
        ttl_seconds: int = 30 * 60,
        timeout: int = 30,
    ) -> None:
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.ttl_seconds = ttl_seconds
        self.timeout = timeout
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": USER_AGENT})

    def _cache_path(self, name: str) -> Path:
        return self.cache_dir / f"{name}.json"

    def _read_cache(self, name: str) -> Any | None:
        path = self._cache_path(name)
        if not path.exists():
            return None
        age = time.time() - path.stat().st_mtime
        if age > self.ttl_seconds:
            return None
        return json.loads(path.read_text())

    def _write_cache(self, name: str, payload: Any) -> None:
        path = self._cache_path(name)
        path.write_text(json.dumps(payload))

    def get(self, path: str, cache_as: str | None = None, force: bool = False) -> Any:
        if cache_as and not force:
            cached = self._read_cache(cache_as)
            if cached is not None:
                return cached
        url = urljoin(BASE_URL, path.lstrip("/"))
        response = self.session.get(url, timeout=self.timeout)
        response.raise_for_status()
        payload = response.json()
        if cache_as:
            self._write_cache(cache_as, payload)
        return payload

    def bootstrap(self, force: bool = False) -> dict[str, Any]:
        return self.get("bootstrap-static/", cache_as="bootstrap-static", force=force)

    def fixtures(self, force: bool = False) -> list[dict[str, Any]]:
        return self.get("fixtures/", cache_as="fixtures", force=force)

    def element_summary(self, player_id: int, force: bool = False) -> dict[str, Any]:
        return self.get(
            f"element-summary/{player_id}/",
            cache_as=f"element-{player_id}",
            force=force,
        )

    def event_live(self, gameweek: int, force: bool = False) -> dict[str, Any]:
        return self.get(
            f"event/{gameweek}/live/",
            cache_as=f"event-{gameweek}-live",
            force=force,
        )

    def entry(self, manager_id: int, force: bool = False) -> dict[str, Any]:
        return self.get(f"entry/{manager_id}/", cache_as=f"entry-{manager_id}", force=force)

    def entry_history(self, manager_id: int, force: bool = False) -> dict[str, Any]:
        return self.get(
            f"entry/{manager_id}/history/",
            cache_as=f"entry-{manager_id}-history",
            force=force,
        )

    def entry_transfers(self, manager_id: int, force: bool = False) -> list[dict[str, Any]]:
        return self.get(
            f"entry/{manager_id}/transfers/",
            cache_as=f"entry-{manager_id}-transfers",
            force=force,
        )

    def entry_picks(
        self, manager_id: int, gameweek: int, force: bool = False
    ) -> dict[str, Any]:
        return self.get(
            f"entry/{manager_id}/event/{gameweek}/picks/",
            cache_as=f"entry-{manager_id}-gw{gameweek}-picks",
            force=force,
        )

    def event_status(self, force: bool = False) -> dict[str, Any]:
        return self.get("event-status/", cache_as="event-status", force=force)

    def set_piece_notes(self, force: bool = False) -> dict[str, Any]:
        return self.get("team/set-piece-notes/", cache_as="set-piece-notes", force=force)

    def refresh(self) -> dict[str, Any]:
        bootstrap = self.bootstrap(force=True)
        fixtures = self.fixtures(force=True)
        return {
            "fetched_at": datetime.now(timezone.utc).isoformat(),
            "players": len(bootstrap.get("elements", [])),
            "fixtures": len(fixtures),
            "cache_dir": str(self.cache_dir),
        }
