"""Official manager GW score and XI from public entry picks."""

from __future__ import annotations

import os
from typing import Any

import pandas as pd

DEFAULT_MANAGER_ID = 5558057


def resolve_manager_id() -> int:
    raw = os.environ.get("FPL_MANAGER_ID")
    if raw not in (None, "", "0"):
        return int(raw)
    return DEFAULT_MANAGER_ID


def players_for_ids(ids: list[int], *frames: pd.DataFrame) -> pd.DataFrame:
    """Resolve player rows in ``ids`` order from the first catalog that has each id."""
    nonempty = [f for f in frames if f is not None and not f.empty and "id" in f.columns]
    if not ids:
        return nonempty[0].iloc[0:0].copy() if nonempty else pd.DataFrame()
    lookups = [f.drop_duplicates(subset=["id"]).set_index("id", drop=False) for f in nonempty]
    rows = []
    for pid in ids:
        key = int(pid)
        for idx in lookups:
            if key in idx.index:
                rows.append(idx.loc[key])
                break
    if not rows:
        return nonempty[0].iloc[0:0].copy() if nonempty else pd.DataFrame()
    return pd.DataFrame(rows).reset_index(drop=True)


def fetch_official_picks(client: Any, manager_id: int, gw: int, force: bool = False) -> dict[str, Any] | None:
    try:
        return parse_entry_picks(client.entry_picks(manager_id, int(gw), force=force))
    except Exception:
        return None


def pick_ids(official: dict[str, Any] | None) -> list[int]:
    if not official:
        return []
    return [int(i) for i in (official.get("xi_ids") or []) + (official.get("bench_ids") or [])]


def bank_from_entry(entry: dict[str, Any] | None) -> float:
    if not entry:
        return 0.0
    raw = entry.get("last_deadline_bank")
    if raw in (None, ""):
        return 0.0
    return float(raw) / 10.0


def free_transfers_from_history(history: dict[str, Any] | None) -> int:
    """FTs available for the next deadline after applying each recorded GW."""
    rows = (history or {}).get("current") or []
    fts = 1
    for row in sorted(rows, key=lambda r: int(r.get("event") or 0)):
        used = int(row.get("event_transfers") or 0)
        fts = min(5, max(0, fts - used) + 1)
    return fts


def load_official_squad(
    client: Any,
    manager_id: int,
    gameweek: int,
    force: bool = False,
) -> tuple[list[int], dict[str, Any] | None, float, int]:
    """Official 15, bank, and free transfers from the public entry endpoints."""
    official = fetch_official_picks(client, manager_id, gameweek, force=force)
    if official is None and gameweek > 1:
        official = fetch_official_picks(client, manager_id, gameweek - 1, force=force)
    ids = pick_ids(official)
    if not ids:
        raise RuntimeError(
            f"Could not load official picks for manager {manager_id} GW{gameweek}."
        )
    try:
        entry = client.entry(manager_id, force=force)
    except Exception:
        entry = None
    try:
        history = client.entry_history(manager_id, force=force)
    except Exception:
        history = None
    return ids, official, bank_from_entry(entry), free_transfers_from_history(history)


def parse_entry_picks(payload: dict[str, Any]) -> dict[str, Any] | None:
    picks = payload.get("picks") or []
    if not picks:
        return None
    history = payload.get("entry_history") or {}
    auto = payload.get("automatic_subs") or []
    on = [p for p in picks if int(p.get("multiplier") or 0) > 0]
    off = [p for p in picks if int(p.get("multiplier") or 0) == 0]
    if on:
        xi_ids = [int(p["element"]) for p in sorted(on, key=lambda p: int(p.get("position") or 0))]
        bench_ids = [int(p["element"]) for p in sorted(off, key=lambda p: int(p.get("position") or 0))]
    else:
        ordered = sorted(picks, key=lambda p: int(p.get("position") or 0))
        xi_ids = [int(p["element"]) for p in ordered if int(p.get("position") or 99) <= 11]
        bench_ids = [int(p["element"]) for p in ordered if int(p.get("position") or 0) > 11]
    captain_id = next((int(p["element"]) for p in picks if p.get("is_captain")), None)
    vice_id = next((int(p["element"]) for p in picks if p.get("is_vice_captain")), None)
    return {
        "xi_ids": xi_ids,
        "bench_ids": bench_ids,
        "captain_id": captain_id,
        "vice_id": vice_id,
        "chip": payload.get("active_chip"),
        "points": history.get("points"),
        "points_on_bench": history.get("points_on_bench"),
        "total_points": history.get("total_points"),
        "rank": history.get("rank"),
        "overall_rank": history.get("overall_rank"),
        "transfers": history.get("event_transfers"),
        "hits": history.get("event_transfers_cost"),
        "auto_subs": [
            {"out_id": int(s["element_out"]), "in_id": int(s["element_in"])}
            for s in auto
            if s.get("element_out") and s.get("element_in")
        ],
    }
