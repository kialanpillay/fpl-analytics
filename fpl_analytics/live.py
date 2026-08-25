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
