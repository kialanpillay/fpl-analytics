"""Normalize official FPL bootstrap + fixtures into analysis tables."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd


def _f(value: Any, default: float = 0.0) -> float:
    if value in (None, "", "None"):
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _i(value: Any, default: int = 0) -> int:
    if value in (None, "", "None"):
        return default
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


@dataclass(frozen=True)
class SeasonMeta:
    next_event: int
    current_event: int | None
    deadline: str | None
    total_managers: int
    budget: float
    team_limit: int
    squad_size: int
    season_started: bool


def season_meta(bootstrap: dict[str, Any]) -> SeasonMeta:
    events = bootstrap["events"]
    nxt = next((e for e in events if e.get("is_next")), None)
    cur = next((e for e in events if e.get("is_current")), None)
    finished_any = any(e.get("finished") for e in events)
    settings = bootstrap.get("game_settings", {})
    return SeasonMeta(
        next_event=_i((nxt or events[0])["id"], 1),
        current_event=_i(cur["id"]) if cur else None,
        deadline=(nxt or events[0]).get("deadline_time"),
        total_managers=_i(bootstrap.get("total_players")),
        budget=_f(settings.get("squad_total_spend"), 1000) / 10,
        team_limit=_i(settings.get("squad_team_limit"), 3),
        squad_size=_i(settings.get("squad_squadsize"), 15),
        season_started=finished_any or cur is not None,
    )


def teams_frame(bootstrap: dict[str, Any]) -> pd.DataFrame:
    rows = []
    for team in bootstrap["teams"]:
        rows.append(
            {
                "team_id": team["id"],
                "team": team["name"],
                "team_short": team["short_name"],
                "team_code": _i(team.get("code")),
                "strength_overall_home": _i(team.get("strength_overall_home")),
                "strength_overall_away": _i(team.get("strength_overall_away")),
                "strength_attack_home": _i(team.get("strength_attack_home")),
                "strength_attack_away": _i(team.get("strength_attack_away")),
                "strength_defence_home": _i(team.get("strength_defence_home")),
                "strength_defence_away": _i(team.get("strength_defence_away")),
            }
        )
    return pd.DataFrame(rows)


def fixtures_frame(raw: list[dict[str, Any]]) -> pd.DataFrame:
    rows = []
    for fixture in raw:
        if fixture.get("event") is None:
            continue
        rows.append(
            {
                "fixture_id": fixture["id"],
                "event": _i(fixture["event"]),
                "kickoff": fixture.get("kickoff_time"),
                "team_h": fixture["team_h"],
                "team_a": fixture["team_a"],
                "fdr_h": _i(fixture.get("team_h_difficulty"), 3),
                "fdr_a": _i(fixture.get("team_a_difficulty"), 3),
                "finished": bool(fixture.get("finished")),
                "started": bool(fixture.get("started")),
            }
        )
    return pd.DataFrame(rows)


def players_frame(bootstrap: dict[str, Any], teams: pd.DataFrame) -> pd.DataFrame:
    types = {t["id"]: t["singular_name_short"] for t in bootstrap["element_types"]}
    team_lookup = teams.set_index("team_id")
    rows = []
    for raw in bootstrap["elements"]:
        team_id = raw["team"]
        team = team_lookup.loc[team_id]
        rows.append(
            {
                "id": raw["id"],
                "web_name": raw["web_name"],
                "first_name": raw["first_name"],
                "second_name": raw["second_name"],
                "full_name": f"{raw['first_name']} {raw['second_name']}",
                "position": types[raw["element_type"]],
                "element_type": raw["element_type"],
                "team_id": team_id,
                "team": team["team"],
                "team_short": team["team_short"],
                "team_code": _i(raw.get("team_code")) or _i(team.get("team_code")),
                "code": _i(raw.get("code")),
                "photo": raw.get("photo") or "",
                "price": raw["now_cost"] / 10.0,
                "ownership": _f(raw.get("selected_by_percent")),
                "status": raw.get("status") or "a",
                "news": raw.get("news") or "",
                "can_select": bool(raw.get("can_select", True)),
                "chance_next": raw.get("chance_of_playing_next_round"),
                "minutes": _i(raw.get("minutes")),
                "starts": _i(raw.get("starts")),
                "event_points": _i(raw.get("event_points")),
                "total_points": _i(raw.get("total_points")),
                "points_per_game": _f(raw.get("points_per_game")),
                "goals": _i(raw.get("goals_scored")),
                "assists": _i(raw.get("assists")),
                "clean_sheets": _i(raw.get("clean_sheets")),
                "goals_conceded": _i(raw.get("goals_conceded")),
                "bonus": _i(raw.get("bonus")),
                "bps": _i(raw.get("bps")),
                "saves": _i(raw.get("saves")),
                "xg": _f(raw.get("expected_goals")),
                "xa": _f(raw.get("expected_assists")),
                "xgi": _f(raw.get("expected_goal_involvements")),
                "xgc": _f(raw.get("expected_goals_conceded")),
                "xg_p90": _f(raw.get("expected_goals_per_90")),
                "xa_p90": _f(raw.get("expected_assists_per_90")),
                "xgi_p90": _f(raw.get("expected_goal_involvements_per_90")),
                "xgc_p90": _f(raw.get("expected_goals_conceded_per_90")),
                "defcon": _f(raw.get("defensive_contribution")),
                "defcon_p90": _f(raw.get("defensive_contribution_per_90")),
                "ep_next": None if raw.get("ep_next") in (None, "") else _f(raw.get("ep_next")),
                "ep_this": None if raw.get("ep_this") in (None, "") else _f(raw.get("ep_this")),
                "form": _f(raw.get("form")),
                "value_season": _f(raw.get("value_season")),
                "ict": _f(raw.get("ict_index")),
                "penalties_order": raw.get("penalties_order"),
                "corners_order": raw.get("corners_and_indirect_freekicks_order"),
                "freekicks_order": raw.get("direct_freekicks_order"),
                "team_join_date": raw.get("team_join_date"),
                "transfers_in": _i(raw.get("transfers_in")),
                "transfers_out": _i(raw.get("transfers_out")),
            }
        )
    return pd.DataFrame(rows)


def apply_event_live(players: pd.DataFrame, live: dict[str, Any] | None) -> pd.DataFrame:
    """Merge /event/{gw}/live/ totals onto the player table."""
    out = players.copy()
    if live:
        by_id = {el["id"]: el.get("stats") or {} for el in live.get("elements", [])}
        points, bonus, bps, minutes = [], [], [], []
        for rec in out.itertuples(index=False):
            stats = by_id.get(int(rec.id), {})
            points.append(_i(stats.get("total_points"), _i(getattr(rec, "event_points", 0))))
            bonus.append(_i(stats.get("bonus"), _i(getattr(rec, "bonus", 0))))
            bps.append(_i(stats.get("bps"), _i(getattr(rec, "bps", 0))))
            minutes.append(_i(stats.get("minutes"), _i(getattr(rec, "minutes", 0))))
        out["event_points"] = points
        out["bonus"] = bonus
        out["bps"] = bps
        out["gw_minutes"] = minutes
    else:
        out["gw_minutes"] = out["minutes"]
    return out

