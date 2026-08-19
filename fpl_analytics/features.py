"""Fixture, minutes and role features used by the scoring models."""

from __future__ import annotations

from datetime import date, datetime

import numpy as np
import pandas as pd

# 2025/26+ FPL: 2 pts when the action threshold is hit.
DEFCON_THRESHOLD = {"GKP": 99, "DEF": 10, "MID": 12, "FWD": 12}

# FDR 2 is a gift, 5 is a brick wall. Mapped onto a multiplicative XP tilt.
FDR_ATTACK = {1: 1.18, 2: 1.10, 3: 1.00, 4: 0.90, 5: 0.80}
FDR_CLEAN = {1: 1.22, 2: 1.12, 3: 1.00, 4: 0.86, 5: 0.72}

GOAL_POINTS = {"GKP": 10, "DEF": 6, "MID": 5, "FWD": 4}
CS_POINTS = {"GKP": 4, "DEF": 4, "MID": 1, "FWD": 0}


def _parse_date(value: object) -> date | None:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    text = str(value)
    if not text or text in {"nan", "None"}:
        return None
    try:
        return date.fromisoformat(text[:10])
    except ValueError:
        return None


def minutes_probability(row: pd.Series, season_started: bool) -> float:
    """Probability the player starts and plays ~60+ minutes."""
    chance = row["chance_next"]
    if pd.isna(chance):
        chance = None
    if chance is not None:
        avail = float(chance) / 100.0
    elif row["status"] == "a":
        avail = 1.0
    elif row["status"] == "d":
        avail = 0.55
    else:
        avail = 0.05

    if row["status"] in {"i", "u", "s", "n"}:
        return 0.05 * avail

    if season_started and row["form"] > 0:
        # In-season: blend recent minutes proxy (form/ep) with start rate.
        start_rate = min(1.0, row["starts"] / max(row.get("events_played", 1), 1))
        return float(np.clip(avail * (0.35 + 0.65 * start_rate), 0.05, 1.0))

    starts = float(row["starts"])
    minutes = float(row["minutes"])
    start_rate = starts / 38.0
    minute_rate = minutes / 3420.0
    base = 0.55 * start_rate + 0.45 * minute_rate

    if minutes < 200:
        base = min(base, 0.18)
    elif minutes < 900:
        base = min(base, 0.45)

    return float(np.clip(avail * (0.12 + 0.88 * base), 0.05, 0.97))


def role_risk(row: pd.Series, as_of: date) -> float:
    """Discount for new-club / rotation / unproven minutes (1 = no extra risk)."""
    risk = 1.0
    joined = _parse_date(row["team_join_date"])
    if joined and (as_of - joined).days < 80:
        risk *= 0.82
    elif joined and (as_of - joined).days < 200:
        risk *= 0.90
    if row["minutes"] < 900:
        risk *= 0.78
    if row["minutes"] < 300:
        risk *= 0.70
    return float(np.clip(risk, 0.35, 1.0))


def _team_fixtures(
    fixtures: pd.DataFrame,
    team_id: int,
    from_event: int,
    horizon: int,
) -> pd.DataFrame:
    end = from_event + horizon - 1
    mask = (
        (fixtures["event"] >= from_event)
        & (fixtures["event"] <= end)
        & ((fixtures["team_h"] == team_id) | (fixtures["team_a"] == team_id))
        & (~fixtures["finished"])
    )
    return fixtures.loc[mask].sort_values("event")


def fixture_features(
    players: pd.DataFrame,
    fixtures: pd.DataFrame,
    teams: pd.DataFrame,
    next_event: int,
    horizon: int = 6,
) -> pd.DataFrame:
    short = teams.set_index("team_id")["team_short"]
    attack_adj: list[float] = []
    clean_adj: list[float] = []
    fdr_mean: list[float] = []
    next_opp: list[str] = []
    run: list[str] = []
    n_fix: list[int] = []

    for team_id in players["team_id"]:
        upcoming = _team_fixtures(fixtures, int(team_id), next_event, horizon)
        if upcoming.empty:
            attack_adj.append(1.0)
            clean_adj.append(1.0)
            fdr_mean.append(3.0)
            next_opp.append("—")
            run.append("")
            n_fix.append(0)
            continue

        fdrs: list[int] = []
        labels: list[str] = []
        atk = []
        cln = []
        for rec in upcoming.itertuples(index=False):
            home = rec.team_h == team_id
            fdr = int(rec.fdr_h if home else rec.fdr_a)
            opp_id = rec.team_a if home else rec.team_h
            opp = short.get(opp_id, "?")
            fdrs.append(fdr)
            labels.append(f"GW{rec.event} {'H' if home else 'A'} {opp} ({fdr})")
            atk.append(FDR_ATTACK.get(fdr, 1.0))
            cln.append(FDR_CLEAN.get(fdr, 1.0))

        attack_adj.append(float(np.mean(atk)))
        clean_adj.append(float(np.mean(cln)))
        fdr_mean.append(float(np.mean(fdrs)))
        next_opp.append(labels[0])
        run.append(" · ".join(labels))
        n_fix.append(len(labels))

    out = players.copy()
    out["attack_adj"] = attack_adj
    out["clean_adj"] = clean_adj
    out["fdr_mean"] = fdr_mean
    out["next_fixture"] = next_opp
    out["fixture_run"] = run
    out["n_fixtures"] = n_fix
    return out


def enrich(
    players: pd.DataFrame,
    fixtures: pd.DataFrame,
    teams: pd.DataFrame,
    next_event: int,
    season_started: bool,
    horizon: int = 6,
    as_of: date | None = None,
) -> pd.DataFrame:
    as_of = as_of or date.today()
    frame = fixture_features(players, fixtures, teams, next_event, horizon)
    frame["minutes_prob"] = frame.apply(
        lambda r: minutes_probability(r, season_started), axis=1
    )
    frame["role_risk"] = frame.apply(lambda r: role_risk(r, as_of), axis=1)
    frame["effective_minutes"] = frame["minutes_prob"] * frame["role_risk"]
    frame["ppm_last"] = np.where(frame["price"] > 0, frame["total_points"] / frame["price"], 0.0)
    frame["set_piece"] = (
        frame["penalties_order"].notna()
        | frame["corners_order"].notna()
        | frame["freekicks_order"].notna()
    )
    return frame
