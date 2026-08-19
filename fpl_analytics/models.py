"""Expected points, consistency, points-per-pound and residual-value models."""

from __future__ import annotations

import numpy as np
import pandas as pd

from fpl_analytics.features import CS_POINTS, DEFCON_THRESHOLD, GOAL_POINTS

# Blend: official FPL EP for the next GW, our model for the horizon.
EP_BLEND = 0.55


def _minmax(series: pd.Series) -> pd.Series:
    lo, hi = series.min(), series.max()
    if hi - lo < 1e-9:
        return pd.Series(0.0, index=series.index)
    return (series - lo) / (hi - lo)


def _appearance_points(minutes_prob: pd.Series) -> pd.Series:
    # 2 pts for 60+ minutes, 1 pt for a cameo. Weighted by start probability.
    return 2.0 * minutes_prob + 0.25 * (1 - minutes_prob)


def _attack_points(frame: pd.DataFrame) -> pd.Series:
    goal_pts = frame["position"].map(GOAL_POINTS)
    return frame["xg_p90"] * goal_pts + frame["xa_p90"] * 3.0


def _clean_sheet_points(frame: pd.DataFrame) -> pd.Series:
    cs_pts = frame["position"].map(CS_POINTS).astype(float)
    games = np.clip(frame["minutes"] / 90.0, 8, 40)
    cs_rate = np.where(games > 0, frame["clean_sheets"] / games, 0.25)
    # Keepers / defenders without history: use a conservative 0.28 prior.
    prior = np.where(frame["position"].isin(["GKP", "DEF"]), 0.28, 0.12)
    shrink = np.clip(frame["minutes"] / 2500.0, 0.15, 0.85)
    rate = shrink * cs_rate + (1 - shrink) * prior
    return rate * cs_pts * frame["clean_adj"]


def _defcon_points(frame: pd.DataFrame) -> pd.Series:
    thresh = frame["position"].map(DEFCON_THRESHOLD).astype(float)
    hit_rate = np.clip(frame["defcon_p90"] / thresh, 0, 1.15)
    points = 2.0 * hit_rate
    points = points.where(frame["position"] != "GKP", 0.0)
    return points


def _bonus_points(frame: pd.DataFrame) -> pd.Series:
    games = np.clip(frame["minutes"] / 90.0, 1, 40)
    return np.clip(frame["bonus"] / games, 0, 1.8)


def _saves_points(frame: pd.DataFrame) -> pd.Series:
    games = np.clip(frame["minutes"] / 90.0, 1, 40)
    per_game = np.where(frame["position"] == "GKP", frame["saves"] / games, 0.0)
    return per_game / 3.0  # 1 pt per 3 saves


def expected_points(frame: pd.DataFrame) -> pd.DataFrame:
    """Per-90 model, then scale by minutes probability, role risk and fixtures."""
    out = frame.copy()
    p90 = (
        _appearance_points(out["effective_minutes"])
        + _attack_points(out) * out["attack_adj"] * out["effective_minutes"]
        + _clean_sheet_points(out) * out["effective_minutes"]
        + _defcon_points(out) * out["effective_minutes"]
        + _bonus_points(out) * out["effective_minutes"]
        + _saves_points(out) * out["effective_minutes"]
    )
    # Penalty / set-piece bump: small, but it is a real extra route.
    pens = out["penalties_order"].fillna(99)
    p90 = p90 + np.where(pens == 1, 0.35, np.where(pens <= 2, 0.12, 0.0))
    p90 = p90 + np.where(out["corners_order"].fillna(99) == 1, 0.10, 0.0)

    out["xp_p90"] = p90
    model_gw = p90.copy()
    official = out["ep_next"].astype(float).fillna(model_gw)
    out["xp_gw"] = EP_BLEND * official + (1 - EP_BLEND) * model_gw
    horizon = out["n_fixtures"].clip(lower=1)
    # Horizon uses our model (fixture-adjusted already via mean FDR) plus a
    # small pull toward last-season points/start for players with a sample.
    last_per_start = np.where(
        out["starts"] >= 10,
        out["total_points"] / out["starts"],
        model_gw,
    )
    out["xp_horizon"] = (
        0.65 * model_gw * horizon
        + 0.20 * official * horizon
        + 0.15 * last_per_start * out["effective_minutes"] * horizon
    )
    return out


def consistency_score(frame: pd.DataFrame) -> pd.Series:
    """0–10. High start rate + DEFCON floor + moderate (not boom-bust) xGI."""
    start_rate = np.clip(frame["starts"] / 38.0, 0, 1)
    minutes_rate = np.clip(frame["minutes"] / 3420.0, 0, 1)
    floor = np.clip(frame["defcon_p90"] / 12.0, 0, 1)
    # Prefer players whose points came without needing a 20-pt haul every week.
    ppg = frame["points_per_game"]
    reliability = np.clip(ppg / 6.0, 0, 1)
    risk_penalty = 1 - frame["role_risk"]
    raw = (
        3.2 * start_rate
        + 2.2 * minutes_rate
        + 2.4 * floor
        + 2.2 * reliability
        - 3.0 * risk_penalty
    )
    return pd.Series(np.clip(raw, 0, 10), index=frame.index)


def points_per_pound(frame: pd.DataFrame) -> pd.Series:
    """Horizon expected points per £1m. This is the PPP the optimiser uses."""
    return frame["xp_horizon"] / frame["price"].clip(lower=0.1)


def underprice_residual(frame: pd.DataFrame) -> pd.Series:
    """Position-wise residual of last-season points versus price.

    Positive = delivered more than the current price band typically bought.
    Early-season this is the main 'is he cheap?' signal; in-season it is
    blended with form residuals in ``score``.
    """
    residual = pd.Series(0.0, index=frame.index)
    for _, group in frame.groupby("position"):
        idx = group.index
        x = group["price"].to_numpy()
        y = group["total_points"].to_numpy(dtype=float)
        if len(x) < 8 or np.allclose(x, x[0]):
            residual.loc[idx] = 0.0
            continue
        slope, intercept = np.polyfit(x, y, 1)
        residual.loc[idx] = y - (slope * x + intercept)
    return residual


def score(frame: pd.DataFrame) -> pd.DataFrame:
    out = expected_points(frame)
    out["consistency"] = consistency_score(out)
    out["ppp"] = points_per_pound(out)
    out["residual"] = underprice_residual(out)
    out["value_flag"] = (out["residual"] > 25) & (out["effective_minutes"] > 0.45)

    # 0–10 style ranks used by the optimiser and reports.
    out["xp_n"] = _minmax(out["xp_horizon"])
    out["ppp_n"] = _minmax(out["ppp"])
    out["cons_n"] = out["consistency"] / 10.0
    out["resid_n"] = _minmax(out["residual"])

    out["balanced"] = (
        0.50 * out["xp_n"] + 0.28 * out["ppp_n"] + 0.22 * out["cons_n"]
    ) * 10
    out["differential"] = out["balanced"] * (
        1.0 + 0.45 * (1.0 - np.clip(out["ownership"] / 25.0, 0, 1))
    )
    # Unorthodox: good model score, ignored by the field.
    out["unorthodox"] = (
        (out["ownership"] < 10)
        & (out["balanced"] >= out.groupby("position")["balanced"].transform("quantile", 0.72))
        & (out["effective_minutes"] >= 0.40)
        & out["can_select"]
        & (out["status"] == "a")
    )
    return out
