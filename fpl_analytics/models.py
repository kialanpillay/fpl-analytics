"""Expected points, consistency, points-per-pound and residual-value models."""

from __future__ import annotations

import numpy as np
import pandas as pd

from fpl_analytics.features import CS_POINTS, DEFCON_THRESHOLD, GOAL_POINTS

# Blend: official FPL EP for the next GW, our model for the horizon.
EP_BLEND = 0.55
EP_BLEND_EARLY = 0.75


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
    # One-week 2–3 BP is not a season rate.
    cap = 0.45 if ("early_season" in frame and bool(frame["early_season"].iloc[0])) else 1.8
    return np.clip(frame["bonus"] / games, 0, cap)


def _saves_points(frame: pd.DataFrame) -> pd.Series:
    games = np.clip(frame["minutes"] / 90.0, 1, 40)
    per_game = np.where(frame["position"] == "GKP", frame["saves"] / games, 0.0)
    return per_game / 3.0  # 1 pt per 3 saves


def expected_points(frame: pd.DataFrame) -> pd.DataFrame:
    """Per-90 model, then scale by minutes probability, role risk and fixtures."""
    out = frame.copy()
    early = "early_season" in out and bool(out["early_season"].iloc[0])
    if early:
        # One-game xG/90 is not a rate (De Cuyper 1.47 xG in 77 ≠ 1.7/90).
        w = np.clip(out["minutes"] / 900.0, 0.10, 1.0)
        prior_xg = out["position"].map({"GKP": 0.0, "DEF": 0.06, "MID": 0.18, "FWD": 0.32}).astype(float)
        prior_xa = out["position"].map({"GKP": 0.0, "DEF": 0.06, "MID": 0.16, "FWD": 0.12}).astype(float)
        out = out.assign(
            xg_p90=w * out["xg_p90"] + (1 - w) * prior_xg,
            xa_p90=w * out["xa_p90"] + (1 - w) * prior_xa,
        )
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
    blend = EP_BLEND_EARLY if early else EP_BLEND
    out["xp_gw"] = blend * official + (1 - blend) * model_gw
    horizon = out["n_fixtures"].clip(lower=1)
    # Do not treat GW1 totals as a season rate (Haaland would be 2 pts/start).
    last_per_start = np.where(
        (not early) & (out["starts"] >= 10),
        out["total_points"] / out["starts"],
        official,
    )
    if early:
        out["xp_horizon"] = (
            0.25 * model_gw * horizon
            + 0.75 * official * horizon
        )
    else:
        out["xp_horizon"] = (
            0.65 * model_gw * horizon
            + 0.20 * official * horizon
            + 0.15 * last_per_start * out["effective_minutes"] * horizon
        )
    return out


def consistency_score(frame: pd.DataFrame) -> pd.Series:
    """0–10. High start rate + DEFCON floor + moderate (not boom-bust) xGI."""
    early = "early_season" in frame and bool(frame["early_season"].iloc[0])
    if early:
        started = (frame["minutes"] >= 60).astype(float)
        start_rate = 0.55 + 0.35 * started
        minutes_rate = np.clip(frame["minutes"] / 90.0, 0, 1)
    else:
        start_rate = np.clip(frame["starts"] / 38.0, 0, 1)
        minutes_rate = np.clip(frame["minutes"] / 3420.0, 0, 1)
    floor = np.clip(frame["defcon_p90"] / 12.0, 0, 1)
    # Prefer players whose points came without needing a 20-pt haul every week.
    if early:
        reliability = np.clip(frame["ep_next"].fillna(0) / 6.0, 0, 1)
    else:
        reliability = np.clip(frame["points_per_game"] / 6.0, 0, 1)
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
        y_col = "total_points"
        if "early_season" in group and bool(group["early_season"].iloc[0]):
            # One-week totals are not a price residual. Use FPL EP as the level.
            y_col = "ep_next"
        y = group[y_col].fillna(0).to_numpy(dtype=float)
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
    # Ceiling: next-GW haul + horizon + attacking threat. No PPP / floor, so
    # premiums are not crowded out by 4.5 DEFCON enablers.
    xgi = out["xgi_p90"] if "xgi_p90" in out else out["xg_p90"] + out["xa_p90"]
    out["xp_gw_n"] = _minmax(out["xp_gw"])
    out["xgi_n"] = _minmax(xgi.fillna(0))
    out["aggressive"] = (0.45 * out["xp_gw_n"] + 0.35 * out["xp_n"] + 0.20 * out["xgi_n"]) * 10
    # Field-following counterpart to differential: points + minutes floor + ownership.
    out["own_n"] = _minmax(out["ownership"].fillna(0))
    out["template"] = (0.50 * out["xp_n"] + 0.20 * out["cons_n"] + 0.30 * out["own_n"]) * 10
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
