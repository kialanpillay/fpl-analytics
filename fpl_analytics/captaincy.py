"""Rank captain and vice options from a squad XI."""

from __future__ import annotations

import pandas as pd

from fpl_analytics.optimiser import pick_xi

SEASON_DEFAULT = "Haaland"


def rank_captaincy(
    squad: pd.DataFrame,
    xi_ids: list[int] | None = None,
    n: int = 11,
) -> pd.DataFrame:
    """Rank by ``xp_gw * minutes_prob``. Default XI from ``pick_xi``."""
    if squad.empty:
        return squad.copy()
    ids = xi_ids if xi_ids is not None else pick_xi(squad)[0]
    frame = squad.loc[squad["id"].isin(ids)].copy()
    minutes = frame["minutes_prob"] if "minutes_prob" in frame.columns else 1.0
    xp = frame["xp_gw"].astype(float)
    frame["captain_score"] = xp * minutes.astype(float)
    frame["captain_ev"] = xp * 2.0
    frame["vice_ev"] = xp
    frame["season_default"] = frame["web_name"] == SEASON_DEFAULT
    return frame.sort_values(
        ["captain_score", "xp_gw"], ascending=False
    ).head(n).reset_index(drop=True)
