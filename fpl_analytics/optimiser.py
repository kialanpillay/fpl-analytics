"""Constrained squad and transfer optimisation.

Respects official FPL rules: 2/5/5/3, £100m, max 3 per club. Objective is a
weighted blend of horizon xPts, points-per-pound and consistency.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

try:
    import pulp
except ImportError:  # pragma: no cover
    pulp = None

POSITION_QUOTAS = {"GKP": 2, "DEF": 5, "MID": 5, "FWD": 3}
XI_BOUNDS = {
    "GKP": (1, 1),
    "DEF": (3, 5),
    "MID": (2, 5),
    "FWD": (1, 3),
}
OBJECTIVES = {
    "balanced": ("balanced", False),
    "xp": ("xp_horizon", False),
    "ppp": ("ppp", False),
    "consistency": ("consistency", False),
    "differential": ("differential", False),
}


@dataclass
class SquadPlan:
    players: pd.DataFrame
    cost: float
    objective: str
    objective_value: float
    xp_horizon: float
    xp_gw: float
    ppp: float
    consistency: float
    xi_ids: list[int]
    bench_ids: list[int]


def _eligible(players: pd.DataFrame) -> pd.DataFrame:
    return players.loc[
        players["can_select"]
        & (players["status"].isin(["a", "d"]))
        & (players["price"] > 0)
    ].copy()


def _solve_ilp(
    pool: pd.DataFrame,
    objective: str,
    budget: float,
    team_limit: int,
    locked_ids: set[int] | None = None,
    banned_ids: set[int] | None = None,
    min_from_ids: set[int] | None = None,
    max_changes_from: pd.DataFrame | None = None,
    max_changes: int | None = None,
) -> pd.DataFrame | None:
    if pulp is None:
        raise RuntimeError("pulp is required for optimisation. pip install pulp")

    col, _ = OBJECTIVES[objective]
    pool = pool.loc[pool[col].notna() & np.isfinite(pool[col])].copy()
    ids = list(pool["id"])
    lookup = pool.set_index("id")
    prob = pulp.LpProblem(f"fpl_{objective}", pulp.LpMaximize)
    x = pulp.LpVariable.dicts("p", ids, cat="Binary")

    prob += pulp.lpSum(float(lookup.loc[i, col]) * x[i] for i in ids)
    prob += pulp.lpSum(float(lookup.loc[i, "price"]) * x[i] for i in ids) <= budget
    for pos, n in POSITION_QUOTAS.items():
        pos_ids = [i for i in ids if lookup.loc[i, "position"] == pos]
        prob += pulp.lpSum(x[i] for i in pos_ids) == n
    for team_id in lookup["team_id"].unique():
        t_ids = [i for i in ids if lookup.loc[i, "team_id"] == team_id]
        prob += pulp.lpSum(x[i] for i in t_ids) <= team_limit

    locked_ids = locked_ids or set()
    banned_ids = banned_ids or set()
    for i in locked_ids:
        if i in x:
            x[i].setInitialValue(1)
            x[i].fixValue()
    for i in banned_ids:
        if i in x:
            x[i].setInitialValue(0)
            x[i].fixValue()
    if min_from_ids:
        present = [i for i in min_from_ids if i in x]
        if present:
            prob += pulp.lpSum(x[i] for i in present) >= min(len(present), 11)

    if max_changes_from is not None and max_changes is not None:
        current = set(max_changes_from["id"])
        keep = [i for i in current if i in x]
        # At least squad_size - max_changes of the current squad must stay.
        keep_n = max(0, 15 - max_changes)
        if keep:
            prob += pulp.lpSum(x[i] for i in keep) >= min(keep_n, len(keep))

    status = prob.solve(pulp.PULP_CBC_CMD(msg=False))
    if pulp.LpStatus[status] != "Optimal":
        return None
    chosen = [i for i in ids if pulp.value(x[i]) > 0.5]
    return lookup.loc[chosen].reset_index()


def pick_xi(squad: pd.DataFrame) -> tuple[list[int], list[int]]:
    """Pick an 11 that maximises next-GW xPts under formation rules."""
    if pulp is None:
        ordered = squad.sort_values("xp_gw", ascending=False)
        return list(ordered["id"].head(11)), list(ordered["id"].tail(4))

    ids = list(squad["id"])
    lookup = squad.set_index("id")
    prob = pulp.LpProblem("xi", pulp.LpMaximize)
    x = pulp.LpVariable.dicts("xi", ids, cat="Binary")
    prob += pulp.lpSum(float(lookup.loc[i, "xp_gw"]) * x[i] for i in ids)
    prob += pulp.lpSum(x[i] for i in ids) == 11
    for pos, (lo, hi) in XI_BOUNDS.items():
        pos_ids = [i for i in ids if lookup.loc[i, "position"] == pos]
        if pos_ids:
            prob += pulp.lpSum(x[i] for i in pos_ids) >= lo
            prob += pulp.lpSum(x[i] for i in pos_ids) <= hi
    status = prob.solve(pulp.PULP_CBC_CMD(msg=False))
    if pulp.LpStatus[status] != "Optimal":
        ordered = squad.sort_values("xp_gw", ascending=False)
        return list(ordered["id"].head(11)), list(ordered["id"].tail(4))
    xi = [i for i in ids if pulp.value(x[i]) > 0.5]
    bench = [i for i in ids if i not in xi]
    bench_sorted = lookup.loc[bench].sort_values("xp_gw", ascending=False).index.tolist()
    return xi, bench_sorted


def _plan(chosen: pd.DataFrame, objective: str) -> SquadPlan:
    xi, bench = pick_xi(chosen)
    return SquadPlan(
        players=chosen.sort_values(["element_type", "xp_horizon"], ascending=[True, False]),
        cost=float(chosen["price"].sum()),
        objective=objective,
        objective_value=float(chosen[OBJECTIVES[objective][0]].sum()),
        xp_horizon=float(chosen["xp_horizon"].sum()),
        xp_gw=float(chosen.loc[chosen["id"].isin(xi), "xp_gw"].sum()),
        ppp=float((chosen["xp_horizon"] / chosen["price"]).mean()),
        consistency=float(chosen["consistency"].mean()),
        xi_ids=xi,
        bench_ids=bench,
    )


def optimise_squad(
    players: pd.DataFrame,
    objective: str = "balanced",
    budget: float = 100.0,
    team_limit: int = 3,
    locked_ids: set[int] | None = None,
) -> SquadPlan:
    if objective not in OBJECTIVES:
        raise ValueError(f"Unknown objective {objective}. Choose from {list(OBJECTIVES)}")
    pool = _eligible(players)
    chosen = _solve_ilp(pool, objective, budget, team_limit, locked_ids=locked_ids)
    if chosen is None:
        raise RuntimeError("Optimiser could not find a feasible 15. Relax locks or budget.")
    return _plan(chosen, objective)


def suggest_transfers(
    squad: pd.DataFrame,
    players: pd.DataFrame,
    max_transfers: int = 2,
    objective: str = "balanced",
    budget: float = 100.0,
    bank: float = 0.0,
    team_limit: int = 3,
) -> SquadPlan:
    """Re-optimise allowing up to ``max_transfers`` changes from the current 15."""
    spend = float(squad["price"].sum()) + bank
    spend = min(budget, spend)
    pool = _eligible(players)
    # Always allow current squad members even if they later become ineligible.
    extra = squad.loc[~squad["id"].isin(pool["id"])]
    if not extra.empty:
        pool = pd.concat([pool, extra], ignore_index=True)
    chosen = _solve_ilp(
        pool,
        objective,
        spend,
        team_limit,
        max_changes_from=squad,
        max_changes=max_transfers,
    )
    if chosen is None:
        raise RuntimeError("No improving transfer set found under the current constraints.")
    return _plan(chosen, objective)


def one_for_one(
    squad: pd.DataFrame,
    players: pd.DataFrame,
    bank: float = 0.0,
    team_limit: int = 3,
    top_n: int = 12,
) -> pd.DataFrame:
    """Rank legal single replacements by balanced-score lift."""
    pool = _eligible(players)
    current_ids = set(squad["id"])
    rows = []
    for _, out_p in squad.iterrows():
        budget = out_p["price"] + bank
        club_counts = squad.loc[squad["id"] != out_p["id"], "team_id"].value_counts().to_dict()
        cands = pool.loc[
            (pool["position"] == out_p["position"])
            & (~pool["id"].isin(current_ids))
            & (pool["price"] <= budget + 1e-9)
        ]
        for _, inn in cands.iterrows():
            after = club_counts.get(inn["team_id"], 0) + 1
            if after > team_limit:
                continue
            d_bal = inn["balanced"] - out_p["balanced"]
            d_xp = inn["xp_horizon"] - out_p["xp_horizon"]
            if not (np.isfinite(d_bal) and np.isfinite(d_xp)):
                continue
            if d_bal <= 0.05 and d_xp <= 0.4:
                continue
            rows.append(
                {
                    "out_id": int(out_p["id"]),
                    "out": out_p["web_name"],
                    "out_team": out_p["team_short"],
                    "in_id": int(inn["id"]),
                    "in": inn["web_name"],
                    "in_team": inn["team_short"],
                    "position": out_p["position"],
                    "out_price": out_p["price"],
                    "in_price": inn["price"],
                    "cost_delta": inn["price"] - out_p["price"],
                    "d_balanced": d_bal,
                    "d_xp": d_xp,
                    "d_ppp": inn["ppp"] - out_p["ppp"],
                    "d_cons": inn["consistency"] - out_p["consistency"],
                    "in_own": inn["ownership"],
                    "unorthodox": bool(inn["unorthodox"]),
                }
            )
    if not rows:
        return pd.DataFrame()
    return (
        pd.DataFrame(rows)
        .sort_values(["d_balanced", "d_xp"], ascending=False)
        .head(top_n)
        .reset_index(drop=True)
    )
