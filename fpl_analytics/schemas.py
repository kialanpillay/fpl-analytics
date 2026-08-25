"""Pydantic payloads shared by CLI export and the FastAPI UI."""

from __future__ import annotations

import math
from typing import Any

import pandas as pd
from pydantic import BaseModel, Field

from fpl_analytics.assets import attach_assets
from fpl_analytics.optimiser import SquadPlan, pair_swaps
from fpl_analytics.pipeline import AnalysisBundle, player_notes, strategies

PLAYER_KEYS = [
    "id",
    "web_name",
    "first_name",
    "second_name",
    "full_name",
    "position",
    "element_type",
    "team_id",
    "team",
    "team_short",
    "team_code",
    "code",
    "photo",
    "price",
    "ownership",
    "status",
    "news",
    "can_select",
    "chance_next",
    "event_points",
    "total_points",
    "xp_gw",
    "xp_horizon",
    "ppp",
    "consistency",
    "balanced",
    "aggressive",
    "template",
    "residual",
    "differential",
    "minutes_prob",
    "role_risk",
    "effective_minutes",
    "next_fixture",
    "fixture_run",
    "fdr_mean",
    "unorthodox",
    "value_flag",
    "xg_p90",
    "xa_p90",
    "xgi_p90",
    "defcon_p90",
    "set_piece",
    "penalties_order",
    "minutes",
    "starts",
    "form",
    "bonus",
    "bps",
    "gw_minutes",
]


def _finite(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, (bool, str)):
        return value
    if pd.isna(value):
        return None
    if isinstance(value, (int,)):
        return int(value)
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return None
        return float(value)
    if hasattr(value, "item"):
        try:
            return _finite(value.item())
        except (ValueError, AttributeError):
            return None
    return value


def round_number(value: Any, ndigits: int = 2) -> float | None:
    number = _finite(value)
    if number is None:
        return None
    return round(float(number), ndigits)


def _round(value: Any, ndigits: int) -> float | None:
    return round_number(value, ndigits)


def _json_value(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(v) for v in value]
    if isinstance(value, str):
        return value
    return _finite(value)


class NoteOut(BaseModel):
    name: str
    tone: str
    note: str


class StrategyOut(BaseModel):
    id: str
    title: str
    detail: str


class MetaOut(BaseModel):
    next_event: int
    current_event: int | None = None
    deadline: str | None = None
    total_managers: int = 0
    budget: float = 100.0
    team_limit: int = 3
    squad_size: int = 15
    season_started: bool = False
    horizon: int = 6
    bank: float = 0.0
    free_transfers: int = 1


class PlanOut(BaseModel):
    objective: str
    cost: float
    xp_gw: float
    xp_horizon: float
    ppp: float
    consistency: float
    xi: list[str]
    bench: list[str]
    xi_ids: list[int]
    bench_ids: list[int]
    players: list[dict[str, Any]]


class AnalysisOut(BaseModel):
    fetched_at: str
    meta: MetaOut
    squad: list[dict[str, Any]]
    squad_eval: dict[str, Any]
    notes: list[NoteOut]
    transfers: list[dict[str, Any]]
    transfer_plan: PlanOut | None
    plans: dict[str, PlanOut]
    underpriced: list[dict[str, Any]]
    unorthodox: list[dict[str, Any]]
    leaders: dict[str, list[dict[str, Any]]]
    strategies: list[StrategyOut]
    xi_ids: list[int]
    bench_ids: list[int]
    warnings: list[str] = Field(default_factory=list)


def player_record(row: Any) -> dict[str, Any]:
    if isinstance(row, pd.Series):
        data = row.to_dict()
    elif hasattr(row, "_asdict"):
        data = row._asdict()
    else:
        data = dict(row)
    out: dict[str, Any] = {}
    for key in PLAYER_KEYS:
        if key not in data:
            continue
        value = data[key]
        if key in {
            "price",
            "ownership",
            "xp_gw",
            "xp_horizon",
            "ppp",
            "consistency",
            "balanced",
            "aggressive",
            "template",
            "residual",
            "differential",
            "minutes_prob",
            "role_risk",
            "effective_minutes",
            "fdr_mean",
            "xg_p90",
            "xa_p90",
            "xgi_p90",
            "defcon_p90",
            "form",
        }:
            out[key] = _round(value, 2 if key != "price" else 1)
        elif key in {"unorthodox", "value_flag", "set_piece", "can_select"}:
            out[key] = bool(value) if not pd.isna(value) else False
        elif key in {
            "id",
            "element_type",
            "team_id",
            "team_code",
            "code",
            "event_points",
            "total_points",
            "minutes",
            "starts",
            "bonus",
            "bps",
            "gw_minutes",
        }:
            number = _finite(value)
            out[key] = int(number) if number is not None else 0
        elif key == "chance_next":
            number = _finite(value)
            out[key] = int(number) if number is not None else None
        elif key == "penalties_order":
            number = _finite(value)
            out[key] = int(number) if number is not None else None
        elif isinstance(value, str):
            out[key] = value
        elif pd.isna(value):
            out[key] = None
        else:
            out[key] = _finite(value)
    return attach_assets(out)


def players_payload(frame: pd.DataFrame) -> list[dict[str, Any]]:
    if frame is None or frame.empty:
        return []
    return [player_record(row) for row in frame.to_dict(orient="records")]


TRANSFER_KEYS = [
    "out_id",
    "out",
    "out_team",
    "in_id",
    "in",
    "in_team",
    "position",
    "out_price",
    "in_price",
    "cost_delta",
    "d_balanced",
    "d_xp",
    "d_ppp",
    "d_cons",
    "in_own",
    "unorthodox",
]


def transfer_records(frame: pd.DataFrame) -> list[dict[str, Any]]:
    if frame is None or frame.empty:
        return []
    rows = []
    for rec in frame.to_dict(orient="records"):
        row: dict[str, Any] = {}
        for key in TRANSFER_KEYS:
            if key not in rec:
                continue
            value = rec[key]
            if key == "unorthodox":
                row[key] = bool(value) if not pd.isna(value) else False
            elif key in {"out", "in", "out_team", "in_team", "position"}:
                row[key] = None if pd.isna(value) else str(value)
            elif key in {"out_id", "in_id"}:
                number = _finite(value)
                row[key] = int(number) if number is not None else 0
            else:
                digits = 1 if key in {"cost_delta", "out_price", "in_price", "in_own"} else 2
                row[key] = _round(value, digits)
        rows.append(row)
    return rows


def swap_payloads(outgoing: pd.DataFrame, incoming: pd.DataFrame) -> list[dict[str, Any]]:
    rows = []
    for out_row, in_row in pair_swaps(outgoing, incoming):
        rows.append(
            {
                "out": player_record(out_row) if out_row is not None else None,
                "in": player_record(in_row) if in_row is not None else None,
            }
        )
    return rows


def squad_diff(current: pd.DataFrame, planned: pd.DataFrame) -> dict[str, Any]:
    current_ids = set(current["id"]) if current is not None and not current.empty else set()
    plan_ids = set(planned["id"]) if planned is not None and not planned.empty else set()
    incoming = planned.loc[~planned["id"].isin(current_ids)] if planned is not None and not planned.empty else planned
    outgoing = current.loc[~current["id"].isin(plan_ids)] if current is not None and not current.empty else current
    incoming_df = incoming if incoming is not None else pd.DataFrame()
    outgoing_df = outgoing if outgoing is not None else pd.DataFrame()
    return {
        "incoming": players_payload(incoming_df),
        "outgoing": players_payload(outgoing_df),
        "swaps": swap_payloads(outgoing_df, incoming_df),
        "n_transfers": int(len(incoming_df)) if incoming_df is not None and not incoming_df.empty else 0,
    }


def plan_payload(plan: SquadPlan) -> PlanOut:
    xi_set = set(plan.xi_ids)
    records = players_payload(plan.players)
    return PlanOut(
        objective=plan.objective,
        cost=round(float(plan.cost), 1),
        xp_gw=round(float(plan.xp_gw), 2),
        xp_horizon=round(float(plan.xp_horizon), 2),
        ppp=round(float(plan.ppp), 2),
        consistency=round(float(plan.consistency), 2),
        xi=plan.players.loc[plan.players["id"].isin(xi_set), "web_name"].tolist(),
        bench=plan.players.loc[plan.players["id"].isin(plan.bench_ids), "web_name"].tolist(),
        xi_ids=[int(i) for i in plan.xi_ids],
        bench_ids=[int(i) for i in plan.bench_ids],
        players=records,
    )


def analysis_payload(bundle: AnalysisBundle) -> AnalysisOut:
    from fpl_analytics.optimiser import pick_xi

    xi_ids = list(getattr(bundle, "xi_ids", None) or [])
    bench_ids = list(getattr(bundle, "bench_ids", None) or [])
    if not xi_ids and not bundle.squad.empty:
        xi_ids, bench_ids = pick_xi(bundle.squad)
    meta = bundle.meta
    return AnalysisOut(
        fetched_at=bundle.fetched_at,
        meta=MetaOut(
            next_event=int(meta.next_event),
            current_event=int(meta.current_event) if meta.current_event else None,
            deadline=meta.deadline,
            total_managers=int(meta.total_managers),
            budget=float(meta.budget),
            team_limit=int(meta.team_limit),
            squad_size=int(meta.squad_size),
            season_started=bool(meta.season_started),
            horizon=bundle.horizon,
            bank=float(bundle.spec.bank),
            free_transfers=int(bundle.spec.free_transfers),
        ),
        squad=players_payload(bundle.squad),
        squad_eval=_json_value(bundle.squad_eval),
        notes=[NoteOut(**n) for n in player_notes(bundle.squad)],
        transfers=transfer_records(bundle.transfers),
        transfer_plan=plan_payload(bundle.transfer_plan) if bundle.transfer_plan else None,
        plans={k: plan_payload(v) for k, v in bundle.plans.items()},
        underpriced=players_payload(bundle.underpriced(15)),
        unorthodox=players_payload(bundle.unorthodox(15)),
        leaders={
            pos: players_payload(bundle.leaders("balanced", 10, pos))
            for pos in ("GKP", "DEF", "MID", "FWD")
        },
        strategies=[StrategyOut(**s) for s in strategies()],
        xi_ids=[int(i) for i in xi_ids],
        bench_ids=[int(i) for i in bench_ids],
        warnings=list(bundle.spec.warnings),
    )
