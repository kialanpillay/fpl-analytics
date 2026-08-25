"""End-to-end load → feature → score → evaluate pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from fpl_analytics.api import FPLClient
from fpl_analytics.live import fetch_official_picks, resolve_manager_id
from fpl_analytics.catalog import (
    apply_event_live,
    fixtures_frame,
    players_frame,
    season_meta,
    teams_frame,
)
from fpl_analytics.features import enrich
from fpl_analytics.models import score
from fpl_analytics.optimiser import (
    DEFAULT_OBJECTIVES,
    SquadPlan,
    one_for_one,
    optimise_squad,
    pick_xi,
    suggest_transfers,
)
from fpl_analytics.research import STRATEGIES, note_for
from fpl_analytics.squad import (
    DEFAULT_SQUAD_PATH,
    ManagerSquad,
    evaluate_squad,
    load_squad,
    resolve_squad,
)


@dataclass
class AnalysisBundle:
    fetched_at: str
    meta: Any
    players: pd.DataFrame
    squad: pd.DataFrame
    squad_eval: dict[str, Any]
    spec: ManagerSquad
    plans: dict[str, SquadPlan] = field(default_factory=dict)
    transfers: pd.DataFrame = field(default_factory=pd.DataFrame)
    transfer_plan: SquadPlan | None = None
    horizon: int = 6
    xi_ids: list[int] = field(default_factory=list)
    bench_ids: list[int] = field(default_factory=list)
    official: dict[str, Any] | None = None

    def available(self) -> pd.DataFrame:
        return self.players.loc[
            self.players["can_select"] & (self.players["status"].isin(["a", "d"]))
        ]

    def leaders(self, sort: str, n: int = 12, position: str | None = None) -> pd.DataFrame:
        frame = self.available()
        if position:
            frame = frame.loc[frame["position"] == position]
        return frame.sort_values(sort, ascending=False).head(n)

    def unorthodox(self, n: int = 15) -> pd.DataFrame:
        pool = self.available()
        return pool.loc[pool["unorthodox"]].sort_values("differential", ascending=False).head(n)

    def underpriced(self, n: int = 15) -> pd.DataFrame:
        return (
            self.available()
            .loc[self.available()["effective_minutes"] >= 0.40]
            .sort_values("residual", ascending=False)
            .head(n)
        )


def run_pipeline(
    squad_path: Path | str = DEFAULT_SQUAD_PATH,
    horizon: int = 6,
    force_refresh: bool = False,
    objectives: tuple[str, ...] = DEFAULT_OBJECTIVES,
    max_transfers: int = 2,
    bank: float | None = None,
    free_transfers: int | None = None,
    transfer_objective: str = "balanced",
) -> AnalysisBundle:
    client = FPLClient()
    bootstrap = client.bootstrap(force=force_refresh)
    fixtures_raw = client.fixtures(force=force_refresh)
    meta = season_meta(bootstrap)
    early_season = (meta.current_event or 0) <= 3
    teams = teams_frame(bootstrap)
    fixtures = fixtures_frame(fixtures_raw)
    gw = meta.current_event or 1
    live = client.event_live(gw, force=force_refresh)
    raw_players = apply_event_live(players_frame(bootstrap, teams), live)
    players = score(
        enrich(
            raw_players,
            fixtures,
            teams,
            next_event=meta.next_event,
            season_started=meta.season_started,
            horizon=horizon,
            early_season=early_season,
        )
    )

    spec = load_squad(squad_path)
    if bank is not None:
        spec.bank = float(bank)
    if free_transfers is not None:
        spec.free_transfers = int(free_transfers)
    squad = resolve_squad(spec, players)
    ev = evaluate_squad(squad, team_limit=meta.team_limit)
    xi_ids, bench_ids = pick_xi(squad) if not squad.empty else ([], [])

    plans: dict[str, SquadPlan] = {}
    for obj in objectives:
        try:
            plans[obj] = optimise_squad(
                players, objective=obj, budget=spec.budget, team_limit=meta.team_limit
            )
        except Exception as exc:  # keep the rest of the report usable
            plans[obj] = exc  # type: ignore[assignment]

    transfers = one_for_one(squad, players, bank=spec.bank, team_limit=meta.team_limit)
    try:
        transfer_plan = suggest_transfers(
            squad,
            players,
            max_transfers=max_transfers,
            objective=transfer_objective,
            budget=spec.budget,
            bank=spec.bank,
            team_limit=meta.team_limit,
        )
    except Exception:
        transfer_plan = None

    official = fetch_official_picks(client, resolve_manager_id(), gw, force=force_refresh)

    return AnalysisBundle(
        fetched_at=datetime.now(timezone.utc).isoformat(),
        meta=meta,
        players=players,
        squad=squad,
        squad_eval=ev,
        spec=spec,
        plans={k: v for k, v in plans.items() if not isinstance(v, Exception)},
        transfers=transfers,
        transfer_plan=transfer_plan,
        horizon=horizon,
        xi_ids=[int(i) for i in xi_ids],
        bench_ids=[int(i) for i in bench_ids],
        official=official,
    )


def player_notes(squad: pd.DataFrame) -> list[dict[str, str]]:
    rows = []
    for rec in squad.itertuples(index=False):
        note = note_for(rec.web_name)
        if note:
            rows.append(
                {
                    "name": rec.web_name,
                    "tone": note.tone,
                    "note": note.note,
                }
            )
    return rows


def strategies() -> list[dict[str, str]]:
    return STRATEGIES
