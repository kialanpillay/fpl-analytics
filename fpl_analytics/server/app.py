"""FastAPI application. Sync analysis; no job queue."""

from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from fpl_analytics import __version__
from fpl_analytics.api import DEFAULT_CACHE_DIR, FPLClient
from fpl_analytics.captaincy import rank_captaincy
from fpl_analytics.live import fetch_official_picks, players_for_ids
from fpl_analytics.optimiser import (
    DEFAULT_OBJECTIVES,
    OBJECTIVES,
    hit_scenarios,
    optimise_squad,
    pulp,
    suggest_transfers,
)
from fpl_analytics.pipeline import player_notes, strategies
from fpl_analytics.research import note_for
from fpl_analytics.schemas import (
    analysis_payload,
    plan_payload,
    player_record,
    players_payload,
    round_number,
    squad_diff,
)
from fpl_analytics.server.settings import load_ui_settings, save_ui_settings
from fpl_analytics.server.state import STATE
from fpl_analytics.squad import SquadEntry

ROOT = Path(__file__).resolve().parent.parent.parent
WEB_DIST = ROOT / "web" / "dist"

app = FastAPI(title="FPL Analytics", version=__version__)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:5174",
        "http://127.0.0.1:5174",
        "http://localhost:8009",
        "http://127.0.0.1:8009",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class RunBody(BaseModel):
    horizon: int | None = None
    refresh: bool = False
    max_transfers: int = 2
    objectives: list[str] = Field(default_factory=lambda: list(DEFAULT_OBJECTIVES))
    bank: float | None = None
    free_transfers: int | None = None


class SquadWriteBody(BaseModel):
    players: list[dict]
    bank: float | None = None
    free_transfers: int | None = None
    budget: float | None = None


class ImportEntryBody(BaseModel):
    manager_id: int


class DraftSolveBody(BaseModel):
    objective: str = "balanced"
    locked_ids: list[int] = Field(default_factory=list)
    banned_ids: list[int] = Field(default_factory=list)
    budget: float | None = None


class SettingsBody(BaseModel):
    horizon: int | None = None


class ApplyTransfersBody(BaseModel):
    use_plan: bool = True
    players: list[dict] | None = None


class TransferSolveBody(BaseModel):
    objective: str = "balanced"
    max_transfers: int | None = None


def _cache_age() -> float | None:
    path = DEFAULT_CACHE_DIR / "bootstrap-static.json"
    if not path.exists():
        return None
    import time

    return time.time() - path.stat().st_mtime


@app.get("/api/health")
def health() -> dict:
    bundle = STATE.bundle
    return {
        "ok": True,
        "version": __version__,
        "cache_age_seconds": _cache_age(),
        "fetched_at": bundle.fetched_at if bundle else None,
        "pulp": pulp is not None,
        "solving": STATE.solving,
        "error": STATE.last_error,
    }


@app.get("/api/meta")
def meta() -> dict:
    bundle = STATE.get_bundle()
    payload = analysis_payload(bundle)
    return payload.meta.model_dump()


@app.get("/api/settings")
def get_settings() -> dict:
    ui = load_ui_settings()
    spec = STATE.bundle.spec if STATE.bundle is not None else None
    return {
        **ui,
        "bank": spec.bank if spec else 0.0,
        "free_transfers": spec.free_transfers if spec else 1,
        "budget": spec.budget if spec else 100.0,
    }


@app.put("/api/settings")
def put_settings(body: SettingsBody) -> dict:
    return save_ui_settings(horizon=body.horizon)


@app.get("/api/analysis")
def get_analysis() -> dict:
    return analysis_payload(STATE.get_bundle()).model_dump()


@app.post("/api/analysis/run")
def run_analysis(body: RunBody | None = None) -> dict:
    body = body or RunBody()
    unknown = [o for o in body.objectives if o not in OBJECTIVES]
    if unknown:
        raise HTTPException(400, f"Unknown objectives: {unknown}")
    bundle = STATE.run(
        horizon=body.horizon,
        force_refresh=body.refresh,
        max_transfers=body.max_transfers,
        objectives=tuple(body.objectives),
        bank=body.bank,
        free_transfers=body.free_transfers,
    )
    return analysis_payload(bundle).model_dump()


@app.get("/api/squad")
def get_squad() -> dict:
    bundle = STATE.get_bundle()
    payload = analysis_payload(bundle)
    return {
        "squad": payload.squad,
        "eval": payload.squad_eval,
        "notes": [n.model_dump() for n in payload.notes],
        "xi_ids": payload.xi_ids,
        "bench_ids": payload.bench_ids,
        "warnings": payload.warnings,
        "bank": bundle.spec.bank,
        "free_transfers": bundle.spec.free_transfers,
        "budget": bundle.spec.budget,
    }


@app.put("/api/squad")
def put_squad(body: SquadWriteBody) -> dict:
    ids = []
    for raw in body.players:
        player_id = raw.get("id")
        if player_id is None:
            raise HTTPException(400, "Each player needs an id")
        ids.append(int(player_id))
    bundle = STATE.run(
        squad_ids=ids,
        bank=body.bank,
        free_transfers=body.free_transfers,
    )
    return analysis_payload(bundle).model_dump()


@app.post("/api/squad/import-entry")
def import_entry(body: ImportEntryBody) -> dict:
    client = FPLClient()
    try:
        entry = client.entry(body.manager_id, force=True)
    except Exception as exc:
        raise HTTPException(404, f"Manager {body.manager_id} not found: {exc}") from exc
    current = entry.get("current_event") or entry.get("started_event") or 1
    picks_payload = None
    last_error = None
    for gw in (current, current - 1 if current > 1 else None):
        if gw is None:
            continue
        try:
            picks_payload = client.entry_picks(body.manager_id, int(gw), force=True)
            break
        except Exception as exc:
            last_error = exc
    if picks_payload is None:
        raise HTTPException(502, f"Could not load picks: {last_error}")
    picks = picks_payload.get("picks") or []
    if len(picks) < 15:
        raise HTTPException(502, f"Expected 15 picks, got {len(picks)}")
    bundle = STATE.run(force_refresh=True)
    return {
        "manager_id": body.manager_id,
        "team_name": entry.get("name"),
        "player_name": f"{entry.get('player_first_name', '')} {entry.get('player_last_name', '')}".strip(),
        "analysis": analysis_payload(bundle).model_dump(),
    }


def _transfers_payload(bundle, plan, objective: str) -> dict:
    payload = analysis_payload(bundle)
    diff = {"incoming": [], "outgoing": [], "swaps": [], "n_transfers": 0}
    hits = 0
    lift = 0.0
    plan_out = None
    if plan:
        diff = squad_diff(bundle.squad, plan.players)
        n = int(diff["n_transfers"])
        hits = max(0, n - int(bundle.spec.free_transfers)) * 4
        lift = float(plan.xp_horizon) - float(bundle.squad_eval["xp_horizon"])
        plan_out = plan_payload(plan).model_dump()
    return {
        "one_for_one": payload.transfers,
        "plan": plan_out,
        "incoming": diff["incoming"],
        "outgoing": diff["outgoing"],
        "swaps": diff["swaps"],
        "n_transfers": diff["n_transfers"],
        "free_transfers": bundle.spec.free_transfers,
        "bank": bundle.spec.bank,
        "hits": hits,
        "horizon_lift": round(lift, 2),
        "hit_table": hit_scenarios(lift, int(diff["n_transfers"])),
        "objective": objective,
    }


@app.get("/api/transfers")
def get_transfers() -> dict:
    bundle = STATE.get_bundle()
    return _transfers_payload(bundle, bundle.transfer_plan, "balanced")


@app.post("/api/transfers/solve")
def solve_transfers(body: TransferSolveBody | None = None) -> dict:
    body = body or TransferSolveBody()
    if body.objective not in OBJECTIVES:
        raise HTTPException(400, f"Unknown objective {body.objective}")
    if pulp is None:
        raise HTTPException(503, "pulp is required for optimisation")
    bundle = STATE.get_bundle()
    max_transfers = body.max_transfers or int(STATE.last_params.get("max_transfers") or 2)
    try:
        plan = suggest_transfers(
            bundle.squad,
            bundle.players,
            max_transfers=max_transfers,
            objective=body.objective,
            budget=bundle.spec.budget,
            bank=bundle.spec.bank,
            team_limit=bundle.meta.team_limit,
        )
    except Exception as exc:
        if "No improving" in str(exc) or "constraints" in str(exc).lower():
            return _transfers_payload(bundle, None, body.objective)
        raise HTTPException(422, str(exc)) from exc
    return _transfers_payload(bundle, plan, body.objective)


@app.post("/api/transfers/apply")
def apply_transfers(body: ApplyTransfersBody | None = None) -> dict:
    body = body or ApplyTransfersBody()
    bundle = STATE.get_bundle()
    if body.players:
        entries = [SquadEntry(int(p["id"]), p.get("name")) for p in body.players]
    elif body.use_plan and bundle.transfer_plan:
        entries = [
            SquadEntry(int(row.id), row.web_name)
            for row in bundle.transfer_plan.players.itertuples(index=False)
        ]
    else:
        raise HTTPException(400, "No transfer plan to apply")
    ids = [int(e.player_id) for e in entries if e.player_id is not None]
    return analysis_payload(STATE.run(squad_ids=ids)).model_dump()


@app.get("/api/captaincy")
def get_captaincy() -> dict:
    bundle = STATE.get_bundle()
    ranked = rank_captaincy(bundle.squad, bundle.xi_ids)
    rows = []
    for i, rec in enumerate(ranked.to_dict(orient="records"), start=1):
        item = player_record(rec)
        item["captain_score"] = round_number(rec.get("captain_score"))
        item["captain_ev"] = round_number(rec.get("captain_ev"))
        item["vice_ev"] = round_number(rec.get("vice_ev"))
        item["season_default"] = bool(rec.get("season_default"))
        item["rank"] = i
        item["role"] = "C" if i == 1 else ("VC" if i == 2 else "—")
        rows.append(item)
    recommended = rows[0] if rows else None
    vice = rows[1] if len(rows) > 1 else None
    return {
        "recommended": recommended,
        "vice": vice,
        "options": rows,
    }


@app.get("/api/players")
def list_players(
    position: str | None = None,
    sort: str = "balanced",
    n: int = 400,
    available: bool = True,
) -> dict:
    bundle = STATE.get_bundle()
    frame = bundle.available() if available else bundle.players
    if position:
        frame = frame.loc[frame["position"] == position]
    if sort not in frame.columns:
        raise HTTPException(400, f"Unknown sort column {sort}")
    frame = frame.sort_values(sort, ascending=False).head(n)
    return {"players": players_payload(frame), "sort": sort, "n": int(len(frame))}


@app.get("/api/players/{player_id}")
def get_player(player_id: int, refresh: bool = False) -> dict:
    bundle = STATE.get_bundle()
    hit = bundle.players.loc[bundle.players["id"] == player_id]
    if hit.empty:
        raise HTTPException(404, f"Player {player_id} not found")
    player = player_record(hit.iloc[0])
    note = note_for(player["web_name"])
    client = FPLClient()
    try:
        summary = client.element_summary(player_id, force=refresh)
    except Exception as exc:
        raise HTTPException(502, f"element-summary failed: {exc}") from exc
    history = summary.get("history") or []
    fixtures = _named_fixtures(summary.get("fixtures") or [], bundle.players)
    history_past = summary.get("history_past") or []
    return {
        "player": player,
        "note": {"tone": note.tone, "note": note.note} if note else None,
        "in_squad": bool(player_id in set(bundle.squad["id"])),
        "history": history,
        "fixtures": fixtures,
        "history_past": history_past,
    }


def _named_fixtures(raw: list[dict], players) -> list[dict]:
    """element-summary fixtures have team_h / team_a ids, not opponent names."""
    names = {
        int(tid): str(short)
        for tid, short in players.drop_duplicates("team_id")
        .set_index("team_id")["team_short"]
        .items()
    }
    rows = []
    for fx in raw:
        home = bool(fx.get("is_home"))
        try:
            opp_id = int(fx["team_a"] if home else fx["team_h"])
        except (KeyError, TypeError, ValueError):
            opp_id = None
        rows.append(
            {
                "event": fx.get("event"),
                "opponent": names.get(opp_id),
                "is_home": home,
                "difficulty": fx.get("difficulty"),
                "kickoff": fx.get("kickoff_time"),
            }
        )
    return rows


@app.get("/api/wildcard")
@app.get("/api/drafts")
def get_wildcard() -> dict:
    bundle = STATE.get_bundle()
    payload = analysis_payload(bundle)
    return {
        "plans": {k: v.model_dump() for k, v in payload.plans.items()},
        "pulp": pulp is not None,
    }


@app.post("/api/wildcard/solve")
@app.post("/api/drafts/solve")
def solve_wildcard(body: DraftSolveBody) -> dict:
    if pulp is None:
        raise HTTPException(503, "pulp is required for optimisation")
    if body.objective not in OBJECTIVES:
        raise HTTPException(400, f"Unknown objective {body.objective}")
    bundle = STATE.get_bundle()
    try:
        plan = optimise_squad(
            bundle.players,
            objective=body.objective,
            budget=body.budget if body.budget is not None else bundle.spec.budget,
            team_limit=bundle.meta.team_limit,
            locked_ids=set(body.locked_ids),
            banned_ids=set(body.banned_ids),
        )
    except Exception as exc:
        raise HTTPException(422, str(exc)) from exc
    diff = squad_diff(bundle.squad, plan.players)
    return {
        "plan": plan_payload(plan).model_dump(),
        **diff,
    }


@app.get("/api/fixtures")
def get_fixtures(horizon: int | None = None) -> dict:
    from fpl_analytics.assets import badge_url
    from fpl_analytics.catalog import fixtures_frame, season_meta, teams_frame

    client = FPLClient()
    bootstrap = client.bootstrap()
    raw = client.fixtures()
    teams = teams_frame(bootstrap)
    fixtures = fixtures_frame(raw)
    meta = season_meta(bootstrap)
    start = meta.next_event
    span = horizon or load_ui_settings()["horizon"]
    events = list(range(start, start + span))
    team_rows = []
    cells = []
    short = teams.set_index("team_id")
    for rec in teams.itertuples(index=False):
        team_rows.append(
            {
                "team_id": int(rec.team_id),
                "team": rec.team,
                "team_short": rec.team_short,
                "team_code": int(rec.team_code),
                "badge_url": badge_url(int(rec.team_code)),
            }
        )
        upcoming = fixtures.loc[
            (fixtures["event"] >= start)
            & (fixtures["event"] < start + span)
            & ((fixtures["team_h"] == rec.team_id) | (fixtures["team_a"] == rec.team_id))
        ]
        for fx in upcoming.itertuples(index=False):
            home = fx.team_h == rec.team_id
            opp_id = fx.team_a if home else fx.team_h
            opp = short.loc[opp_id, "team_short"] if opp_id in short.index else "?"
            cells.append(
                {
                    "team_id": int(rec.team_id),
                    "event": int(fx.event),
                    "fdr": int(fx.fdr_h if home else fx.fdr_a),
                    "home": bool(home),
                    "opponent": str(opp),
                    "kickoff": fx.kickoff,
                    "finished": bool(fx.finished),
                }
            )
    return {
        "next_event": start,
        "events": events,
        "teams": team_rows,
        "cells": cells,
    }


@app.get("/api/live")
def get_live(refresh: bool = False) -> dict:
    from fpl_analytics.catalog import apply_event_live

    bundle = STATE.get_bundle()
    client = FPLClient()
    gw = bundle.meta.current_event or bundle.meta.next_event
    live = client.event_live(int(gw), force=refresh)
    status = {}
    try:
        status = client.event_status(force=refresh)
    except Exception:
        status = {}
    ui = load_ui_settings()
    entry = None
    if ui.get("manager_id"):
        entry = fetch_official_picks(client, int(ui["manager_id"]), int(gw), force=refresh)
    pick_ids = (entry["xi_ids"] + entry["bench_ids"]) if entry else [int(i) for i in bundle.squad["id"]]
    resolved = players_for_ids(pick_ids, bundle.players, bundle.squad)
    squad = apply_event_live(resolved if not resolved.empty else bundle.squad, live)
    return {
        "event": int(gw),
        "deadline": bundle.meta.deadline,
        "status": status,
        "squad": players_payload(squad),
        "xi_ids": entry["xi_ids"] if entry else bundle.xi_ids,
        "bench_ids": entry["bench_ids"] if entry else bundle.bench_ids,
        "captain_id": entry["captain_id"] if entry else None,
        "vice_id": entry["vice_id"] if entry else None,
        "source": "entry" if entry else "model",
        "manager_id": ui.get("manager_id"),
        "official": (
            {
                "points": entry["points"],
                "points_on_bench": entry["points_on_bench"],
                "total_points": entry["total_points"],
                "rank": entry["rank"],
                "overall_rank": entry["overall_rank"],
                "transfers": entry["transfers"],
                "hits": entry["hits"],
                "chip": entry["chip"],
                "auto_subs": entry["auto_subs"],
            }
            if entry
            else None
        ),
        "notes": player_notes(squad),
    }


@app.get("/api/strategies")
def get_strategies() -> list[dict]:
    return strategies()


def main() -> None:
    import os
    import uvicorn

    reload = os.environ.get("FPL_WEB_RELOAD", "").lower() in {"1", "true", "yes"}
    uvicorn.run("fpl_analytics.server.app:app", host="127.0.0.1", port=8009, reload=reload)


if WEB_DIST.is_dir():
    app.mount("/", StaticFiles(directory=WEB_DIST, html=True), name="ui")
else:

    @app.get("/", response_class=HTMLResponse)
    def root_hint() -> str:
        return (
            "<!doctype html><meta charset=utf-8><title>FPL Analytics</title>"
            "<body style='font:14px/1.5 system-ui;background:#0b0f14;color:#e8eef7;padding:2rem'>"
            "<p>API is up. In development run <code>cd web && npm run dev</code> and open "
            "<a href='http://127.0.0.1:5173' style='color:#3ee0a2'>http://127.0.0.1:5173</a>.</p>"
            "<p>Or <code>cd web && npm run build</code> and restart this server to serve the UI here.</p>"
        )
