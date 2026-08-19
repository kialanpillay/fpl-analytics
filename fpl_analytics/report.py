"""Human-readable reports and JSON export for season-long use."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from fpl_analytics.optimiser import SquadPlan
from fpl_analytics.pipeline import AnalysisBundle, player_notes, strategies

COLS = [
    "web_name",
    "position",
    "team_short",
    "price",
    "ownership",
    "xp_gw",
    "xp_horizon",
    "ppp",
    "consistency",
    "balanced",
    "residual",
    "minutes_prob",
    "next_fixture",
]


def _fmt(frame: pd.DataFrame, extra: list[str] | None = None) -> str:
    cols = COLS + [c for c in (extra or []) if c not in COLS]
    view = frame.loc[:, [c for c in cols if c in frame.columns]].copy()
    for col in ("price", "ownership", "xp_gw", "xp_horizon", "ppp", "consistency", "balanced", "residual", "minutes_prob"):
        if col in view:
            view[col] = view[col].map(lambda x: f"{x:.2f}")
    return view.to_string(index=False)


OBJECTIVE_LABELS = {
    "balanced": "Balanced",
    "ppp": "PPP",
    "consistency": "Consistency",
    "differential": "Differential",
    "xp": "xPts",
}


def _plan_lines(plan: SquadPlan) -> str:
    names = []
    xi = set(plan.xi_ids)
    for rec in plan.players.itertuples(index=False):
        mark = "*" if rec.id in xi else " "
        names.append(f"{mark} {rec.web_name:16} {rec.position} {rec.team_short:3} £{rec.price:.1f}")
    label = OBJECTIVE_LABELS.get(plan.objective, plan.objective)
    header = (
        f"{label}: £{plan.cost:.1f}  XI xPts {plan.xp_gw:.1f}  "
        f"Horizon {plan.xp_horizon:.1f}  PPP {plan.ppp:.2f}  Cons {plan.consistency:.2f}"
    )
    return header + "\n" + "\n".join(names)


def render_text(bundle: AnalysisBundle) -> str:
    meta = bundle.meta
    ev = bundle.squad_eval
    parts = [
        "FPL Analytics",
        "=" * 88,
        f"Fetched {bundle.fetched_at}   Next GW {meta.next_event}   Deadline {meta.deadline}",
        f"Managers {meta.total_managers:,}   Budget £{meta.budget:.1f}m   Horizon {bundle.horizon} GW",
        "",
        "Squad",
        "-" * 88,
        _fmt(bundle.squad),
        "",
        (
            f"Cost £{ev['cost']:.1f}m  Bank £{bundle.spec.bank:.1f}m  "
            f"Horizon xPts {ev['xp_horizon']:.1f}  Mean PPP {ev['ppp']:.2f}  "
            f"Mean Cons {ev['consistency']:.2f}"
        ),
        f"Club Counts: {ev['club_counts']}",
        f"Dead Slots: {ev['dead_slots'] or 'none'}",
        f"Role Risk: {ev['risk_names'] or 'none'}",
    ]
    if ev["illegal_clubs"]:
        parts.append(f"Illegal Club Cap: {ev['illegal_clubs']}")
    if bundle.spec.warnings:
        parts += [f"Squad File: {w}" for w in bundle.spec.warnings]

    notes = player_notes(bundle.squad)
    if notes:
        parts += ["", "Squad Notes", "-" * 88]
        for n in notes:
            parts.append(f"[{n['tone']}] {n['name']}: {n['note']}")

    if not bundle.transfers.empty:
        parts += ["", "Legal 1-For-1 Transfers", "-" * 88]
        show = bundle.transfers[
            ["out", "in", "position", "cost_delta", "d_balanced", "d_xp", "d_ppp", "in_own", "unorthodox"]
        ].copy()
        parts.append(show.to_string(index=False, formatters={
            "cost_delta": "{:+.1f}".format,
            "d_balanced": "{:+.2f}".format,
            "d_xp": "{:+.2f}".format,
            "d_ppp": "{:+.2f}".format,
            "in_own": "{:.1f}".format,
        }))

    if bundle.transfer_plan:
        current_ids = set(bundle.squad["id"])
        incoming = bundle.transfer_plan.players.loc[~bundle.transfer_plan.players["id"].isin(current_ids)]
        outgoing = bundle.squad.loc[~bundle.squad["id"].isin(set(bundle.transfer_plan.players["id"]))]
        parts += ["", f"Optimised {len(incoming)}-Transfer Plan (Balanced)", "-" * 88]
        for o, i in zip(outgoing.itertuples(), incoming.itertuples()):
            parts.append(
                f"  {o.web_name} £{o.price:.1f}  ->  {i.web_name} {i.team_short} £{i.price:.1f}  "
                f"(xH {i.xp_horizon:.1f} vs {o.xp_horizon:.1f})"
            )
        parts.append(_plan_lines(bundle.transfer_plan))

    for key, title in (
        ("balanced", "Wildcard Draft — Balanced"),
        ("ppp", "Wildcard Draft — Points Per Pound"),
        ("consistency", "Wildcard Draft — Consistency"),
        ("differential", "Wildcard Draft — Differential"),
    ):
        plan = bundle.plans.get(key)
        if plan:
            parts += ["", title, "-" * 88, _plan_lines(plan)]

    parts += ["", "Underpriced vs Price Band", "-" * 88]
    parts.append(_fmt(bundle.underpriced(12)))
    parts += ["", "Unorthodox (<10% Owned)", "-" * 88]
    parts.append(_fmt(bundle.unorthodox(12)))
    parts += ["", "Position Leaders — Balanced", "-" * 88]
    for pos in ("GKP", "DEF", "MID", "FWD"):
        parts += [f"\n{pos}", _fmt(bundle.leaders("balanced", 8, pos))]

    parts += ["", "Season Strategies", "-" * 88]
    for s in strategies():
        parts.append(f"* {s['title']}\n    {s['detail']}")
    return "\n".join(parts) + "\n"


def _plan_json(plan: SquadPlan) -> dict:
    return {
        "objective": plan.objective,
        "cost": plan.cost,
        "xp_gw": plan.xp_gw,
        "xp_horizon": plan.xp_horizon,
        "ppp": plan.ppp,
        "consistency": plan.consistency,
        "xi": plan.players.loc[plan.players["id"].isin(plan.xi_ids), "web_name"].tolist(),
        "bench": plan.players.loc[plan.players["id"].isin(plan.bench_ids), "web_name"].tolist(),
        "players": plan.players[["id", "web_name", "position", "team_short", "price", "ownership", "xp_gw", "xp_horizon", "ppp", "consistency", "balanced"]].to_dict(orient="records"),
    }


def export_json(bundle: AnalysisBundle, path: Path | str) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "fetched_at": bundle.fetched_at,
        "meta": {
            "next_event": bundle.meta.next_event,
            "deadline": bundle.meta.deadline,
            "total_managers": bundle.meta.total_managers,
            "horizon": bundle.horizon,
            "season_started": bundle.meta.season_started,
        },
        "squad": bundle.squad[COLS + ["id", "role_risk", "effective_minutes", "fixture_run", "unorthodox"]].to_dict(orient="records"),
        "squad_eval": bundle.squad_eval,
        "notes": player_notes(bundle.squad),
        "transfers": bundle.transfers.to_dict(orient="records") if not bundle.transfers.empty else [],
        "transfer_plan": _plan_json(bundle.transfer_plan) if bundle.transfer_plan else None,
        "plans": {k: _plan_json(v) for k, v in bundle.plans.items()},
        "underpriced": bundle.underpriced(15)[COLS + ["id", "unorthodox"]].to_dict(orient="records"),
        "unorthodox": bundle.unorthodox(15)[COLS + ["id"]].to_dict(orient="records"),
        "leaders": {
            pos: bundle.leaders("balanced", 10, pos)[COLS + ["id"]].to_dict(orient="records")
            for pos in ("GKP", "DEF", "MID", "FWD")
        },
        "strategies": strategies(),
    }
    path.write_text(json.dumps(payload, indent=2, default=str))
    return path
