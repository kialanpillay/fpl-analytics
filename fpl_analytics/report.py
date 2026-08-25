"""Human-readable reports and JSON export for season-long use."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from fpl_analytics.captaincy import rank_captaincy
from fpl_analytics.live import players_for_ids
from fpl_analytics.optimiser import SquadPlan, pair_swaps, pick_xi
from fpl_analytics.pipeline import AnalysisBundle, player_notes, strategies
from fpl_analytics.schemas import analysis_payload

COLS = [
    "web_name",
    "position",
    "team_short",
    "price",
    "ownership",
    "event_points",
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
    "aggressive": "Aggressive",
    "template": "Template",
    "ppp": "PPP",
    "consistency": "Consistency",
    "differential": "Differential",
    "xp": "xPts",
}


def _gw_review(bundle: AnalysisBundle) -> list[str]:
    """Finished-GW actuals. Prefer official picks/score when the pipeline fetched them."""
    squad = bundle.squad.copy()
    if "event_points" not in squad.columns or squad.empty:
        return []
    gw = bundle.meta.current_event or bundle.meta.next_event
    official = getattr(bundle, "official", None)
    catalog = bundle.players if bundle.players is not None and not bundle.players.empty else squad
    recs = rank_captaincy(squad, list(bundle.xi_ids or pick_xi(squad)[0]), n=2)
    rec_c = recs.iloc[0] if not recs.empty else None
    rec_label = rec_c.web_name if rec_c is not None else "—"
    parts = [
        "",
        f"GW{gw} Actuals",
        "-" * 88,
        _fmt(squad.sort_values(["element_type", "event_points"], ascending=[True, False]), extra=["bonus", "bps"]),
    ]
    if official and official.get("xi_ids"):
        xi = players_for_ids(official["xi_ids"], catalog, squad)
        bench = players_for_ids(official.get("bench_ids") or [], catalog, squad)
        cap_id = official.get("captain_id")
        vice_id = official.get("vice_id")
        parts += ["", "Official XI"]
        for rec in xi.itertuples(index=False):
            badge = " (C)" if rec.id == cap_id else (" (VC)" if rec.id == vice_id else "")
            parts.append(f"  * {rec.web_name:16} {rec.position} {rec.team_short:3}  {int(rec.event_points):2d} pts{badge}")
        for rec in bench.itertuples(index=False):
            badge = " (C)" if rec.id == cap_id else (" (VC)" if rec.id == vice_id else "")
            parts.append(f"    {rec.web_name:16} {rec.position} {rec.team_short:3}  {int(rec.event_points):2d} pts{badge}")
        pts = official.get("points")
        bench_pts = official.get("points_on_bench")
        hits = official.get("hits") or 0
        chip = official.get("chip")
        line = f"Official {pts if pts is not None else '—'} Pts"
        if bench_pts is not None:
            line += f"  ·  Bench {bench_pts}"
        if hits:
            line += f"  ·  −{hits} Hits"
        if chip:
            line += f"  ·  {chip}"
        line += f"  ·  {rec_label} (C rec)"
        parts.append(line)
        for sub in official.get("auto_subs") or []:
            out_name = _name_for(catalog, squad, sub.get("out_id"))
            in_name = _name_for(catalog, squad, sub.get("in_id"))
            parts.append(f"  Auto-sub {out_name} -> {in_name}")
        return parts

    scored = squad.copy()
    scored["xp_gw"] = scored["event_points"].astype(float)
    xi_ids, bench_ids = pick_xi(scored)
    xi = squad.loc[squad["id"].isin(xi_ids)].sort_values("event_points", ascending=False)
    bench = squad.loc[squad["id"].isin(bench_ids)].sort_values("event_points", ascending=False)
    raw = float(xi["event_points"].sum())
    best_row = squad.loc[squad["event_points"].idxmax()]
    rec_pts = float(squad.loc[squad["id"] == rec_c.id, "event_points"].iloc[0]) if rec_c is not None else 0.0
    parts += ["", "Retrospective best XI (formation rules, by GW points)"]
    for rec in xi.itertuples(index=False):
        parts.append(f"  * {rec.web_name:16} {rec.position} {rec.team_short:3}  {int(rec.event_points):2d} pts")
    for rec in bench.itertuples(index=False):
        parts.append(f"    {rec.web_name:16} {rec.position} {rec.team_short:3}  {int(rec.event_points):2d} pts")
    parts.append(
        f"XI {raw:.0f}  ·  {rec_label} (C rec) {raw + rec_pts:.0f}  ·  "
        f"{best_row.web_name} (C actual) {raw + float(best_row.event_points):.0f}"
    )
    return parts


def _name_for(catalog: pd.DataFrame, squad: pd.DataFrame, player_id: int | None) -> str:
    if player_id is None:
        return "—"
    hit = players_for_ids([int(player_id)], catalog, squad)
    if hit.empty:
        return str(player_id)
    return str(hit.iloc[0]["web_name"])


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
    parts += _gw_review(bundle)
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
        for out_row, in_row in pair_swaps(outgoing, incoming):
            if out_row is None or in_row is None:
                name = (in_row or out_row)["web_name"]
                parts.append(f"  {name}")
                continue
            parts.append(
                f"  {out_row.web_name} £{out_row.price:.1f}  ->  {in_row.web_name} {in_row.team_short} £{in_row.price:.1f}  "
                f"(xH {in_row.xp_horizon:.1f} vs {out_row.xp_horizon:.1f})"
            )
        parts.append(_plan_lines(bundle.transfer_plan))

    for key, title in (
        ("balanced", "Wildcard — Balanced"),
        ("aggressive", "Wildcard — Aggressive"),
        ("template", "Wildcard — Template"),
        ("ppp", "Wildcard — Points Per Pound"),
        ("consistency", "Wildcard — Consistency"),
        ("differential", "Wildcard — Differential"),
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


def export_json(bundle: AnalysisBundle, path: Path | str) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = analysis_payload(bundle)
    path.write_text(payload.model_dump_json(indent=2))
    return path
