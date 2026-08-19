"""Command-line interface for weekly use through the season."""

from __future__ import annotations

import argparse
from pathlib import Path

from fpl_analytics.api import FPLClient
from fpl_analytics.pipeline import run_pipeline
from fpl_analytics.report import export_json, render_text
from fpl_analytics.squad import DEFAULT_SQUAD_PATH


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="fpl",
        description="FPL modelling, value analysis and squad optimisation.",
    )
    p.add_argument("--squad", type=Path, default=DEFAULT_SQUAD_PATH, help="YAML squad file")
    p.add_argument("--horizon", type=int, default=6, help="Gameweeks ahead to score")
    p.add_argument("--refresh", action="store_true", help="Bypass the 30-minute API cache")
    sub = p.add_subparsers(dest="cmd")

    sub.add_parser("refresh", help="Force-download bootstrap + fixtures")
    sub.add_parser("analyse", help="Full squad report (default)")

    rank = sub.add_parser("rank", help="Rank the player pool")
    rank.add_argument("--position", choices=["GKP", "DEF", "MID", "FWD"])
    rank.add_argument(
        "--sort",
        default="balanced",
        choices=["balanced", "xp_horizon", "xp_gw", "ppp", "consistency", "residual", "differential"],
    )
    rank.add_argument("-n", type=int, default=15)

    opt = sub.add_parser("optimise", help="Build a fresh 15 under FPL constraints")
    opt.add_argument(
        "--mode",
        default="balanced",
        choices=["balanced", "ppp", "consistency", "differential", "xp"],
    )

    tr = sub.add_parser("transfers", help="Suggest 1-for-1 and N-transfer plans")
    tr.add_argument("--max", type=int, default=2)

    sub.add_parser("differentials", help="Low-owned players the model likes")
    sub.add_parser("value", help="Underpriced vs current price band")

    exp = sub.add_parser("export", help="Write the analysis bundle to JSON")
    exp.add_argument("-o", "--out", type=Path, default=Path("data/exports/latest.json"))
    for child in sub.choices.values():
        child.add_argument("--refresh", action="store_true", help=argparse.SUPPRESS)
    return p


def main(argv: list[str] | None = None) -> None:
    args = _build_parser().parse_args(argv)
    cmd = args.cmd or "analyse"

    if cmd == "refresh":
        info = FPLClient().refresh()
        print(f"Refreshed {info['players']} players, {info['fixtures']} fixtures → {info['cache_dir']}")
        return

    bundle = run_pipeline(
        squad_path=args.squad,
        horizon=args.horizon,
        force_refresh=args.refresh,
        max_transfers=getattr(args, "max", 2),
    )

    if cmd == "analyse":
        print(render_text(bundle))
        return

    if cmd == "rank":
        print(bundle.leaders(args.sort, args.n, args.position).to_string(index=False))
        return

    if cmd == "optimise":
        plan = bundle.plans.get(args.mode) or bundle.plans.get("balanced")
        if not plan:
            raise SystemExit("Optimiser produced no plan. Is pulp installed?")
        print(render_text(bundle))
        return

    if cmd == "transfers":
        print(render_text(bundle))
        return

    if cmd == "differentials":
        print(bundle.unorthodox(20).to_string(index=False))
        return

    if cmd == "value":
        print(bundle.underpriced(20).to_string(index=False))
        return

    if cmd == "export":
        path = export_json(bundle, args.out)
        print(f"Wrote {path}")
        return
