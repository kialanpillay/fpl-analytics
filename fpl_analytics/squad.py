"""Load a manager squad from YAML and evaluate it against the live table.

Canonical identity is the FPL element ``id`` (what every API endpoint uses).
``name`` is a human label — usually FPL ``web_name``, not the legal name —
checked against the live table so a stale id is obvious after a transfer.
"""

from __future__ import annotations

import unicodedata
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

DEFAULT_SQUAD_PATH = Path(__file__).resolve().parent.parent / "config" / "squad.yaml"

ALIASES = {
    "Guehi": "Guéhi",
    "Guehí": "Guéhi",
    "Joao Pedro": "João Pedro",
    "JoaoPedro": "João Pedro",
    "Van Hecke": "Van Hecke",
    "van Hecke": "Van Hecke",
    "Calvert Lewin": "Calvert-Lewin",
}


def _fold(value: str) -> str:
    decomposed = unicodedata.normalize("NFD", value)
    stripped = "".join(ch for ch in decomposed if unicodedata.category(ch) != "Mn")
    return " ".join(stripped.casefold().replace("-", " ").replace(".", " ").split())


def _names_match(label: str, player: pd.Series) -> bool:
    folded = _fold(ALIASES.get(label, label))
    candidates = (
        _fold(player["web_name"]),
        _fold(player["full_name"]),
        _fold(f"{player['first_name']} {player['second_name']}"),
    )
    return any(folded == cand or folded in cand or cand in folded for cand in candidates)


@dataclass
class SquadEntry:
    player_id: int | None
    name: str | None


@dataclass
class ManagerSquad:
    entries: list[SquadEntry]
    budget: float
    bank: float
    free_transfers: int
    path: Path
    warnings: list[str] = field(default_factory=list)


def _parse_entry(raw: Any) -> SquadEntry:
    if isinstance(raw, int):
        return SquadEntry(player_id=int(raw), name=None)
    if isinstance(raw, str):
        return SquadEntry(player_id=None, name=ALIASES.get(raw, raw))
    if isinstance(raw, dict):
        player_id = raw.get("id")
        name = raw.get("name")
        if player_id is None and name is None:
            raise ValueError(f"Squad entry needs id or name: {raw!r}")
        return SquadEntry(
            player_id=int(player_id) if player_id is not None else None,
            name=ALIASES.get(str(name), str(name)) if name is not None else None,
        )
    raise ValueError(f"Unsupported squad entry: {raw!r}")


def load_squad(path: Path | str = DEFAULT_SQUAD_PATH) -> ManagerSquad:
    raw: dict[str, Any] = yaml.safe_load(Path(path).read_text()) or {}
    players_raw = raw.get("players", [])
    legacy_ids = {ALIASES.get(k, k): int(v) for k, v in (raw.get("ids") or {}).items()}

    entries: list[SquadEntry] = []
    for item in players_raw:
        entry = _parse_entry(item)
        if entry.player_id is None and entry.name and entry.name in legacy_ids:
            entry = SquadEntry(player_id=legacy_ids[entry.name], name=entry.name)
        entries.append(entry)

    return ManagerSquad(
        entries=entries,
        budget=float(raw.get("budget", 100.0)),
        bank=float(raw.get("bank", 0.0)),
        free_transfers=int(raw.get("free_transfers", 1)),
        path=Path(path),
    )


def _match_by_name(name: str, players: pd.DataFrame) -> pd.Series:
    exact = players.loc[players["web_name"].str.lower() == name.lower()]
    if len(exact) == 1:
        return exact.iloc[0]
    if len(exact) > 1:
        raise ValueError(
            f"{name!r} is ambiguous ({', '.join(exact['web_name'] + ' ' + exact['team_short'])}). "
            "Use the FPL id in config/squad.yaml."
        )
    contains = players.loc[
        players["web_name"].str.contains(name, case=False, regex=False)
        | players["full_name"].str.contains(name, case=False, regex=False)
    ]
    if len(contains) == 1:
        return contains.iloc[0]
    options = ", ".join(contains["web_name"].head(8)) if not contains.empty else "no close match"
    raise ValueError(f"Could not resolve {name!r} uniquely ({options}).")


def resolve_squad(spec: ManagerSquad, players: pd.DataFrame) -> pd.DataFrame:
    rows = []
    warnings: list[str] = []
    for entry in spec.entries:
        if entry.player_id is not None:
            hit = players.loc[players["id"] == entry.player_id]
            if hit.empty:
                raise ValueError(f"FPL id {entry.player_id} is not in the live API.")
            player = hit.iloc[0]
            if entry.name and not _names_match(entry.name, player):
                warnings.append(
                    f"id {entry.player_id} is {player['web_name']} ({player['team_short']}), "
                    f"not {entry.name!r} — update the label or the id."
                )
        elif entry.name:
            player = _match_by_name(entry.name, players)
            warnings.append(
                f"{entry.name!r} resolved by name to id {int(player['id'])}. "
                "Prefer pinning the id in config/squad.yaml."
            )
        else:
            raise ValueError("Squad entry is empty.")
        rows.append(player)

    frame = pd.DataFrame(rows)
    if frame["id"].duplicated().any():
        dupes = frame.loc[frame["id"].duplicated(keep=False), "web_name"].tolist()
        raise ValueError(f"Duplicate squad entries: {dupes}")
    spec.warnings = warnings
    return frame.reset_index(drop=True)


def save_squad(
    spec: ManagerSquad,
    path: Path | str | None = None,
    players: pd.DataFrame | None = None,
) -> Path:
    """Write ``{id, name}`` rows plus bank / FTs / budget."""
    dest = Path(path) if path is not None else spec.path
    rows: list[dict[str, Any]] = []
    lookup = None
    if players is not None and not players.empty:
        lookup = players.set_index("id")
    for entry in spec.entries:
        player_id = entry.player_id
        name = entry.name
        if player_id is not None and lookup is not None and player_id in lookup.index:
            name = str(lookup.loc[player_id, "web_name"])
        if player_id is None:
            continue
        rows.append({"id": int(player_id), "name": name or str(player_id)})
    payload = {
        "budget": float(spec.budget),
        "bank": float(spec.bank),
        "free_transfers": int(spec.free_transfers),
        "players": rows,
    }
    dest.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# FPL element id is the identity. name is a label (web_name, not the legal name).",
        "# After a transfer, change the id — if the label no longer matches, analyse warns.",
        "",
        f"budget: {payload['budget']}",
        f"bank: {payload['bank']}",
        f"free_transfers: {payload['free_transfers']}",
        "",
        "players:",
    ]
    for row in rows:
        name = yaml.safe_dump(str(row["name"]), default_style='"').strip()
        lines.append(f"  - {{id: {row['id']}, name: {name}}}")
    dest.write_text("\n".join(lines) + "\n")
    spec.path = dest
    return dest


def evaluate_squad(squad: pd.DataFrame, team_limit: int = 3) -> dict[str, Any]:
    club_counts = squad.groupby("team_short").size().sort_values(ascending=False)
    pos_counts = squad.groupby("position").size().reindex(["GKP", "DEF", "MID", "FWD"]).fillna(0)
    over_club = club_counts[club_counts > team_limit]
    return {
        "n": int(len(squad)),
        "cost": float(squad["price"].sum()),
        "xp_gw": float(squad["xp_gw"].sum()),
        "xp_horizon": float(squad["xp_horizon"].sum()),
        "ppp": float(squad["ppp"].mean()),
        "consistency": float(squad["consistency"].mean()),
        "balanced": float(squad["balanced"].mean()),
        "ownership_xi_proxy": float(squad.nlargest(11, "xp_gw")["ownership"].mean()),
        "club_counts": club_counts.to_dict(),
        "pos_counts": {k: int(v) for k, v in pos_counts.items()},
        "illegal_clubs": over_club.index.tolist(),
        "dead_slots": squad.loc[squad["effective_minutes"] < 0.25, "web_name"].tolist(),
        "risk_names": squad.loc[squad["role_risk"] < 0.85, "web_name"].tolist(),
    }
