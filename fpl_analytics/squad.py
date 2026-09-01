"""Resolve a manager squad from official FPL ids and evaluate it."""

from __future__ import annotations

import unicodedata
from dataclasses import dataclass, field
from typing import Any

import pandas as pd

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
    warnings: list[str] = field(default_factory=list)


def spec_from_ids(
    ids: list[int],
    *,
    budget: float = 100.0,
    bank: float = 0.0,
    free_transfers: int = 1,
    warnings: list[str] | None = None,
) -> ManagerSquad:
    return ManagerSquad(
        entries=[SquadEntry(player_id=int(i), name=None) for i in ids],
        budget=budget,
        bank=bank,
        free_transfers=free_transfers,
        warnings=list(warnings or []),
    )


def _match_by_name(name: str, players: pd.DataFrame) -> pd.Series:
    exact = players.loc[players["web_name"].str.lower() == name.lower()]
    if len(exact) == 1:
        return exact.iloc[0]
    if len(exact) > 1:
        raise ValueError(
            f"{name!r} is ambiguous ({', '.join(exact['web_name'] + ' ' + exact['team_short'])}). "
            "Use the FPL element id."
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
                f"{entry.name!r} resolved by name to id {int(player['id'])}."
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
