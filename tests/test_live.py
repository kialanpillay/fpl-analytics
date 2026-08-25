import pandas as pd

from fpl_analytics.live import parse_entry_picks, players_for_ids, resolve_manager_id


def test_parse_entry_picks_uses_multiplier_and_official_points():
    payload = {
        "active_chip": None,
        "automatic_subs": [{"element_out": 10, "element_in": 20}],
        "entry_history": {
            "points": 56,
            "points_on_bench": 8,
            "event_transfers_cost": 0,
        },
        "picks": [
            {"element": 1, "position": 1, "multiplier": 1, "is_captain": False, "is_vice_captain": False},
            {"element": 2, "position": 2, "multiplier": 2, "is_captain": True, "is_vice_captain": False},
            {"element": 20, "position": 12, "multiplier": 1, "is_captain": False, "is_vice_captain": False},
            {"element": 10, "position": 13, "multiplier": 0, "is_captain": False, "is_vice_captain": True},
        ],
    }
    parsed = parse_entry_picks(payload)
    assert parsed is not None
    assert parsed["points"] == 56
    assert parsed["points_on_bench"] == 8
    assert parsed["captain_id"] == 2
    assert parsed["vice_id"] == 10
    assert parsed["xi_ids"] == [1, 2, 20]
    assert parsed["bench_ids"] == [10]
    assert parsed["auto_subs"] == [{"out_id": 10, "in_id": 20}]


def test_players_for_ids_prefers_catalog_and_keeps_order():
    squad = pd.DataFrame([{"id": 1, "web_name": "YamlOnly", "position": "MID"}])
    catalog = pd.DataFrame(
        [
            {"id": 20, "web_name": "AutoSub", "position": "DEF"},
            {"id": 1, "web_name": "CatalogOne", "position": "MID"},
            {"id": 2, "web_name": "Captain", "position": "FWD"},
        ]
    )
    resolved = players_for_ids([2, 20, 99], catalog, squad)
    assert list(resolved["id"]) == [2, 20]
    assert list(resolved["web_name"]) == ["Captain", "AutoSub"]


def test_resolve_manager_id_defaults(monkeypatch):
    monkeypatch.delenv("FPL_MANAGER_ID", raising=False)
    assert resolve_manager_id() == 5558057
    monkeypatch.setenv("FPL_MANAGER_ID", "99")
    assert resolve_manager_id() == 99
