import pandas as pd

from fpl_analytics.catalog import SeasonMeta
from fpl_analytics.optimiser import SquadPlan
from fpl_analytics.pipeline import AnalysisBundle
from fpl_analytics.schemas import analysis_payload, player_record
from fpl_analytics.squad import ManagerSquad, SquadEntry


def _player(**overrides):
    row = {
        "id": 411,
        "web_name": "Haaland",
        "first_name": "Erling",
        "second_name": "Haaland",
        "full_name": "Erling Haaland",
        "position": "FWD",
        "element_type": 4,
        "team_id": 13,
        "team": "Man City",
        "team_short": "MCI",
        "team_code": 43,
        "code": 223094,
        "photo": "223094.jpg",
        "price": 14.5,
        "ownership": 60.0,
        "status": "a",
        "news": "",
        "can_select": True,
        "chance_next": 100,
        "event_points": 2,
        "total_points": 2,
        "xp_gw": 7.2,
        "xp_horizon": 42.0,
        "ppp": 2.9,
        "consistency": 8.1,
        "balanced": 8.4,
        "residual": 12.0,
        "differential": 8.0,
        "minutes_prob": 0.94,
        "role_risk": 1.0,
        "effective_minutes": 0.94,
        "next_fixture": "GW2 H WOL (2)",
        "fixture_run": "GW2 H WOL (2)",
        "fdr_mean": 2.5,
        "unorthodox": False,
        "value_flag": False,
        "xg_p90": 0.7,
        "xa_p90": 0.1,
        "xgi_p90": 0.8,
        "defcon_p90": 1.0,
        "set_piece": False,
        "penalties_order": 1,
        "minutes": 90,
        "starts": 1,
        "form": 2.0,
        "bonus": 0,
        "bps": 12,
        "gw_minutes": 90,
    }
    row.update(overrides)
    return row


def test_player_record_attaches_assets_and_sanitises():
    rec = player_record(_player(xp_gw=float("nan")))
    assert rec["id"] == 411
    assert rec["xp_gw"] is None
    assert rec["photo_url"].endswith("p223094.png")
    assert rec["shirt_url"].endswith("shirt_43-66.webp")
    assert rec["badge_url"].endswith("t43.png")


def test_analysis_payload_round_trip():
    players = pd.DataFrame([_player(), _player(id=1, web_name="Raya", position="GKP", element_type=1)])
    spec = ManagerSquad(
        entries=[SquadEntry(411, "Haaland"), SquadEntry(1, "Raya")],
        budget=100.0,
        bank=0.5,
        free_transfers=1,
        path=__file__,
        warnings=[],
    )
    plan = SquadPlan(
        players=players,
        cost=20.0,
        objective="balanced",
        objective_value=10.0,
        xp_horizon=50.0,
        xp_gw=12.0,
        ppp=2.5,
        consistency=7.0,
        xi_ids=[411],
        bench_ids=[1],
    )
    bundle = AnalysisBundle(
        fetched_at="2026-08-25T00:00:00+00:00",
        meta=SeasonMeta(
            next_event=2,
            current_event=1,
            deadline="2026-08-29T17:30:00Z",
            total_managers=11,
            budget=100.0,
            team_limit=3,
            squad_size=15,
            season_started=True,
        ),
        players=players,
        squad=players,
        squad_eval={"n": 2, "cost": 20.0, "xp_gw": 12.0, "xp_horizon": 50.0, "ppp": 2.5, "consistency": 7.0, "balanced": 8.0, "ownership_xi_proxy": 20.0, "club_counts": {"MCI": 1}, "pos_counts": {"FWD": 1}, "illegal_clubs": [], "dead_slots": [], "risk_names": []},
        spec=spec,
        plans={"balanced": plan},
        transfers=pd.DataFrame(
            [
                {
                    "out_id": 1,
                    "out": "Raya",
                    "in_id": 2,
                    "in": "Sels",
                    "position": "GKP",
                    "cost_delta": 0.5,
                    "d_balanced": float("nan"),
                    "d_xp": 1.2,
                    "unorthodox": False,
                }
            ]
        ),
        transfer_plan=None,
        horizon=6,
        xi_ids=[411],
        bench_ids=[1],
    )
    payload = analysis_payload(bundle)
    dumped = payload.model_dump()
    assert dumped["meta"]["next_event"] == 2
    assert dumped["meta"]["bank"] == 0.5
    assert dumped["squad"][0]["photo_url"]
    assert dumped["plans"]["balanced"]["xi_ids"] == [411]
    assert dumped["transfers"][0]["d_balanced"] is None
    assert dumped["transfers"][0]["d_xp"] == 1.2
    restored = payload.model_validate(dumped)
    assert restored.fetched_at == bundle.fetched_at
