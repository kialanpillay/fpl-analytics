import pandas as pd

from fpl_analytics.models import score


def _row(**overrides):
    row = {
        "id": 1,
        "position": "MID",
        "minutes": 90,
        "xg_p90": 0.15,
        "xa_p90": 0.10,
        "xgi_p90": 0.25,
        "clean_sheets": 0,
        "defcon_p90": 6.0,
        "bonus": 0,
        "saves": 0,
        "penalties_order": 99,
        "corners_order": 99,
        "effective_minutes": 0.90,
        "attack_adj": 1.0,
        "clean_adj": 1.0,
        "ep_next": 4.0,
        "n_fixtures": 6,
        "starts": 1,
        "total_points": 4,
        "early_season": True,
        "minutes_prob": 0.90,
        "role_risk": 1.0,
        "ownership": 15.0,
        "price": 7.0,
        "can_select": True,
        "status": "a",
    }
    row.update(overrides)
    return row


def test_aggressive_prefers_next_gw_attack_over_floor():
    frame = pd.DataFrame(
        [
            _row(id=1, ep_next=8.5, xg_p90=0.7, xa_p90=0.3, xgi_p90=1.0, price=13.0, ownership=60, defcon_p90=3),
            _row(id=2, ep_next=3.0, xg_p90=0.02, xa_p90=0.02, xgi_p90=0.04, price=4.5, ownership=8, defcon_p90=14),
            _row(id=3, ep_next=5.0, xg_p90=0.2, xa_p90=0.2, xgi_p90=0.4, price=7.5, ownership=20, defcon_p90=8),
        ]
    )
    out = score(frame)
    assert out.loc[out["id"] == 1, "aggressive"].iloc[0] > out.loc[out["id"] == 2, "aggressive"].iloc[0]


def test_template_prefers_high_ownership_at_similar_quality():
    frame = pd.DataFrame(
        [
            _row(id=1, ep_next=6.0, ownership=55.0, price=10.0),
            _row(id=2, ep_next=6.0, ownership=4.0, price=10.0),
            _row(id=3, ep_next=4.0, ownership=20.0, price=7.0),
        ]
    )
    out = score(frame)
    assert out.loc[out["id"] == 1, "template"].iloc[0] > out.loc[out["id"] == 2, "template"].iloc[0]
