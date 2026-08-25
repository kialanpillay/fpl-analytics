import pandas as pd

from fpl_analytics.captaincy import rank_captaincy


def test_rank_captaincy_orders_by_xp_and_minutes():
    squad = pd.DataFrame(
        [
            {"id": 1, "web_name": "Safe", "position": "MID", "xp_gw": 6.0, "minutes_prob": 0.95},
            {"id": 2, "web_name": "Haaland", "position": "FWD", "xp_gw": 8.0, "minutes_prob": 0.90},
            {"id": 3, "web_name": "Risk", "position": "MID", "xp_gw": 9.0, "minutes_prob": 0.40},
            {"id": 4, "web_name": "Bench", "position": "DEF", "xp_gw": 4.0, "minutes_prob": 0.80},
        ]
    )
    ranked = rank_captaincy(squad, xi_ids=[1, 2, 3], n=3)
    assert list(ranked["web_name"]) == ["Haaland", "Safe", "Risk"]
    assert bool(ranked.iloc[0]["season_default"]) is True
    assert ranked.iloc[0]["captain_ev"] == 16.0
    assert ranked.iloc[0]["captain_score"] == 7.2
