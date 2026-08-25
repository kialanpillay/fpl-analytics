import pandas as pd

from fpl_analytics.optimiser import hit_scenarios, pair_swaps
from fpl_analytics.schemas import squad_diff, transfer_records


def _player(pid, name, position, xp_horizon):
    return {"id": pid, "web_name": name, "position": position, "team_short": "TST", "price": 6.0, "xp_horizon": xp_horizon}


def test_pair_swaps_matches_by_position_not_row_order():
    outgoing = pd.DataFrame(
        [
            _player(1, "MidOut", "MID", 5.0),
            _player(2, "FwdOut", "FWD", 8.0),
        ]
    )
    incoming = pd.DataFrame(
        [
            _player(3, "FwdIn", "FWD", 10.0),
            _player(4, "MidIn", "MID", 7.0),
        ]
    )
    pairs = pair_swaps(outgoing, incoming)
    assert [(o.web_name, i.web_name) for o, i in pairs] == [("MidOut", "MidIn"), ("FwdOut", "FwdIn")]


def test_hit_scenarios_follows_n_transfers():
    assert hit_scenarios(9.0, 0) == []
    assert hit_scenarios(9.0, 1) == [
        {"hits": 0, "cost": 0, "net_horizon": 9.0},
        {"hits": 1, "cost": 4, "net_horizon": 5.0},
    ]


def test_squad_diff_exposes_swaps():
    current = pd.DataFrame([_player(1, "MidOut", "MID", 5.0), _player(2, "Keep", "DEF", 4.0)])
    planned = pd.DataFrame([_player(4, "MidIn", "MID", 7.0), _player(2, "Keep", "DEF", 4.0)])
    diff = squad_diff(current, planned)
    assert diff["n_transfers"] == 1
    assert diff["swaps"][0]["out"]["web_name"] == "MidOut"
    assert diff["swaps"][0]["in"]["web_name"] == "MidIn"


def test_transfer_records_drop_nan():
    frame = pd.DataFrame(
        [
            {
                "out_id": 1,
                "out": "A",
                "in_id": 2,
                "in": "B",
                "position": "MID",
                "cost_delta": 0.5,
                "d_balanced": float("nan"),
                "d_xp": 1.25,
                "unorthodox": False,
            }
        ]
    )
    rows = transfer_records(frame)
    assert rows[0]["d_balanced"] is None
    assert rows[0]["d_xp"] == 1.25
    assert rows[0]["cost_delta"] == 0.5
