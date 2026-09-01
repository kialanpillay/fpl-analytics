# FPL Analytics

Season-long FPL modelling on the official public API:

`https://fantasy.premierleague.com/api/`

Each deadline the package scores the pool and solves a legal 15 (2/5/5/3, £100m, max 3/club).

- **Expected Points** — next GW and a 6-GW horizon (xG/xA, CS, DEFCON, bonus, `ep_next`, FDR, minutes)
- **Points Per Pound** — `xp_horizon / price`
- **Consistency** — start rate, minutes, DEFCON floor, role risk
- **Price Residual** — last-season points vs current price band
- **Unorthodox** — high model score, ownership &lt; 10%
- **Transfers** — 1-for-1 and N-swap plans from the official FPL 15

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[web,dev]"
```

Frontend (once):

```bash
cd web && npm install
```

## Web UI

Alternate to the CLI. Vite on `:5173`, FastAPI on `:8009`.

```bash
pip install -e ".[web]"
fpl-web
# other terminal
cd web && npm run dev
```

Open `http://127.0.0.1:5173`. Re-run analysis, inspect the pitch, transfers, captaincy, wildcard, fixtures, and live GW. The squad is the official 15 (`FPL_MANAGER_ID` or the default entry). Settings → Import Picks reloads it.

Production-style (API serves `web/dist`):

```bash
cd web && npm run build
fpl-web
```

## Weekly Loop

```bash
python -m fpl_analytics refresh
python main.py analyse

python -m fpl_analytics rank --position MID --sort ppp
python -m fpl_analytics differentials
python -m fpl_analytics value

python -m fpl_analytics optimise --mode balanced
python -m fpl_analytics optimise --mode differential
```

`refresh` bypasses the 30-minute cache. After a transfer on FPL, re-run with `--refresh` or Import Picks.

## Model Signals

| Signal | Definition |
| --- | --- |
| `xp_gw` | `0.55 * ep_next + 0.45 *` per-90 model (xG/xA, CS, DEFCON, bonus, saves) × minutes × FDR |
| `xp_horizon` | Same model over the next N fixtures (default 6) |
| `ppp` | `xp_horizon / price` |
| `consistency` | Starts, minutes, DEFCON floor, minus new-club / &lt;900-minute risk |
| `residual` | Per-position OLS residual of last-season points on price |
| `balanced` | `0.50 * xp_horizon + 0.28 * ppp + 0.22 * consistency` (minmax-scaled) |

Pre-season, `form` is 0; 2025/26 minutes, xG, and DEFCON remain on the element. Role risk discounts summer moves and &lt;900 minutes. In-season, `refresh` picks up live `ep_next`, `form`, and availability.

## API Surface

| Endpoint | Payload |
| --- | --- |
| `/bootstrap-static/` | Elements, prices, ownership, xG, DEFCON, events |
| `/fixtures/` | FDR, horizon run |
| `/element-summary/{id}/` | Per-GW history |
| `/event/{gw}/live/` | Live points |
| `/entry/{id}/` | Public manager profile |
| `/entry/{id}/event/{gw}/picks/` | Public GW picks |
| `/event-status/` | Bonus / league processing |
| `/team/set-piece-notes/` | Set-piece notes |

No auth. Responses cache to `data/cache/` (TTL 30 min).

Tests: `pytest`.
