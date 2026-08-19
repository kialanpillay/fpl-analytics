"""Qualitative flags and 2026/27 strategy notes used alongside the numbers.

Sources: official FPL API, BBC FPL 2026-27 expert guide (14 Aug 2026),
Fantasy Football Fix bargains / differentials, FPL Pulse value rankings.
Revisit after each international break — minutes and set-piece roles move.
"""

from __future__ import annotations

from dataclasses import dataclass

STRATEGIES = [
    {
        "id": "defcon-value",
        "title": "Underpriced DEFCON Assets",
        "detail": (
            "DEFCON flattened 2025/26 VAPM. Do not rebuy last year's leaders if "
            "already re-priced. Target £4.5–£5.5 defenders and cheap mids who "
            "clear the DEFCON threshold before ownership moves."
        ),
    },
    {
        "id": "premium-plus-enablers",
        "title": "Premium Captain Plus Enablers",
        "detail": (
            "Haaland (£15.5m) is the default captain. A second premium is optional "
            "without Salah. Fund with £4.0–£4.5 nailed starters (Mitchell; "
            "Dubravka if £0.5 is required)."
        ),
    },
    {
        "id": "minutes-over-upside",
        "title": "Minutes Over Upside",
        "detail": (
            "Early-season error is rotation and new-club roles (Guéhi, Semenyo, "
            "Rogers, Ngumoha 547 min). A £6.0 who starts 90 dominates a £6.0 on 20."
        ),
    },
    {
        "id": "fixture-swings",
        "title": "Fixture Swings, Not Season Averages",
        "detail": (
            "Best GW1–2 attack FDR: MUN (HUL, IPS), SUN (IPS, FUL), ARS (COV). "
            "Fade HUL. Re-score the horizon when FDR turns."
        ),
    },
    {
        "id": "three-club-cap",
        "title": "Keep a Free Slot at MCI and CHE",
        "detail": (
            "A City triple (Haaland, Semenyo, Guéhi) plus a Chelsea triple "
            "(Pedro, Rogers, Lacroix) blocks Foden, Gvardiol, Colwill, Palmer. "
            "Hold one slot until minutes settle."
        ),
    },
    {
        "id": "chips",
        "title": "Chip Windows, 2026/27",
        "detail": (
            "BB and TC live from GW1. WC/FH: GW2–19 and GW20–38. Do not BB a "
            "non-starter. First WC is post-minutes, not GW1."
        ),
    },
]


@dataclass(frozen=True)
class PlayerNote:
    tone: str  # risk | value | watch | avoid
    note: str


PLAYER_NOTES: dict[str, PlayerNote] = {
    "Haaland": PlayerNote(
        "value",
        "Default captain. £15.5m / ~70% owned. GW1 90 vs Bournemouth is not guaranteed after a short pre-season; still the season-long talisman.",
    ),
    "João Pedro": PlayerNote(
        "value",
        "Template Chelsea forward (~61% owned). 177 pts, 15g/9a last season. Minutes risk only if Chelsea add another 9.",
    ),
    "Calvert-Lewin": PlayerNote(
        "value",
        "Nailed Leeds 9 on penalties. 14 goals last season, 35 apps. Best of the £6.0 forward band for minutes certainty.",
    ),
    "Raya": PlayerNote(
        "value",
        "Arsenal's 19-CS defence last season. Expensive vs Verbruggen/Dubravka, but GW1 home Coventry is a clean-sheet spot.",
    ),
    "Verbruggen": PlayerNote(
        "value",
        "Best-value nailed keeper last season (130 pts at £4.5). Pair with a 4.0 if you ever need 0.5, otherwise keep.",
    ),
    "Mitchell": PlayerNote(
        "value",
        "Elite 4.5: 135 pts, 36 starts. The enabler who actually returns. Low ownership for the output.",
    ),
    "Calafiori": PlayerNote(
        "watch",
        "Arsenal route at £5.5 but only 22 starts last season. Rotation with White/Timber is the risk; CS upside is real.",
    ),
    "Lacroix": PlayerNote(
        "watch",
        "New to Chelsea (Jul 2026). Strong DEFCON (10.8/90) but Colwill is £1.0 cheaper if he starts. Sage/Alonso minutes TBD.",
    ),
    "Guéhi": PlayerNote(
        "risk",
        "BBC avoid: not guaranteed under Maresca with Dias and Gvardiol. 179 pts were mostly at Palace. City slot is expensive uncertainty.",
    ),
    "Van Hecke": PlayerNote(
        "watch",
        "June move to Spurs. 36 starts / 148 pts at Brighton. De Zerbi CB pool is deep (Senesi etc.) — confirm the pairing.",
    ),
    "Rogers": PlayerNote(
        "watch",
        "Villa talisman moved to Chelsea (Jul 2026). 169 pts / 37 starts last year, but new-system minutes are the open question.",
    ),
    "Semenyo": PlayerNote(
        "watch",
        "202 pts from Bournemouth, now City at £8.5. Mid-season move already happened; rotation with Foden/Cherki/Haaland is the ceiling cap.",
    ),
    "Ndiaye": PlayerNote(
        "value",
        "Everton pens, 7.19 xG vs 6 goals. £6.0 bargain if he stays — Al-Hilal rumours were the pre-season caveat.",
    ),
    "Ngumoha": PlayerNote(
        "risk",
        "£6.0 for 547 minutes / 5 starts. xGI/90 (0.55) is exciting, but this is a lottery ticket, not a midfield pillar.",
    ),
    "Fletcher": PlayerNote(
        "risk",
        "Tyler Fletcher: 17 minutes, 0 starts. Dead slot. Fine only as 4.5 fodder if you never need the bench.",
    ),
    "J.Fletcher": PlayerNote(
        "risk",
        "Jack Fletcher: 107 minutes. Still not a GW1 starter. Prefer a 4.5 who actually plays.",
    ),
    "Tarkowski": PlayerNote(
        "value",
        "Algorithm favourite differential. 170 pts, 376 DEFCON, fixture-agnostic floor. ~9% owned.",
    ),
    "Truffert": PlayerNote(
        "value",
        "165 pts at £5.5, attacking FB + DEFCON. ~5% owned. Strong unorthodox defender.",
    ),
    "Anderson": PlayerNote(
        "watch",
        "180 pts / 515 DEFCON then moved to City at £6.5. Elite last year; City minutes are the whole question.",
    ),
    "Szoboszlai": PlayerNote(
        "value",
        "Liverpool talisman at £7.0. Set pieces, possible pens post-Salah, 20 DEFCON pts. Template (~41%).",
    ),
    "Mosquera": PlayerNote(
        "watch",
        "£5.5 Arsenal CB while Saliba is out. 2.5m cheaper than Gabriel — only if he actually starts.",
    ),
    "Sarr": PlayerNote(
        "value",
        "Palace attacker, ~6% owned, pens when Mateta is off. High-upside 6.5 mid.",
    ),
    "Tavernier": PlayerNote(
        "value",
        "Bournemouth creator / set pieces after Semenyo left. 13.7 xGI, 1.7% owned.",
    ),
    "Colwill": PlayerNote(
        "watch",
        "£5.0 vs Lacroix £6.0. Injury-shortened last season but cheaper DEFCON route into Chelsea.",
    ),
    "Zubimendi": PlayerNote(
        "value",
        "133 pts, 307 DEFCON at £5.5 / 1.3% owned. Floor mid who funds premiums.",
    ),
    "Gravenberch": PlayerNote(
        "value",
        "144 pts, 298 DEFCON, £6.0, <2% owned. Liverpool minutes look safer than Ngumoha.",
    ),
    "Foden": PlayerNote(
        "watch",
        "7% owned City mid. High ceiling, high drop risk. Needs a free City slot.",
    ),
    "Brobbey": PlayerNote(
        "watch",
        "Short-term Sunderland fixtures (Ipswich, Fulham) then a brutal run. Early-season punt, not set-and-forget.",
    ),
}


def note_for(web_name: str) -> PlayerNote | None:
    return PLAYER_NOTES.get(web_name)
