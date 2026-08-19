"""Qualitative flags used with the live scores.

Sources: FPL API; BBC 2026/27 guide (14 Aug 2026); Community Shield
ARS 3–0 MCI (16 Aug 2026); FFF / FPL Pulse pre-season. Revisit after
each international break — minutes and set pieces move.
"""

from __future__ import annotations

from dataclasses import dataclass

STRATEGIES = [
    {
        "id": "defcon-value",
        "title": "Underpriced DEFCON Assets",
        "detail": (
            "2025/26 VAPM was DEFCON-driven. Do not rebuy last year's leaders "
            "if already re-priced. Target £4.5–£6.0 floors (Tarkowski, "
            "Truffert, Mitchell, Ampadu, Rice) before ownership moves."
        ),
    },
    {
        "id": "premium-plus-enablers",
        "title": "Premium Captain Plus Enablers",
        "detail": (
            "Haaland is the 38-GW captain. A second premium is optional "
            "without Salah. Fund with nailed 4.5s (Mitchell; Dubravka if "
            "£0.5 is required). Bruno is a GW1–2 FDR dart, not a late pivot."
        ),
    },
    {
        "id": "minutes-over-upside",
        "title": "Minutes Over Upside",
        "detail": (
            "Sell rotation and new-club roles first (Calafiori 22 starts, "
            "Rogers CHE, Van Hecke TOT, Guéhi vs Dias/Gvardiol). Semenyo "
            "started CS 90 — treat as nailed until benched."
        ),
    },
    {
        "id": "fixture-swings",
        "title": "Fixture Swings, Not Season Averages",
        "detail": (
            "GW1–2 attack FDR: MUN (HUL, IPS), SUN (IPS, FUL), ARS (COV). "
            "MUN fades GW4 (H MCI). Fade HUL. Re-solve when FDR turns."
        ),
    },
    {
        "id": "three-club-cap",
        "title": "Club Cap",
        "detail": (
            "Squad A is MCI 3 (Haaland, Semenyo, Guéhi) and ARS 3 (Raya, "
            "Calafiori, Rice). A CHE triple blocks Enzo/Palmer. Free a slot "
            "before adding Foden, Gvardiol, or Virgil-funded moves."
        ),
    },
    {
        "id": "chips",
        "title": "Chip Windows, 2026/27",
        "detail": (
            "BB and TC live from GW1. WC/FH: GW2–19 and GW20–38. Do not BB "
            "a non-starter. First WC is post-minutes, not GW1."
        ),
    },
]


@dataclass(frozen=True)
class PlayerNote:
    tone: str  # risk | value | watch | avoid
    note: str


PLAYER_NOTES: dict[str, PlayerNote] = {
    # Squad A
    "Raya": PlayerNote(
        "value",
        "ARS 19 CS in 2025/26. £6.0. GW1 H COV (2). CS started vs MCI.",
    ),
    "Verbruggen": PlayerNote(
        "value",
        "130 pts / 38 starts at £4.5. Best-value nailed GK. Keep unless £0.5 needed.",
    ),
    "Mitchell": PlayerNote(
        "value",
        "135 pts / 36 starts at £4.5. DEFCON enabler. 6.7% owned. First bench, not a sale.",
    ),
    "Calafiori": PlayerNote(
        "watch",
        "22 starts / 109 pts. CS goal vs MCI; still rotates with White/Timber. First DEF FT (Virgil/Truffert).",
    ),
    "Tarkowski": PlayerNote(
        "value",
        "170 pts, 376 DEFCON, 38-start floor. FDR-agnostic. ~9% owned.",
    ),
    "Guéhi": PlayerNote(
        "risk",
        "179 pts mostly CRY. Model keep (xH 31); minutes risk vs Dias/Gvardiol. Sell only if benched.",
    ),
    "Van Hecke": PlayerNote(
        "watch",
        "Jun 2026 → TOT. 36 starts / 148 pts at BHA. De Zerbi CB pool (Senesi). Role risk 0.82.",
    ),
    "Semenyo": PlayerNote(
        "value",
        "202 pts / 37 starts. CS: started RW, 90 vs ARS. City since Jan. Do not sell for Rogers.",
    ),
    "Rice": PlayerNote(
        "value",
        "184 pts / 35 starts, DEFCON floor. £7.5. Caps ARS at 3 with Raya+Calafiori.",
    ),
    "Ndiaye": PlayerNote(
        "value",
        "Pens. 7.19 xG vs 6 goals. £6.0. Hold unless a confirmed exit.",
    ),
    "E.Le Fée": PlayerNote(
        "value",
        "SUN £6.0. GW1–2 IPS/FUL. Pens possible. Multiple routes (xGI + DEFCON).",
    ),
    "Ampadu": PlayerNote(
        "value",
        "LEE £5.5. Nailed DEFCON mid, 1.5% owned. Bench 8; not a sale.",
    ),
    "Haaland": PlayerNote(
        "value",
        "£15.5m / ~70%. CS start; Raya saved a chance. 38-GW captain. GW1 H BOU not guaranteed 90.",
    ),
    "Calvert-Lewin": PlayerNote(
        "value",
        "LEE 9, pens, 35 apps / 14 goals. £6.0 minutes-certain forward.",
    ),
    "João Pedro": PlayerNote(
        "value",
        "177 pts, 15g/9a. ~61% owned. Pre-season haul. Minutes risk only if CHE add a 9.",
    ),
    # Sold / alternatives
    "Rogers": PlayerNote(
        "watch",
        "Jul 2026 → CHE. 169/37 at AVL. Sociedad: 62 min, debut goal, late back. New-system risk. xH 22 vs Semenyo 30.",
    ),
    "Lacroix": PlayerNote(
        "watch",
        "Jul 2026 → CHE. DEFCON 10.8/90. Tarkowski is the same-price floor upgrade.",
    ),
    "Ngumoha": PlayerNote(
        "risk",
        "£6.0 / 5 starts / 547 min. xGI/90 0.55. Lottery, not a MID pillar.",
    ),
    "Fletcher": PlayerNote(
        "risk",
        "Tyler Fletcher: 17 min, 0 starts. Dead 4.5.",
    ),
    "J.Fletcher": PlayerNote(
        "risk",
        "Jack Fletcher: 107 min. Not a GW1 starter.",
    ),
    # Model targets
    "Virgil": PlayerNote(
        "value",
        "175 pts / 38 starts. £6.5. GW3–4 IPS/FUL CS. +8 xH vs Calafiori; −1 vs Guéhi. Needs £1.0.",
    ),
    "Truffert": PlayerNote(
        "value",
        "165 pts at £5.5. FB + DEFCON, 4.9% owned. Hard GW1 (A MCI). Same-price Calafiori swap.",
    ),
    "N.Williams": PlayerNote(
        "value",
        "Glasner WB: attack + DEFCON + set pieces. £5.0. ~11% owned.",
    ),
    "Gabriel": PlayerNote(
        "value",
        "209 pts, record £8.0. Aggressive ILP pick. CS + attack. Price already reflects it.",
    ),
    "Mosquera": PlayerNote(
        "watch",
        "£5.5 ARS CB while Saliba is out. Only if he starts. £2.5 cheaper than Gabriel.",
    ),
    "White": PlayerNote(
        "watch",
        "£5.5. Minutes while Timber is out. Short-term ARS DEF, not a season hold.",
    ),
    "Senesi": PlayerNote(
        "avoid",
        "175 pts at BOU; now TOT. De Zerbi CB depth. Do not pay last year's residual.",
    ),
    "Richards": PlayerNote(
        "value",
        "CRY £5.0, 0.8% owned. DEFCON floor. Unorthodox Calafiori replacement.",
    ),
    "Thiaw": PlayerNote(
        "value",
        "NEW £5.0. 126 pts, 12 DEFCON hits, 4 goals. 1.9% owned.",
    ),
    "Collins": PlayerNote(
        "value",
        "BRE £5.5. DEFCON + CS. 2% owned.",
    ),
    "Shaw": PlayerNote(
        "watch",
        "£4.5 MUN. High owned enabler. Minutes more secure than output.",
    ),
    "B.Fernandes": PlayerNote(
        "value",
        "235 pts, pens/set pieces. £12.0. GW1–2 HUL/IPS. Model GW1 captain. Do not buy after GW3.",
    ),
    "Szoboszlai": PlayerNote(
        "value",
        "LIV £7.0. Set pieces, pens candidate post-Salah, 20 DEFCON pts. ~41% owned.",
    ),
    "Enzo": PlayerNote(
        "watch",
        "Model MID (xH 30) at £7.0 / 5% owned. Exit rumours. Do not take if locking 3 CHE.",
    ),
    "Anderson": PlayerNote(
        "watch",
        "180 pts / 515 DEFCON → MCI £6.5. Last-year residual; City minutes unknown.",
    ),
    "Zubimendi": PlayerNote(
        "value",
        "133 pts, 307 DEFCON, £5.5, 1.3% owned. Floor mid. Blocked if ARS already at 3.",
    ),
    "Gravenberch": PlayerNote(
        "value",
        "144 pts, 298 DEFCON, £6.0, <2% owned. Safer LIV minutes than Ngumoha.",
    ),
    "Tavernier": PlayerNote(
        "value",
        "BOU creator / set pieces. 13.7 xGI, 1.7% owned. Pens if Kluivert off.",
    ),
    "Xhaka": PlayerNote(
        "value",
        "SUN £5.5. 26 DEFCON pts / 32 starts. GW1 A IPS. Floor, not ceiling.",
    ),
    "Wharton": PlayerNote(
        "value",
        "CRY £5.5. High residual, 0.7% owned. DEFCON mid. Minutes TBD.",
    ),
    "Foden": PlayerNote(
        "watch",
        "MCI £7.0, ~5% owned. Ceiling + drop risk. Needs a free City slot.",
    ),
    "Mbeumo": PlayerNote(
        "watch",
        "MUN £8.0. Template-adjacent. Inferior GW1–2 to Bruno; more minutes risk.",
    ),
    "Cunha": PlayerNote(
        "watch",
        "MUN £8.0, ~11% owned. Front-line flexibility. Minutes shared.",
    ),
    "Wirtz": PlayerNote(
        "watch",
        "LIV £7.5. Higher ceiling under Iraola; 2025/26 under-returned.",
    ),
    "Thiago": PlayerNote(
        "value",
        "BRE £8.0. 22 league goals / 181 pts. Aggressive/balanced FWD over DCL if unlocking £2.0.",
    ),
    "Watkins": PlayerNote(
        "value",
        "AVL £8.0 after −£1.0. Pens candidate post-Tielemans. Consistency prior.",
    ),
    "Mateta": PlayerNote(
        "watch",
        "CRY £6.5, pens. Minutes vs Strand Larsen. Mid-price FWD, not a Haaland replacement.",
    ),
    "Brobbey": PlayerNote(
        "watch",
        "SUN £6.0. GW1–2 only (IPS, FUL); then BRE/ARS/MCI. Do not set-and-forget.",
    ),
    "Kelleher": PlayerNote(
        "value",
        "BRE £5.0. 143 pts. Aggressive GK over Verbruggen if spending the £0.5 on outfield.",
    ),
    "Pickford": PlayerNote(
        "watch",
        "EVE £5.5. Nailed. Inferior PPP to Verbruggen/Kelleher.",
    ),
    "Dubravka": PlayerNote(
        "value",
        "TOT £4.0. Cheapest nailed GK. Use only to free £0.5.",
    ),
    "Colwill": PlayerNote(
        "watch",
        "CHE £5.0. Injury-short 2025/26. Cheaper DEFCON than Lacroix if he starts.",
    ),
    "Sarr": PlayerNote(
        "value",
        "CRY £6.5. Shots + pens if Mateta off. ~6% owned.",
    ),
    "Pedro Porro": PlayerNote(
        "watch",
        "TOT £5.5. Highest-owned TOT asset. Attack + CS. Template, not a differential.",
    ),
    "O'Reilly": PlayerNote(
        "watch",
        "MCI DEF £6.5. High owned. Needs a free City slot. CS + DEFCON.",
    ),
}


_ALIASES = {
    "Guehi": "Guéhi",
    "Joao Pedro": "João Pedro",
    "Van Dijk": "Virgil",
    "Le Fee": "E.Le Fée",
    "Le Fée": "E.Le Fée",
    "Fernandes": "B.Fernandes",
    "Bruno": "B.Fernandes",
    "Williams": "N.Williams",
}


def note_for(web_name: str) -> PlayerNote | None:
    return PLAYER_NOTES.get(web_name) or PLAYER_NOTES.get(_ALIASES.get(web_name, ""))
