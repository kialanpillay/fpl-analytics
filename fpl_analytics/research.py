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
            "2025/26 VAPM was DEFCON-driven. Last year's leaders are already "
            "re-priced. Remaining floors: Tarkowski, Mitchell, Ampadu. Rice "
            "has been sold for Gakpo."
        ),
    },
    {
        "id": "premium-plus-enablers",
        "title": "Premium Captain Plus Enablers",
        "detail": (
            "Haaland remains the season captain. GW3 home to Coventry. "
            "Bruno's 23-point GW2 does not fit the budget without selling "
            "Haaland. Mitchell is the 4.5 enabler; Verbruggen starts vs Leeds."
        ),
    },
    {
        "id": "minutes-over-upside",
        "title": "Minutes Over Upside",
        "detail": (
            "Senesi recorded 0 minutes in GW2 and has been sold. Van Hecke "
            "started that match and stays as Tottenham cover. Calafiori "
            "remains the later defensive upgrade once £1.0 is free."
        ),
    },
    {
        "id": "fixture-swings",
        "title": "Fixture Swings, Not Season Averages",
        "detail": (
            "GW3: City (Haaland, Semenyo, Gvardiol) home to Coventry. "
            "Gakpo away at Ipswich. Pedro away at Arsenal loses the armband. "
            "Fade Hull after one result. Re-solve when FDR turns."
        ),
    },
    {
        "id": "three-club-cap",
        "title": "Club Cap",
        "detail": (
            "Manchester City is at three (Haaland, Semenyo, Gvardiol). "
            "Arsenal is at two (Raya, Calafiori). Liverpool is Gakpo. "
            "Cherki and Foden remain blocked."
        ),
    },
    {
        "id": "chips",
        "title": "Chip Windows, 2026/27",
        "detail": (
            "BB and TC from GW1. WC/FH: GW2–19 and GW20–38. First wildcard "
            "waits until minutes settle; two gameweeks remains early."
        ),
    },
]


@dataclass(frozen=True)
class PlayerNote:
    tone: str  # risk | value | watch | avoid
    note: str


PLAYER_NOTES: dict[str, PlayerNote] = {
    # Current 15 (Senesi → Gvardiol, Rice → Gakpo)
    "Raya": PlayerNote(
        "value",
        "GW2: 90 minutes, clean sheet, 6 points at Villa. GW3 home to Chelsea. Bench if starting Verbruggen.",
    ),
    "Verbruggen": PlayerNote(
        "value",
        "GW2: 90 minutes, 0 points (four conceded). GW3 home to Leeds. Preferred goalkeeper this week.",
    ),
    "Mitchell": PlayerNote(
        "value",
        "GW2: 90 minutes, 0 points at Manchester City. First substitute. Away at Fulham.",
    ),
    "Calafiori": PlayerNote(
        "value",
        "GW2: 90 minutes, assist, clean sheet, two bonus, 11 points. £5.6. Home to Chelsea. Start.",
    ),
    "Tarkowski": PlayerNote(
        "value",
        "GW2: 90 minutes, one goal, two bonus, 12 points. Home to Manchester United. Start. Retain.",
    ),
    "Gvardiol": PlayerNote(
        "value",
        "In for Senesi at £5.6. GW2: 74 minutes, assist, 5 points. Home to Coventry. Start. Completes the City allocation.",
    ),
    "Van Hecke": PlayerNote(
        "watch",
        "GW2: started, 90 minutes, 1 point. Away at Nottingham Forest. Bench this week.",
    ),
    "Semenyo": PlayerNote(
        "value",
        "GW2: 90 minutes, assist, 5 points. Home to Coventry with Haaland and Gvardiol. Start. Retain.",
    ),
    "Rice": PlayerNote(
        "watch",
        "Sold for Gakpo (£7.5 → £7.0). GW2: 84 minutes, clean sheet, 3 points, 11 DEFCON. Arsenal allocation is now Raya and Calafiori.",
    ),
    "Ndiaye": PlayerNote(
        "value",
        "GW2: 90 minutes, 4 points, 13 DEFCON. Penalty taker. Home to Manchester United. Start. Retain.",
    ),
    "E.Le Fée": PlayerNote(
        "watch",
        "GW2: 89 minutes, clean sheet, 3 points. Away at Brentford. Bench; Gakpo starts.",
    ),
    "Ampadu": PlayerNote(
        "value",
        "GW2: 90 minutes, 2 points, 10 DEFCON. Away at Brighton. Start. Retain.",
    ),
    "Haaland": PlayerNote(
        "value",
        "GW2: 90 minutes, two goals, three bonus, 13 points. Season captain. Home to Coventry.",
    ),
    "Calvert-Lewin": PlayerNote(
        "value",
        "GW2: 90 minutes, one goal, two bonus, 8 points. Away at Brighton. Starts in a 3-4-3.",
    ),
    "João Pedro": PlayerNote(
        "value",
        "GW2 captain: 90 minutes, goal, assist, two bonus, 9 points (18 with armband). Away at Arsenal. Start; Haaland takes the armband.",
    ),
    "Senesi": PlayerNote(
        "avoid",
        "Sold for Gvardiol. GW2: unused (0 minutes). Official score 75 does not yet include the Calvert-Lewin automatic substitute.",
    ),
    "Guéhi": PlayerNote(
        "watch",
        "Sold earlier. GW2: 90 minutes, 2 points. City slot is Gvardiol.",
    ),
    # Sold / alternatives
    "Rogers": PlayerNote(
        "watch",
        "Joined Chelsea in July 2026. 169 points from 37 starts at Aston Villa. Sociedad: 62 minutes and a debut goal. New-system risk. Horizon 22 against Semenyo 30.",
    ),
    "Lacroix": PlayerNote(
        "watch",
        "Joined Chelsea in July 2026. DEFCON 10.8 per 90. Tarkowski is the same-price minutes upgrade.",
    ),
    "Ngumoha": PlayerNote(
        "risk",
        "£6.0. Five starts, 547 minutes, 0.55 xGI per 90. Insufficient sample for a midfield place.",
    ),
    "Fletcher": PlayerNote(
        "risk",
        "Tyler Fletcher: 17 minutes, no starts. Unusable at £4.5.",
    ),
    "J.Fletcher": PlayerNote(
        "risk",
        "Jack Fletcher: 107 minutes. Did not start in GW1.",
    ),
    # Model targets
    "Virgil": PlayerNote(
        "value",
        "GW2: 90 minutes, 1 point. £6.5. Away at Ipswich, then Fulham. Alternative to Calafiori once £1.0 is available.",
    ),
    "Truffert": PlayerNote(
        "value",
        "165 points at £5.5. Full-back with DEFCON, 4.9% owned. Difficult GW1 (away at Manchester City). Same-price alternative to Calafiori.",
    ),
    "N.Williams": PlayerNote(
        "value",
        "Crystal Palace wing-back: attack, DEFCON and set pieces. £5.0. Approximately 11% owned.",
    ),
    "Gabriel": PlayerNote(
        "value",
        "209 points. Priced at a record £8.0. Clean sheets and attacking returns. The price already reflects last season.",
    ),
    "Mosquera": PlayerNote(
        "watch",
        "£5.5 Arsenal centre-back while Saliba is unavailable. Relevant only if selected. £2.5 below Gabriel.",
    ),
    "White": PlayerNote(
        "watch",
        "£5.5. Minutes available while Timber is absent. Short-term Arsenal defender, not a season holding.",
    ),
    "Richards": PlayerNote(
        "value",
        "Crystal Palace, £5.0, 0.8% owned. DEFCON floor. Low-owned alternative to Calafiori.",
    ),
    "Thiaw": PlayerNote(
        "value",
        "Newcastle, £5.0. 126 points, 12 DEFCON returns, 4 goals. 1.9% owned.",
    ),
    "Collins": PlayerNote(
        "value",
        "Brentford, £5.5. DEFCON and clean sheets. 2% owned.",
    ),
    "Shaw": PlayerNote(
        "watch",
        "£4.5 Manchester United. Widely owned enabler. Minutes more reliable than attacking output.",
    ),
    "B.Fernandes": PlayerNote(
        "watch",
        "GW2: 90 minutes, 23 points, three bonus. Away at Everton. £12.0. Requires selling Haaland to fund. Not a transfer.",
    ),
    "Palmer": PlayerNote(
        "watch",
        "GW2: 90 minutes, 7 points. £9.6. Away at Arsenal. Requires £2.0, a midfield sale, and a second Chelsea place.",
    ),
    "Gakpo": PlayerNote(
        "value",
        "In for Rice at £7.0. 160 minutes, 5 points. Away at Ipswich. Starts in the midfield four.",
    ),
    "Cherki": PlayerNote(
        "watch",
        "GW2: 108 minutes, 14 points. Home to Coventry. City allocation is full.",
    ),
    "Hall": PlayerNote(
        "value",
        "£5.0. 180 minutes, 11 points. Home to Bournemouth. Alternative to Gvardiol; the City move was taken.",
    ),
    "Szoboszlai": PlayerNote(
        "watch",
        "GW2: 90 minutes, 4 points. £7.0. Away at Ipswich. Same fixture as Gakpo; not required.",
    ),
    "Enzo": PlayerNote(
        "watch",
        "Model midfield (horizon 30) at £7.0, 5% owned. Transfer speculation. Incompatible with a three-man Chelsea allocation.",
    ),
    "Anderson": PlayerNote(
        "watch",
        "180 points and 515 DEFCON, now Manchester City at £6.5. Residual reflects last season; City minutes are unconfirmed.",
    ),
    "Zubimendi": PlayerNote(
        "value",
        "133 points, 307 DEFCON, £5.5, 1.3% owned. Defensive midfield floor. Arsenal is at two; one slot remains.",
    ),
    "Gravenberch": PlayerNote(
        "value",
        "144 points, 298 DEFCON, £6.0, under 2% owned. More secure Liverpool minutes than Ngumoha.",
    ),
    "Tavernier": PlayerNote(
        "value",
        "Bournemouth creator and set-piece taker. 13.7 xGI, 1.7% owned. Penalties contingent on Kluivert.",
    ),
    "Xhaka": PlayerNote(
        "value",
        "Sunderland, £5.5. 26 DEFCON points from 32 starts. GW1 away at Ipswich. Minutes floor rather than attacking ceiling.",
    ),
    "Wharton": PlayerNote(
        "value",
        "Crystal Palace, £5.5. High residual, 0.7% owned. DEFCON midfielder. Minutes still to be established.",
    ),
    "Foden": PlayerNote(
        "watch",
        "Manchester City, £7.0. High ceiling and rotation risk. City allocation is full.",
    ),
    "Mbeumo": PlayerNote(
        "watch",
        "Manchester United, £8.0. Template-adjacent. Inferior GW1–2 fixtures to Fernandes; greater minutes uncertainty.",
    ),
    "Cunha": PlayerNote(
        "watch",
        "Manchester United, £8.0, approximately 11% owned. Flexible forward. Minutes are shared.",
    ),
    "Wirtz": PlayerNote(
        "watch",
        "Liverpool, £7.5. Higher attacking ceiling under Iraola; under-returned in 2025/26.",
    ),
    "Thiago": PlayerNote(
        "value",
        "Brentford, £8.0. 22 league goals, 181 points. Preferred balanced forward over Calvert-Lewin if £2.0 can be raised.",
    ),
    "Watkins": PlayerNote(
        "value",
        "Aston Villa, £8.0 after a £1.0 fall. Penalty candidate following Tielemans. Strong consistency prior.",
    ),
    "Mateta": PlayerNote(
        "watch",
        "Crystal Palace, £6.5, penalties. Minutes shared with Strand Larsen. Mid-price forward; not a substitute for Haaland.",
    ),
    "Brobbey": PlayerNote(
        "watch",
        "Sunderland, £6.0. Favourable only in GW1–2 (Ipswich, Fulham); thereafter Brentford, Arsenal and Manchester City.",
    ),
    "Kelleher": PlayerNote(
        "value",
        "Brentford, £5.0. 143 points. Alternative to Verbruggen if the £0.5 is required in the outfield.",
    ),
    "Pickford": PlayerNote(
        "watch",
        "Everton, £5.5. Regular starter. Inferior points per pound to Verbruggen and Kelleher.",
    ),
    "Dubravka": PlayerNote(
        "value",
        "Tottenham, £4.0. Lowest-priced regular goalkeeper. Use only to release £0.5.",
    ),
    "Colwill": PlayerNote(
        "watch",
        "Chelsea, £5.0. Injury-shortened 2025/26. Cheaper DEFCON than Lacroix if selected.",
    ),
    "Sarr": PlayerNote(
        "value",
        "Crystal Palace, £6.5. Shot volume; penalties if Mateta is absent. Approximately 6% owned.",
    ),
    "Pedro Porro": PlayerNote(
        "watch",
        "Tottenham, £5.5. Highest-owned Tottenham asset. Attack and clean sheets. High ownership, limited differential.",
    ),
    "O'Reilly": PlayerNote(
        "watch",
        "Manchester City defender, £6.5. City allocation is full (Gvardiol).",
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
