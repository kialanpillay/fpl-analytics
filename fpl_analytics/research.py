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
        "19 clean sheets in 2025/26. £6.0. GW1: started, clean sheet. GW2 away at Aston Villa.",
    ),
    "Verbruggen": PlayerNote(
        "value",
        "130 points from 38 starts at £4.5. Regular starter. Retain unless £0.5 is required elsewhere.",
    ),
    "Mitchell": PlayerNote(
        "value",
        "135 points from 36 starts at £4.5. DEFCON coverage, 6.7% owned. Preferred first substitute.",
    ),
    "Calafiori": PlayerNote(
        "watch",
        "GW1: 80 minutes, assist and clean sheet, 9 points. First defensive transfer once £1.0 is available (Virgil).",
    ),
    "Tarkowski": PlayerNote(
        "value",
        "170 points, 376 DEFCON, 38 starts. Fixture-insensitive floor. Approximately 9% owned.",
    ),
    "Guéhi": PlayerNote(
        "value",
        "GW1: 90 minutes, one goal, two bonus, 10 points. Starting minutes confirmed. Retain for away at Crystal Palace.",
    ),
    "Van Hecke": PlayerNote(
        "watch",
        "GW1: started, 1 point. Competition for places remains. Home to Newcastle: retain.",
    ),
    "Semenyo": PlayerNote(
        "value",
        "GW1: 90 minutes, 2 points. Starting role confirmed. Retain.",
    ),
    "Rice": PlayerNote(
        "value",
        "184 points from 35 starts. DEFCON floor at £7.5. Completes the Arsenal allocation with Raya and Calafiori.",
    ),
    "Ndiaye": PlayerNote(
        "value",
        "Penalty taker. 7.19 xG against 6 goals. £6.0. Retain pending any confirmed departure.",
    ),
    "E.Le Fée": PlayerNote(
        "watch",
        "GW1: 79 minutes, 2 points. Home to Fulham is the remaining favourable fixture. Retain the free transfer.",
    ),
    "Ampadu": PlayerNote(
        "value",
        "Leeds, £5.5. Regular DEFCON midfielder, 1.5% owned. Suitable on the bench. Retain.",
    ),
    "Haaland": PlayerNote(
        "value",
        "GW1: 90 minutes, 0.74 xG, 2 points. Season captain. Retain.",
    ),
    "Calvert-Lewin": PlayerNote(
        "watch",
        "GW1: 90 minutes, 1 point. Minutes confirmed. Forward upgrade (Thiago or Wissa) is deferred.",
    ),
    "João Pedro": PlayerNote(
        "value",
        "GW1: 90 minutes, one goal, one assist, two bonus, 11 points. Retain. Palmer requires £2.0 and a second Chelsea place.",
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
        "175 points from 38 starts. £6.5. Favourable clean-sheet fixtures in GW3–4 (Ipswich, Fulham). +8 horizon versus Calafiori; −1 versus Guéhi. Requires £1.0.",
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
    "Senesi": PlayerNote(
        "avoid",
        "175 points at Bournemouth; now Tottenham. Centre-back depth under De Zerbi. Last season's residual does not transfer.",
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
        "value",
        "GW1: 2 points. GW2 home to Ipswich remains favourable. £12.0 would require a sale that is not justified after one gameweek.",
    ),
    "Palmer": PlayerNote(
        "watch",
        "GW1: 82 minutes, one goal, one assist, three bonus, 13 points. GW2 home to Brighton. £9.5. Requires £2.0, a midfield sale, and a second Chelsea place.",
    ),
    "Szoboszlai": PlayerNote(
        "value",
        "GW1: 90 minutes, 8 points. £7.0, home to Nottingham Forest. Preferred midfield upgrade once £1.5 is available.",
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
        "133 points, 307 DEFCON, £5.5, 1.3% owned. Defensive midfield floor. Unavailable if Arsenal is already at three.",
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
        "Manchester City, £7.0, approximately 5% owned. High ceiling and rotation risk. Requires a vacant City place.",
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
        "Manchester City defender, £6.5. High ownership. Requires a vacant City place. Clean sheets and DEFCON.",
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
