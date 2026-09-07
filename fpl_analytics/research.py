"""Qualitative flags attached to live model scores.

Sources: official FPL API; BBC 2026/27 guide (14 Aug 2026); Community
Shield ARS 3–0 MCI (16 Aug 2026). Recalibrate after each international
break: minutes, set pieces, and club registrations move.

Note format (keep all new text on this contract):

* Strategies: two to four declarative sentences. Evidence first;
  implication second. No slogans, no second-person imperatives.
* Squad notes: ``GW{n}: {min} min; {events}; {pts} pts. GW{n+1}: {venue}
  {opp}. {role}.``
* Watch / sold notes: ``{club}, £{price}. {sample}. {constraint}.``
* Role vocabulary: selected, omitted, retained, sold, ineligible,
  blocked by club cap, insufficient sample.
"""

from __future__ import annotations

from dataclasses import dataclass

STRATEGIES = [
    {
        "id": "defcon-value",
        "title": "Underpriced DEFCON Assets",
        "detail": (
            "2025/26 value-above-replacement was DEFCON-weighted; those "
            "leaders have been re-priced. Mitchell recorded 15 points from "
            "the GW3 bench and is selected at home to Ipswich. Tarkowski "
            "remains the minutes-and-DEFCON floor in defence."
        ),
    },
    {
        "id": "premium-plus-enablers",
        "title": "Premium Captain Plus Enablers",
        "detail": (
            "Haaland is the season captain under the aggressive objective "
            "(next-GW, horizon, attack; no price or minutes floor). GW4 is "
            "away at Manchester United; João Pedro at home to Hull is the "
            "week captain. B.Fernandes at £12.0 is unfunded without a "
            "Haaland sale."
        ),
    },
    {
        "id": "minutes-over-upside",
        "title": "Minutes Over Upside",
        "detail": (
            "Senesi has been sold. GW3 left 34 points on the bench "
            "(Mitchell 15, Van Hecke 8, E.Le Fée 8). Ampadu is omitted "
            "from the GW4 XI. No free transfer remains after Semenyo → "
            "Ødegaard."
        ),
    },
    {
        "id": "fixture-swings",
        "title": "Fixture Swings, Not Season Averages",
        "detail": (
            "GW4 attack fixtures: João Pedro home to Hull, Gakpo home to "
            "Fulham, Mitchell home to Ipswich, Calvert-Lewin home to "
            "Newcastle. Manchester City assets are away at Manchester "
            "United. Hull assets are away at Chelsea; one-week residuals "
            "are not treated as rates. Re-solve when FDR turns."
        ),
    },
    {
        "id": "three-club-cap",
        "title": "Club Cap",
        "detail": (
            "Manchester City is at three (Haaland, Ndiaye, Gvardiol) after "
            "Semenyo → Ødegaard. Arsenal is at three (Raya, Calafiori, "
            "Ødegaard). Cherki, Foden and a fourth City player are blocked."
        ),
    },
    {
        "id": "chips",
        "title": "Chip Windows, 2026/27",
        "detail": (
            "Bench Boost and Triple Captain are available from GW1. "
            "Wildcard and Free Hit windows are GW2–19 and GW20–38. Three "
            "completed gameweeks is an insufficient minutes sample for "
            "the first wildcard. No free transfer remains this deadline."
        ),
    },
]


@dataclass(frozen=True)
class PlayerNote:
    tone: str  # risk | value | watch | avoid
    note: str


PLAYER_NOTES: dict[str, PlayerNote] = {
    # Current 15
    "Raya": PlayerNote(
        "value",
        "GW3: 90 min; 1 GC; 3 pts. GW4: A SUN. Omitted; Verbruggen is selected versus Coventry.",
    ),
    "Verbruggen": PlayerNote(
        "value",
        "GW3: 90 min; 1 GC; 3 pts. GW4: A COV. Selected in goal.",
    ),
    "Mitchell": PlayerNote(
        "value",
        "GW3: 73 min; 2 goals, 2 BP; 15 pts (bench). GW4: H IPS. Selected.",
    ),
    "Calafiori": PlayerNote(
        "value",
        "GW3: 66 min; 2 pts. GW4: A SUN. Selected in a back four.",
    ),
    "Tarkowski": PlayerNote(
        "value",
        "GW3: 90 min; 12 DEFCON; 3 pts. GW4: A TOT. Selected. Retained.",
    ),
    "Gvardiol": PlayerNote(
        "value",
        "GW3: 90 min; CS, 2 BP; 8 pts. GW4: A MUN. Selected. Retained as the City defender.",
    ),
    "Van Hecke": PlayerNote(
        "watch",
        "GW3: 90 min; CS, 2 BP; 8 pts (bench). GW4: H EVE. GW xPts 3.57 versus Calvert-Lewin 3.54. Omitted from a 4-3-3.",
    ),
    "Ndiaye": PlayerNote(
        "watch",
        "GW3: 86 min; CS; 3 pts. GW4: A MUN. Selected. Third Manchester City player.",
    ),
    "E.Le Fée": PlayerNote(
        "watch",
        "GW3: 90 min; 1 goal, 1 BP; 8 pts (bench). GW4: H ARS. Omitted.",
    ),
    "Ampadu": PlayerNote(
        "watch",
        "GW3: 90 min; 11 DEFCON; 2 pts. GW4: H NEW. GW xPts 3.38. Omitted.",
    ),
    "Gakpo": PlayerNote(
        "value",
        "GW3: 90 min; 2 assists, CS, 2 BP; 11 pts. GW4: H FUL. Selected. Retained.",
    ),
    "Ødegaard": PlayerNote(
        "value",
        "In for Semenyo at £6.6. GW3: 10 pts. GW4: A SUN. Selected. Completes the Arsenal allocation.",
    ),
    "Haaland": PlayerNote(
        "value",
        "GW3 (C): 90 min; 1 goal, CS, 3 BP; 9 pts (18 with multiplier). GW4: A MUN. Selected; vice-captain.",
    ),
    "Calvert-Lewin": PlayerNote(
        "watch",
        "GW3: 72 min; yellow card; 1 pt. GW4: H NEW. GW xPts 3.54 versus Van Hecke 3.57. Selected in a 4-3-3 under the aggressive objective.",
    ),
    "João Pedro": PlayerNote(
        "value",
        "GW3: 90 min; yellow card; 1 pt. GW4: H HUL. Selected; week captain.",
    ),
    # Recent sales
    "Semenyo": PlayerNote(
        "avoid",
        "Manchester City, £8.4. Sold for Ødegaard. GW3: 90 min; 1 assist, CS; 6 pts. Lowest City horizon among the previous four.",
    ),
    "Rice": PlayerNote(
        "watch",
        "Arsenal, £7.4. Sold for Gakpo. GW3: 90 min; 1 assist; 5 pts. Reverse transfer is not indicated.",
    ),
    "Senesi": PlayerNote(
        "avoid",
        "Tottenham, £5.9. Sold for Gvardiol. Unused in GW2 and GW3.",
    ),
    "Guéhi": PlayerNote(
        "watch",
        "Sold prior to GW2. City defensive slot is Gvardiol.",
    ),
    # Watchlist
    "Scott": PlayerNote(
        "watch",
        "Bournemouth, £6.0, 3.6% owned. Aggressive 1-for-1 replacement for Semenyo (Δ horizon +18.8). GW4: H BRE. Early-season sample.",
    ),
    "Rogers": PlayerNote(
        "watch",
        "Chelsea, July 2026 arrival. 169 pts from 37 starts at Aston Villa. Sociedad: 62 min, debut goal. New-system minutes risk.",
    ),
    "Lacroix": PlayerNote(
        "watch",
        "Chelsea, July 2026 arrival. DEFCON 10.8 per 90. Tarkowski is the same-price minutes upgrade.",
    ),
    "Ngumoha": PlayerNote(
        "risk",
        "Liverpool, £6.0. Five starts, 547 min, 0.55 xGI per 90. Insufficient sample for a midfield place.",
    ),
    "Fletcher": PlayerNote(
        "risk",
        "Manchester United (Tyler), £4.5. 17 min, no starts. Insufficient sample.",
    ),
    "J.Fletcher": PlayerNote(
        "risk",
        "Manchester United (Jack). 107 min. Did not start GW1. Insufficient sample.",
    ),
    "Virgil": PlayerNote(
        "value",
        "Liverpool, £6.5. GW2: 90 min; 1 pt. Defensive alternative to Calafiori once £1.0 is available.",
    ),
    "Truffert": PlayerNote(
        "value",
        "Bournemouth, £5.5, 4.9% owned. 165 pts; full-back DEFCON. Same-price alternative to Calafiori.",
    ),
    "N.Williams": PlayerNote(
        "value",
        "Crystal Palace, £5.0, ~11% owned. Wing-back: attack, DEFCON, set pieces.",
    ),
    "Gabriel": PlayerNote(
        "value",
        "Arsenal, £8.0. 209 pts. Price already capitalises last-season clean sheets and attacking returns.",
    ),
    "Mosquera": PlayerNote(
        "watch",
        "Arsenal, £5.5. Centre-back cover while Saliba is unavailable. Relevant only if selected. £2.5 below Gabriel.",
    ),
    "White": PlayerNote(
        "watch",
        "Arsenal, £5.5. Minutes available while Timber is absent. Short-horizon defender, not a season holding.",
    ),
    "Richards": PlayerNote(
        "value",
        "Crystal Palace, £5.0, 0.8% owned. DEFCON floor. Low-owned alternative to Calafiori.",
    ),
    "Thiaw": PlayerNote(
        "value",
        "Newcastle, £5.0, 1.9% owned. 126 pts, 12 DEFCON returns, 4 goals.",
    ),
    "Collins": PlayerNote(
        "value",
        "Brentford, £5.5, 2% owned. DEFCON and clean sheets.",
    ),
    "Shaw": PlayerNote(
        "watch",
        "Manchester United, £4.5. Widely owned enabler. Minutes more reliable than attacking output.",
    ),
    "B.Fernandes": PlayerNote(
        "watch",
        "Manchester United, £12.0. GW3: 2 pts. GW4: H MCI. Requires a Haaland sale. Not a transfer.",
    ),
    "Palmer": PlayerNote(
        "watch",
        "Chelsea, £9.6. GW3: 1 pt. GW4: H HUL. Semenyo proceeds plus bank were £9.3; short £0.3 at the time of the Ødegaard move.",
    ),
    "Cherki": PlayerNote(
        "watch",
        "Manchester City, £7.8. GW3: 3 pts. GW4: A MUN. Blocked by the club cap.",
    ),
    "Hall": PlayerNote(
        "value",
        "Newcastle, £5.1. 180 min, 11 pts through GW2. Alternative City-exit defender; Gvardiol was selected.",
    ),
    "Szoboszlai": PlayerNote(
        "watch",
        "Liverpool, £7.0. GW2: 90 min; 4 pts. Same club as Gakpo; not required.",
    ),
    "Enzo": PlayerNote(
        "watch",
        "Listed as Manchester City, £6.9, 3.3% owned. Minutes unconfirmed. Ineligible alongside three City outfielders.",
    ),
    "Anderson": PlayerNote(
        "watch",
        "Manchester City, £6.5. 180 pts and 515 DEFCON last season. Residual is prior-club; City minutes unconfirmed. Blocked by the club cap.",
    ),
    "Zubimendi": PlayerNote(
        "value",
        "Arsenal, £5.5. Defensive-midfield floor. Ineligible; Arsenal is at three after Ødegaard.",
    ),
    "Gravenberch": PlayerNote(
        "value",
        "Liverpool, £6.0, <2% owned. 144 pts, 298 DEFCON. More secure Liverpool minutes than Ngumoha.",
    ),
    "Tavernier": PlayerNote(
        "value",
        "Bournemouth, £6.0, 2.9% owned. Creator and set-piece taker. Penalties contingent on Kluivert.",
    ),
    "Xhaka": PlayerNote(
        "value",
        "Sunderland, £5.5. 26 DEFCON points from 32 starts. Minutes floor rather than attacking ceiling.",
    ),
    "Wharton": PlayerNote(
        "value",
        "Crystal Palace, £5.5, 0.7% owned. High residual. DEFCON midfielder; minutes still to be established.",
    ),
    "Foden": PlayerNote(
        "watch",
        "Manchester City. High ceiling, rotation risk. Blocked by the club cap.",
    ),
    "Mbeumo": PlayerNote(
        "watch",
        "Manchester United, £7.9. Template-adjacent. Minutes less certain than B.Fernandes.",
    ),
    "Cunha": PlayerNote(
        "watch",
        "Manchester United, £8.0, ~11% owned. Flexible forward. Minutes are shared.",
    ),
    "Wirtz": PlayerNote(
        "watch",
        "Liverpool, £7.4. Attacking ceiling under Iraola; under-returned in 2025/26.",
    ),
    "Thiago": PlayerNote(
        "value",
        "Brentford, £8.0. 22 league goals, 181 pts. Preferred mid-price forward over Calvert-Lewin if £2.0 can be raised.",
    ),
    "Watkins": PlayerNote(
        "value",
        "Aston Villa, £8.0 after a £1.0 fall. Penalty candidate following Tielemans.",
    ),
    "Mateta": PlayerNote(
        "watch",
        "Crystal Palace, £6.5. Penalties. Minutes shared with Strand Larsen. Not a Haaland substitute.",
    ),
    "Brobbey": PlayerNote(
        "watch",
        "Sunderland, £6.0. Opening fixtures (Ipswich, Fulham) have passed; remaining run includes Arsenal and Manchester City.",
    ),
    "Kelleher": PlayerNote(
        "value",
        "Brentford, £5.0. 143 pts. Alternative to Verbruggen if £0.5 is required in the outfield.",
    ),
    "Pickford": PlayerNote(
        "watch",
        "Everton, £5.5. Regular starter. Inferior points per pound to Verbruggen and Kelleher.",
    ),
    "Dubravka": PlayerNote(
        "value",
        "Tottenham, £4.0. Lowest-priced regular goalkeeper. Used only to release £0.5.",
    ),
    "Colwill": PlayerNote(
        "watch",
        "Chelsea, £5.0. Injury-shortened 2025/26. Cheaper DEFCON than Lacroix if selected.",
    ),
    "Sarr": PlayerNote(
        "value",
        "Crystal Palace, £6.5, ~6% owned. Shot volume; penalties if Mateta is absent.",
    ),
    "Pedro Porro": PlayerNote(
        "watch",
        "Tottenham, £5.5. Highest-owned Tottenham asset. Attack and clean sheets. High ownership, limited differential.",
    ),
    "O'Reilly": PlayerNote(
        "watch",
        "Manchester City defender. Blocked while Gvardiol, Haaland and Ndiaye remain.",
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
    "Odegaard": "Ødegaard",
}


def note_for(web_name: str) -> PlayerNote | None:
    return PLAYER_NOTES.get(web_name) or PLAYER_NOTES.get(_ALIASES.get(web_name, ""))
