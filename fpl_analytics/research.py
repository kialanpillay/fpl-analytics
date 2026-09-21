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
            "leaders have been re-priced. Tarkowski recorded 14 points "
            "(assist, CS, 12 DEFCON, 3 BP) at home to Ipswich. Mitchell "
            "recorded 7 from the bench. Both remain the defensive floor."
        ),
    },
    {
        "id": "premium-plus-enablers",
        "title": "Premium Captain Plus Enablers",
        "detail": (
            "Haaland is the season captain under the aggressive objective. "
            "GW6 is away at Liverpool. João Pedro recorded 0 minutes "
            "(knee, 75%). Isak at £9.1 is funded from Pedro plus bank. "
            "The sale is deferred until after the international break."
        ),
    },
    {
        "id": "minutes-over-upside",
        "title": "Minutes Over Upside",
        "detail": (
            "Gakpo recorded 90 minutes after a 30-minute GW4. Ndiaye "
            "recorded 74 minutes and an assist. Ampadu recorded 90 and "
            "14 DEFCON. Fábio Carvalho's 51-point horizon is 3 minutes "
            "and 12.9 xG per 90; that residual is not treated as a rate."
        ),
    },
    {
        "id": "fixture-swings",
        "title": "Fixture Swings, Not Season Averages",
        "detail": (
            "GW6: Arsenal home to Leeds; Tarkowski away at Hull; "
            "Manchester City away at Liverpool. Groß is away at "
            "Sunderland after 14 points against Arsenal. Hull residuals "
            "remain one-week samples. Re-solve when FDR turns."
        ),
    },
    {
        "id": "three-club-cap",
        "title": "Club Cap",
        "detail": (
            "Manchester City is at two (Haaland, Gvardiol) after Ndiaye "
            "was sold for Groß. Arsenal is at three (Raya, Calafiori, "
            "Ødegaard). Cherki is legal; Havertz is blocked by the "
            "Arsenal cap."
        ),
    },
    {
        "id": "chips",
        "title": "Chip Windows, 2026/27",
        "detail": (
            "Bench Boost and Triple Captain are available from GW1. "
            "Wildcard and Free Hit windows are GW2–19 and GW20–38. Five "
            "completed gameweeks plus an international break is still "
            "an insufficient minutes sample for the first wildcard. One "
            "free transfer remains after Ndiaye → Groß."
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
        "GW5: 90 min; 3 GC; 1 pt. GW6: H LEE. Selected in goal.",
    ),
    "Verbruggen": PlayerNote(
        "watch",
        "GW5: 90 min; CS; 6 pts (bench). GW6: A SUN. Omitted; Raya is selected versus Leeds.",
    ),
    "Mitchell": PlayerNote(
        "value",
        "GW5: 90 min; CS, 1 BP; 7 pts (bench). GW6: H NFO. Selected in a back three.",
    ),
    "Calafiori": PlayerNote(
        "value",
        "GW5: 90 min; 3 GC; 1 pt. GW6: H LEE. Selected.",
    ),
    "Tarkowski": PlayerNote(
        "value",
        "GW5: 90 min; 1 assist, CS, 12 DEFCON, 3 BP; 14 pts. GW6: A HUL. Selected. Retained.",
    ),
    "Gvardiol": PlayerNote(
        "value",
        "GW5: 90 min; 1 assist, 3 GC; 4 pts. GW6: A LIV. Selected. Retained.",
    ),
    "Van Hecke": PlayerNote(
        "watch",
        "GW5: 90 min; 1 goal, 3 GC; 6 pts (auto-sub for João Pedro). GW6: A MUN. First defensive substitute.",
    ),
    "Groß": PlayerNote(
        "value",
        "GW5: 90 min; 14 pts. GW6: A SUN. Selected. Incoming for Ndiaye.",
    ),
    "E.Le Fée": PlayerNote(
        "watch",
        "GW5: 90 min; 1 assist, 5 GC; 5 pts (bench). GW6: H BHA. Cover if Ampadu is retained.",
    ),
    "Ampadu": PlayerNote(
        "watch",
        "GW5: 90 min; CS, 14 DEFCON; 5 pts. GW6: A ARS. Retained.",
    ),
    "Gakpo": PlayerNote(
        "value",
        "GW5: 90 min; CS; 3 pts. GW6: H MCI. Selected. Retained.",
    ),
    "Ødegaard": PlayerNote(
        "value",
        "GW5: 72 min; 3 GC; 2 pts. GW6: H LEE. Selected. Retained.",
    ),
    "Haaland": PlayerNote(
        "value",
        "GW5 (C): 90 min; 1 goal, 3 GC; 6 pts (12 with multiplier). GW6: A LIV. Selected; season captain.",
    ),
    "Calvert-Lewin": PlayerNote(
        "watch",
        "GW5: 90 min; CS; 2 pts. GW6: A ARS. Selected until a João Pedro replacement arrives.",
    ),
    "João Pedro": PlayerNote(
        "avoid",
        "GW5: 0 min; knee, 75%; 0 pts. GW6: H BOU. Sale indicated after the break if the flag holds; Isak is the replacement.",
    ),
    # Recent sales
    "Ndiaye": PlayerNote(
        "watch",
        "Manchester City, £5.8. Sold for Groß. GW5: 74 min; 1 assist; 5 pts. City is now Haaland and Gvardiol.",
    ),
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
        "Manchester City, £7.8, 28.1% owned. GW5: 8 pts. GW6: A LIV. Blocked by the club cap until Ndiaye is sold.",
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
        "Manchester City defender. Legal after the Ndiaye sale; Gvardiol is the City defender.",
    ),
    "Isak": PlayerNote(
        "value",
        "Liverpool, £9.1, 22.1% owned. 413 min, 5 starts, 4 goals, 33 pts. Horizon 31.3 versus João Pedro 19.6 after the minutes discount. Funded from Pedro plus bank.",
    ),
    "Carvalho": PlayerNote(
        "avoid",
        "Brentford, £5.0, 0.0% owned. 3 min, 0 starts, 12.9 xG per 90. Insufficient sample. Residual is not treated as a rate.",
    ),
    "Schade": PlayerNote(
        "watch",
        "Brentford, £6.1, 10.0% owned. GW5: 9 pts. Not required after Groß.",
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
    "Gross": "Groß",
}


def note_for(web_name: str) -> PlayerNote | None:
    return PLAYER_NOTES.get(web_name) or PLAYER_NOTES.get(_ALIASES.get(web_name, ""))
