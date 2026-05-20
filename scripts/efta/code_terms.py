"""
Grep-able set of documented Epstein-network code terms / euphemisms.

Sources are court filings, sworn depositions, Palm Beach PD affidavits,
and reputable press (Bloomberg, CNN, ITV, Newsweek, NPR, Puck, Yahoo,
NZ Herald, CBS). See WORK_LOG.md for full source list.

Categories:
  A — court/affidavit/deposition documented
  B — journalist-decoded from primary documents
  C — generic suspicious phrasing worth flagging on face
"""
from collections import OrderedDict

# (category, term, one-line annotation)
TERMS = [
    # (a) operational euphemisms — court/depo documented
    ('A', 'massage',           "Giuffre 2016 depo: 'their code word' for commercial sex/abuse"),
    ('A', 'erotic massage',    "Giuffre depo gloss"),
    ('A', 'give a massage',    "Maxwell instruction phrasing per Giuffre"),
    ('A', 'masseuse',          "Black-book index heading"),
    ('A', 'masseur',           "Black-book index heading"),
    ('A', 'ugly back up',      "Verbatim Maxwell black-book section heading"),
    ('A', 'exercise people',   "8-name Maxwell black-book section"),
    ('A', 'schoolgirl',        "Court-ordered search term, Giuffre v Maxwell"),
    ('A', 'servitude',         "Same court-ordered search term"),
    ('A', 'career advice',     "Maxwell depo gloss for paying women to 'massage' Epstein"),
    ('A', 'pyramid scheme',    "Prosecution framing in US v Maxwell sentencing memo"),
    ('A', 'pyramid of abuse',  "Carolyn testimony framing"),
    ('A', 'the Principal',     "LSJ house manual: required staff term for Epstein"),
    # (a) initials — flight logs / NPA co-conspirator lists
    ('A', 'JE',                "Jeffrey Epstein"),
    ('A', 'GM',                "Ghislaine Maxwell"),
    ('A', 'G. Maxwell',        "GM expansion in logs"),
    ('A', 'LG',                "Lesley Groff — exec asst, DOJ 2019 co-conspirator list"),
    ('A', 'SK',                "Sarah Kellen — 2008 NPA unindicted co-conspirator"),
    ('A', 'NM',                "Nadia Marcinkova — 2008 NPA unindicted co-conspirator"),
    ('A', 'AR',                "Adriana Ross — 2008 NPA unindicted co-conspirator"),
    ('A', 'JLB',               "Jean-Luc Brunel — 2019 FBI co-conspirator list"),
    ('A', 'LW',                "Les Wexner — same 2019 FBI co-conspirator list"),
    ('A', 'DI',                "Darren Indyke, attorney — same list"),
    ('A', 'RK',                "Richard Kahn, accountant — same list"),
    # (b) journalist-decoded nicknames
    ('B', 'Mr. Big',           "Epstein's email nickname for Leon Black (Puck)"),
    ('B', 'Mr Big',            "Same, alt punctuation"),
    ('B', 'Invisible Man',     "Maxwell contact label for 'A' (believed Prince Andrew) (CNN/ITV/ABC)"),
    ('B', 'inappropriate friends', "Verbatim 'A'-to-Maxwell email Aug 2001 (ITV/CNN)"),
    # (a/b) geographic codes
    ('A', 'LSJ',               "Little Saint James"),
    ('A', 'Little St. James',  ""),
    ('A', 'Little Saint James',""),
    ('A', 'TIST',              "ICAO: Cyril E. King, St Thomas USVI — LSJ gateway"),
    ('A', 'TISX',              "ICAO: Henry Rohlsen, St Croix"),
    ('A', 'KTEB',              "ICAO: Teterboro NJ"),
    ('A', ' TEB ',             "TEB shorthand"),
    ('A', 'KPBI',              "ICAO: Palm Beach Intl"),
    ('A', ' PBI ',             "PBI shorthand"),
    ('A', 'KSAF',              "ICAO: Santa Fe Muni — Zorro Ranch gateway"),
    ('A', ' SAF ',             "SAF shorthand"),
    ('A', 'KCMH',              "ICAO: Columbus OH — Wexner hub"),
    ('A', 'Zorro',             "Zorro Ranch NM"),
    ('A', 'the Ranch',         "Zorro Ranch"),
    ('A', 'the Island',        "LSJ"),
    ('A', 'El Brillo',         "Palm Beach residence street"),
    # (b/c) scheduling / event tokens
    ('B', 'Alert -',           "Calendar-system subject prefix"),
    ('B', 'Alert—',            "Em-dash variant"),
    ('B', 'appointment',       "Groff/Kellen scheduled 'massage appointments'"),
    ('B', 'tea',               "Recurring social-invite token (CNN, Couric/Epstein 2010 emails)"),
    ('B', 'stop by',           "Informal-visit token"),
    # (c) generic flags
    ('C', 'underage',          ""),
    ('C', 'minor',             ""),
    ('C', 'young girl',        ""),
    ('C', 'young friend',      ""),
    ('C', 'new girl',          ""),
    ('C', 'new friend',        ""),
    ('C', 'high school',       ""),
    ('C', '8th grade',         "cf. Jane Doe Interlochen testimony"),
    ('C', 'interlochen',       "Camp name from Jane Doe testimony"),
    ('C', 'model',             "Recruitment cover (Brunel/MC2 context)"),
    ('C', 'models',            ""),
    ('C', 'spa',               "Recruitment venue (Mar-a-Lago spa)"),
    ('C', 'discreet',          ""),
    ('C', 'discretion',        ""),
    ('C', 'NDA',               ""),
    ('C', 'hush',              ""),
    ('C', 'keep quiet',        ""),
    ('C', 'never speak',       ""),
    ('C', 'co-conspirator',    ""),
]


def all_terms():
    """{term: (category, annotation)} preserving original order."""
    out = OrderedDict()
    for cat, term, note in TERMS:
        out[term] = (cat, note)
    return out
