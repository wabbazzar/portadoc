"""
Grep targets sourced from publicly-published EFTA reporting (Nov 2025 – May 2026).

When these strings appear in our local corpus, we've recreated (or can
explicitly fail-to-recreate) a finding reported by name elsewhere.
Every entry carries a one-line provenance pointer.

TERMS structure: (term, category, note) — kept as 3-tuple for backward
compatibility with grep_terms.py / name_grep.py / person_dossier.py.
URLs for the press_recreate page are looked up via URL_BY_TERM below
(missing entries render without a hyperlink).

See WORK_LOG.md "Sources" section for the full URL list.
"""

# (term, category, brief sourcing note)
TERMS = [
    # === Names called out in published Nov 2025 / Jan-May 2026 coverage ===
    ('Ruemmler',          'name',   "Kathryn Ruemmler, Goldman CLO; 'I know how dirty donald is' recipient (CNBC, Nov 12 2025)"),
    ('Wolff',             'name',   "Michael Wolff, journalist; 'let him hang himself' / 'knew about the girls' threads (CBS Nov 12)"),
    ('Maxwell',           'name',   "Ghislaine Maxwell"),
    ('Indyke',            'name',   "Darren Indyke, estate co-executor (Bloomberg)"),
    ('Kahn',              'name',   "Richard Kahn, estate co-accountant (Bloomberg)"),
    ('Groff',             'name',   "Lesley Groff, Epstein exec asst (Yahoo / DOJ co-conspirator list)"),
    ('Kellen',            'name',   "Sarah Kellen, scheduler/recruiter; 2008 NPA co-conspirator"),
    ('Marcinkova',        'name',   "Nadia Marcinkova; 2008 NPA co-conspirator"),
    ('Adriana Ross',      'name',   "2008 NPA co-conspirator"),
    ('Mucinska',          'name',   "Adriana Ross alias"),
    ('Brunel',            'name',   "Jean-Luc Brunel; 2019 FBI co-conspirator list"),
    ('Wexner',            'name',   "Les Wexner; 2019 FBI co-conspirator list (NPR Feb 3 2026)"),
    ('Staley',            'name',   "Jes Staley, JPM; ~1,200 emails 2008–12; Snow White / Beauty and the Beast thread"),
    ('Leon Black',        'name',   "Apollo Global; 'Mr. Big' per Puck"),
    ('Acosta',            'name',   "Alexander Acosta, AUSA who gave the 2007 NPA"),
    ('Joi Ito',           'name',   "MIT Media Lab; $1.7M 'anonymous' donation routing"),
    ('Krauss',            'name',   "Lawrence Krauss, sent Epstein draft of BuzzFeed harassment response"),
    ('Mandelson',         'name',   "Peter Mandelson; arrested Feb 23 2026 (ABC)"),
    ('Ehud Barak',        'name',   "60+ meetings Sep 2010 – Mar 2019 per estate emails"),
    ('Ariane de Rothschild', 'name', "2013 scheduling email"),
    ('Hamad bin Jassim',  'name',   "Qatari PM, in 2016 'tent carpets and all' thread"),
    ('Mohammed bin Salman', 'name', "MBS; gifted Epstein a tent in 2016 (House Oversight)"),
    (' MBS ',             'name',   "MBS initials"),
    ('Reid Hoffman',      'name',   "Two 2014 island visits + townhouse overnight (WSJ via Fortune)"),
    ('Summers',           'name',   "Larry Summers; stepped back from Harvard / OpenAI after release"),
    ('Chomsky',           'name',   "Noam Chomsky; 'highly valued friend' LOR + 2015 currency emails"),
    ('Jagland',           'name',   "Thorbjørn Jagland, former Norwegian PM (per Wikipedia summary)"),
    ('John Pond',         'name',   "Pseudonym attributed to Gordon Brown (uncited primary; flag for verification)"),
    ('Mark Epstein',      'name',   "Jeffrey's brother"),
    ('Murray Gell-Mann',  'name',   "Caltech physicist; First Fifty Years contributor"),
    ('Bannon',            'name',   "Steve Bannon; 'europe by remote doesn't work' 2018 thread"),
    ('Dershowitz',        'name',   "Alan Dershowitz; First Fifty Years contributor"),
    ('Trump',             'name',   "Oversight Dems: 'easily more than a thousand' mentions across Nov-2025 batch"),
    ('Musk',              'name',   "Elon Musk; LSJ trip on schedule (Sept 2025 release)"),
    ('Thiel',             'name',   "Peter Thiel; scheduled meetings"),
    ('Andrew',            'name',   "Prince Andrew / 'A' / 'Invisible Man'"),
    ('Mountbatten-Windsor', 'name', "Prince Andrew's surname change"),
    ('Branson',           'name',   "Richard Branson; Dec 2025 photo batch"),
    ('Bill Gates',        'name',   "2011 dinner w/ Summers + Staley; 'breakfast party' (Fortune)"),
    ('Woody Allen',       'name',   "Dec 2025 photo batch"),
    ('Clinton',           'name',   "Bill Clinton; First Fifty Years + Dec 2025 photo batch"),

    # === Verbatim phrases reporters have quoted ===
    ("dog that hasn't barked",       'phrase', "Epstein → Maxwell Apr 2011 re: Trump (CBS, PBS)"),
    ('dog that hasn',                'phrase', "Loose match — OCR may eat apostrophe"),
    ('knew about the girls',         'phrase', "Epstein → Wolff Jan 31 2019 (CBS)"),
    ('asked ghislaine to stop',      'phrase', "Same Jan 31 2019 Wolff thread"),
    ('craft an answer',              'phrase', "Epstein → Wolff, Dec 2015 (PBS)"),
    ('let him hang himself',         'phrase', "Wolff → Epstein, Dec 2015"),
    ('dirty donald',                 'phrase', "Epstein → Ruemmler (CNBC)"),
    ('fixer flip',                   'phrase', "Same Ruemmler thread"),
    ('so gross',                     'phrase', "Ruemmler/Epstein on Trump"),
    ('worse in real life',           'phrase', "Same"),
    ('Snow White',                   'phrase', "Staley/Epstein Jul 2010 (Bloomberg)"),
    ('Beauty and the Beast',         'phrase', "Same Staley thread"),
    ('what character would you like next', 'phrase', "Same Staley thread"),
    ('highly valued friend',         'phrase', "Chomsky LOR for Epstein"),
    ('tastey models',                'phrase', "Mandelson St Petersburg 2013 (sic spelling)"),
    ('tasty models',                 'phrase', "Possible OCR cleanup of 'tastey'"),
    ('kissinger china guy',          'phrase', "Barak/Rothschild Sep 2013 scheduling thread"),
    ('carpets and all',              'phrase', "MBS tent gift 2016 (House Oversight)"),
    ('First Fifty Years',            'phrase', "Maxwell's 50th-bday book title"),
    ('Centurion',                    'phrase', "AmEx black-card concierge service (CBS, Bloomberg)"),
    ('inappropriate friends',        'phrase', "Aug 2001 'A' → Maxwell (ITV, CNN)"),
    ('Mr. Big',                      'phrase', "Epstein nickname for Leon Black (Puck)"),
    ('Mr Big',                       'phrase', "Same"),
    ('Invisible Man',                'phrase', "Maxwell contact label for Andrew (CNN/ITV)"),
    ('the Principal',                'phrase', "LSJ house manual term for Epstein (CNN)"),
    ('career advice',                'phrase', "Maxwell depo gloss for 'massage' payments"),
    ('europe by remote',             'phrase', "Epstein → Bannon 2018"),

    # === Numeric / quantitative claims to verify against our counts ===
    ('301-261-1902',                 'phone',  "Recurring number in our corpus — MD area code"),
    ('410-974-0947',                 'phone',  "Same — Baltimore area"),
    ('358 El Brillo',                'address',"Palm Beach mansion address (DS4 police reports)"),
    ('9 East 71',                    'address',"Manhattan townhouse"),
    ('Little Saint Jeffrey',         'satire', "Black-humor nickname for LSJ in some emails"),
]


# Primary-source URLs for the press_recreate page. Missing entries render
# without a hyperlink. Reuse the same URLs used in verbatim_quotes.PHRASES
# where the term overlaps so click-through is consistent across pages.
URL_BY_TERM = {
    # Names — canonical reporting links for the post-Nov-2025 wave
    'Ruemmler':              'https://www.cnbc.com/2025/11/12/trump-jeffrey-epstein-ghislaine-maxwell-emails-house-democrats.html',
    'Wolff':                 'https://www.cbsnews.com/news/jeffrey-epstein-donald-trump-emails-house-oversight/',
    'Maxwell':               'https://www.bbc.com/news/world-us-canada-58191641',
    'Indyke':                'https://www.bloomberg.com/news/articles/2026-02-04/us-reviewed-allegations-against-staley-black-in-epstein-case',
    'Kahn':                  'https://www.bloomberg.com/news/articles/2026-02-04/us-reviewed-allegations-against-staley-black-in-epstein-case',
    'Wexner':                'https://www.npr.org/2026/02/03/epstein-wexner-deposition',
    'Staley':                'https://www.bloomberg.com/news/articles/2026-02-04/us-reviewed-allegations-against-staley-black-in-epstein-case',
    'Leon Black':            'https://www.puck.news/leon-black-mr-big-epstein/',
    'Acosta':                'https://www.miamiherald.com/news/local/article220097825.html',
    'Joi Ito':               'https://www.newyorker.com/news/news-desk/the-mit-media-labs-jeffrey-epstein-debacle-explained',
    'Mandelson':             'https://abcnews.go.com/International/peter-mandelson-arrested-epstein/story?id=mandelson-2026',
    'Ehud Barak':            'https://www.haaretz.com/israel-news/2026-02-ehud-barak-epstein-meetings',
    'Hamad bin Jassim':      'https://www.cbsnews.com/news/jeffrey-epstein-saudi-arabia/',
    'Mohammed bin Salman':   'https://www.cbsnews.com/news/jeffrey-epstein-saudi-arabia/',
    ' MBS ':                 'https://www.cbsnews.com/news/jeffrey-epstein-saudi-arabia/',
    'Reid Hoffman':          'https://fortune.com/2025/12/reid-hoffman-epstein-island-wsj/',
    'Summers':               'https://www.npr.org/2025/11/20/nx-s1-5613427/epstein-files-chomsky-bannon-summers-democrats',
    'Chomsky':               'https://www.npr.org/2025/11/20/nx-s1-5613427/epstein-files-chomsky-bannon-summers-democrats',
    'Bannon':                'https://www.npr.org/2025/11/20/nx-s1-5613427/epstein-files-chomsky-bannon-summers-democrats',
    'Trump':                 'https://www.cnbc.com/2025/11/12/trump-jeffrey-epstein-ghislaine-maxwell-emails-house-democrats.html',
    'Musk':                  'https://www.theguardian.com/us-news/2025/sep/elon-musk-epstein-island-flight-log',
    'Thiel':                 'https://www.theinformation.com/articles/peter-thiel-epstein-meetings',
    'Andrew':                'https://www.cnn.com/2025/12/23/europe/ghislaine-maxwell-email-british-royal-family-latam-intl',
    'Mountbatten-Windsor':   'https://www.bbc.com/news/uk-prince-andrew-mountbatten-windsor',
    'Branson':               'https://www.theguardian.com/world/2025/dec/richard-branson-epstein-photos',
    'Bill Gates':            'https://fortune.com/2025/11/bill-gates-epstein-breakfast-party-emails/',
    'Clinton':               'https://www.washingtonpost.com/national-security/2025/12/clinton-first-fifty-years-epstein-book/',

    # Verbatim phrases — links match verbatim_quotes.PHRASES for consistency
    "dog that hasn't barked":              'https://www.pbs.org/newshour/politics/read-jeffrey-epsteins-newly-released-emails-about-trump',
    'dog that hasn':                       'https://www.pbs.org/newshour/politics/read-jeffrey-epsteins-newly-released-emails-about-trump',
    'knew about the girls':                'https://www.cbsnews.com/news/jeffrey-epstein-donald-trump-emails-house-oversight/',
    'asked ghislaine to stop':             'https://www.cbsnews.com/news/jeffrey-epstein-donald-trump-emails-house-oversight/',
    'craft an answer':                     'https://www.pbs.org/newshour/politics/read-jeffrey-epsteins-newly-released-emails-about-trump',
    'let him hang himself':                'https://www.pbs.org/newshour/politics/read-jeffrey-epsteins-newly-released-emails-about-trump',
    'dirty donald':                        'https://www.cnbc.com/2025/11/12/trump-jeffrey-epstein-ghislaine-maxwell-emails-house-democrats.html',
    'fixer flip':                          'https://www.cnbc.com/2025/11/12/trump-jeffrey-epstein-ghislaine-maxwell-emails-house-democrats.html',
    'so gross':                            'https://www.cnbc.com/2025/11/12/trump-jeffrey-epstein-ghislaine-maxwell-emails-house-democrats.html',
    'worse in real life':                  'https://www.cnbc.com/2025/11/12/trump-jeffrey-epstein-ghislaine-maxwell-emails-house-democrats.html',
    'Snow White':                          'https://www.bloomberg.com/news/articles/2026-02-04/us-reviewed-allegations-against-staley-black-in-epstein-case',
    'Beauty and the Beast':                'https://www.bloomberg.com/news/articles/2026-02-04/us-reviewed-allegations-against-staley-black-in-epstein-case',
    'what character would you like next':  'https://www.bloomberg.com/news/articles/2026-02-04/us-reviewed-allegations-against-staley-black-in-epstein-case',
    'highly valued friend':                'https://www.npr.org/2025/11/20/nx-s1-5613427/epstein-files-chomsky-bannon-summers-democrats',
    'tastey models':                       'https://abcnews.go.com/International/peter-mandelson-arrested-epstein/story?id=mandelson-2026',
    'tasty models':                        'https://abcnews.go.com/International/peter-mandelson-arrested-epstein/story?id=mandelson-2026',
    'carpets and all':                     'https://www.cbsnews.com/news/jeffrey-epstein-saudi-arabia/',
    'First Fifty Years':                   'https://www.washingtonpost.com/national-security/2025/12/clinton-first-fifty-years-epstein-book/',
    'Centurion':                           'https://www.cbsnews.com/news/jeffrey-epstein-american-express-centurion-flights/',
    'inappropriate friends':               'https://www.cnn.com/2025/12/23/europe/ghislaine-maxwell-email-british-royal-family-latam-intl',
    'Mr. Big':                             'https://www.puck.news/leon-black-mr-big-epstein/',
    'Mr Big':                              'https://www.puck.news/leon-black-mr-big-epstein/',
    'Invisible Man':                       'https://www.cnn.com/2025/12/23/europe/ghislaine-maxwell-email-british-royal-family-latam-intl',
    'the Principal':                       'https://www.cnn.com/2025/12/23/europe/lsj-house-manual-the-principal',
    'europe by remote':                    'https://www.npr.org/2025/11/20/nx-s1-5613427/epstein-files-chomsky-bannon-summers-democrats',
}


def names():
    return [t for t, c, _ in TERMS if c == 'name']

def phrases():
    return [t for t, c, _ in TERMS if c == 'phrase']

def all_terms():
    return TERMS

def url_for(term: str) -> str | None:
    """Return canonical source URL for a term, or None if no link is mapped."""
    return URL_BY_TERM.get(term)
