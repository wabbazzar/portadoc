"""
Grep targets sourced from publicly-published EFTA reporting (Nov 2025 – May 2026).

When these strings appear in our local corpus, we've recreated (or can
explicitly fail-to-recreate) a finding reported by name elsewhere.
Every entry carries a one-line provenance pointer.

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


def names():
    return [t for t, c, _ in TERMS if c == 'name']

def phrases():
    return [t for t, c, _ in TERMS if c == 'phrase']

def all_terms():
    return TERMS
