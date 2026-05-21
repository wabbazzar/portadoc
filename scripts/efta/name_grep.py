#!/usr/bin/env python3
"""
Fast grep-based name ranker across the FULL corpus — not waiting on NER.

NER (Piiranha) is CPU-bound and was only ~7% through the corpus when we
needed top-name rankings. This script greps for a curated list of
Epstein-network names (case-insensitive, word-boundary) across every
doc in corpus.jsonl and emits doc + mention counts.

The name list combines:
  - Press-cited names from journalist_grep.TERMS (Trump, Clinton, Maxwell, ...)
  - Documented co-conspirators from the 2008 NPA + 2019 FBI lists
  - Confirmed real surfaces from our prior NER + DS10 analyzer runs
    (Alessi, Brunel, Visoski, Firetog, Groff, Wigdor, Acosta, Indyke,
    Kahn, Shuliak, Best/Amanda, etc.)
  - Senior politicians/staff that appear in the published emails

Each entry: (canonical_name, [regex_aliases], short_note).

Output:
  names_full.json   sorted by docs-containing-mention desc, then total mentions
  names_full.csv    full ranking

Usage:
    name_grep.py <corpus.jsonl> <out_dir> [--min-docs 1]
"""
import sys, json, re, csv, argparse
from pathlib import Path
from collections import Counter, defaultdict

# (canonical_label, aliases_regex_list, note)
NAMES = [
    # === Principals ===
    ('Jeffrey Epstein',         [r'Jeffrey\s+E\.?\s+Epstein', r'Jeffrey\s+Epstein', r'\bEpstein\b', r'\bJE\b'],
        'Subject'),
    ('Ghislaine Maxwell',       [r'Ghislaine\s+Maxwell', r'Ghislaine', r'\bGhislaine\b', r'G\.\s*Maxwell', r'\bGM\b'],
        'Co-defendant (convicted)'),
    # === 2008 NPA / 2019 FBI unindicted co-conspirators ===
    ('Sarah Kellen',            [r'Sarah\s+Kellen', r'\bKellen\b', r'\bSK\b'],     '2008 NPA co-conspirator'),
    ('Nadia Marcinkova',        [r'Nadia\s+Marcinkova', r'\bMarcinkova\b'],         '2008 NPA co-conspirator'),
    ('Adriana Ross',            [r'Adriana\s+Ross', r'\bMucinska\b'],               '2008 NPA co-conspirator (alias)'),
    ('Lesley Groff',            [r'Lesley\s+Groff', r'\bGroff\b', r'lesley\.jee'],  'Exec asst; 2019 FBI list'),
    ('Jean-Luc Brunel',         [r'Jean[\s\-]Luc\s+Brunel', r'\bBrunel\b', r'\bJLB\b'], 'Deceased modeling agent'),
    ('Les Wexner',              [r'Les\s+Wexner', r'Leslie\s+Wexner', r'\bWexner\b', r'\bLW\b'], 'Limited Brands; longest financial backer'),
    ('Darren Indyke',           [r'Darren\s+(?:K\.\s+)?Indyke', r'\bIndyke\b'],     'Attorney; estate co-executor'),
    ('Richard Kahn',            [r'Richard\s+Kahn', r'\bKahn\b'],                   'Accountant; estate co-executor (HBRK Associates)'),
    ('Karyna Shuliak',          [r'Karyna\s+Shuliak', r'\bShuliak\b'],              'Longtime girlfriend'),
    # === Executors / lawyers in Epstein orbit ===
    ('Alan Dershowitz',         [r'Alan\s+Dershowitz', r'\bDershowitz\b'],          'Former defense atty; accused'),
    ('Douglas Wigdor',          [r'Douglas?\s+Wigdor', r'\bWigdor\b'],              'Plaintiffs\' attorney'),
    ('Paul Everdell',           [r'Paul\s+Everdell', r'\bEverdell\b'],              'Maxwell\'s longtime atty'),
    ('Kenneth Starr',           [r'Kenneth\s+Starr', r'\bKen\s+Starr\b'],           'Former defense atty'),
    ('Roy Black',               [r'Roy\s+Black'],                                    'Lead defense in 2008'),
    ('Marc Kasowitz',           [r'Marc\s+Kasowitz', r'\bKasowitz\b'],              'Defense atty'),
    ('Reid Weingarten',         [r'Reid\s+Weingarten', r'\bWeingarten\b'],          'Defense atty'),
    # === Judiciary / DOJ ===
    ('Alexander Acosta',        [r'Alex(?:ander)?\s+Acosta', r'\bAcosta\b'],        'AUSA who gave 2007 NPA'),
    ('Neil Firetog',            [r'(?:Judge\s+)?Neil\s+Firetog', r'\bFiretog\b'],   'NY judge'),
    ('Geoffrey Berman',         [r'Geoffrey\s+S?\.?\s*Berman', r'\bGeoffrey\s+Berman\b'], 'SDNY US Atty 2019'),
    # === Household staff / pilots / witnesses ===
    ('Alfredo Alessi',          [r'Alfredo\s+Alessi', r'\bAlfredo\b'],              'Palm Beach houseman who testified'),
    ('Maria Alessi',            [r'Maria\s+Alessi'],                                 'Palm Beach household staff'),
    ('Larry Visoski',           [r'Larry\s+Visoski', r'\bVisoski\b'],               'Epstein\'s pilot; Maxwell-trial witness'),
    ('David Rodgers',           [r'David\s+Rodgers'],                                'Second pilot; flight log'),
    # === Politicians / world figures cited in the released emails ===
    ('Donald Trump',            [r'Donald\s+J?\.?\s*Trump', r'\bDJT\b', r'\bTrump\b'], 'Mentioned 1000+× per Oversight Dems'),
    ('Bill Clinton',            [r'Bill\s+Clinton', r'William\s+J?\.?\s*Clinton', r'\bClinton\b'], 'Multiple flight-log entries'),
    ('Hillary Clinton',         [r'Hillary\s+(?:R\.\s+|Rodham\s+)?Clinton'],         ''),
    ('Prince Andrew',           [r'Prince\s+Andrew', r'\bAndrew\s+Mountbatten[\s\-]Windsor\b'], '"Invisible Man" / "A" in Maxwell emails'),
    ('Bill Gates',              [r'Bill\s+Gates', r'William\s+H?\.?\s*Gates'],       '2011+ meetings + breakfast party'),
    ('Larry Summers',           [r'Lar+y\s+Summers', r'L\.\s*H\.\s*Summers', r'lhsummers'], 'Ex-Harvard/Treasury'),
    ('Noam Chomsky',            [r'Noam\s+Chomsky', r'\bChomsky\b'],                'LOR for Epstein'),
    ('Steve Bannon',            [r'Steve(?:n)?\s+Bannon', r'\bBannon\b'],           '2018+ PR coaching thread'),
    ('Michael Wolff',           [r'Michael\s+Wolff', r'\bM\.?\s*Wolff\b'],          'Journalist; "let him hang himself" / "knew about the girls" threads'),
    ('Kathryn Ruemmler',        [r'Kathryn\s+Ruemmler', r'\bRuemmler\b'],           'Goldman CLO; "I know how dirty donald is" recipient'),
    ('Leon Black',              [r'Leon\s+Black'],                                   'Apollo Global; "Mr. Big"'),
    ('Jes Staley',              [r'Jes\s+Staley', r'James\s+E?\.?\s*Staley', r'\bStaley\b'], 'JPM; Snow White/Beauty&Beast thread'),
    ('Reid Hoffman',            [r'Reid\s+Hoffman'],                                 'Two 2014 island visits'),
    ('Joi Ito',                 [r'Joi\s+Ito'],                                       'MIT Media Lab; $1.7M "anonymous"'),
    ('Lawrence Krauss',         [r'(?:Lawrence|Larry)\s+Krauss', r'\bKrauss\b'],     'Theoretical physicist'),
    ('Peter Mandelson',         [r'Peter\s+Mandelson', r'\bMandelson\b'],            'UK Labour; arrested Feb 2026'),
    ('Ehud Barak',              [r'Ehud\s+Barak'],                                   'Former Israeli PM; 60+ meetings'),
    ('Ariane de Rothschild',    [r'Ariane\s+(?:de\s+)?Rothschild'],                  ''),
    ('Hamad bin Jassim',        [r'Hamad\s+bin\s+Jassim', r'\bHBJ\b'],               'Former Qatari PM'),
    ('Mohammed bin Salman',     [r'Mohammed\s+bin\s+Salman', r'\bMBS\b'],            'Saudi Crown Prince; 2016 "tent carpets and all"'),
    ('Elon Musk',               [r'Elon\s+Musk'],                                    'Scheduled LSJ trip (Sept 2025 release)'),
    ('Peter Thiel',             [r'Peter\s+Thiel'],                                  'Scheduled meetings'),
    ('Tom Barrack',             [r'Tom\s+Barrack', r'Thomas\s+J?\.?\s*Barrack'],     'Trump Inaugural Cmte chair'),
    ('Jared Kushner',           [r'Jared\s+Kushner', r'\bKushner\b'],                ''),
    ('Murray Gell-Mann',        [r'Murray\s+Gell[\s\-]Mann'],                        'Caltech physicist'),
    ('Boris Nikolic',           [r'Boris\s+Nikolic', r'\bNikolic\b'],                'Sci advisor; in iMessages'),
    ('Mark Epstein',            [r'Mark\s+Epstein'],                                 'Jeffrey\'s brother'),
    # === Known JPM bankers ===
    ('Amanda Best',             [r'Amanda\s+Best', r'Best,\s+Amanda'],               'JPM banker on Epstein accounts'),
    ('Matthew Hiltzik',         [r'Matt(?:hew)?\s+Hiltzik', r'mhiltzik'],            'PR fixer (Hiltzik Strategies)'),
    ('Natalia Molotkova',       [r'Natalia\s+Molotkova', r'\bMolotkova\b', r'\bNatasha\s+Molotkova\b'],
        'AmEx Centurion concierge'),
    # === Companies / shells ===
    ('Hyperion Air',            [r'Hyperion\s+Air'],                                 'Epstein jet company'),
    ('Southern Financial LLC',  [r'Southern\s+Financial\s+L?\.?L?\.?C?\.?'],         'Epstein shell company'),
    ('HBRK Associates',         [r'HBRK\s+Associates'],                              'Richard Kahn\'s firm'),
    ('FIE LLC',                 [r'\bFIE\s+L?\.?L?\.?C?\.?\b'],                      'Epstein entity (wire beneficiary)'),
    ('Apollo Global',           [r'Apollo\s+Global', r'Apollo\s+Management'],        'Leon Black\'s firm'),
    ('Deutsche Bank',           [r'Deutsche\s+Bank'],                                'Epstein\'s post-JPM bank'),
    ('JPMorgan',                [r'JPMorgan', r'J\.?\s*P\.?\s*Morgan'],              'Epstein\'s pre-2013 bank'),
    ('Mar-a-Lago',              [r'Mar[\s\-]a[\s\-]Lago', r'\bMaraLago\b'],          'Trump\'s Palm Beach property; recruitment venue'),
    # === Geography / venues ===
    ('Little Saint James',      [r'Little\s+S(?:aint|t\.?)\s+James', r'\bLSJ\b'],    'Epstein\'s USVI island'),
    ('Zorro Ranch',             [r'Zorro\s+Ranch', r'\bZorro\b'],                    'New Mexico ranch'),
    ('El Brillo',               [r'El\s+Brillo'],                                    'Palm Beach mansion street'),
    # === AI / science / mathematics circle (surfaced via Amanda Palasciano reel) ===
    ('Robert Lawrence Kuhn',    [r'Robert\s+Lawrence\s+Kuhn', r'\bRobert\s+Kuhn\b'], 'PBS "Closer to Truth" host; Kuhn Foundation chair'),
    ('Kuhn Foundation',         [r'Kuhn\s+Foundation'],                              'Science/philosophy/China-relations foundation'),
    ('Misha Gromov',            [r'Misha\s+Gromov', r'Mikhail\s+Gromov'],            'Abel Prize mathematician (IHES)'),
    ('Joscha Bach',             [r'Joscha\s+Bach', r'\bJoscha\b'],                   'AI cognitive scientist'),
    ('Greg Borenstein',         [r'Greg\s+Borenstein'],                              'AI researcher / MIT Media Lab'),
    ('Marvin Minsky',           [r'Marvin\s+Minsky'],                                'AI pioneer; on flight logs'),
    ('Stephen Hawking',         [r'Stephen\s+Hawking'],                              'Physicist; LSJ visitor (2006)'),
    ('Lawrence Krauss',         [r'(?:Lawrence|Larry)\s+Krauss', r'\bKrauss\b'],     'Theoretical physicist (dup of Krauss above)'),
    ('Edge.org',                [r'\bEdge\.org\b', r'edge\.org'],                    'Brockman science salon'),
    ('John Brockman',           [r'John\s+Brockman'],                                'Edge.org founder'),
    ('Murray Gell-Mann',        [r'Murray\s+Gell[\s\-]Mann'],                        'Caltech physicist (dup above)'),
]


def compile_aliases():
    out = []
    for canon, aliases, note in NAMES:
        patterns = []
        for a in aliases:
            try:
                patterns.append(re.compile(a, re.IGNORECASE))
            except re.error:
                patterns.append(re.compile(re.escape(a), re.IGNORECASE))
        out.append((canon, patterns, note))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('corpus_jsonl')
    ap.add_argument('out_dir')
    ap.add_argument('--min-docs', type=int, default=1)
    args = ap.parse_args()
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)

    compiled = compile_aliases()
    # canonical -> {count, docs:set, datasets:Counter, sample_doc_ids:set, note}
    stats = {canon: {'count': 0, 'docs': set(), 'datasets': Counter(),
                     'sample_doc_ids': set(), 'note': note}
             for canon, _, note in compiled}

    n_docs = 0
    with open(args.corpus_jsonl) as f:
        for line in f:
            try: rec = json.loads(line)
            except: continue
            text = rec.get('text', '')
            if not text: continue
            n_docs += 1
            ds = rec.get('dataset', 'unknown')
            doc_id = rec.get('id', '?')
            for canon, patterns, _ in compiled:
                total = 0
                for rx in patterns:
                    total += len(rx.findall(text))
                if total > 0:
                    s = stats[canon]
                    s['count'] += total
                    s['docs'].add(doc_id)
                    s['datasets'][ds] += total
                    if len(s['sample_doc_ids']) < 5:
                        s['sample_doc_ids'].add(doc_id)

    rows = []
    for canon, s in stats.items():
        if len(s['docs']) < args.min_docs: continue
        rows.append({
            'name': canon,
            'note': s['note'],
            'mentions': s['count'],
            'docs': len(s['docs']),
            'datasets': dict(s['datasets']),
            'sample_doc_ids': sorted(s['sample_doc_ids']),
        })
    # Sort by docs (broader reach matters more), then by mentions, then alphabetical
    rows.sort(key=lambda r: (-r['docs'], -r['mentions'], r['name']))

    with open(out / 'names_full.json', 'w', encoding='utf-8') as f:
        json.dump({'n_documents_scanned': n_docs, 'rows': rows}, f, indent=2, ensure_ascii=False)
    with open(out / 'names_full.csv', 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['rank','name','mentions','docs','datasets','note'])
        for i, r in enumerate(rows, 1):
            w.writerow([i, r['name'], r['mentions'], r['docs'],
                        '|'.join(f'{k}:{v}' for k,v in sorted(r['datasets'].items(), key=lambda kv:-kv[1])[:5]),
                        r['note']])

    print(f'corpus docs scanned: {n_docs}')
    print(f'names with ≥{args.min_docs} doc(s): {len(rows)} / {len(NAMES)}')
    print(f'\nTop 30:')
    for r in rows[:30]:
        print(f"  {r['mentions']:>7} mentions in {r['docs']:>5} docs  ::  {r['name']}")


if __name__ == '__main__':
    main()
