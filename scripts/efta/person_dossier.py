#!/usr/bin/env python3
"""
Build per-person dossier records from the NER tagged.jsonl + a curated
known-person list (journalist_grep.TERMS for cross-reference).

Each dossier card includes:
  - canonical name + ocr_variants
  - total mention count (cross-corpus, full doc text)
  - per-dataset breakdown
  - mention timeline (year histogram)
  - top 5 sample doc_ids with one-line context snippets
  - wikipedia url + thumbnail (best-effort, lazy-fetched at view-time)

Output: persons.json with shape
  {"persons": [
      {"name", "aliases", "mentions", "docs", "by_dataset", "by_year",
       "samples": [{doc_id, dataset, date, context}], "wiki": {...}}
  ]}

Aliases merging: combines GIVENNAME + SURNAME tags of likely-same person
(e.g., "Ghislaine" + "Maxwell" → "Ghislaine Maxwell" when both appear in
≥3 same docs together).

Usage:
    person_dossier.py <corpus.jsonl> <tagged.jsonl> <out.json>
"""
import sys, json, re, argparse
from pathlib import Path
from collections import Counter, defaultdict


# Curated canonical name list (extends journalist_grep). Each:
#   (canonical, [aliases regex], "one-line relation to Jeffrey Epstein")
KNOWN_PERSONS = [
    ('Jeffrey Epstein',     [r'Jeffrey\s+E?\.?\s*Epstein', r'\bJeffrey\s+Epstein\b'],
        "The subject of the DOJ disclosure."),
    ('Ghislaine Maxwell',   [r'Ghislaine\s+Maxwell', r'\bGhislaine\b'],
        "Longtime girlfriend and co-conspirator; convicted of sex trafficking 2021."),
    ('Donald Trump',        [r'Donald\s+J?\.?\s*Trump', r'\bDJT\b', r'\bDonald\s+Trump\b'],
        "Longtime acquaintance from 1990s NYC and Mar-a-Lago social circuit."),
    ('Bill Clinton',        [r'Bill\s+Clinton', r'William\s+J?\.?\s*Clinton'],
        "Took multiple flights on Epstein's plane in the early 2000s."),
    ('Hillary Clinton',     [r'Hillary\s+(?:R\.\s+|Rodham\s+)?Clinton'],
        "Surfaces in flight log marginalia."),
    ('Prince Andrew',       [r'Prince\s+Andrew', r'Andrew\s+Mountbatten[\s\-]Windsor'],
        "Photographed with victim Virginia Giuffre; settled civil suit 2022."),
    ('Alan Dershowitz',     [r'Alan\s+Dershowitz', r'\bDershowitz\b'],
        "Defense attorney on Epstein's 2008 plea deal; later named (and settled) in Giuffre suit."),
    ('Alexander Acosta',    [r'Alex(?:ander)?\s+Acosta', r'\bAcosta\b'],
        "AUSA who signed off on Epstein's 2007 non-prosecution agreement; later Trump Labor Secretary."),
    ('Jean-Luc Brunel',     [r'Jean[\s\-]Luc\s+Brunel', r'\bBrunel\b'],
        "Modeling agent and Epstein associate accused of trafficking; died in French custody 2022."),
    ('Les Wexner',          [r'Les\s+Wexner', r'Leslie\s+Wexner', r'\bWexner\b'],
        "Longtime financial patron; transferred the 9 E 71st St townhouse to Epstein."),
    ('Darren Indyke',       [r'Darren\s+(?:K\.\s+)?Indyke', r'\bIndyke\b'],
        "Longtime personal attorney; co-executor of Epstein's estate."),
    ('Richard Kahn',        [r'Richard\s+Kahn', r'\bKahn\b'],
        "Longtime accountant (HBRK Associates); co-executor of Epstein's estate."),
    ('Lesley Groff',        [r'Lesley\s+Groff', r'\bGroff\b'],
        "Executive assistant for 20+ years; on the 2019 FBI co-conspirator list."),
    ('Karyna Shuliak',      [r'Karyna\s+Shuliak', r'\bShuliak\b'],
        "Longtime girlfriend in Epstein's later years."),
    ('Larry Visoski',       [r'Larry\s+Visoski', r'\bVisoski\b'],
        "Epstein's personal pilot for ~28 years; testified at Maxwell trial."),
    ('Steve Bannon',        [r'Steve(?:n)?\s+Bannon', r'\bBannon\b'],
        "Post-2018 PR and legal-strategy coaching contact (recorded interviews)."),
    ('Michael Wolff',       [r'Michael\s+Wolff', r'\bM\.?\s*Wolff\b'],
        "Journalist; recipient of the 'knew about the girls' and 'let him hang himself' emails."),
    ('Kathryn Ruemmler',    [r'Kathryn\s+Ruemmler', r'\bRuemmler\b'],
        "Goldman Sachs CLO and ex-Obama WH Counsel; recipient of the 'dirty donald' email."),
    ('Leon Black',          [r'Leon\s+Black'],
        "Apollo Global Mgmt founder; paid Epstein ~$170M in advisory fees ('Mr. Big')."),
    ('Jes Staley',          [r'Jes\s+Staley', r'\bStaley\b'],
        "JPMorgan private banker who managed Epstein's accounts; later Barclays CEO."),
    ('Reid Hoffman',        [r'Reid\s+Hoffman'],
        "LinkedIn co-founder; two 2014 trips to Little Saint James, also brought Joi Ito."),
    ('Joi Ito',             [r'Joi\s+Ito'],
        "MIT Media Lab director; took ~$1.7M Epstein donations laundered as 'anonymous'."),
    ('Lawrence Krauss',     [r'(?:Lawrence|Larry)\s+Krauss', r'\bKrauss\b'],
        "Theoretical physicist; LSJ visitor; sent Epstein his BuzzFeed-harassment response draft."),
    ('Peter Mandelson',     [r'Peter\s+Mandelson', r'\bMandelson\b'],
        "UK Labour politician; Epstein-defender emails post-2008; arrested Feb 2026."),
    ('Ehud Barak',          [r'Ehud\s+Barak'],
        "Former Israeli PM; 60+ documented meetings with Epstein 2010-19."),
    ('Mohammed bin Salman', [r'Mohammed\s+bin\s+Salman', r'\bMBS\b'],
        "Saudi Crown Prince; gifted Epstein a tent 'carpets and all' in 2016."),
    ('Elon Musk',           [r'Elon\s+Musk'],
        "Trip to Little Saint James scheduled per the Sept 2025 Estate release."),
    ('Peter Thiel',         [r'Peter\s+Thiel'],
        "Scheduled meetings with Epstein per the Estate release."),
    ('Tom Barrack',         [r'Tom\s+Barrack', r'Thomas\s+J?\.?\s*Barrack'],
        "Trump's longtime friend and Inaugural Committee chair; in Epstein's iMessages Jan 2017."),
    ('Jared Kushner',       [r'Jared\s+Kushner', r'\bKushner\b'],
        "Trump son-in-law; named in Epstein's iMessages with redacted counterpart Jan 2017."),
    ('Bill Gates',          [r'Bill\s+Gates', r'William\s+H?\.?\s*Gates'],
        "Met with Epstein post-conviction including a 2011 dinner with Summers and Staley."),
    ('Noam Chomsky',        [r'Noam\s+Chomsky', r'\bChomsky\b'],
        "MIT linguist; wrote a letter calling Epstein a 'highly valued friend'."),
    ('Larry Summers',       [r'Lar+y\s+Summers', r'L\.\s*H\.\s*Summers'],
        "Ex-Treasury Secretary and Harvard President; close email correspondent through 2018."),
    ('Robert Lawrence Kuhn',[r'Robert\s+Lawrence\s+Kuhn', r'\bRobert\s+Kuhn\b'],
        "PBS 'Closer to Truth' host; AI / probability / Misha Gromov correspondence 2016-17."),
    ('Misha Gromov',        [r'Misha\s+Gromov', r'Mikhail\s+Gromov'],
        "Abel Prize-winning mathematician (IHES); featured in Epstein's 'Radical Breakthroughs' book."),
    ('Steven Hoffenberg',   [r'Stev(?:e|en)\s+Hoffenberg', r'\bHoffenberg\b'],
        "Towers Financial Ponzi schemer; Epstein's 1980s-90s associate and claimed mentor."),
    ('Virginia Giuffre',    [r'Virginia\s+Giuffre', r'\bGiuffre\b'],
        "Maxwell-trafficking victim turned principal accuser; sued Prince Andrew and Dershowitz."),
    ('Sarah Kellen',        [r'Sarah\s+Kellen', r'\bKellen\b'],
        "Maxwell's assistant; named co-conspirator in Epstein's 2008 non-prosecution agreement."),
    ('Nadia Marcinkova',    [r'Nadia\s+Marcinkova', r'\bMarcinkova\b'],
        "Named co-conspirator in Epstein's 2008 non-prosecution agreement."),
    ('Adriana Ross',        [r'Adriana\s+Ross', r'\bMucinska\b'],
        "Named co-conspirator in Epstein's 2008 non-prosecution agreement."),
    ('Boris Nikolic',       [r'Boris\s+Nikolic', r'\bNikolic\b'],
        "Bill Gates's science adviser; recurring Epstein iMessage correspondent."),
    ('Anthony Scaramucci',  [r'Anthony\s+Scaramucci', r'\bScaramucci\b'],
        "Briefly Trump's communications director; SkyBridge Capital founder; in Epstein's orbit."),
    ('Joe Recarey',         [r'Joe\s+Recarey', r'\bRecarey\b'],
        "Lead Palm Beach Police detective on the 2005-06 Epstein investigation."),
    ('Bruce Krischer',      [r'Bruce\s+Krischer', r'\bKrischer\b'],
        "Palm Beach County State Attorney 2005-09; declined to upgrade Epstein's charges."),
    ('Alfredo Alessi',      [r'Alfredo\s+Alessi', r'\bAlessi\b'],
        "Epstein's Palm Beach household manager; testified for the prosecution."),
    ('Matthew Hiltzik',     [r'Matt(?:hew)?\s+Hiltzik', r'\bHiltzik\b'],
        "NYC PR fixer (Hiltzik Strategies); Epstein's post-conviction PR contact."),
    ('Natalia Molotkova',   [r'Natalia\s+Molotkova', r'\bMolotkova\b'],
        "American Express Centurion concierge who booked Epstein's international flights."),
    ('Amanda Best',         [r'Amanda\s+Best', r'Best,\s+Amanda'],
        "JPMorgan Private Bank relationship officer on the Epstein accounts."),
    ('Yitzhak Rabin',       [r'Yitzhak\s+Rabin', r'\bRabin\b'],
        "Late Israeli Prime Minister (assassinated 1995); referenced in Israeli-context docs."),
    ('Ariane de Rothschild',[r'Ariane\s+(?:de\s+)?Rothschild'],
        "Edmond de Rothschild banker; in 2013 scheduling thread with Ehud Barak."),
    ('Hamad bin Jassim',    [r'Hamad\s+bin\s+Jassim', r'\bHBJ\b'],
        "Former Qatari PM (HBJ); subject of Apr 2018 'HBJ sun. MBS mon.' iMessage."),
    ('Murray Gell-Mann',    [r'Murray\s+Gell[\s\-]Mann'],
        "Caltech Nobel-laureate physicist; contributor to Epstein's 50th birthday book."),
]


def find_doc_date(text: str) -> str | None:
    head = text[:1500]
    m = re.search(r'Time:\s*(\d{1,2}/\d{1,2}/\d{2,4}\s+\d{1,2}:\d{2}(?::\d{2})?\s*(?:AM|PM)?)', head, re.IGNORECASE)
    if m: return _ts(m.group(1))
    m = re.search(r'(?i)Sent[:\s]+(?:[A-Za-z]+,?\s+)?(\d{1,2}/\d{1,2}/\d{2,4}(?:\s+\d{1,2}:\d{2})?)', head)
    if m: return _ts(m.group(1))
    m = re.search(r'\b(19[5-9][0-9]|20[0-2][0-9])-(\d{2})-(\d{2})\b', head)
    if m: return f'{m.group(1)}-{m.group(2)}-{m.group(3)}'
    return None


def _ts(s: str):
    from datetime import datetime
    for fmt in ('%m/%d/%y %I:%M:%S %p','%m/%d/%y %I:%M %p','%m/%d/%Y %I:%M:%S %p',
                '%m/%d/%Y %I:%M %p','%m/%d/%y %H:%M:%S','%m/%d/%y %H:%M',
                '%m/%d/%Y %H:%M:%S','%m/%d/%Y %H:%M','%m/%d/%y','%m/%d/%Y'):
        try: return datetime.strptime(s.strip(), fmt).isoformat()
        except ValueError: pass
    return None


def wiki_slug(name: str) -> str:
    """Convert "Jean-Luc Brunel" → "Jean-Luc_Brunel"."""
    return name.replace(' ', '_')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('corpus_jsonl')
    ap.add_argument('out_json')
    ap.add_argument('--max-samples', type=int, default=5)
    args = ap.parse_args()

    compiled = []
    for name, pats, relation in KNOWN_PERSONS:
        compiled.append((name, [re.compile(p, re.IGNORECASE) for p in pats], relation))

    persons = {n: {
        'name': n,
        'relation': rel,
        'aliases': [p.pattern for p in pats],
        'mentions': 0,
        'docs': set(),
        'by_dataset': Counter(),
        'by_year': Counter(),
        'by_year_dataset': defaultdict(Counter),
        'samples': [],
        'sample_seen_docs': set(),
        'wiki': {'url': f'https://en.wikipedia.org/wiki/{wiki_slug(n)}',
                 'slug': wiki_slug(n)},
    } for n, pats, rel in compiled}

    n_docs = 0
    with open(args.corpus_jsonl) as f:
        for line in f:
            try: rec = json.loads(line)
            except: continue
            n_docs += 1
            text = rec.get('text', '')
            if not text: continue
            ds = rec.get('dataset', 'unknown')
            doc_id = rec.get('id', '?')
            doc_date = find_doc_date(text)
            for name, pats, _rel in compiled:
                hits = 0
                first_idx = None
                for rx in pats:
                    for m in rx.finditer(text):
                        hits += 1
                        if first_idx is None: first_idx = m.start()
                if not hits: continue
                p = persons[name]
                p['mentions'] += hits
                p['docs'].add(doc_id)
                p['by_dataset'][ds] += hits
                if doc_date:
                    yr = doc_date[:4]
                    if yr.isdigit():
                        yr_i = int(yr)
                        p['by_year'][yr_i] += 1
                        p['by_year_dataset'][yr_i][ds] += 1
                if doc_id not in p['sample_seen_docs'] and len(p['samples']) < args.max_samples:
                    start = max(0, (first_idx or 0) - 100)
                    end = min(len(text), (first_idx or 0) + 250)
                    ctx = re.sub(r'\s+', ' ', text[start:end]).strip()
                    p['samples'].append({
                        'doc_id': doc_id,
                        'dataset': ds,
                        'date': doc_date,
                        'context': ctx,
                    })
                    p['sample_seen_docs'].add(doc_id)

    out_persons = []
    for name, p in persons.items():
        if p['mentions'] == 0: continue
        out_persons.append({
            'name': name,
            'relation': p['relation'],
            'aliases': p['aliases'],
            'mentions': p['mentions'],
            'docs': len(p['docs']),
            'by_dataset': dict(p['by_dataset']),
            'by_year': dict(sorted(p['by_year'].items())),
            'by_year_dataset': {y: dict(p['by_year_dataset'][y])
                                for y in sorted(p['by_year_dataset'].keys())},
            'samples': p['samples'],
            'wiki': p['wiki'],
        })
    out_persons.sort(key=lambda p: -p['docs'])

    out_data = {
        'n_documents_scanned': n_docs,
        'n_persons': len(out_persons),
        'persons': out_persons,
    }
    with open(args.out_json, 'w', encoding='utf-8') as f:
        json.dump(out_data, f, indent=2, ensure_ascii=False)

    print(f'docs={n_docs}  persons_with_hits={len(out_persons)}')
    for p in out_persons[:20]:
        ds = max(p['by_dataset'].items(), key=lambda kv:kv[1])[0] if p['by_dataset'] else '?'
        print(f"  {p['mentions']:>6} mentions / {p['docs']:>5} docs  ::  {p['name']}  (top ds: {ds})")


if __name__ == '__main__':
    main()
