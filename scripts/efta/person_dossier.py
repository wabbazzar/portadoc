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
#   canonical: regex patterns (case-insensitive, word-bound)
KNOWN_PERSONS = [
    ('Jeffrey Epstein',     [r'Jeffrey\s+E?\.?\s*Epstein', r'\bJeffrey\s+Epstein\b']),
    ('Ghislaine Maxwell',   [r'Ghislaine\s+Maxwell', r'\bGhislaine\b']),
    ('Donald Trump',        [r'Donald\s+J?\.?\s*Trump', r'\bDJT\b', r'\bDonald\s+Trump\b']),
    ('Bill Clinton',        [r'Bill\s+Clinton', r'William\s+J?\.?\s*Clinton']),
    ('Hillary Clinton',     [r'Hillary\s+(?:R\.\s+|Rodham\s+)?Clinton']),
    ('Prince Andrew',       [r'Prince\s+Andrew', r'Andrew\s+Mountbatten[\s\-]Windsor']),
    ('Alan Dershowitz',     [r'Alan\s+Dershowitz', r'\bDershowitz\b']),
    ('Alexander Acosta',    [r'Alex(?:ander)?\s+Acosta', r'\bAcosta\b']),
    ('Jean-Luc Brunel',     [r'Jean[\s\-]Luc\s+Brunel', r'\bBrunel\b']),
    ('Les Wexner',          [r'Les\s+Wexner', r'Leslie\s+Wexner', r'\bWexner\b']),
    ('Darren Indyke',       [r'Darren\s+(?:K\.\s+)?Indyke', r'\bIndyke\b']),
    ('Richard Kahn',        [r'Richard\s+Kahn', r'\bKahn\b']),
    ('Lesley Groff',        [r'Lesley\s+Groff', r'\bGroff\b']),
    ('Karyna Shuliak',      [r'Karyna\s+Shuliak', r'\bShuliak\b']),
    ('Larry Visoski',       [r'Larry\s+Visoski', r'\bVisoski\b']),
    ('Steve Bannon',        [r'Steve(?:n)?\s+Bannon', r'\bBannon\b']),
    ('Michael Wolff',       [r'Michael\s+Wolff', r'\bM\.?\s*Wolff\b']),
    ('Kathryn Ruemmler',    [r'Kathryn\s+Ruemmler', r'\bRuemmler\b']),
    ('Leon Black',          [r'Leon\s+Black']),
    ('Jes Staley',          [r'Jes\s+Staley', r'\bStaley\b']),
    ('Reid Hoffman',        [r'Reid\s+Hoffman']),
    ('Joi Ito',             [r'Joi\s+Ito']),
    ('Lawrence Krauss',     [r'(?:Lawrence|Larry)\s+Krauss', r'\bKrauss\b']),
    ('Peter Mandelson',     [r'Peter\s+Mandelson', r'\bMandelson\b']),
    ('Ehud Barak',          [r'Ehud\s+Barak']),
    ('Mohammed bin Salman', [r'Mohammed\s+bin\s+Salman', r'\bMBS\b']),
    ('Elon Musk',           [r'Elon\s+Musk']),
    ('Peter Thiel',         [r'Peter\s+Thiel']),
    ('Tom Barrack',         [r'Tom\s+Barrack', r'Thomas\s+J?\.?\s*Barrack']),
    ('Jared Kushner',       [r'Jared\s+Kushner', r'\bKushner\b']),
    ('Bill Gates',          [r'Bill\s+Gates', r'William\s+H?\.?\s*Gates']),
    ('Noam Chomsky',        [r'Noam\s+Chomsky', r'\bChomsky\b']),
    ('Larry Summers',       [r'Lar+y\s+Summers', r'L\.\s*H\.\s*Summers']),
    ('Robert Lawrence Kuhn',[r'Robert\s+Lawrence\s+Kuhn', r'\bRobert\s+Kuhn\b']),
    ('Misha Gromov',        [r'Misha\s+Gromov', r'Mikhail\s+Gromov']),
    ('Steven Hoffenberg',   [r'Stev(?:e|en)\s+Hoffenberg', r'\bHoffenberg\b']),
    ('Virginia Giuffre',    [r'Virginia\s+Giuffre', r'\bGiuffre\b']),
    ('Sarah Kellen',        [r'Sarah\s+Kellen', r'\bKellen\b']),
    ('Nadia Marcinkova',    [r'Nadia\s+Marcinkova', r'\bMarcinkova\b']),
    ('Adriana Ross',        [r'Adriana\s+Ross', r'\bMucinska\b']),
    ('Boris Nikolic',       [r'Boris\s+Nikolic', r'\bNikolic\b']),
    ('Anthony Scaramucci',  [r'Anthony\s+Scaramucci', r'\bScaramucci\b']),
    ('Joe Recarey',         [r'Joe\s+Recarey', r'\bRecarey\b']),
    ('Bruce Krischer',      [r'Bruce\s+Krischer', r'\bKrischer\b']),
    ('Alfredo Alessi',      [r'Alfredo\s+Alessi', r'\bAlessi\b']),
    ('Matthew Hiltzik',     [r'Matt(?:hew)?\s+Hiltzik', r'\bHiltzik\b']),
    ('Natalia Molotkova',   [r'Natalia\s+Molotkova', r'\bMolotkova\b']),
    ('Amanda Best',         [r'Amanda\s+Best', r'Best,\s+Amanda']),
    ('Yitzhak Rabin',       [r'Yitzhak\s+Rabin', r'\bRabin\b']),
    ('Ariane de Rothschild',[r'Ariane\s+(?:de\s+)?Rothschild']),
    ('Hamad bin Jassim',    [r'Hamad\s+bin\s+Jassim', r'\bHBJ\b']),
    ('Murray Gell-Mann',    [r'Murray\s+Gell[\s\-]Mann']),
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
    for name, pats in KNOWN_PERSONS:
        compiled.append((name, [re.compile(p, re.IGNORECASE) for p in pats]))

    persons = {n: {
        'name': n,
        'aliases': [p.pattern for p in pats],
        'mentions': 0,
        'docs': set(),
        'by_dataset': Counter(),
        'by_year': Counter(),
        'samples': [],
        'sample_seen_docs': set(),
        'wiki': {'url': f'https://en.wikipedia.org/wiki/{wiki_slug(n)}',
                 'slug': wiki_slug(n)},
    } for n, pats in compiled}

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
            for name, pats in compiled:
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
                    if yr.isdigit(): p['by_year'][int(yr)] += 1
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
            'aliases': p['aliases'],
            'mentions': p['mentions'],
            'docs': len(p['docs']),
            'by_dataset': dict(p['by_dataset']),
            'by_year': dict(sorted(p['by_year'].items())),
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
