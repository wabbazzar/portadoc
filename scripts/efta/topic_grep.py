#!/usr/bin/env python3
"""
Topic-search grep over the email/iMessage subset, returning per-match records
(not just aggregate counts) so the viewer can show date + sender + context +
clickable doc-id for each individual hit.

Each topic is a list of regex variants. For each match we capture:
  - doc_id, dataset
  - timestamp (parsed from email Sent: header or iMessage Time: field if present)
  - matched_text (the verbatim hit)
  - context (~200 chars surrounding)
  - source_url or source_path (so the viewer can link out)

Output: topic_search.json with per-topic per-match list, sorted by date.

Usage:
    topic_grep.py <corpus.emails.jsonl> <out_dir>
"""
import sys, json, re, csv, argparse
from pathlib import Path
from datetime import datetime
from collections import Counter

# (topic_label, regex pattern list, note)
TOPICS = [
    ('World War 3', [
        r'\bworld\s+war\s+(?:iii|3|three)\b',
        r'\bww\s*(?:iii|3)\b',
        r'\bwwiii\b',
        r'\bnuclear\s+war\b',
        r'\bnuclear\s+winter\b',
        r'\bnuclear\s+armageddon\b',
        r'\bnuclear\s+conflict\b',
        r'\bnuclear\s+exchange\b',
        r'\bthird\s+world\s+war\b',
        r'\bthird\s+war\b',
        r'\bgreat\s+power\s+conflict\b',
        r'\bkinetic\s+war\b',
        r'\bhot\s+war\b',
        r'\barmageddon\b',
        r'\bextinction\s+event\b',
        r'\bdoomsday\b',
    ], 'War-scare phrasing — global / nuclear / extinction'),
    ('Simulation', [
        r'\bsimulation\b',
        r'\bsimulated\b',
        r'\bsimulating\b',
        r'\bsimulation\s+hypothesis\b',
        r'\bsim\s+hypothesis\b',
        r'\bsim\s+theory\b',
        r'\bin\s+a\s+simulation\b',
        r'\bare\s+we\s+in\s+a\s+simulation\b',
    ], 'Simulation theory / hypothesis references'),
    ('Pandemic', [
        r'\bpandemic\b',
        r'\bpandemics\b',
        r'\bbioweapon\b',
        r'\bbiological\s+warfare\b',
        r'\bgain\s+of\s+function\b',
        r'\bvirus\s+leak\b',
        r'\blab\s+leak\b',
    ], 'Pandemic / bio-threat phrasing'),
]


def find_doc_date(text: str) -> str | None:
    """Extract a canonical doc date — try email 'Sent:' header first, then iMessage 'Time:'."""
    head = text[:1500]
    # iMessage forensic timestamp
    m = re.search(r'Time:\s*(\d{1,2}/\d{1,2}/\d{2,4}\s+\d{1,2}:\d{2}(?::\d{2})?\s*(?:AM|PM)?)', head, re.IGNORECASE)
    if m:
        return _parse_ts(m.group(1))
    # Email Sent: header
    m = re.search(r'(?i)Sent[:\s]+(?:[A-Za-z]+,?\s+)?(\d{1,2}/\d{1,2}/\d{2,4}(?:\s+\d{1,2}:\d{2})?)', head)
    if m:
        return _parse_ts(m.group(1))
    # First yyyy-mm-dd
    m = re.search(r'\b(\d{4})-(\d{2})-(\d{2})\b', head)
    if m:
        return f'{m.group(1)}-{m.group(2)}-{m.group(3)}'
    # First Month-DD-YYYY
    m = re.search(r'\b(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec)\w*\s+(\d{1,2}),?\s+(\d{4})\b', head, re.IGNORECASE)
    if m:
        mo_name = m.group(1).lower()[:3]
        mo = {'jan':1,'feb':2,'mar':3,'apr':4,'may':5,'jun':6,'jul':7,
              'aug':8,'sep':9,'oct':10,'nov':11,'dec':12}.get(mo_name)
        if mo:
            return f"{m.group(3)}-{mo:02d}-{int(m.group(2)):02d}"
    return None


def _parse_ts(s: str) -> str | None:
    s = s.strip()
    for fmt in ('%m/%d/%y %I:%M:%S %p','%m/%d/%y %I:%M %p',
                '%m/%d/%Y %I:%M:%S %p','%m/%d/%Y %I:%M %p',
                '%m/%d/%y %H:%M:%S','%m/%d/%y %H:%M',
                '%m/%d/%Y %H:%M:%S','%m/%d/%Y %H:%M',
                '%m/%d/%y','%m/%d/%Y'):
        try:
            return datetime.strptime(s, fmt).isoformat()
        except ValueError:
            pass
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('corpus_jsonl')
    ap.add_argument('out_dir')
    ap.add_argument('--max-matches-per-topic', type=int, default=500)
    args = ap.parse_args()
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)

    compiled = [(label, [re.compile(p, re.IGNORECASE) for p in pats], note)
                for label, pats, note in TOPICS]

    per_topic = {label: {
        'topic': label, 'note': note, 'patterns': [p.pattern for p in pats],
        'matches': [], 'docs': set(), 'datasets': Counter(),
    } for label, pats, note in compiled}

    n_docs = 0
    with open(args.corpus_jsonl) as f:
        for line in f:
            n_docs += 1
            try: rec = json.loads(line)
            except: continue
            text = rec.get('text','')
            if not text: continue
            doc_date = find_doc_date(text)
            doc_id = rec.get('id','?')
            ds = rec.get('dataset','?')
            for label, pats, _ in compiled:
                slot = per_topic[label]
                for rx in pats:
                    for m in rx.finditer(text):
                        if len(slot['matches']) >= args.max_matches_per_topic:
                            break
                        start = max(0, m.start() - 120)
                        end   = min(len(text), m.end() + 220)
                        ctx = re.sub(r'\s+', ' ', text[start:end]).strip()
                        slot['matches'].append({
                            'doc_id': doc_id,
                            'dataset': ds,
                            'date': doc_date,
                            'matched': m.group(0),
                            'context': ctx,
                            'source': rec.get('source',''),
                        })
                        slot['docs'].add(doc_id)
                        slot['datasets'][ds] += 1

    # Serialize — sort matches by date (None at end)
    out_topics = []
    for label, slot in per_topic.items():
        slot['matches'].sort(key=lambda r: (r['date'] is None, r['date'] or ''))
        out_topics.append({
            'topic': slot['topic'],
            'note': slot['note'],
            'patterns': slot['patterns'],
            'total_matches': len(slot['matches']),
            'n_docs': len(slot['docs']),
            'datasets': dict(slot['datasets']),
            'matches': slot['matches'],
        })

    with open(out / 'topic_search.json', 'w', encoding='utf-8') as f:
        json.dump({'n_documents_scanned': n_docs, 'topics': out_topics},
                  f, indent=2, ensure_ascii=False)

    # Per-topic CSV
    for t in out_topics:
        slug = re.sub(r'\W+','_', t['topic'].lower()).strip('_')
        with open(out / f'topic_{slug}.csv', 'w', newline='', encoding='utf-8') as cf:
            w = csv.writer(cf)
            w.writerow(['date','dataset','doc_id','matched','context'])
            for m in t['matches']:
                w.writerow([m['date'] or '', m['dataset'], m['doc_id'],
                            m['matched'], m['context']])

    print(f'docs scanned: {n_docs}')
    for t in out_topics:
        print(f"  {t['topic']:>16}: {t['total_matches']:>4} matches in {t['n_docs']:>4} docs")
        # Show first 5 dated matches
        for m in [x for x in t['matches'] if x['date']][:3]:
            print(f"      [{m['date'][:10]}] {m['doc_id']}: …{m['matched']}…")


if __name__ == '__main__':
    main()
