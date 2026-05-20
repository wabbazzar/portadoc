#!/usr/bin/env python3
"""
Entity co-occurrence: pairs of named entities appearing in the same document.

For each document, collect distinct (SURNAME ∪ GIVENNAME) entities (after the
same canonicalization rank_entities uses). Emit pair counts and the implied
adjacency graph as an edgelist.

Output:
  - cooccur.json: top pairs + per-node degree
  - cooccur_pairs.csv

Usage:
    cooccur.py <tagged.jsonl> <out_dir> [--top 100] [--min-pair 2]
"""
import sys, json, argparse, itertools
from pathlib import Path
from collections import Counter, defaultdict

sys.path.insert(0, str(Path(__file__).resolve().parent))
import canonicalize


def canonical_name(text: str, label: str) -> str:
    """Mirror of rank_entities normalization for PERSON-like labels."""
    t = text.replace('##', '').strip()
    if not t: return ''
    if len(t) < 3: return ''  # filter noise
    return t.title()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('tagged_jsonl')
    ap.add_argument('out_dir')
    ap.add_argument('--top', type=int, default=100)
    ap.add_argument('--min-pair', type=int, default=2)
    ap.add_argument('--min-score', type=float, default=0.5)
    args = ap.parse_args()

    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)

    # First pass: collect per-doc name sets + per-label counts for canonicalization
    doc_names = []  # list of (doc_id, dataset, set(names))
    person_counts = defaultdict(lambda: {'count': 0, 'docs': set(), 'datasets': Counter(), 'max_score': 0.0})

    with open(args.tagged_jsonl) as f:
        for line in f:
            try: rec = json.loads(line)
            except: continue
            ds = rec.get('dataset','unknown')
            doc_id = rec.get('id','?')
            names_here = set()
            for e in rec.get('entities', []):
                if e['score'] < args.min_score: continue
                lbl = (e['label'] or '').upper().lstrip('I-').lstrip('B-')
                if lbl not in ('GIVENNAME', 'SURNAME', 'PERSON', 'NAME'): continue
                n = canonical_name(e['text'], lbl)
                if not n: continue
                names_here.add(n)
                slot = person_counts[n]
                slot['count'] += 1
                slot['docs'].add(doc_id)
                slot['datasets'][ds] += 1
                slot['max_score'] = max(slot['max_score'], e['score'])
            doc_names.append((doc_id, ds, names_here))

    # Canonicalize across all names (fuzzy-merge OCR variants)
    person_canon = canonicalize.cluster_persons(dict(person_counts))
    # Build name→canonical map
    name_to_canon = {}
    for canon, slot in person_canon.items():
        name_to_canon[canon] = canon
        for v in slot.get('ocr_variants', []):
            name_to_canon[v] = canon

    # Recompute per-doc name sets using canonical forms
    doc_canon_names = []
    for doc_id, ds, names_here in doc_names:
        canon = set()
        for n in names_here:
            c = name_to_canon.get(n, n)
            canon.add(c)
        doc_canon_names.append((doc_id, ds, canon))

    # Pair counts
    pair_count = Counter()
    pair_docs = defaultdict(set)
    pair_datasets = defaultdict(Counter)
    for doc_id, ds, names in doc_canon_names:
        # Sort to canonicalize pair direction
        for a, b in itertools.combinations(sorted(names), 2):
            if a == b: continue
            # Skip pairs where one is obvious noise (single token "Ep" et al.)
            if len(a) < 3 or len(b) < 3: continue
            pair_count[(a, b)] += 1
            pair_docs[(a, b)].add(doc_id)
            pair_datasets[(a, b)][ds] += 1

    # Filter
    pairs = [(a, b, c) for (a, b), c in pair_count.items() if c >= args.min_pair]
    pairs.sort(key=lambda r: (-r[2], r[0], r[1]))

    # Per-node degree (number of distinct partners)
    degree = Counter()
    for (a, b), c in pair_count.items():
        if c < args.min_pair: continue
        degree[a] += 1
        degree[b] += 1

    out_data = {
        'n_documents': len(doc_canon_names),
        'n_distinct_names': len(person_canon),
        'n_pairs': len(pairs),
        'top_pairs': [
            {
                'a': a, 'b': b,
                'cooccur_docs': c,
                'sample_doc_ids': sorted(pair_docs[(a, b)])[:5],
                'datasets': dict(pair_datasets[(a, b)]),
            }
            for a, b, c in pairs[:args.top]
        ],
        'top_degree': [
            {'name': n, 'degree': d, 'mentions': person_canon.get(n, {}).get('count', 0)}
            for n, d in degree.most_common(50)
        ],
    }
    with open(out / 'cooccur.json', 'w', encoding='utf-8') as f:
        json.dump(out_data, f, indent=2, ensure_ascii=False)

    print(f'docs: {len(doc_canon_names)}  distinct names: {len(person_canon)}  pairs: {len(pairs)}')
    print(f'\nTop 25 co-occurring pairs:')
    for a, b, c in pairs[:25]:
        print(f'  {c:>3}× {a} ↔ {b}')


if __name__ == '__main__':
    main()
