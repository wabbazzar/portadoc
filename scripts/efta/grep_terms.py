#!/usr/bin/env python3
"""
Grep a list of terms (code-language + journalist findings) across a corpus.jsonl
and emit per-term hit counts with per-dataset breakdown + sample doc IDs.

Output:
  - greplist.json  (consumed by export_findings.py)
  - greplist.csv

Each row: term, category, source, total_count, total_docs, datasets, sample_doc_ids

Search is case-insensitive substring (after lowercase). Whole-word for short
terms (≤4 chars or all-uppercase initials) to avoid spurious sub-matches
('JE' inside 'project', etc).

Usage:
    grep_terms.py <corpus.jsonl> <out_dir>
"""
import sys, json, re, csv, argparse
from pathlib import Path
from collections import defaultdict, Counter

# Make local imports work when run from anywhere
sys.path.insert(0, str(Path(__file__).resolve().parent))
import code_terms
import journalist_grep


def needs_word_boundary(term: str) -> bool:
    """Short or all-uppercase terms get word-boundary matching."""
    t = term.strip()
    if len(t) <= 4: return True
    if t.upper() == t and t.isalpha(): return True
    return False


def compile_term(term: str):
    t = term.strip()
    flags = re.IGNORECASE
    if needs_word_boundary(t):
        # Use word-boundary, escape special chars
        return re.compile(rf'(?<![A-Za-z0-9]){re.escape(t)}(?![A-Za-z0-9])', flags)
    return re.compile(re.escape(t), flags)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('corpus_jsonl')
    ap.add_argument('out_dir')
    args = ap.parse_args()

    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)

    # Build unified term list with (term, category, source) provenance
    all_terms = []
    for cat, term, note in code_terms.TERMS:
        all_terms.append((term, f'code:{cat}', note))
    for term, jcat, note in journalist_grep.TERMS:
        all_terms.append((term, f'press:{jcat}', note))

    # Pre-compile
    compiled = [(term, cat, note, compile_term(term)) for term, cat, note in all_terms]

    counters = {(t, c): {'count': 0, 'docs': set(), 'datasets': Counter(), 'sample': set(), 'note': n}
                for t, c, n, _ in compiled}

    n_docs = 0
    with open(args.corpus_jsonl) as f:
        for line in f:
            n_docs += 1
            try: rec = json.loads(line)
            except: continue
            text = rec.get('text', '')
            if not text: continue
            ds = rec.get('dataset', 'unknown')
            docid = rec.get('id', '?')
            for term, cat, note, rx in compiled:
                hits = rx.findall(text)
                if not hits: continue
                slot = counters[(term, cat)]
                slot['count'] += len(hits)
                slot['docs'].add(docid)
                slot['datasets'][ds] += len(hits)
                if len(slot['sample']) < 5:
                    slot['sample'].add(docid)

    # Build output sorted by total count desc
    rows = []
    for (term, cat), slot in counters.items():
        if slot['count'] == 0: continue
        rows.append({
            'term': term,
            'category': cat,
            'count': slot['count'],
            'docs': len(slot['docs']),
            'datasets': dict(slot['datasets']),
            'sample_doc_ids': sorted(slot['sample']),
            'note': slot['note'],
        })
    rows.sort(key=lambda r: (-r['count'], r['term']))

    with open(out / 'greplist.json', 'w', encoding='utf-8') as f:
        json.dump({'n_documents_scanned': n_docs, 'rows': rows}, f, indent=2, ensure_ascii=False)

    with open(out / 'greplist.csv', 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['rank','term','category','count','docs','datasets','sample_doc_ids','note'])
        for i, r in enumerate(rows, 1):
            w.writerow([i, r['term'], r['category'], r['count'], r['docs'],
                        '|'.join(f'{k}:{v}' for k,v in sorted(r['datasets'].items(), key=lambda kv: -kv[1])),
                        '|'.join(r['sample_doc_ids']),
                        r['note']])

    # Also a small report to stdout
    print(f'docs scanned: {n_docs}')
    print(f'terms with ≥1 hit: {len(rows)} / {len(all_terms)}')
    print(f'\nTop 25:')
    for r in rows[:25]:
        print(f"  {r['count']:>5} ({r['docs']:>3} docs)  [{r['category']:<14}] {r['term']}")


if __name__ == '__main__':
    main()
