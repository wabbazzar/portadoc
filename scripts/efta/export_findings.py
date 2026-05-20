#!/usr/bin/env python3
"""
Consolidate rank + ngram output into a single findings.json for the website.

Output shape:
{
  "generated_at": "<ISO8601>",
  "corpus": {
    "n_documents": <int>,
    "n_entity_hits": <int>,
    "datasets_present": ["dataset_3", "dataset_11", ...]
  },
  "pages": [
    {
      "kind": "names_top20",
      "title": "Top 20 names by mention count",
      "rows": [{"rank":1, "text":"...", "count":N, "docs":M}, ...]
    },
    {
      "kind": "label_top",
      "label": "EMAIL",
      "title": "Top email addresses",
      "rows": [...]
    },
    {
      "kind": "ngram",
      "n": 6,
      "title": "Top 6-grams",
      "rows": [...]
    },
    ...
  ]
}

Usage:
    export_findings.py <rank_dir> <ngram_dir> <out.json> [--per-page 20]
"""
import sys, json, argparse, datetime
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('rank_dir')
    ap.add_argument('ngram_dir')
    ap.add_argument('out_json')
    ap.add_argument('--per-page', type=int, default=20)
    args = ap.parse_args()

    rank = json.load(open(Path(args.rank_dir) / 'rankings.json'))
    ng   = json.load(open(Path(args.ngram_dir) / 'ngrams.json'))

    datasets_seen = set()
    for label_info in rank.get('labels', {}).values():
        for row in label_info.get('top', []):
            datasets_seen.update(row.get('datasets', {}).keys())

    pages = []

    # Page 1: Top 20 names (combine PERSON / GIVENNAME / SURNAME — different models use
    # different label names; Piiranha-v1 uses GIVENNAME + SURNAME separately).
    name_labels = [l for l in ('PERSON','NAME','GIVENNAME','SURNAME') if l in rank.get('labels', {})]
    if name_labels:
        merged = {}
        for lbl in name_labels:
            for r in rank['labels'][lbl]['top']:
                key = r['text']
                if key not in merged:
                    merged[key] = {
                        'text': key,
                        'count': 0,
                        'docs': 0,
                        'datasets': {},
                        'sample_doc_ids': r.get('sample_doc_ids', [])[:5],
                        'label_breakdown': {},
                    }
                m = merged[key]
                m['count'] += r['count']
                m['docs'] = max(m['docs'], r['docs'])  # docs is unique-doc count per label
                for ds, n in (r.get('datasets') or {}).items():
                    m['datasets'][ds] = m['datasets'].get(ds, 0) + n
                m['label_breakdown'][lbl] = r['count']
        ranked = sorted(merged.values(), key=lambda v: (-v['count'], v['text']))[:args.per_page]
        pages.append({
            'kind': 'names_top20',
            'title': f'Top {args.per_page} names by mention count',
            'subtitle': f'merged from labels: {", ".join(name_labels)}',
            'rows': [{
                'rank': i+1,
                'text': r['text'],
                'count': r['count'],
                'docs': r['docs'],
                'datasets': r['datasets'],
                'sample_doc_ids': r['sample_doc_ids'],
                'label_breakdown': r['label_breakdown'],
            } for i, r in enumerate(ranked)],
        })

    # Pages for other labels (paginate top-100 by per-page)
    label_order = sorted(rank.get('labels', {}).keys(), key=lambda l: -rank['labels'][l]['total_mentions'])
    for label in label_order:
        if label in name_labels:
            continue
        top = rank['labels'][label]['top']
        if not top:
            continue
        # Paginate
        for i in range(0, min(len(top), args.per_page * 5), args.per_page):
            page_rows = top[i:i+args.per_page]
            pages.append({
                'kind': 'label_top',
                'label': label,
                'page': i // args.per_page + 1,
                'title': f'Top {label} (page {i // args.per_page + 1})',
                'rows': [{
                    'rank': i + j + 1,
                    'text': r['text'],
                    'count': r['count'],
                    'docs': r['docs'],
                    'datasets': r['datasets'],
                    'sample_doc_ids': r['sample_doc_ids'],
                } for j, r in enumerate(page_rows)],
            })

    # N-gram pages
    for n in sorted(ng.get('sizes', {}).keys(), key=int, reverse=True):
        info = ng['sizes'][n]
        # Paginate top
        for i in range(0, min(len(info['top']), args.per_page * 5), args.per_page):
            page_rows = info['top'][i:i+args.per_page]
            pages.append({
                'kind': 'ngram',
                'n': int(n),
                'page': i // args.per_page + 1,
                'title': f'Top {n}-grams (page {i // args.per_page + 1})',
                'rows': [{
                    'rank': i + j + 1,
                    'text': r['text'],
                    'count': r['count'],
                    'docs': r['docs'],
                } for j, r in enumerate(page_rows)],
            })

    out = {
        'generated_at': datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'corpus': {
            'n_documents': rank.get('n_documents_scanned'),
            'n_entity_hits': rank.get('n_hits_total'),
            'datasets_present': sorted(datasets_seen),
        },
        'model': 'iiiorg/piiranha-v1-detect-personal-information',
        'pages': pages,
    }
    Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_json, 'w', encoding='utf-8') as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f'findings.json written: {len(pages)} pages, {sum(len(p["rows"]) for p in pages)} rows total')


if __name__ == '__main__':
    main()
