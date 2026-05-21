#!/usr/bin/env python3
"""
Build per-doc snippet JSON files for the EFTA viewer's clickable doc modal.

Scans findings.json for every referenced doc_id, then writes one
`efta/docs/<id>.json` per doc with {id, dataset, text, n_chars, n_words}.

Per-doc files are bounded to ~8KB (text capped at 5000 chars centered around
the most relevant match where applicable). A "load full" link in the UI can
later fetch a larger version on demand.

Usage:
    build_doc_snippets.py <corpus.jsonl> <findings.json> <out_dir>
"""
import sys, json, re
from pathlib import Path
from collections import Counter

MAX_SNIPPET = 5000


def collect_doc_ids(findings: dict) -> set[str]:
    """Walk findings.json and pull every doc_id referenced anywhere."""
    ids = set()
    for page in findings.get('pages', []):
        for row in page.get('rows', []):
            # Direct doc_id field (topic_search, iMessages)
            if row.get('doc_id'):
                ids.add(row['doc_id'])
            # sample_doc_ids array (NER ranks, press_recreate)
            for sid in row.get('sample_doc_ids', []) or []:
                if sid: ids.add(sid)
            # peak_doc (TF-IDF)
            if row.get('peak_doc'):
                ids.add(row['peak_doc'])
            # nested samples (verbatim_quote)
            for s in row.get('samples', []) or []:
                if s.get('doc_id'): ids.add(s['doc_id'])
    return ids


def smart_snippet(text: str, max_chars: int = MAX_SNIPPET) -> str:
    """Return first max_chars chars, trimmed to word boundary."""
    if len(text) <= max_chars:
        return text
    s = text[:max_chars]
    # back off to last whitespace
    sp = s.rfind(' ')
    if sp > max_chars - 200:
        s = s[:sp]
    return s + '…'


def main():
    if len(sys.argv) < 4:
        print(__doc__); sys.exit(2)
    corpus_path, findings_path, out_dir = sys.argv[1], sys.argv[2], sys.argv[3]
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)

    findings = json.load(open(findings_path))
    target_ids = collect_doc_ids(findings)
    print(f'doc IDs referenced in findings.json: {len(target_ids)}')

    written = 0
    by_ds = Counter()
    with open(corpus_path) as f:
        for line in f:
            try: rec = json.loads(line)
            except: continue
            if rec.get('id') not in target_ids: continue
            text = rec.get('text', '') or ''
            doc = {
                'id': rec['id'],
                'dataset': rec.get('dataset', 'unknown'),
                'n_chars': len(text),
                'n_words': rec.get('n_words', len(text.split())),
                'text': smart_snippet(text),
                'source': rec.get('source', ''),
            }
            outfile = out / f'{rec["id"]}.json'
            outfile.write_text(json.dumps(doc, ensure_ascii=False), encoding='utf-8')
            written += 1
            by_ds[doc['dataset']] += 1

    print(f'wrote {written} doc snippets')
    for k, v in sorted(by_ds.items(), key=lambda kv: -kv[1]):
        print(f'  {k}: {v}')


if __name__ == '__main__':
    main()
