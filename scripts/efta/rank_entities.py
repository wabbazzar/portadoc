#!/usr/bin/env python3
"""
Rank entities by occurrence across a tagged corpus.

Reads tagged.jsonl (from ner_tag.py), produces:
  - rankings.json with per-label counts and top entries
  - rankings_PERSON.csv (one CSV per label, sorted)

Entity-text normalization:
  - lowercase + strip
  - collapse internal whitespace
  - strip subword markers (## from BERT tokenization)
  - canonicalize OCR @ ↔ © for emails

Filters out low-confidence (<0.5) hits and obvious garbage (<2 chars, all-digits-of-len-3).

Usage:
    rank_entities.py <tagged.jsonl> <out_dir> [--min-score 0.5] [--top 100]
"""
import sys, json, re, csv, argparse
from pathlib import Path
from collections import Counter, defaultdict

# OCR-aware canonicalization (Levenshtein-clustered merge for EMAIL + PERSON)
import canonicalize


def normalize_text(text: str, label: str) -> str:
    t = text.replace('##', '').strip()
    t = re.sub(r'\s+', ' ', t)
    if label.upper() in ('EMAIL', 'I-EMAIL', 'EMAIL_ADDRESS', 'I-EMAIL_ADDRESS'):
        t = t.replace('©', '@').lower()
    elif label.upper() in ('PERSON', 'I-PERSON', 'GIVENNAME', 'SURNAME', 'I-GIVENNAME', 'I-SURNAME'):
        # Title-case people names so "JOHN" and "John" merge
        t = t.title()
    else:
        t = t.lower()
    return t


def looks_garbage(text: str, label: str) -> bool:
    if len(text) < 2:
        return True
    if text.isdigit() and len(text) < 4:
        return True
    if re.match(r'^[\W_]+$', text):
        return True
    # Single letters and 1-char fragments are useless except for SSN-fragments / digits
    if label.upper().startswith('PERSON') and len(text) < 3:
        return True
    return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('tagged_jsonl')
    ap.add_argument('out_dir')
    ap.add_argument('--min-score', type=float, default=0.5)
    ap.add_argument('--top', type=int, default=100)
    args = ap.parse_args()

    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)

    # entity_text -> {count, docs:set, datasets:Counter, sample_score:max}
    counts = defaultdict(lambda: defaultdict(lambda: {
        'count': 0,
        'docs': set(),
        'datasets': Counter(),
        'max_score': 0.0,
    }))

    n_docs = 0
    n_hits = 0
    with open(args.tagged_jsonl) as f:
        for line in f:
            try:
                rec = json.loads(line)
            except Exception:
                continue
            n_docs += 1
            ds = rec.get('dataset', 'unknown')
            for e in rec.get('entities', []):
                if e['score'] < args.min_score:
                    continue
                label = (e['label'] or 'UNKNOWN').upper().lstrip('I-').lstrip('B-')
                norm = normalize_text(e['text'], label)
                if looks_garbage(norm, label):
                    continue
                slot = counts[label][norm]
                slot['count'] += 1
                slot['docs'].add(rec['id'])
                slot['datasets'][ds] += 1
                slot['max_score'] = max(slot['max_score'], e['score'])
                n_hits += 1

    # Canonicalize EMAIL + name-flavored labels before ranking
    for label in list(counts.keys()):
        items = counts[label]
        if label in ('EMAIL', 'EMAIL_ADDRESS'):
            counts[label] = canonicalize.cluster_emails(dict(items))
        elif label in ('GIVENNAME', 'SURNAME', 'PERSON', 'NAME'):
            counts[label] = canonicalize.cluster_persons(dict(items))

    # Serialize
    rankings = {
        'n_documents_scanned': n_docs,
        'n_hits_total': n_hits,
        'labels': {},
    }
    for label, items in counts.items():
        ranked = sorted(items.items(), key=lambda kv: (-kv[1]['count'], kv[0]))
        rankings['labels'][label] = {
            'total_unique': len(ranked),
            'total_mentions': sum(v['count'] for _, v in ranked),
            'top': [
                {
                    'text': text,
                    'count': v['count'],
                    'docs': len(v['docs']),
                    'sample_doc_ids': sorted(v['docs'])[:5],
                    'datasets': dict(v['datasets']),
                    'max_score': round(v['max_score'], 3),
                    'ocr_variants': v.get('ocr_variants', []),
                }
                for text, v in ranked[:args.top]
            ],
        }
        # Per-label CSV
        with open(out / f'rankings_{label}.csv', 'w', newline='', encoding='utf-8') as cf:
            w = csv.writer(cf)
            w.writerow(['rank','text','count','docs','datasets','max_score'])
            for i, (text, v) in enumerate(ranked, 1):
                w.writerow([i, text, v['count'], len(v['docs']),
                            '|'.join(f'{k}:{n}' for k,n in v['datasets'].most_common()),
                            f'{v["max_score"]:.3f}'])

    with open(out / 'rankings.json', 'w', encoding='utf-8') as f:
        json.dump(rankings, f, indent=2, ensure_ascii=False)

    print(f'documents: {n_docs}  hits: {n_hits}  labels: {len(counts)}')
    for label in sorted(counts.keys()):
        info = rankings['labels'][label]
        print(f"  {label:>20}: {info['total_unique']:>6} unique, {info['total_mentions']:>7} mentions")


if __name__ == '__main__':
    main()
