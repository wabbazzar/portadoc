#!/usr/bin/env python3
"""
Compute character-stripped, stopword-filtered n-grams across the corpus.

Reads corpus.jsonl (from build_corpus.py), writes:
  - ngrams.json with top-N for each n in {4,5,6} (default)
  - per-n CSV files for full data

Normalization:
  - lowercase
  - strip non-alphanumerics except spaces
  - tokenize on whitespace
  - drop tokens shorter than 2 chars
  - drop very-common English stopwords (small in-file list)
  - drop tokens that are part of common email/footer boilerplate
    ("centurion relationship manager", etc — found in DS11 emails)

Usage:
    ngrams.py <corpus.jsonl> <out_dir> [--n 4 5 6] [--top 200]
"""
import sys, json, re, csv, argparse
from pathlib import Path
from collections import Counter

STOPWORDS = set('''
a an and are as at be by for from has have he her him his i if in is it its me my
no not of on or our she so than that the their them they this to us was we were what
when which who will with you your yours yourself ours ourselves but had been being
about into through during before after above below up down out off over under again
further then once here there all any both each few more most other some such only own
same too very can just don should now had has have having do does did doing would could
should might may must shall will i'm i've you're you've we're we've they're they've
'''.split())

# Email boilerplate words that show up in 80% of DS11 emails (centurion concierge footer)
# + legal disclaimer footer + OCR address-stamp artifacts seen in DS4 police reports
BOILERPLATE = set('''
centurion relationship manager regards hours mon tue wed thu fri thursday friday
saturday sunday monday tuesday wednesday est please thanks thank visit privacy
statement email security suspicious american express subject sent forwarded
phishing rights reserved service provider purchases shipping fees authorize
profile preference servicing remedy claims relating products provided
attorney client privileged confidential communication received error notify
immediately return mail destroy copies attachments intended recipient unlawful
contained part thereof strictly prohibited unauthorized review use disclosure
distribution reproduction reading dissemination forwarding action reliance
herein replying telephone deleting computer system network advised
white plains pis pls ny york legal disclaimer notice information
'''.split())

DROP = STOPWORDS | BOILERPLATE


def tokens(text: str):
    t = text.lower()
    t = re.sub(r'[^a-z0-9 ]+', ' ', t)
    for w in t.split():
        if len(w) >= 2 and w not in DROP:
            yield w


def ngrams_iter(tokens_list, n):
    for i in range(len(tokens_list) - n + 1):
        yield ' '.join(tokens_list[i:i+n])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('corpus_jsonl')
    ap.add_argument('out_dir')
    ap.add_argument('--n', type=int, nargs='+', default=[4, 5, 6])
    ap.add_argument('--top', type=int, default=200)
    ap.add_argument('--exclude-dataset', action='append', default=[],
                    help='Skip docs whose dataset field matches (repeatable). Useful for '
                         'memory-hungry corpora like DS10 financial bundles.')
    args = ap.parse_args()
    EXCLUDE = set(args.exclude_dataset)

    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)

    counters = {n: Counter() for n in args.n}
    doc_counters = {n: {} for n in args.n}  # ngram -> set(doc_id)

    n_docs = 0
    n_skipped = 0
    with open(args.corpus_jsonl) as f:
        for line in f:
            try:
                rec = json.loads(line)
            except Exception:
                continue
            if rec.get('dataset') in EXCLUDE:
                n_skipped += 1
                continue
            n_docs += 1
            toks = list(tokens(rec.get('text', '')))
            for n in args.n:
                seen_in_doc = set()
                for g in ngrams_iter(toks, n):
                    counters[n][g] += 1
                    seen_in_doc.add(g)
                for g in seen_in_doc:
                    doc_counters[n].setdefault(g, set()).add(rec['id'])

    summary = {
        'n_documents': n_docs,
        'sizes': {},
    }
    for n in args.n:
        c = counters[n]
        # Sort by document-spread (how widely seen across distinct docs) descending,
        # then by raw count. Filters out within-doc repetition (form-template stamps).
        items = sorted(
            ((g, ct) for g, ct in c.items() if len(doc_counters[n][g]) >= 3),
            key=lambda kv: (-len(doc_counters[n][kv[0]]), -kv[1]),
        )
        summary['sizes'][n] = {
            'total_distinct': len(c),
            'top': [
                {
                    'text': g,
                    'count': ct,
                    'docs': len(doc_counters[n][g]),
                }
                for g, ct in items[:args.top]
            ],
        }
        with open(out / f'ngrams_n{n}.csv', 'w', newline='', encoding='utf-8') as cf:
            w = csv.writer(cf)
            w.writerow(['rank','ngram','count','docs'])
            for i, (g, ct) in enumerate(items, 1):
                w.writerow([i, g, ct, len(doc_counters[n][g])])

    summary['n_skipped_excluded'] = n_skipped
    summary['excluded_datasets'] = sorted(EXCLUDE)
    with open(out / 'ngrams.json', 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f'documents: {n_docs}  (skipped {n_skipped} excluded; datasets={sorted(EXCLUDE)})')
    for n in args.n:
        info = summary['sizes'][n]
        print(f'  n={n}: {info["total_distinct"]} distinct, top showing {len(info["top"])}')


if __name__ == '__main__':
    main()
