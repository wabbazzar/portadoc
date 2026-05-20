#!/usr/bin/env python3
"""
TF-IDF n-gram ranking.

Surfaces phrases that are RARE globally but CONCENTRATED in specific documents
(the inverse of the doc-spread filter in ngrams.py, which surfaces ubiquitous
phrases).

For each n-gram g:
  score(g) = sum_{d : g in d} tf(g, d) × log(N / df(g))

Where tf is term-frequency in that doc and df is document-frequency.
Result: phrases that are characteristic of FEW docs but appear A LOT in them.

Usage:
    tfidf.py <corpus.jsonl> <out_dir> [--n 4 5 6] [--top 100]
"""
import sys, json, re, math, csv, argparse
from pathlib import Path
from collections import Counter, defaultdict

# Reuse stopword + boilerplate filter from ngrams.py
sys.path.insert(0, str(Path(__file__).resolve().parent))
from ngrams import tokens, ngrams_iter


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('corpus_jsonl')
    ap.add_argument('out_dir')
    ap.add_argument('--n', type=int, nargs='+', default=[4, 5, 6])
    ap.add_argument('--top', type=int, default=150)
    ap.add_argument('--min-doc-tf', type=int, default=3,
                    help='Minimum tf in a single doc for n-gram to count')
    args = ap.parse_args()
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)

    # Per-doc tf counters + global df
    docs = []  # list of (id, dataset, {n: Counter})
    with open(args.corpus_jsonl) as f:
        for line in f:
            try: rec = json.loads(line)
            except: continue
            toks = list(tokens(rec.get('text', '')))
            per_n = {}
            for n in args.n:
                c = Counter(ngrams_iter(toks, n))
                per_n[n] = c
            docs.append((rec.get('id','?'), rec.get('dataset','unknown'), per_n))
    N = len(docs)

    summary = {'n_documents': N, 'sizes': {}}

    for n in args.n:
        df = Counter()
        for _, _, per_n in docs:
            for g in per_n[n].keys():
                df[g] += 1
        # score = sum_d (tf(g,d) * log(N/df(g))) for g with df ≥ 2 and at least one doc tf ≥ min-doc-tf
        scores = defaultdict(float)
        peak_doc = {}
        peak_tf = {}
        peak_doc_count = {}
        for docid, ds, per_n in docs:
            for g, tf in per_n[n].items():
                if df[g] < 2: continue  # global hapaxes — useless
                if tf < args.min_doc_tf and df[g] > 5:
                    # if it's not rare-and-concentrated, skip
                    continue
                idf = math.log(N / df[g])
                scores[g] += tf * idf
                if tf > peak_tf.get(g, 0):
                    peak_tf[g] = tf
                    peak_doc[g] = docid
                    peak_doc_count[g] = df[g]

        ranked = sorted(scores.items(), key=lambda kv: -kv[1])
        summary['sizes'][n] = {
            'total_distinct': len(scores),
            'top': [
                {
                    'text': g,
                    'score': round(scores[g], 2),
                    'peak_tf': peak_tf[g],
                    'peak_doc': peak_doc[g],
                    'docs': peak_doc_count[g],
                }
                for g, _ in ranked[:args.top]
            ],
        }
        with open(out / f'tfidf_n{n}.csv', 'w', newline='', encoding='utf-8') as cf:
            w = csv.writer(cf)
            w.writerow(['rank','ngram','score','peak_tf','peak_doc','docs'])
            for i, (g, sc) in enumerate(ranked, 1):
                w.writerow([i, g, f'{sc:.2f}', peak_tf[g], peak_doc[g], peak_doc_count[g]])

    with open(out / 'tfidf.json', 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f'docs: {N}')
    for n in args.n:
        print(f'  n={n}: {summary["sizes"][n]["total_distinct"]} scored, top shown {len(summary["sizes"][n]["top"])}')


if __name__ == '__main__':
    main()
