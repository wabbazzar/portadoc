#!/usr/bin/env python3
"""
Two date analyses:
  1. **Doc-date**: assign ONE canonical date per document via priority chain
     (email Sent: > police-report Date: > grand-jury heading > first-body date).
     Emit year/month histogram.
  2. **Mention-date**: every date pattern in document bodies, deduped per-doc.
     Shows what years the corpus REFERS TO vs. what years it was WRITTEN IN.

Outputs:
  - dates.json   {"doc_dates": {...}, "mention_dates": {...}}
  - doc_dates.csv, mention_dates.csv

Usage:
    dates.py <corpus.jsonl> <out_dir>
"""
import sys, json, re, csv, argparse
from pathlib import Path
from collections import Counter, defaultdict

MONTHS = ('january february march april may june july august '
          'september october november december').split()
MONTH_ABBR = ('jan feb mar apr may jun jul aug sep sept oct nov dec').split()
MONTH_MAP = {m: i+1 for i, m in enumerate(MONTHS)}
for i, m in enumerate(MONTH_ABBR):
    if m == 'sept': MONTH_MAP[m] = 9
    else: MONTH_MAP[m] = i+1 if i < 12 else 12

# (regex, parser-fn) — parser returns (year:int, month:int|None) or None
# Year ceiling: documents in the Epstein corpus cannot legitimately post-date the
# 2026 release. Pre-1990 documents exist but are vanishingly rare (Epstein's
# career didn't take off until the mid-90s); 1953-dated false-positives from
# OCR-mangled SSNs and serial numbers were polluting the histogram.
DATE_PATTERNS = [
    # 12/31/2017, 12-31-2017, 12.31.2017
    (re.compile(r'\b(0?[1-9]|1[0-2])[/\-.](0?[1-9]|[12][0-9]|3[01])[/\-.](199[0-9]|20[0-2][0-9])\b'),
     lambda m: (int(m.group(3)), int(m.group(1)))),
    # 12/31/17 → 2017
    (re.compile(r'\b(0?[1-9]|1[0-2])[/\-.](0?[1-9]|[12][0-9]|3[01])[/\-.]([0-9]{2})\b'),
     lambda m: (2000 + int(m.group(3)) if int(m.group(3)) < 50 else 1900 + int(m.group(3)),
                int(m.group(1)))),
    # 2017-12-31
    (re.compile(r'\b(199[0-9]|20[0-2][0-9])-(0[1-9]|1[0-2])-(0[1-9]|[12][0-9]|3[01])\b'),
     lambda m: (int(m.group(1)), int(m.group(2)))),
    # January 31, 2017  |  Jan. 31, 2017  |  Jan 31 2017
    (re.compile(r'\b(January|February|March|April|May|June|July|August|September|October|November|December|Jan\.?|Feb\.?|Mar\.?|Apr\.?|Jun\.?|Jul\.?|Aug\.?|Sept?\.?|Oct\.?|Nov\.?|Dec\.?)\s+([0-3]?[0-9])(?:,)?\s+(199[0-9]|20[0-2][0-9])\b',
                re.IGNORECASE),
     lambda m: (int(m.group(3)), MONTH_MAP.get(m.group(1).lower().rstrip('.').replace('sept','sep')[:3], None))),
    # 31 January 2017 (DD Mon YYYY)
    (re.compile(r'\b([0-3]?[0-9])\s+(January|February|March|April|May|June|July|August|September|October|November|December|Jan\.?|Feb\.?|Mar\.?|Apr\.?|Jun\.?|Jul\.?|Aug\.?|Sept?\.?|Oct\.?|Nov\.?|Dec\.?)\s+(199[0-9]|20[0-2][0-9])\b',
                re.IGNORECASE),
     lambda m: (int(m.group(3)), MONTH_MAP.get(m.group(2).lower().rstrip('.').replace('sept','sep')[:3], None))),
    # Bare year, last resort: only used if nothing better matched
]
BARE_YEAR = re.compile(r'\b(199[0-9]|20[0-2][0-9])\b')


def all_dates_in(text: str):
    """Yield (year, month) for every parseable date in text."""
    for rx, parser in DATE_PATTERNS:
        for m in rx.finditer(text):
            try:
                y, mo = parser(m)
                if 1990 <= y <= 2026 and (mo is None or 1 <= mo <= 12):
                    yield (y, mo)
            except Exception:
                pass


def doc_canonical_date(text: str, dataset: str):
    """
    Priority chain (first hit wins, only inspect the first ~1500 chars per layer):
      1. 'Sent:' header line (emails)
      2. 'Date:' or 'Report Date' (police reports)
      3. First Month-DD-YYYY pattern in first 800 chars (transcripts, letters)
      4. First numeric date in first 800 chars
      5. Earliest year in first 800 chars (bare-year fallback)
      6. None
    """
    head = text[:2000]
    # Layer 1: Sent header
    m = re.search(r'(?i)^\s*Sent\s*[:.]\s*([A-Za-z]{0,8}\s*[0-3]?[0-9][/\-.][0-3]?[0-9][/\-.](?:19|20)?[0-9]{2,4})',
                  head, re.MULTILINE)
    if m:
        for d in all_dates_in(m.group(1)):
            return d, 'sent_header'
    # Layer 1b: standalone date line right after 'Sent:' label
    m = re.search(r'(?i)Sent[:\s]+\n?\s*[A-Za-z]{0,12}\s*([0-3]?[0-9][/\-.][0-3]?[0-9][/\-.](?:19|20)?[0-9]{2,4})',
                  head)
    if m:
        for d in all_dates_in(m.group(1)):
            return d, 'sent_header'
    # Layer 2: police report 'Date:' / 'Report Date'
    m = re.search(r'(?i)(?:Report\s*Date|Date)\s*[.:]\s*([0-3]?[0-9][/\-.][0-3]?[0-9][/\-.](?:19|20)?[0-9]{2,4})',
                  head)
    if m:
        for d in all_dates_in(m.group(1)):
            return d, 'report_date_header'
    # Layer 3: First Month-DD-YYYY in first 800 chars
    for d in all_dates_in(head[:800]):
        return d, 'first_body_date'
    # Layer 4: bare year fallback
    m = BARE_YEAR.search(head[:800])
    if m:
        y = int(m.group(1))
        if 1990 <= y <= 2026:
            return (y, None), 'bare_year_fallback'
    return None, 'none'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('corpus_jsonl')
    ap.add_argument('out_dir')
    args = ap.parse_args()
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)

    doc_year_counts = Counter()
    doc_yearmonth_counts = Counter()
    method_counts = Counter()
    mention_year_counts = Counter()
    mention_year_counts_unique_doc = Counter()  # one vote per doc per year

    doc_date_rows = []
    n_docs = 0
    n_with_date = 0
    with open(args.corpus_jsonl) as f:
        for line in f:
            n_docs += 1
            try: rec = json.loads(line)
            except: continue
            text = rec.get('text', '')
            ds = rec.get('dataset', 'unknown')
            d, method = doc_canonical_date(text, ds)
            method_counts[method] += 1
            if d is not None:
                y, mo = d
                n_with_date += 1
                doc_year_counts[y] += 1
                if mo:
                    doc_yearmonth_counts[f'{y}-{mo:02d}'] += 1
                doc_date_rows.append({
                    'id': rec.get('id','?'),
                    'dataset': ds,
                    'year': y, 'month': mo,
                    'method': method,
                })
            # mention dates
            seen_years = set()
            for (y, _) in all_dates_in(text):
                mention_year_counts[y] += 1
                seen_years.add(y)
            for y in seen_years:
                mention_year_counts_unique_doc[y] += 1

    out_data = {
        'n_documents': n_docs,
        'n_with_canonical_date': n_with_date,
        'doc_dates': {
            'by_year': dict(sorted(doc_year_counts.items())),
            'by_yearmonth': dict(sorted(doc_yearmonth_counts.items())),
            'method_breakdown': dict(method_counts),
        },
        'mention_dates': {
            'by_year_total_mentions': dict(sorted(mention_year_counts.items())),
            'by_year_distinct_docs': dict(sorted(mention_year_counts_unique_doc.items())),
        },
    }
    with open(out / 'dates.json', 'w', encoding='utf-8') as f:
        json.dump(out_data, f, indent=2, ensure_ascii=False)
    with open(out / 'doc_dates.csv', 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=['id','dataset','year','month','method'])
        w.writeheader(); w.writerows(doc_date_rows)

    print(f'docs: {n_docs}  with_canonical_date: {n_with_date}')
    print(f'method breakdown: {dict(method_counts)}')
    print(f'\ndoc-date year histogram:')
    for y, c in sorted(doc_year_counts.items()):
        bar = '#' * min(60, c)
        print(f'  {y}: {c:>5}  {bar}')
    print(f'\nmention-date year histogram (unique-doc votes):')
    for y, c in sorted(mention_year_counts_unique_doc.items()):
        if c < 5: continue
        bar = '#' * min(60, c)
        print(f'  {y}: {c:>5}  {bar}')


if __name__ == '__main__':
    main()
