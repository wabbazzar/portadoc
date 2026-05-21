#!/usr/bin/env python3
"""
Consolidate rank + ngram + grep + dates + tfidf + cooccur into findings.json.

Page kinds:
  - names_top20          (merged GIVENNAME+SURNAME, top 20)
  - email_top            (EMAIL label, promoted to early position)
  - press_recreate       (journalist-cited terms our corpus hits)
  - codeword_top         (Epstein-network euphemisms our corpus hits)
  - doc_dates_year       (canonical doc-date histogram by year)
  - mention_dates_year   (in-body date mentions histogram)
  - label_top            (other PII labels)
  - cooccur_pairs        (top entity co-occurrence pairs)
  - tfidf                (TF-IDF n-grams — phrases concentrated in few docs)
  - ngram                (doc-spread n-grams — phrases across many docs)

Usage:
    export_findings.py [--ngram-dir D] [--grep-dir D] [--dates-dir D]
                       [--tfidf-dir D] [--cooccur-dir D] [--per-page 20]
                       <rank_dir> <out.json>
"""
import sys, json, argparse, datetime
from pathlib import Path


def safe_load(path):
    try:
        return json.load(open(path))
    except (FileNotFoundError, json.JSONDecodeError):
        return None


def page_names_top20(rank, per_page):
    """Merged GIVENNAME+SURNAME top-N."""
    name_labels = [l for l in ('PERSON','NAME','GIVENNAME','SURNAME') if l in rank.get('labels', {})]
    if not name_labels: return None
    merged = {}
    for lbl in name_labels:
        for r in rank['labels'][lbl]['top']:
            key = r['text']
            if key not in merged:
                merged[key] = {
                    'text': key, 'count': 0, 'docs': 0, 'datasets': {},
                    'sample_doc_ids': r.get('sample_doc_ids', [])[:5],
                    'label_breakdown': {}, 'ocr_variants': r.get('ocr_variants', []),
                }
            m = merged[key]
            m['count'] += r['count']
            m['docs'] = max(m['docs'], r['docs'])
            for ds, n in (r.get('datasets') or {}).items():
                m['datasets'][ds] = m['datasets'].get(ds, 0) + n
            m['label_breakdown'][lbl] = r['count']
    ranked = sorted(merged.values(), key=lambda v: (-v['count'], v['text']))[:per_page]
    return {
        'kind': 'names_top20',
        'title': f'Top {per_page} names by mention count',
        'subtitle': f'merged from labels: {", ".join(name_labels)}',
        'rows': [{
            'rank': i+1, 'text': r['text'], 'count': r['count'],
            'docs': r['docs'], 'datasets': r['datasets'],
            'sample_doc_ids': r['sample_doc_ids'],
            'label_breakdown': r['label_breakdown'],
            'ocr_variants': r['ocr_variants'],
        } for i, r in enumerate(ranked)],
    }


def page_label(rank, label, per_page, page_idx=0, kind='label_top'):
    top = rank.get('labels', {}).get(label, {}).get('top', [])
    if not top: return None
    start = page_idx * per_page
    rows = top[start:start+per_page]
    if not rows: return None
    return {
        'kind': kind,
        'label': label,
        'page': page_idx + 1,
        'title': (f'Top {label}' + (f' (page {page_idx+1})' if page_idx > 0 else '')),
        'rows': [{
            'rank': start + i + 1, 'text': r['text'], 'count': r['count'],
            'docs': r['docs'], 'datasets': r['datasets'],
            'sample_doc_ids': r['sample_doc_ids'],
            'ocr_variants': r.get('ocr_variants', []),
        } for i, r in enumerate(rows)],
    }


def page_grep(grep, category_filter, title, kind, per_page, page_idx=0):
    """Pick grep rows whose category startswith category_filter prefix."""
    rows_all = [r for r in grep.get('rows', []) if r['category'].startswith(category_filter)]
    rows_all.sort(key=lambda r: (-r['count'], r['term']))
    start = page_idx * per_page
    rows = rows_all[start:start+per_page]
    if not rows: return None
    return {
        'kind': kind,
        'page': page_idx + 1,
        'title': f'{title}' + (f' (page {page_idx+1})' if page_idx > 0 else ''),
        'rows': [{
            'rank': start + i + 1, 'text': r['term'], 'count': r['count'],
            'docs': r['docs'], 'datasets': r['datasets'],
            'sample_doc_ids': r['sample_doc_ids'],
            'note': r.get('note', ''),
        } for i, r in enumerate(rows)],
    }


def page_dates(dates, kind):
    if not dates: return None
    if kind == 'doc_dates_year':
        data = dates.get('doc_dates', {}).get('by_year', {})
        title = 'Document dates by year (when each doc was written)'
        subtitle = f'method breakdown: {dates.get("doc_dates",{}).get("method_breakdown",{})}'
    elif kind == 'mention_dates_year':
        data = dates.get('mention_dates', {}).get('by_year_distinct_docs', {})
        title = 'Mention dates by year (years referenced in document bodies)'
        subtitle = 'unique-doc votes per year (de-duplicated within a doc)'
    else:
        return None
    rows = sorted(data.items(), key=lambda kv: kv[0])
    if not rows: return None
    max_count = max(c for _, c in rows)
    return {
        'kind': kind,
        'title': title,
        'subtitle': subtitle,
        'rows': [{
            'rank': i+1, 'text': str(y), 'count': c,
            'bar_pct': int(100 * c / max_count) if max_count else 0,
        } for i, (y, c) in enumerate(rows)],
    }


def page_tfidf(tfidf, n, per_page, page_idx=0):
    info = tfidf.get('sizes', {}).get(str(n)) or tfidf.get('sizes', {}).get(n)
    if not info: return None
    top = info.get('top', [])
    start = page_idx * per_page
    rows = top[start:start+per_page]
    if not rows: return None
    return {
        'kind': 'tfidf',
        'n': n,
        'page': page_idx + 1,
        'title': f'Top {n}-grams by TF-IDF (page {page_idx+1})',
        'subtitle': 'phrases concentrated in few docs (the inverse of doc-spread ngrams)',
        'rows': [{
            'rank': start + i + 1, 'text': r['text'],
            'count': r.get('peak_tf', 0),
            'docs': r.get('docs', 0),
            'score': r.get('score', 0),
            'peak_doc': r.get('peak_doc', ''),
        } for i, r in enumerate(rows)],
    }


def page_cooccur(cooccur, per_page, page_idx=0):
    pairs = cooccur.get('top_pairs', [])
    start = page_idx * per_page
    rows = pairs[start:start+per_page]
    if not rows: return None
    return {
        'kind': 'cooccur_pairs',
        'page': page_idx + 1,
        'title': f'Top entity co-occurrence pairs (page {page_idx+1})',
        'subtitle': 'pairs of names appearing together in the same document',
        'rows': [{
            'rank': start + i + 1,
            'text': f'{r["a"]} ↔ {r["b"]}',
            'count': r['cooccur_docs'],
            'docs': r['cooccur_docs'],
            'datasets': r.get('datasets', {}),
            'sample_doc_ids': r.get('sample_doc_ids', []),
        } for i, r in enumerate(rows)],
    }


def page_ngram(ng, n, per_page, page_idx=0):
    info = ng.get('sizes', {}).get(str(n)) or ng.get('sizes', {}).get(n)
    if not info: return None
    top = info.get('top', [])
    start = page_idx * per_page
    rows = top[start:start+per_page]
    if not rows: return None
    return {
        'kind': 'ngram',
        'n': n,
        'page': page_idx + 1,
        'title': f'Top {n}-grams by doc-spread (page {page_idx+1})',
        'rows': [{
            'rank': start + i + 1, 'text': r['text'],
            'count': r['count'], 'docs': r.get('docs', 0),
        } for i, r in enumerate(rows)],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('rank_dir')
    ap.add_argument('out_json')
    ap.add_argument('--ngram-dir')
    ap.add_argument('--grep-dir')
    ap.add_argument('--dates-dir')
    ap.add_argument('--tfidf-dir')
    ap.add_argument('--cooccur-dir')
    ap.add_argument('--quotes-dir')
    ap.add_argument('--threads-dir')
    ap.add_argument('--per-page', type=int, default=20)
    args = ap.parse_args()

    rank = safe_load(Path(args.rank_dir) / 'rankings.json') or {'labels': {}}
    ng    = safe_load(Path(args.ngram_dir)   / 'ngrams.json')   if args.ngram_dir   else None
    grep  = safe_load(Path(args.grep_dir)    / 'greplist.json') if args.grep_dir    else None
    dates = safe_load(Path(args.dates_dir)   / 'dates.json')    if args.dates_dir   else None
    tfidf = safe_load(Path(args.tfidf_dir)   / 'tfidf.json')    if args.tfidf_dir   else None
    coo   = safe_load(Path(args.cooccur_dir) / 'cooccur.json')  if args.cooccur_dir else None
    quotes= safe_load(Path(args.quotes_dir)  / 'quotes.json')   if args.quotes_dir  else None
    threads = safe_load(Path(args.threads_dir) / 'threads.json') if args.threads_dir else None

    pages = []

    # PAGE 1: top 20 names
    p = page_names_top20(rank, args.per_page)
    if p: pages.append(p)

    # PAGE 2: emails (promoted)
    p = page_label(rank, 'EMAIL', args.per_page)
    if p:
        p['title'] = 'Top email addresses observed in document bodies'
        pages.append(p)

    # Verbatim quote pages (most journalist-useful — show the actual quote)
    if quotes:
        # Split into hit/miss for cleaner reader experience
        hit = [p for p in quotes.get('phrases', []) if p['n_total_hits'] > 0]
        miss = [p for p in quotes.get('phrases', []) if p['n_total_hits'] == 0]
        if hit:
            pages.append({
                'kind': 'verbatim_quote',
                'title': 'Verbatim press-finding recreations',
                'subtitle': f'{len(hit)} of {len(hit)+len(miss)} journalist-cited phrases located in our corpus',
                'rows': [{
                    'rank': i+1,
                    'text': p['phrase'],
                    'count': p['n_total_hits'],
                    'docs': p['n_docs'],
                    'note': p['source'],
                    'samples': p['samples'],
                } for i, p in enumerate(hit)],
            })
        if miss:
            pages.append({
                'kind': 'verbatim_quote',
                'title': 'Press-finding phrases NOT (yet) in our corpus',
                'subtitle': 'likely live in House-Oversight-Dems-specific releases we have not pulled',
                'rows': [{
                    'rank': i+1, 'text': p['phrase'], 'count': 0, 'docs': 0,
                    'note': p['source'], 'samples': [],
                } for i, p in enumerate(miss)],
            })

    # PAGE 3-5: press recreate / codeword / doc-dates / mention-dates
    if grep:
        p = page_grep(grep, 'press:', 'Press findings recreated in our corpus', 'press_recreate', args.per_page)
        if p: pages.append(p)
        p = page_grep(grep, 'code:', 'Documented code-language hit counts', 'codeword_top', args.per_page)
        if p: pages.append(p)
    if dates:
        p = page_dates(dates, 'doc_dates_year')
        if p: pages.append(p)
        p = page_dates(dates, 'mention_dates_year')
        if p: pages.append(p)
    if coo:
        for i in range(3):
            p = page_cooccur(coo, args.per_page, i)
            if p: pages.append(p)
            else: break
    if threads:
        top = threads.get('top_threads', [])
        if top:
            pages.append({
                'kind': 'email_threads',
                'title': 'Longest reconstructed email threads',
                'subtitle': f'{threads.get("n_emails_scanned",0)} emails grouped by normalized subject ({threads.get("n_threads",0)} distinct threads)',
                'rows': [{
                    'rank': i+1,
                    'text': r['subject_sample'][:80],
                    'count': r['n_messages'],
                    'docs': r['n_messages'],
                    'note': f"{r.get('first_sent','?')[:10] if r.get('first_sent') else '?'} → {r.get('last_sent','?')[:10] if r.get('last_sent') else '?'}  ·  from: {(r['participants_from'][0] if r['participants_from'] else '?')[:40]}",
                } for i, r in enumerate(top[:args.per_page])],
            })

    # PAGE 6+: other PII labels paginated
    name_labels = {'PERSON','NAME','GIVENNAME','SURNAME','EMAIL'}
    label_order = sorted(
        (l for l in rank.get('labels', {}) if l not in name_labels),
        key=lambda l: -rank['labels'][l]['total_mentions']
    )
    for label in label_order:
        for i in range(5):
            p = page_label(rank, label, args.per_page, i)
            if p: pages.append(p)
            else: break

    # TF-IDF and doc-spread n-grams
    if tfidf:
        for n in (6, 5, 4):
            for i in range(3):
                p = page_tfidf(tfidf, n, args.per_page, i)
                if p: pages.append(p)
                else: break
    if ng:
        for n in (6, 5, 4):
            for i in range(5):
                p = page_ngram(ng, n, args.per_page, i)
                if p: pages.append(p)
                else: break

    # Datasets seen across all PII labels
    datasets_seen = set()
    for label_info in rank.get('labels', {}).values():
        for row in label_info.get('top', []):
            datasets_seen.update((row.get('datasets') or {}).keys())

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
