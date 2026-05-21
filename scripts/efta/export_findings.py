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
    ap.add_argument('--imessages-dir')
    ap.add_argument('--ds10-dir')
    ap.add_argument('--names-dir')
    ap.add_argument('--topics-dir')
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
    imsgs = safe_load(Path(args.imessages_dir) / 'imessages.json') if args.imessages_dir else None
    ds10 = safe_load(Path(args.ds10_dir) / 'ds10_financial.json') if args.ds10_dir else None
    names_full = safe_load(Path(args.names_dir) / 'names_full.json') if args.names_dir else None
    topics = safe_load(Path(args.topics_dir) / 'topic_search.json') if args.topics_dir else None

    pages = []

    # PAGE 1: top names by full-corpus grep (preferred over NER which is partial)
    if names_full and names_full.get('rows'):
        top = names_full['rows']
        per_page = max(args.per_page, 50)  # show more — user wanted full ranking
        for i in range(0, min(len(top), per_page * 3), per_page):
            chunk = top[i:i+per_page]
            pages.append({
                'kind': 'names_grep',
                'page': i // per_page + 1,
                'title': 'Top names by mention count (full-corpus grep)' + (
                    f' — page {i//per_page+1}' if i > 0 else ''),
                'subtitle': f'{names_full.get("n_documents_scanned",0)} docs scanned. Curated alias list; case-insensitive word-boundary regex.',
                'explainer': 'Counts mentions of each named entity across the FULL corpus (~173K docs). Uses a curated alias list (e.g. "Maxwell" matches "Ghislaine Maxwell", "G. Maxwell", "GM"). Sorted by number of distinct documents containing a mention.',
                'rows': [{
                    'rank': i+j+1, 'text': r['name'], 'count': r['mentions'],
                    'docs': r['docs'], 'datasets': r['datasets'],
                    'sample_doc_ids': r['sample_doc_ids'],
                    'note': r['note'],
                } for j, r in enumerate(chunk)],
            })

    # Topic-search pages (WW3 / simulation / pandemic / antarctica / reptilian /
    # sacrifice / ritual / pedophilia — clickable per-match with year histogram)
    if topics:
        for t in topics.get('topics', []):
            if t['total_matches'] == 0: continue
            pages.append({
                'kind': 'topic_search',
                'topic': t['topic'],
                'title': f"Topic search: {t['topic']}",
                'subtitle': f"{t['total_matches']} matches in {t['n_docs']} docs · {t.get('n_dated',0)} dated, {t.get('n_undated',0)} undated · patterns: {', '.join(t['patterns'][:3])}…",
                'explainer': f"Per-match records (not aggregate) for '{t['topic']}'. {t['note']}. Year histogram at top shows when these messages were written (pre-2020 vs post). Each row is one hit: message date, dataset, doc-id, verbatim matched phrase, 300-char context.",
                'by_year': t.get('by_year', {}),
                'rows': [{
                    'rank': i+1,
                    'date': m['date'],
                    'doc_id': m['doc_id'],
                    'dataset': m['dataset'],
                    'matched': m['matched'],
                    'context': m['context'],
                } for i, m in enumerate(t['matches'])],
            })

    # Emails page (from NER EMAIL label — may be partial but still useful)
    p = page_label(rank, 'EMAIL', args.per_page)
    if p:
        p['title'] = 'Top email addresses observed in document bodies'
        p['explainer'] = 'Email addresses tagged by the PII model (Piiranha-v1), after OCR-aware fuzzy-merge that collapses near-duplicates of the same address (Levenshtein ≤2 on local part). Note: NER is partial — only ~7% of the corpus has been tagged so far.'
        pages.append(p)

    # Verbatim quote pages (most journalist-useful — show the actual quote)
    if quotes:
        hit = [p for p in quotes.get('phrases', []) if p['n_total_hits'] > 0]
        miss = [p for p in quotes.get('phrases', []) if p['n_total_hits'] == 0]
        if hit:
            pages.append({
                'kind': 'verbatim_quote',
                'title': 'Verbatim press-finding recreations',
                'subtitle': f'{len(hit)} of {len(hit)+len(miss)} journalist-cited phrases located in our corpus',
                'explainer': 'Phrases that named journalists / press releases have publicly quoted from these EFTA documents. Each entry shows where we found that exact phrase in our locally-mirrored copy. The blockquote is verbatim — same text reporters were quoting.',
                'rows': [{
                    'rank': i+1, 'text': p['phrase'], 'count': p['n_total_hits'],
                    'docs': p['n_docs'], 'note': p['source'], 'samples': p['samples'],
                } for i, p in enumerate(hit)],
            })
        if miss:
            pages.append({
                'kind': 'verbatim_quote',
                'title': 'Press-finding phrases NOT (yet) in our corpus',
                'subtitle': 'likely live in House-Oversight-Dems-specific releases we have not pulled',
                'explainer': 'Press-cited phrases the pipeline could NOT locate in our local corpus. These almost certainly live in House Oversight Democrats releases distributed as individual Google Drive previews rather than bulk downloads. Pull plan documented in WORK_LOG.',
                'rows': [{
                    'rank': i+1, 'text': p['phrase'], 'count': 0, 'docs': 0,
                    'note': p['source'], 'samples': [],
                } for i, p in enumerate(miss)],
            })

    # iMessage chronological pages (forensic Mac export)
    if imsgs and imsgs.get('messages'):
        msgs = imsgs['messages']
        msgs_per_page = 60
        for i in range(0, min(len(msgs), msgs_per_page * 20), msgs_per_page):
            page_msgs = msgs[i:i+msgs_per_page]
            pages.append({
                'kind': 'imessages',
                'page': i // msgs_per_page + 1,
                'title': f'Epstein iMessages (chronological) — page {i//msgs_per_page + 1}',
                'subtitle': f'{imsgs["n_messages"]} messages from {imsgs["n_source_docs"]} forensic-export docs, sender breakdown: {imsgs["by_sender"]}',
                'rows': [{
                    'rank': i + j + 1,
                    'text': m['message'],
                    'note': m.get('timestamp', '?'),
                    'sender': m['sender'],
                    'doc_id': m['doc_id'],
                } for j, m in enumerate(page_msgs)],
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
    # DS10 financial pages (one per category)
    if ds10:
        CAT_LABELS = {
            'banker': 'JPMorgan banker names',
            'entity': 'Counterparty entities (LLC / Inc / Corp / Trust)',
            'bank': 'External bank mentions',
            'beneficiary': 'Wire-transfer beneficiaries',
            'account_holder': 'Account holder mentions',
            'poa_grantee': 'Power-of-attorney grantees',
            'money': 'Dollar amounts',
            'acct_type': 'Account type tokens',
        }
        cat_order = ['poa_grantee','beneficiary','account_holder','entity','banker','bank','money','acct_type']
        for cat in cat_order:
            info = ds10.get('categories', {}).get(cat)
            if not info or not info.get('top'): continue
            top = info['top']
            for i in range(0, min(len(top), args.per_page * 3), args.per_page):
                rows = top[i:i+args.per_page]
                pages.append({
                    'kind': 'ds10_financial',
                    'category': cat,
                    'page': i // args.per_page + 1,
                    'title': f'DS10 financial — {CAT_LABELS.get(cat, cat)}' + (
                        f' (page {i//args.per_page + 1})' if i > 0 else ''),
                    'subtitle': f'{ds10.get("n_documents_scanned",0)} DS10 docs scanned · {info["unique_values"]} unique values',
                    'rows': [{
                        'rank': i + j + 1,
                        'text': r['text'],
                        'count': r['count'],
                        'docs': r['docs'],
                        'sample_doc_ids': r['sample_doc_ids'],
                    } for j, r in enumerate(rows)],
                })

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

    # Inject per-kind explainer text for any page missing it
    EXPLAINERS = {
        'names_grep': 'Counts mentions of each named entity across the FULL corpus (~173K docs). Uses a curated alias list (e.g. "Maxwell" matches "Ghislaine Maxwell", "G. Maxwell", "GM"). Sorted by distinct documents containing a mention.',
        'topic_search': 'Per-match topic search across the full corpus. Each row is ONE hit (not aggregate) with the message date, source doc id, the matched phrase, and surrounding context.',
        'verbatim_quote': 'Phrases reporters have publicly quoted from the EFTA release. Blockquotes are verbatim from our local copy.',
        'press_recreate': 'Counts of journalist-cited names and phrases across the full corpus. Each row links to its source (which article first surfaced the quote/name).',
        'codeword_top': 'Documented Epstein-network code language from court filings + journalist reporting. Counts the literal terms across the corpus.',
        'doc_dates_year': 'Each document gets ONE canonical date via a priority chain: email Sent header > police report date > grand-jury heading > first body date > bare-year fallback. Bar shows how many documents are from each year.',
        'mention_dates_year': 'Every date-shaped string in document bodies. Per-doc deduplicated, then bucketed by year. Shows what years the corpus REFERS TO (vs. what years it was WRITTEN IN, see "Doc dates" page).',
        'cooccur_pairs': 'Pairs of named entities (SURNAME/GIVENNAME tagged by the PII model) that appear in the same document. High co-occurrence implies real connection.',
        'email_threads': 'Emails grouped by normalized Subject: line (Re:/Fwd: stripped). Each row is a distinct conversation thread, sorted by message count.',
        'imessages': 'Forensic iMessage exports from Epstein\'s Mac (captured day-of-arrest July 6 2019). JE = sender Jeffrey Epstein. ◼ = REDACTED counterpart. Yellow row = JE sent; red = counterpart sent. Chronological.',
        'tfidf': 'TF-IDF ranks phrases that are RARE globally but CONCENTRATED in a few documents — the opposite of doc-spread n-grams. Surfaces what specific docs are uniquely about.',
        'ngram': 'Most repeated word sequences across the corpus, sorted by how many documents contain the phrase (not raw count). Boilerplate phrases are filtered.',
        'label_top': 'Per-label PII rankings from the PII model (Piiranha-v1). NOTE: NER is currently partial (~7% of corpus tagged) — counts will grow as it completes.',
        'ds10_financial': 'Regex-based extractors run on DS10 (the JPM Private Bank + Deutsche Bank correspondence dossier, 158K docs). Captures counterparty entities, JPM banker names, money amounts, beneficiaries, and account holders.',
    }
    for p in pages:
        if 'explainer' not in p:
            p['explainer'] = EXPLAINERS.get(p['kind'], '')

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
