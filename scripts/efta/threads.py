#!/usr/bin/env python3
"""
Email-thread reconstruction from corpus.jsonl.

For docs that look like emails (have To:/From:/Sent:/Subject:), normalize the
subject (drop Re:/Fwd: prefixes, brackets, whitespace) and group by it. Order
each thread by Sent: date. Top-K longest threads emitted.

Output:
  - threads.json
  - threads.csv (one row per thread)

Usage:
    threads.py <corpus.jsonl> <out_dir> [--top 50]
"""
import sys, json, re, csv, argparse
from pathlib import Path
from collections import defaultdict
from datetime import datetime

SUBJECT_PREFIX_RE = re.compile(r'^\s*(re\s*[:\-]\s*|fwd?\s*[:\-]\s*|fw\s*[:\-]\s*)+', re.IGNORECASE)
BRACKETS_RE = re.compile(r'\[[^\]]*\]')


def normalize_subject(s: str) -> str:
    s = s.strip()
    # Strip repeated Re:/Fwd: prefixes
    prev = None
    while prev != s:
        prev = s
        s = SUBJECT_PREFIX_RE.sub('', s).strip()
    s = BRACKETS_RE.sub('', s).strip()
    s = re.sub(r'\s+', ' ', s)
    return s.lower()


def parse_email_headers(text: str):
    """Pull To/From/Sent/Subject from a doc. Handles both layouts:
       (a) labels and values inline ("To: foo@bar"),
       (b) labels stacked then values stacked.
    """
    lines = text.splitlines()
    # Try inline form first
    headers = {}
    for line in lines[:30]:
        m = re.match(r'^\s*(To|From|Sent|Subject)\s*[:.]?\s*(.+?)\s*$', line, re.IGNORECASE)
        if m and len(m.group(2).strip()) > 1:
            k = m.group(1).capitalize()
            headers.setdefault(k, m.group(2).strip())
    if 'Subject' in headers and ('From' in headers or 'To' in headers):
        return headers
    # Fall back to stacked form
    label_order = []
    i = 0
    while i < len(lines[:30]):
        m = re.match(r'^\s*(To|From|Sent|Subject|Cc|Bcc)\s*[:.]?\s*$', lines[i].strip(), re.IGNORECASE)
        if m:
            label_order.append(m.group(1).capitalize())
            i += 1
        else:
            if label_order: break
            i += 1
    if label_order:
        vals = []
        j = (i if i else 0)
        while j < len(lines) and len(vals) < len(label_order):
            if lines[j].strip():
                vals.append(lines[j].strip())
            j += 1
        if len(vals) >= len(label_order):
            return {lbl: val for lbl, val in zip(label_order, vals)}
    return None


SENT_PATTERNS = [
    re.compile(r'(\w{3,9})\s+(\d{1,2})/(\d{1,2})/(\d{4})\s+(\d{1,2}):(\d{2}):(\d{2})\s*([AP]M)?', re.IGNORECASE),
    re.compile(r'(\d{1,2})/(\d{1,2})/(\d{4})\s+(\d{1,2}):(\d{2})', re.IGNORECASE),
    re.compile(r'(\d{4})-(\d{2})-(\d{2})\s+(\d{1,2}):(\d{2})'),
]


def parse_sent(s: str):
    """Return (datetime, raw) or (None, raw)."""
    if not s: return None, s
    for rx in SENT_PATTERNS:
        m = rx.search(s)
        if m:
            try:
                g = m.groups()
                # weekday MM/DD/YYYY h:mm:ss AM/PM
                if len(g) >= 7 and g[0] and not g[0].isdigit():
                    mo, d, y, hh, mm, ss = int(g[1]), int(g[2]), int(g[3]), int(g[4]), int(g[5]), int(g[6] or 0)
                    if g[7] and g[7].upper() == 'PM' and hh < 12: hh += 12
                    if g[7] and g[7].upper() == 'AM' and hh == 12: hh = 0
                    return datetime(y, mo, d, hh, mm, ss), s
                # MM/DD/YYYY h:mm
                if len(g) >= 5 and g[0].isdigit():
                    mo, d, y, hh, mm = int(g[0]), int(g[1]), int(g[2]), int(g[3]), int(g[4])
                    return datetime(y, mo, d, hh, mm), s
                # YYYY-MM-DD h:mm
                if len(g) >= 5:
                    y, mo, d, hh, mm = int(g[0]), int(g[1]), int(g[2]), int(g[3]), int(g[4])
                    return datetime(y, mo, d, hh, mm), s
            except ValueError:
                pass
    return None, s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('corpus_jsonl')
    ap.add_argument('out_dir')
    ap.add_argument('--top', type=int, default=50)
    args = ap.parse_args()
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)

    threads = defaultdict(list)  # normalized_subject -> list of msg dicts

    n_emails = 0
    with open(args.corpus_jsonl) as f:
        for line in f:
            try: rec = json.loads(line)
            except: continue
            text = rec.get('text', '')
            if 'Subject' not in text and 'subject' not in text[:200].lower():
                continue
            h = parse_email_headers(text)
            if not h or 'Subject' not in h:
                continue
            n_emails += 1
            subj = h.get('Subject', '')
            norm = normalize_subject(subj)
            if not norm or len(norm) < 4:
                continue
            dt, raw_sent = parse_sent(h.get('Sent', ''))
            threads[norm].append({
                'id': rec.get('id', '?'),
                'dataset': rec.get('dataset', '?'),
                'subject': subj,
                'from': h.get('From', ''),
                'to': h.get('To', ''),
                'sent_raw': raw_sent,
                'sent_iso': dt.isoformat() if dt else None,
                'sent_ts': dt.timestamp() if dt else None,
            })

    # Sort each thread by sent date
    for subj, msgs in threads.items():
        msgs.sort(key=lambda m: (m['sent_ts'] is None, m['sent_ts'] or 0))

    # Rank threads by length
    ranked = sorted(threads.items(), key=lambda kv: -len(kv[1]))

    out_data = {
        'n_emails_scanned': n_emails,
        'n_threads': len(threads),
        'top_threads': [
            {
                'subject_normalized': subj,
                'subject_sample': msgs[0]['subject'],
                'n_messages': len(msgs),
                'first_sent': msgs[0].get('sent_iso'),
                'last_sent': msgs[-1].get('sent_iso'),
                'participants_from': sorted({m['from'] for m in msgs if m['from']})[:5],
                'sample_ids': [m['id'] for m in msgs[:5]],
                'datasets': sorted({m['dataset'] for m in msgs}),
            }
            for subj, msgs in ranked[:args.top]
        ],
    }
    with open(out / 'threads.json', 'w', encoding='utf-8') as f:
        json.dump(out_data, f, indent=2, ensure_ascii=False)
    with open(out / 'threads.csv', 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['rank','n_messages','subject','first_sent','last_sent','from_sample','sample_ids'])
        for i, (subj, msgs) in enumerate(ranked, 1):
            from_first = msgs[0]['from'][:40] if msgs[0]['from'] else ''
            w.writerow([i, len(msgs), subj, msgs[0].get('sent_iso',''), msgs[-1].get('sent_iso',''),
                        from_first, '|'.join(m['id'] for m in msgs[:5])])

    print(f'emails scanned: {n_emails}  distinct threads: {len(threads)}')
    print(f'\nTop 20 longest threads:')
    for subj, msgs in ranked[:20]:
        first = msgs[0].get('sent_iso','?')[:10] if msgs[0].get('sent_iso') else '?'
        last  = msgs[-1].get('sent_iso','?')[:10] if msgs[-1].get('sent_iso') else '?'
        print(f'  {len(msgs):>3} msgs  [{first} → {last}]  {subj[:60]}')


if __name__ == '__main__':
    main()
