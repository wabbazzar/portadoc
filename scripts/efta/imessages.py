#!/usr/bin/env python3
"""
Extract Epstein's iMessage records from forensic dump docs in the Estate.

These docs (HOUSE_OVERSIGHT_025408..027794) are forensic exports from
Epstein's Mac, captured day-of-arrest July 6 2019. Each record:

  Source Entry: H\\Macintosh HD\\root\\Users\\jee\\Library\\Messages\\...
  Sender: e:jeeitunes@gmail.com
  Time: 06/14/19 01:26:08 PM (582236768)
  Flags: ...
  Is Read: No
  Message: <message body>

This parses them and emits chronological JSON + plain log.

Usage:
    imessages.py <corpus.jsonl> <out_dir>
"""
import sys, json, re, argparse, csv
from pathlib import Path
from datetime import datetime
from collections import Counter

# Per-record block separated by "Time: …" followed by "Message: ..."
TIME_RE = re.compile(
    r'Sender:\s*([^\n\r]*?)\n\s*Time:\s*(\d{1,2}/\d{1,2}/\d{2,4}\s+\d{1,2}:\d{2}(?::\d{2})?\s*(?:AM|PM)?)[^\n]*\n[^M]*?Message:\s*(.+?)(?=\n\s*Sender:|\n\s*Source\s|$)',
    re.IGNORECASE | re.DOTALL,
)


def parse_timestamp(ts: str):
    ts = ts.strip()
    for fmt in ('%m/%d/%y %I:%M:%S %p', '%m/%d/%y %I:%M %p', '%m/%d/%Y %I:%M:%S %p', '%m/%d/%Y %I:%M %p',
                '%m/%d/%y %H:%M:%S', '%m/%d/%y %H:%M', '%m/%d/%Y %H:%M:%S', '%m/%d/%Y %H:%M'):
        try:
            return datetime.strptime(ts, fmt)
        except ValueError:
            continue
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('corpus_jsonl')
    ap.add_argument('out_dir')
    args = ap.parse_args()
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)

    messages = []
    seen_docs = set()

    with open(args.corpus_jsonl) as f:
        for line in f:
            try: rec = json.loads(line)
            except: continue
            text = rec.get('text', '')
            if 'Presentity IDs' not in text and 'Last Message ID' not in text:
                continue
            seen_docs.add(rec['id'])
            for m in TIME_RE.finditer(text):
                sender_raw = m.group(1).strip()
                ts_raw = m.group(2).strip()
                msg = m.group(3).strip()
                msg = re.sub(r'\s+', ' ', msg)
                if len(msg) < 1: continue
                # Truncate runaway captures
                if len(msg) > 600: msg = msg[:600] + '…'
                dt = parse_timestamp(ts_raw)
                is_epstein = 'jeeitunes' in sender_raw.lower() or 'jee' in sender_raw.lower()
                messages.append({
                    'doc_id': rec['id'],
                    'timestamp': dt.isoformat() if dt else None,
                    'epoch': dt.timestamp() if dt else None,
                    'sender': 'epstein' if is_epstein else 'counterpart',
                    'sender_raw': sender_raw,
                    'message': msg,
                })

    # Sort chronologically (None goes to end)
    messages.sort(key=lambda m: (m['epoch'] is None, m['epoch'] or 0))

    # Stats
    by_year = Counter()
    by_sender = Counter()
    n_with_url = 0
    keywords = Counter()
    INTERESTING = ['ranch','island','paris','harvard','epstein','trump','darren','indyke',
                   'china','extradition','schedule','meet','tomorrow','today','flight',
                   'plane','call','dinner','attorney','lawyer']
    for m in messages:
        if m['timestamp']:
            by_year[m['timestamp'][:4]] += 1
        by_sender[m['sender']] += 1
        if 'http' in m['message']:
            n_with_url += 1
        m_lc = m['message'].lower()
        for kw in INTERESTING:
            if re.search(rf'\b{re.escape(kw)}\b', m_lc):
                keywords[kw] += 1

    out_data = {
        'n_source_docs': len(seen_docs),
        'n_messages': len(messages),
        'by_year': dict(sorted(by_year.items())),
        'by_sender': dict(by_sender),
        'n_with_url': n_with_url,
        'keyword_hits': dict(keywords.most_common(20)),
        'messages': messages,  # full chronological list
    }
    with open(out / 'imessages.json', 'w', encoding='utf-8') as f:
        json.dump(out_data, f, indent=2, ensure_ascii=False)
    with open(out / 'imessages.csv', 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=['timestamp','sender','message','doc_id'])
        w.writeheader()
        for m in messages:
            w.writerow({'timestamp': m['timestamp'] or '',
                        'sender': m['sender'],
                        'message': m['message'],
                        'doc_id': m['doc_id']})

    print(f'source docs: {len(seen_docs)}')
    print(f'messages parsed: {len(messages)}')
    print(f'  by sender: {dict(by_sender)}')
    print(f'  by year: {dict(sorted(by_year.items()))}')
    print(f'  with URLs: {n_with_url}')
    print(f'\\ninteresting keyword hits:')
    for kw, c in keywords.most_common(15):
        print(f'  {kw:>10}: {c}')
    print(f'\\nfirst 5 messages chronologically:')
    for m in messages[:5]:
        print(f'  [{m["timestamp"] or "?"}] {m["sender"]:>11}: {m["message"][:80]}')


if __name__ == '__main__':
    main()
