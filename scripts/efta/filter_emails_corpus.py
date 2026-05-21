#!/usr/bin/env python3
"""
Emit a subset of corpus.jsonl containing only email-shaped or iMessage-shaped
documents — for n-gram / TF-IDF analyses where court-document boilerplate
would dominate.

A doc is kept if its first ~600 chars contains either:
  - Email headers: From: / To: / Sent: / Subject: pattern (with both From AND
    Subject present, to avoid false matches like a single 'To:' field on a form)
  - iMessage forensic export markers: "Last Message ID" or "Presentity IDs"

This catches:
  - DS11 (operational 2017 emails)
  - DS12 (mixed emails + scans)
  - emails/ (parsed)
  - estate bundled docs that ARE emails
  - DS8 docs that are emails (subset of the 8993)
  - The 25 iMessage forensic dump docs

And excludes:
  - DS1/2/5 (photos)
  - DS3/4/6/7 (notes, police reports, grand jury transcripts)
  - DS10 (financial statements, wire-transfer forms)
  - non-email portions of estate / dems / DS8

Usage:
    filter_emails_corpus.py <corpus.jsonl> <out.jsonl>
"""
import sys, json, re
from collections import Counter

EMAIL_FROM = re.compile(r'(?im)^\s*From\s*[:.]')
EMAIL_SUBJ = re.compile(r'(?im)^\s*Subject\s*[:.]')
EMAIL_SENT = re.compile(r'(?im)^\s*Sent\s*[:.]')
EMAIL_TO   = re.compile(r'(?im)^\s*To\s*[:.]')
IMSG       = re.compile(r'(?:Last Message ID|Presentity IDs)')


def is_email_like(text: str) -> str | None:
    """Return 'email' / 'imessage' / None depending on shape of first 800 chars."""
    if not text: return None
    head = text[:800]
    if IMSG.search(head):
        return 'imessage'
    has_from = bool(EMAIL_FROM.search(head))
    has_subj = bool(EMAIL_SUBJ.search(head))
    has_sent = bool(EMAIL_SENT.search(head))
    has_to   = bool(EMAIL_TO.search(head))
    # Need at least 2 of: From, Subject, Sent (To-only is too loose)
    score = sum([has_from, has_subj, has_sent])
    if score >= 2: return 'email'
    if has_from and has_to: return 'email'
    return None


def main():
    if len(sys.argv) < 3:
        print(__doc__); sys.exit(2)
    in_path, out_path = sys.argv[1], sys.argv[2]

    # Personal correspondence + iMessage forensic dumps only.
    # Exclude DS10 (internal JPM-bank correspondence — still boilerplate)
    # and the court-document datasets entirely.
    KEEP_DATASETS = {'dataset_11','dataset_12','emails','estate','dems'}

    n_in = 0
    kept_by_ds = Counter()
    kept_by_kind = Counter()
    with open(in_path) as f, open(out_path, 'w') as g:
        for line in f:
            n_in += 1
            try: rec = json.loads(line)
            except: continue
            if rec.get('dataset') not in KEEP_DATASETS: continue
            kind = is_email_like(rec.get('text',''))
            if not kind: continue
            rec['email_shape'] = kind
            g.write(json.dumps(rec, ensure_ascii=False) + '\n')
            ds = rec.get('dataset','unknown')
            kept_by_ds[ds] += 1
            kept_by_kind[kind] += 1

    total = sum(kept_by_ds.values())
    print(f'in: {n_in}  kept: {total}  ({100*total/n_in:.1f}%)')
    print(f'  by shape: email={kept_by_kind["email"]}  imessage={kept_by_kind["imessage"]}')
    print(f'  by dataset:')
    for ds, c in sorted(kept_by_ds.items(), key=lambda kv: -kv[1]):
        print(f'    {ds}: {c}')


if __name__ == '__main__':
    main()
