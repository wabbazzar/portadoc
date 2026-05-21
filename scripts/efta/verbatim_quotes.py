#!/usr/bin/env python3
"""
Extract verbatim contexts for journalist-cited phrases from the corpus.

For each phrase reporters have quoted, return the first 3 actual hits with
~150-char surrounding context + doc id + dataset. Lets the website show the
underlying quote, not just a count.

Output: quotes.json with shape
  { "quotes": [
      {"phrase": "...", "n_total_hits": N, "n_docs": M,
       "samples": [{"doc_id":..., "dataset":..., "context":...}, ...]}
  ] }

Usage:
    verbatim_quotes.py <corpus.jsonl> <out_dir>
"""
import sys, json, re, argparse
from pathlib import Path

# Phrases to chase verbatim. Each: (phrase, fuzzy_regex_or_None, source_note).
# fuzzy_regex_or_None lets us match OCR'd variants when the exact phrase fails.
PHRASES = [
    ("Snow White",
     r"snow\s*white",
     "Jul 9 2010 Jes Staley → Epstein (Bloomberg)"),
    ("Beauty and the Beast",
     r"beauty\s+and\s+the\s+beast",
     "Same Staley thread / also victim's description of Epstein's library"),
    ("highly valued friend",
     r"highly\s+valued\s+friend",
     "Noam Chomsky LOR (NPR Nov 20 2025)"),
    ("Invisible Man",
     r"invisible\s+man",
     "Aug 18 2001 Maxwell contact label for 'A' / Andrew (CNN/ITV Dec 2025)"),
    ("Mohammed bin Salman",
     r"mohammed\s+bin\s+salman",
     "2016 House Oversight estate emails (CBS Nov 12 2025)"),
    ("dog that hasn't barked",
     r"dog\s+that\s+hasn'?t\s+barked",
     "Apr 2011 Epstein → Maxwell re Trump (PBS, CBS, Senator Reed)"),
    ("knew about the girls",
     r"knew\s+about\s+the\s+girls",
     "Jan 31 2019 Epstein → Wolff (CBS Nov 12 2025)"),
    ("let him hang himself",
     r"let\s+him\s+hang\s+himself",
     "Dec 2015 Wolff → Epstein on Trump CNN interview (PBS)"),
    ("dirty donald",
     r"dirty\s+donald",
     "Epstein → Ruemmler (CNBC)"),
    ("Ruemmler",
     r"ruemmler",
     "Kathryn Ruemmler — Goldman CLO (CNBC)"),
    ("Operation Leap Year",
     r"operation\s+leap\s+year",
     "Federal Grand Jury 07-103, West Palm Beach, May 8 2007 (DS7)"),
    ("Free State Reporting",
     r"free\s+state\s+reporting",
     "Court-reporting company that transcribed SDNY Maxwell grand jury"),
    ("Centurion",
     r"centurion",
     "AmEx black-card concierge (CBS, Bloomberg May 2026)"),
    ("Little Saint James",
     r"little\s+s(?:aint|t\.?)\s+james|\bLSJ\b",
     "Epstein's USVI island"),
    ("Zorro Ranch",
     r"zorro\s+ranch",
     "Epstein's New Mexico ranch"),
    ("El Brillo",
     r"el\s+brillo",
     "358 El Brillo Way, Palm Beach mansion"),
    ("Darren awol",
     r"darren\s+awol",
     "From Epstein's iMessages, Jun-Jul 2019 — 'Darren' = Darren Indyke, his attorney"),
    ("Ranch island. Paris. Harvard",
     r"ranch\s+island.{0,3}paris.{0,3}harvard",
     "Epstein iMessage to redacted counterpart, Jun 15 2019 — travel schedule request"),
    ("Chinese blink on extradition",
     r"chinese\s+blink\s+on\s+extradition",
     "Epstein iMessage Jun 15 2019"),
    ("Aka MBS",
     r"aka\s+MBS",
     "House Oversight 019874 — Mohammed bin Salman decode"),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('corpus_jsonl')
    ap.add_argument('out_dir')
    ap.add_argument('--ctx-chars', type=int, default=160)
    ap.add_argument('--max-samples', type=int, default=3)
    args = ap.parse_args()
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)

    accum = {}
    for phrase, fuzzy, note in PHRASES:
        accum[phrase] = {
            'phrase': phrase,
            'pattern': fuzzy,
            'source': note,
            'n_total_hits': 0,
            'doc_ids': set(),
            'samples': [],
        }

    compiled = [(phrase, re.compile(fuzzy, re.IGNORECASE)) for phrase, fuzzy, _ in PHRASES]

    with open(args.corpus_jsonl) as f:
        for line in f:
            try: rec = json.loads(line)
            except: continue
            text = rec.get('text', '')
            if not text: continue
            ds = rec.get('dataset', '?')
            doc_id = rec.get('id', '?')
            for phrase, rx in compiled:
                slot = accum[phrase]
                for m in rx.finditer(text):
                    slot['n_total_hits'] += 1
                    slot['doc_ids'].add(doc_id)
                    if len(slot['samples']) < args.max_samples:
                        start = max(0, m.start() - args.ctx_chars)
                        end = min(len(text), m.end() + args.ctx_chars)
                        ctx = text[start:end]
                        # Light cleanup
                        ctx = re.sub(r'\s+', ' ', ctx).strip()
                        slot['samples'].append({
                            'doc_id': doc_id,
                            'dataset': ds,
                            'context': ctx,
                            'matched': text[m.start():m.end()],
                        })

    out_data = {
        'phrases': [
            {
                'phrase': v['phrase'],
                'source': v['source'],
                'n_total_hits': v['n_total_hits'],
                'n_docs': len(v['doc_ids']),
                'samples': v['samples'],
            }
            for k, v in accum.items()
        ],
    }
    with open(out / 'quotes.json', 'w', encoding='utf-8') as f:
        json.dump(out_data, f, indent=2, ensure_ascii=False)

    print(f'phrases evaluated: {len(PHRASES)}')
    for p in out_data['phrases']:
        sym = '✓' if p['n_total_hits'] else '✗'
        print(f"  {sym} {p['n_total_hits']:>4} hits in {p['n_docs']:>3} docs  :: {p['phrase']}")


if __name__ == '__main__':
    main()
