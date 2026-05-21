#!/usr/bin/env python3
"""
DS10-specific extractors for the financial dossier (JPMorgan Private Bank
internal correspondence + wire transfer requests + AmEx statements).

N-grams + generic NER are the wrong shape for this surface — DS10 is
heavily-templated financial paperwork. Instead we regex for:

  - counterparty entities (LLC / Inc / Corp / LP / Trust / Foundation / NA)
  - dollar amounts (with denomination + 'amount of' context)
  - JPM banker names (Lastname, Firstname-letter pattern in From/To/Cc)
  - external bank names (Deutsche Bank, Wells Fargo, etc.)
  - explicit beneficiary / payer / payee tokens
  - power-of-attorney / signatory grantees
  - account-type tokens (DDA, CD, LOC, ACH, Domestic Wire, etc.)

Output:
  ds10_financial.json with per-category ranked tables + per-year transaction count

Usage:
    ds10_financial.py <corpus.jsonl> <out_dir>
"""
import sys, json, re, csv, argparse
from pathlib import Path
from collections import Counter, defaultdict


# Money: $1,234,567.89  |  $1.2 million  |  1,234.56 (only if preceded by amount cue)
MONEY_RE = re.compile(
    r'(?:\$\s?[\d]{1,3}(?:,\d{3})*(?:\.\d{2})?(?:\s?(?:million|billion|K|M|B))?'
    r'|(?:amount[\s\-]of|sum[\s\-]of|total\s+fare[\s\-]+(?:USD|usd)|USD)\s*\$?\s?[\d]{1,3}(?:,\d{3})*(?:\.\d{2})?)',
)

# Counterparty entities — capture an entity expression ending in a corp suffix.
# Be conservative: ≥3 char preceding word, capture surrounding 1-4 capitalized tokens.
ENTITY_RE = re.compile(
    r'\b((?:[A-Z][A-Za-z&\.\-]{2,}(?:\s+[A-Z][A-Za-z&\.\-]{2,}){0,4})\s+'
    r'(?:LLC|L\.L\.C\.|Inc\.?|Corp\.?|Corporation|LP|L\.P\.|LLP|N\.A\.|NA|Trust|Foundation|Fund|Partners|Holdings|Holding|Capital|Group|Ltd\.?))\b'
)

# Banker name pattern: "Lastname, Firstname [Middle Initial]" — common in JPM correspondence
BANKER_RE = re.compile(
    r'\b([A-Z][a-z]{2,}),\s+([A-Z][a-z]{2,})(?:\s+([A-Z])\.?)?\b'
)

# External bank names — well-known list, case-insensitive
KNOWN_BANKS = [
    'Deutsche Bank', 'Deutsche Bank Trust', 'Wells Fargo', 'Bank of America', 'BofA',
    'Citibank', 'Citi', 'HSBC', 'Barclays', 'UBS', 'Credit Suisse', 'Goldman Sachs',
    'Morgan Stanley', 'JPMorgan', 'JP Morgan', 'JPM Chase', 'JPMC', 'Chase',
    'BNP Paribas', 'Societe Generale', 'Standard Chartered', 'Royal Bank',
    'Bank of NY Mellon', 'BNY Mellon', 'State Street', 'PNC', 'Capital One',
    'American Express', 'AmEx', 'Centurion', 'Apollo', 'Highbridge',
    'BankUnited', 'M&T Bank', 'TD Bank',
]
BANK_RE = re.compile(r'\b(' + '|'.join(re.escape(b) for b in KNOWN_BANKS) + r')\b', re.IGNORECASE)

# Account-type tokens
ACCT_TYPES = [
    'DDA', 'CD', 'IRA', 'Letter of Credit', 'LOC', 'ACH', 'Domestic Wire',
    'International Wire', 'Wire Transfer', 'Power of Attorney', 'POA', 'POAs',
    'Trustee', 'Custody', 'Brokerage', 'Money Market', 'Money Mkt',
    'Funds Transfer Request', 'Beneficiary', 'Pay To', 'Pay to:',
    'Confidential Treatment Request',
]
ACCT_RE = re.compile(r'(?<![A-Z])(' + '|'.join(re.escape(a) for a in ACCT_TYPES) + r')(?![A-Z])', re.IGNORECASE)

# Beneficiary / payee labels — capture the value on same line
BENEFICIARY_RE = re.compile(
    r'(?:Beneficiary\s*Name|Beneficiary|Pay\s*[Tt]o|Payee)\s*[:.]?\s*'
    r'([A-Z][A-Za-z0-9&\.\,\-\s]{2,60}?)(?:\n|\s{4,}|$)'
)

# Account holder mentions
ACCOUNT_HOLDER_RE = re.compile(
    r'(?:Account\s+(?:[Nn]ame|[Hh]older)|Account\s+for|on\s+behalf\s+of)\s*[:.]?\s*'
    r'([A-Z][A-Za-z0-9&\.\,\-\s]{2,60}?)(?:\n|\s{4,}|$)'
)

# Power-of-attorney grantees — "granted ... authority to <NAME>"
POA_RE = re.compile(
    r'(?:granted\s+(?:full\s+)?authority\s+to|Power\s+of\s+Attorney\s+for|signatory\s+authority\s+to)\s+'
    r'([A-Z][A-Za-z\.\s,]{4,80}?)(?:\.|,\s*and|\s*$|\n)',
    re.IGNORECASE
)

# Year extraction (for transaction-date histogram)
YEAR_RE = re.compile(r'\b(19[5-9][0-9]|20[0-2][0-9])\b')

# Tokens that, when they appear as the "Firstname" half of Lastname,Firstname,
# mean the match is a company/place suffix, not a person.
BANKER_FIRSTNAME_DENYLIST = {
    'New', 'Inc', 'Corp', 'Llc', 'Ltd', 'Lp', 'Llp', 'Na', 'Co', 'Of',
    'And', 'The', 'York', 'Beach', 'James', 'Andrew', 'Saint', 'Mary',
    'County', 'States', 'America', 'United', 'International', 'National',
    'Bank', 'Trust', 'Partners', 'Holdings', 'Capital', 'Group', 'Fund',
    'Securities', 'Insurance', 'Financial', 'Investments', 'Services',
    'Avenue', 'Street', 'Boulevard', 'Drive', 'Road', 'Lane',
    'Limited', 'Company', 'Plc',
}

# Tokens that, as the Lastname half, are clearly not names.
BANKER_LASTNAME_DENYLIST = {
    'Inc', 'Corp', 'Llc', 'Co', 'Ltd', 'Lp', 'Llp', 'Na', 'Trust',
    'Avenue', 'Street', 'Road', 'Lane', 'Drive', 'Court',
    'York', 'Beach', 'County', 'States', 'America',
    'And', 'The', 'Of', 'For', 'With', 'From', 'Subject',
    'Fund', 'Group', 'Holdings', 'Partners', 'Bank', 'Company', 'Services',
    'Date', 'Time', 'Page', 'Note', 'Attn', 'Re',
}


def extract(text: str):
    """Run all regexes on a doc, return dict of {category: list_of_strings}."""
    out = defaultdict(list)
    for m in MONEY_RE.finditer(text):
        v = m.group(0).strip()
        # Drop trivial / zero amounts — $0, $1, $0.00, $1.00 etc. are noise from
        # financial-doc templates (balance fields with zero values).
        if re.match(r'^\$?[01](?:\.0+)?$', v): continue
        if v in ('$0.00','$0','$1.00','$1','$0.0','$1.0'): continue
        out['money'].append(v)
    for m in ENTITY_RE.finditer(text):
        ent = re.sub(r'\s+', ' ', m.group(1).strip())
        # Avoid swallowing extremely long captures
        if 4 <= len(ent) <= 80:
            out['entity'].append(ent)
    for m in BANKER_RE.finditer(text):
        last, first, mi = m.group(1), m.group(2), m.group(3)
        if last in BANKER_LASTNAME_DENYLIST: continue
        if first in BANKER_FIRSTNAME_DENYLIST: continue
        full = f'{last}, {first}'
        if mi:
            full += f' {mi}'
        out['banker'].append(full)
    for m in BANK_RE.finditer(text):
        # Normalize to canonical form
        canon = m.group(1)
        if canon.lower() in {'jpm chase', 'jpmorgan chase'}: canon = 'JPMorgan Chase'
        out['bank'].append(canon)
    for m in ACCT_RE.finditer(text):
        out['acct_type'].append(m.group(1))
    for m in BENEFICIARY_RE.finditer(text):
        val = re.sub(r'\s+', ' ', m.group(1)).strip(' :,-.')
        if 4 <= len(val) <= 80:
            out['beneficiary'].append(val)
    for m in ACCOUNT_HOLDER_RE.finditer(text):
        val = re.sub(r'\s+', ' ', m.group(1)).strip(' :,-.')
        if 4 <= len(val) <= 80:
            out['account_holder'].append(val)
    for m in POA_RE.finditer(text):
        val = re.sub(r'\s+', ' ', m.group(1)).strip(' :,-.')
        # Filter boilerplate POA legalese ("a specific or limited purpose" etc.)
        if len(val) < 6 or len(val) > 120: continue
        if val.lower().startswith(('a ', 'an ', 'the ', 'us ', 'see ', 'lori ', 'itr')):
            continue
        if not re.search(r'[A-Z][a-z]', val):  # must contain a proper-noun-looking word
            continue
        out['poa_grantee'].append(val)
    # Year mentions
    for m in YEAR_RE.finditer(text):
        y = int(m.group(1))
        if 1990 <= y <= 2026:
            out['year'].append(y)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('corpus_jsonl')
    ap.add_argument('out_dir')
    ap.add_argument('--datasets', default='dataset_10',
                    help='comma-separated list of dataset names to analyze (default: dataset_10)')
    ap.add_argument('--top', type=int, default=100)
    args = ap.parse_args()
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    target_ds = set(args.datasets.split(','))

    # category -> {value: {count, docs: set}}
    cats = defaultdict(lambda: defaultdict(lambda: {'count': 0, 'docs': set()}))
    year_counts = Counter()
    n_docs = 0
    with open(args.corpus_jsonl) as f:
        for line in f:
            try: rec = json.loads(line)
            except: continue
            if rec.get('dataset') not in target_ds: continue
            n_docs += 1
            text = rec.get('text', '')
            if not text: continue
            ext = extract(text)
            for cat, vals in ext.items():
                if cat == 'year':
                    for y in vals:
                        year_counts[y] += 1
                    continue
                seen_in_doc = set()
                for v in vals:
                    cats[cat][v]['count'] += 1
                    seen_in_doc.add(v)
                for v in seen_in_doc:
                    cats[cat][v]['docs'].add(rec['id'])

    # Serialize
    result = {
        'n_documents_scanned': n_docs,
        'datasets': sorted(target_ds),
        'year_mentions': dict(sorted(year_counts.items())),
        'categories': {},
    }
    for cat, items in cats.items():
        ranked = sorted(
            items.items(),
            key=lambda kv: (-len(kv[1]['docs']), -kv[1]['count'], kv[0]),
        )
        result['categories'][cat] = {
            'unique_values': len(ranked),
            'top': [
                {
                    'text': v,
                    'count': info['count'],
                    'docs': len(info['docs']),
                    'sample_doc_ids': sorted(info['docs'])[:5],
                }
                for v, info in ranked[:args.top]
            ],
        }
        # Per-cat CSV
        with open(out / f'ds10_{cat}.csv', 'w', newline='', encoding='utf-8') as cf:
            w = csv.writer(cf)
            w.writerow(['rank','text','count','docs','sample_doc_ids'])
            for i, (v, info) in enumerate(ranked, 1):
                w.writerow([i, v, info['count'], len(info['docs']),
                            '|'.join(sorted(info['docs'])[:5])])

    with open(out / 'ds10_financial.json', 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    print(f'DS10 docs scanned: {n_docs}')
    print(f'Categories surfaced:')
    for cat, info in result['categories'].items():
        print(f"  {cat:>15}: {info['unique_values']:>5} unique")
        for r in info['top'][:6]:
            print(f"    {r['count']:>4} ({r['docs']:>3} docs)  {r['text'][:60]}")


if __name__ == '__main__':
    main()
