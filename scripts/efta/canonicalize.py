"""
OCR-aware canonicalization for EFTA PII entities.

Two passes:
  1. EMAIL: normalize OCR-broken @/©/A/a; cluster by edit-distance on local-part
            with same domain. Pick the highest-count form as canonical.
  2. PERSON (GIVENNAME, SURNAME): cluster by edit-distance ≤1 within same soundex.

Used by rank_entities.py (imported, not invoked standalone).
"""
import re
from collections import Counter, defaultdict

import Levenshtein
import jellyfish

# (re-imported above is fine — keep import re at module top for the regex use below)

# Common OCR character confusions, asymmetric: key is what OCR produced,
# value is the likely intended character. Apply both directions during clustering.
OCR_SUBS = [
    ('0', 'o'), ('o', '0'),
    ('1', 'l'), ('l', '1'), ('1', 'i'), ('i', '1'),
    ('5', 's'), ('s', '5'),
    ('8', 'b'), ('b', '8'),
    ('rn', 'm'), ('m', 'rn'),
    ('cl', 'd'), ('d', 'cl'),
]


def normalize_email(text: str) -> str:
    """Pre-cluster email cleanup: collapse OCR @-substitutes inside the
    domain-separator position to '@', lowercase, strip whitespace."""
    t = text.strip().lower()
    # Replace common OCR @-substitutes when they're followed by a typical email-domain pattern
    t = re.sub(r'([a-z0-9._%+\-])(©|\(c\)|A|a|@)([a-z0-9\-]+\.[a-z]{2,})',
               r'\1@\3', t)
    # Tighten whitespace
    t = re.sub(r'\s+', '', t)
    return t


def cluster_emails(items: dict) -> dict:
    """items: {email_text: stats_dict}. Returns merged {canonical_email: merged_stats}.

    Same-domain emails with Levenshtein distance ≤2 on local-part cluster together.
    Canonical = the highest-count member of each cluster.
    """
    pre = {}
    for raw, stats in items.items():
        norm = normalize_email(raw)
        # Aggregate into pre-cluster bucket
        if norm not in pre:
            pre[norm] = _empty_stats()
        _merge_into(pre[norm], stats)

    by_domain = defaultdict(list)
    for addr in pre:
        if '@' in addr:
            local, dom = addr.rsplit('@', 1)
            by_domain[dom].append((addr, local))
        else:
            by_domain['__nodomain__'].append((addr, addr))

    out = {}
    for dom, members in by_domain.items():
        members.sort(key=lambda m: -pre[m[0]]['count'])
        assigned = {}
        for addr, local in members:
            placed = False
            for canon_addr, canon_local in list(assigned.keys()):
                if Levenshtein.distance(local, canon_local) <= 2 \
                   and abs(len(local) - len(canon_local)) <= 2:
                    assigned[(canon_addr, canon_local)].append(addr)
                    placed = True
                    break
            if not placed:
                assigned[(addr, local)] = [addr]
        for (canon_addr, _), members_list in assigned.items():
            merged = _empty_stats()
            variants = []
            for m in members_list:
                _merge_into(merged, pre[m])
                if m != canon_addr:
                    variants.append(m)
            merged['ocr_variants'] = sorted(variants)[:10]
            out[canon_addr] = merged
    return out


def cluster_persons(items: dict) -> dict:
    """items: {name_text: stats_dict}. Returns merged {canonical_name: merged_stats}.

    Clusters within the same soundex code by Levenshtein ≤1 (very tight to avoid
    false merges of unrelated short names).
    """
    # Hard filter: drop single-token noise and obvious non-names.
    NAME_NOISE = {
        'Ep','Mr','Ms','Mrs','Dr','Sir','Re','Fwd','Fw','Cc','Bcc',
        'And','The','For','But','Not','Yes','No','Of','To','From','By','In','On','At',
        'Pm','Am','Re:','Inc','Llc','Corp','Llp','Lp','Co','Na',
        'Ai','Ar','Sk','Nm','Lg','Gm','Je','Jlb','Lw','Di','Rk',
    }
    by_sx = defaultdict(list)
    for name in items:
        if len(name) < 4: continue          # bump from 3 to 4
        if name in NAME_NOISE: continue
        if not re.search(r'[A-Za-z]', name): continue  # no letters → garbage
        sx = jellyfish.soundex(name)
        by_sx[sx].append(name)
    # Items unchanged for short names → emit as-is later
    out = {}
    seen = set()
    for sx, members in by_sx.items():
        members.sort(key=lambda n: -items[n]['count'])
        assigned = {}
        for name in members:
            placed = False
            for canon in list(assigned.keys()):
                # Tighter: distance ≤ 1, length difference ≤ 2
                if Levenshtein.distance(name.lower(), canon.lower()) <= 1 \
                   and abs(len(name) - len(canon)) <= 2:
                    assigned[canon].append(name)
                    placed = True
                    break
            if not placed:
                assigned[name] = [name]
        for canon, members_list in assigned.items():
            merged = _empty_stats()
            variants = []
            for m in members_list:
                _merge_into(merged, items[m])
                if m != canon:
                    variants.append(m)
                seen.add(m)
            merged['ocr_variants'] = sorted(variants)[:10]
            out[canon] = merged
    # Add unclustered names — but apply the same noise filter
    NAME_NOISE = {
        'Ep','Mr','Ms','Mrs','Dr','Sir','Re','Fwd','Fw','Cc','Bcc',
        'And','The','For','But','Not','Yes','No','Of','To','From','By','In','On','At',
        'Pm','Am','Re:','Inc','Llc','Corp','Llp','Lp','Co','Na',
        'Ai','Ar','Sk','Nm','Lg','Gm','Je','Jlb','Lw','Di','Rk',
    }
    for name, stats in items.items():
        if name in seen: continue
        if name in out: continue
        if len(name) < 4: continue
        if name in NAME_NOISE: continue
        if not re.search(r'[A-Za-z]', name): continue
        merged = dict(stats)
        merged.setdefault('ocr_variants', [])
        out[name] = merged
    return out


def _empty_stats():
    return {
        'count': 0,
        'docs': set(),
        'datasets': Counter(),
        'max_score': 0.0,
        'ocr_variants': [],
    }


def _merge_into(dst, src):
    dst['count'] += src.get('count', 0)
    dst_docs = dst['docs']
    src_docs = src.get('docs', set())
    if isinstance(src_docs, list):
        src_docs = set(src_docs)
    dst_docs.update(src_docs)
    src_ds = src.get('datasets', {})
    if not isinstance(src_ds, Counter):
        src_ds = Counter(src_ds)
    dst['datasets'].update(src_ds)
    dst['max_score'] = max(dst['max_score'], src.get('max_score', 0.0))
