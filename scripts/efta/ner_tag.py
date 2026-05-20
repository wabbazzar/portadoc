#!/usr/bin/env python3
"""
Run a PII NER model over a JSONL corpus and emit a parallel JSONL of taggings.

Output records:
  {"id": "<corpus id>",
   "dataset": "<bucket>",
   "n_words": <int>,
   "entities": [
     {"label": "PERSON|EMAIL|...", "text": "...", "start": <char>, "end": <char>, "score": <float>}, ...
   ]}

Model default: iiiorg/piiranha-v1-detect-personal-information (DeBERTa-base fine-tune,
~278M params, but the only well-supported small PII model on HF; user requested
'~44M openmed' which doesn't exist as a PII model — Piiranha is the closest fit).

Long inputs are sliced into overlapping windows (default 400 tokens, 50 token stride)
and entity offsets are remapped to global char positions, then merged across windows
when they overlap.

Idempotent: skips ids that already appear in the output JSONL.

Usage:
    ner_tag.py <corpus.jsonl> <tagged.jsonl> [--model HF_ID] [--device cpu|cuda]
"""
import sys, json, argparse
from pathlib import Path

from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline
import torch


def load_pipeline(model_id: str, device: str):
    tok = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForTokenClassification.from_pretrained(model_id)
    dev_arg = 0 if (device == 'cuda' and torch.cuda.is_available()) else -1
    return pipeline(
        'token-classification', model=model, tokenizer=tok,
        aggregation_strategy='simple', device=dev_arg,
    )


def chunk_text(text: str, max_chars: int = 1800, overlap: int = 200):
    """Yield (offset, chunk) pairs. Char-based windowing avoids tokenizer-dependent
    arithmetic and keeps offset remap trivial."""
    if len(text) <= max_chars:
        yield 0, text
        return
    i = 0
    while i < len(text):
        end = min(i + max_chars, len(text))
        # try not to split mid-word
        if end < len(text):
            sp = text.rfind(' ', i + max_chars - 100, end)
            if sp > i:
                end = sp
        yield i, text[i:end]
        if end >= len(text):
            return
        i = end - overlap if end - overlap > i else end


def dedup_entities(ents):
    """Merge entities that share (label, text, start) — windows can overlap."""
    seen = set()
    out = []
    for e in ents:
        key = (e['label'], e['text'].strip().lower(), e['start'])
        if key in seen:
            continue
        seen.add(key)
        out.append(e)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('corpus_jsonl')
    ap.add_argument('out_jsonl')
    ap.add_argument('--model', default='iiiorg/piiranha-v1-detect-personal-information')
    ap.add_argument('--device', default='cpu', choices=['cpu', 'cuda'])
    ap.add_argument('--limit', type=int, default=None,
                    help='Only process first N corpus records (for smoke tests)')
    args = ap.parse_args()

    seen_ids = set()
    out_path = Path(args.out_jsonl)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists():
        with open(out_path) as f:
            for line in f:
                try:
                    rec = json.loads(line)
                    seen_ids.add(rec.get('id', ''))
                except Exception:
                    pass
        print(f'  found {len(seen_ids)} already-tagged ids; will skip', flush=True)

    print(f'loading model {args.model} on {args.device}...', flush=True)
    nlp = load_pipeline(args.model, args.device)
    print(f'model loaded', flush=True)

    n_done = n_skip = 0
    with open(args.corpus_jsonl) as cin, open(out_path, 'a') as cout:
        for i, line in enumerate(cin):
            if args.limit is not None and n_done >= args.limit:
                break
            try:
                rec = json.loads(line)
            except Exception:
                continue
            if rec['id'] in seen_ids:
                n_skip += 1
                continue
            text = rec.get('text', '')
            ents_all = []
            for offset, chunk in chunk_text(text):
                try:
                    chunk_ents = nlp(chunk)
                except Exception as ex:
                    print(f'  ner error on {rec["id"]} chunk@{offset}: {ex}', flush=True)
                    continue
                for e in chunk_ents:
                    ents_all.append({
                        'label': e.get('entity_group') or e.get('entity'),
                        'text': e['word'],
                        'start': int(e['start']) + offset,
                        'end': int(e['end']) + offset,
                        'score': float(e['score']),
                    })
            ents_all = dedup_entities(ents_all)
            out_rec = {
                'id': rec['id'],
                'dataset': rec.get('dataset', 'unknown'),
                'n_words': rec.get('n_words', 0),
                'entities': ents_all,
            }
            cout.write(json.dumps(out_rec, ensure_ascii=False) + '\n')
            cout.flush()
            seen_ids.add(rec['id'])
            n_done += 1
            if n_done % 25 == 0:
                print(f'  tagged {n_done} (last: {rec["id"]}, {len(ents_all)} ents)', flush=True)
    print(f'done. new_tagged={n_done}  skipped_already_tagged={n_skip}')


if __name__ == '__main__':
    main()
