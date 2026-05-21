#!/usr/bin/env python3
"""
DS10-specific ingest: filter to medium-sized PDFs (skip image-only tiny ones AND
skip the multi-thousand-page bundles which would choke pdftotext).

Wraps build_corpus.py behavior but pre-filters paths by file size.
"""
import sys, json, hashlib, subprocess
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_corpus import sha256_file, extract_text, infer_dataset


def main():
    out_path = sys.argv[1]
    in_dir = sys.argv[2]
    min_kb = int(sys.argv[3]) if len(sys.argv) > 3 else 50
    max_mb = int(sys.argv[4]) if len(sys.argv) > 4 else 50

    out = Path(out_path)
    seen = set()
    if out.exists():
        with open(out) as f:
            for line in f:
                try:
                    rec = json.loads(line)
                    seen.add(rec.get('sha256', ''))
                except Exception:
                    pass
    print(f'  existing records: {len(seen)}', flush=True)

    paths = []
    for p in Path(in_dir).rglob('*.pdf'):
        sz = p.stat().st_size
        if sz < min_kb * 1024 or sz > max_mb * 1024 * 1024:
            continue
        paths.append(p)
    print(f'  candidates after size filter ({min_kb}KB..{max_mb}MB): {len(paths)}', flush=True)

    n_new = n_skip = n_empty = 0
    with open(out, 'a') as f:
        for i, path in enumerate(paths):
            digest = sha256_file(path)
            if digest in seen:
                n_skip += 1
                continue
            text = extract_text(path)
            if not text.strip():
                n_empty += 1
                seen.add(digest)
                continue
            rec = {
                'id': path.stem,
                'source': str(path),
                'dataset': infer_dataset(path),
                'format': path.suffix.lower().lstrip('.'),
                'text': text,
                'n_words': len(text.split()),
                'sha256': digest,
            }
            f.write(json.dumps(rec, ensure_ascii=False) + '\n')
            seen.add(digest)
            n_new += 1
            if n_new % 200 == 0:
                print(f'  +{n_new} new records...', flush=True)
    print(f'done. new={n_new}  skipped_seen={n_skip}  empty={n_empty}')


if __name__ == '__main__':
    main()
