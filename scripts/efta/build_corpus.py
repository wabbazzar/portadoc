#!/usr/bin/env python3
"""
Build a normalized JSONL corpus from a directory of PDFs and TXT files.

Each output record:
  {"id": "<filename-stem>",
   "source": "<absolute path>",
   "dataset": "<inferred top-level bucket>",
   "format": "pdf|txt",
   "text": "<extracted plaintext>",
   "n_words": <int>,
   "sha256": "<hash of source>"}

Idempotent: skips files whose sha256 already appears in the existing JSONL.
Drop new files in, re-run, only the new ones get processed.

Usage:
    build_corpus.py <output.jsonl> <input_dir> [<input_dir> ...]
"""
import sys, os, json, hashlib, subprocess
from pathlib import Path


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def extract_text(path: Path) -> str:
    if path.suffix.lower() == '.txt':
        try:
            return path.read_text(encoding='utf-8', errors='replace')
        except Exception:
            return ''
    if path.suffix.lower() == '.pdf':
        try:
            out = subprocess.run(
                ['pdftotext', '-layout', '-q', str(path), '-'],
                capture_output=True, timeout=300, text=True,
            )
            return out.stdout
        except Exception:
            return ''
    if path.suffix.lower() == '.xml' and '_djvu' in path.stem:
        # IA djvu OCR XML — extract every <WORD>text</WORD>
        try:
            import xml.etree.ElementTree as ET
            tree = ET.parse(path)
            words = [w.text or '' for w in tree.findall('.//WORD')]
            return ' '.join(words)
        except Exception:
            return ''
    return ''


def infer_dataset(path: Path) -> str:
    """Bucket files into dataset_N / emails / estate / etc. based on path."""
    parts = path.parts
    for p in parts:
        if p.startswith('dataset_'):
            return p
        if p == 'emails':
            return 'emails'
        if p == 'estate':
            return 'estate'
        if p == 'text_only':
            return 'estate'
    return 'unknown'


def main(out_path: str, in_dirs: list[str]):
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    seen_hashes = set()
    if out.exists():
        with open(out) as f:
            for line in f:
                try:
                    rec = json.loads(line)
                    seen_hashes.add(rec.get('sha256', ''))
                except Exception:
                    pass
        print(f'  found {len(seen_hashes)} existing records in {out}; will skip re-processing those', flush=True)

    candidates = []
    for d in in_dirs:
        root = Path(d)
        if not root.exists():
            print(f'  skipping missing dir: {root}', flush=True)
            continue
        for ext in ('*.pdf', '*.txt', '*_djvu.xml'):
            candidates += list(root.rglob(ext))
    print(f'scanning {len(candidates)} candidate files...', flush=True)

    n_new = n_skip = n_empty = 0
    with open(out, 'a') as f:
        for path in candidates:
            digest = sha256_file(path)
            if digest in seen_hashes:
                n_skip += 1
                continue
            text = extract_text(path)
            if not text.strip():
                n_empty += 1
                seen_hashes.add(digest)
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
            seen_hashes.add(digest)
            n_new += 1
            if n_new % 100 == 0:
                print(f'  +{n_new} new records...', flush=True)
    print(f'done. new={n_new}  skipped_seen={n_skip}  empty={n_empty}  total_in_corpus={len(seen_hashes)}')


if __name__ == '__main__':
    if len(sys.argv) < 3:
        print(__doc__)
        sys.exit(2)
    main(sys.argv[1], sys.argv[2:])
