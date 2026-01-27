#!/usr/bin/env python3
"""Convert text dictionaries to JSON arrays for browser client."""
import json
from pathlib import Path

def convert_wordlist(input_path: Path, output_path: Path):
    """Convert .txt wordlist to JSON array."""
    words = []
    with open(input_path, encoding='utf-8') as f:
        for line in f:
            word = line.strip()
            if word and not word.startswith('#'):
                words.append(word)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(words, f)

    print(f"Converted {len(words):,} words: {input_path.name} -> {output_path.name}")

def main():
    repo_root = Path(__file__).parent.parent
    dict_dir = repo_root / 'data' / 'dictionaries'
    out_dir = repo_root / 'src' / 'portadoc' / 'browser' / 'public' / 'dictionaries'

    conversions = [
        ('english_words.txt', 'english.json'),
        ('us_names.txt', 'names.json'),
        ('medical_terms.txt', 'medical.json'),
        ('custom.txt', 'custom.json'),
    ]

    for src, dest in conversions:
        src_path = dict_dir / src
        dest_path = out_dir / dest
        if src_path.exists():
            convert_wordlist(src_path, dest_path)
        else:
            print(f"WARNING: {src} not found, skipping")

    print(f"\nDictionaries written to {out_dir}")

if __name__ == '__main__':
    main()
