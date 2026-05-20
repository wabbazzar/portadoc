#!/usr/bin/env bash
# Re-runnable EFTA PII pipeline.
# Drop new PDFs/TXT into ~/data/epstein-files/{extracted,raw,emails}/ and re-run.
# Idempotent: only new files get OCR'd, only new docs get NER'd.

set -e
ROOT="${EFTA_ROOT:-$HOME/data/epstein-files}"
WORK="${EFTA_WORK:-$HOME/data/epstein-files/work}"
PORTADOC="$HOME/code/portadoc"
PYBIN="$PORTADOC/.venv/bin/python"

mkdir -p "$WORK"

echo "===> [1/5] Building corpus.jsonl"
"$PYBIN" "$PORTADOC/scripts/efta/build_corpus.py" \
  "$WORK/corpus.jsonl" \
  "$ROOT/extracted" \
  "$ROOT/emails/txt" \
  "$ROOT/raw/estate/pdfs" \
  "$ROOT/raw/estate/text_only" \
  "$ROOT/raw/dataset_8"

echo
echo "===> [2/5] Running NER tagging with Piiranha-v1"
"$PYBIN" "$PORTADOC/scripts/efta/ner_tag.py" \
  "$WORK/corpus.jsonl" \
  "$WORK/tagged.jsonl" \
  --device "${EFTA_DEVICE:-cpu}"

echo
echo "===> [3/5] Ranking entities"
"$PYBIN" "$PORTADOC/scripts/efta/rank_entities.py" \
  "$WORK/tagged.jsonl" \
  "$WORK/rankings"

echo
echo "===> [4/5] Computing n-grams (4, 5, 6)"
"$PYBIN" "$PORTADOC/scripts/efta/ngrams.py" \
  "$WORK/corpus.jsonl" \
  "$WORK/ngrams"

echo
echo "===> [5/5] Exporting findings.json"
"$PYBIN" "$PORTADOC/scripts/efta/export_findings.py" \
  "$WORK/rankings" \
  "$WORK/ngrams" \
  "$WORK/findings.json"

echo
echo "DONE. findings.json at $WORK/findings.json"
echo "Copy to 2pizzaclub: cp $WORK/findings.json $HOME/code/2pizzaclub/efta/findings.json"
