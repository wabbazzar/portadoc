#!/usr/bin/env bash
# Re-runnable EFTA PII pipeline.
# Drop new PDFs/TXT into ~/data/epstein-files/{extracted,raw,emails}/ and re-run.
# Idempotent: only new files get OCR'd, only new docs get NER'd, only changed
# analyses regenerate their JSON outputs.

set -e
ROOT="${EFTA_ROOT:-$HOME/data/epstein-files}"
WORK="${EFTA_WORK:-$ROOT/work}"
PORTADOC="$HOME/code/portadoc"
SITE="$HOME/code/2pizzaclub"
PYBIN="$PORTADOC/.venv/bin/python"
EFTA_DIR="$PORTADOC/scripts/efta"
PYPATH="PYTHONPATH=$EFTA_DIR"

mkdir -p "$WORK" "$WORK/ngrams" "$WORK/rankings" "$WORK/grep" "$WORK/dates" \
         "$WORK/tfidf" "$WORK/cooccur" "$WORK/quotes" "$WORK/threads" "$WORK/imessages"

echo "===> [1/11] Build corpus.jsonl from all input dirs"
"$PYBIN" "$EFTA_DIR/build_corpus.py" \
  "$WORK/corpus.jsonl" \
  "$ROOT/extracted" \
  "$ROOT/emails/txt" \
  "$ROOT/raw/estate/pdfs" \
  "$ROOT/raw/estate/text_only" \
  "$ROOT/raw/dataset_8"

echo
echo "===> [2/11] Filter corpus to high-value text-heavy docs"
"$PYBIN" - <<'PY'
import json
KEEP = {'emails','dataset_3','dataset_4','dataset_6','dataset_7','dataset_8',
        'dataset_10','dataset_11','dataset_12','estate'}
n_in=n_out=0
with open('${WORK}/corpus.jsonl'.replace('${WORK}','$WORK'.replace('$WORK','$WORK')).replace('$WORK','/home/wabbazzar/data/epstein-files/work')) as f, \
     open('/home/wabbazzar/data/epstein-files/work/corpus.hi.jsonl','w') as g:
    for line in f:
        n_in+=1
        try: r=json.loads(line)
        except: continue
        if r.get('n_words',0) < 30: continue
        ds = r.get('dataset','unknown')
        if ds not in KEEP: continue
        if ds == 'estate' and r.get('n_words',0) > 5000:
            # Keep large estate bundles too (they have everything)
            pass
        g.write(line); n_out+=1
print(f'filtered: {n_out}/{n_in}')
PY

echo
echo "===> [3/11] N-grams (4, 5, 6, doc-spread sort)"
"$PYBIN" "$EFTA_DIR/ngrams.py" "$WORK/corpus.hi.jsonl" "$WORK/ngrams"

echo
echo "===> [4/11] TF-IDF n-grams"
"$PYBIN" "$EFTA_DIR/tfidf.py" "$WORK/corpus.hi.jsonl" "$WORK/tfidf"

echo
echo "===> [5/11] Date histograms (doc-date + mention-date)"
"$PYBIN" "$EFTA_DIR/dates.py" "$WORK/corpus.hi.jsonl" "$WORK/dates"

echo
echo "===> [6/11] Grep code-language + journalist terms"
"$PYBIN" "$EFTA_DIR/grep_terms.py" "$WORK/corpus.hi.jsonl" "$WORK/grep"

echo
echo "===> [7/11] Verbatim press quote extraction"
"$PYBIN" "$EFTA_DIR/verbatim_quotes.py" "$WORK/corpus.hi.jsonl" "$WORK/quotes"

echo
echo "===> [8/11] Email-thread reconstruction"
"$PYBIN" "$EFTA_DIR/threads.py" "$WORK/corpus.hi.jsonl" "$WORK/threads"

echo
echo "===> [9/11] iMessage chronological extract"
"$PYBIN" "$EFTA_DIR/imessages.py" "$WORK/corpus.jsonl" "$WORK/imessages"

echo
echo "===> [10/11] NER + co-occurrence (skipped if tagged.jsonl absent; run ner_tag.py separately)"
if [ -f "$WORK/tagged.jsonl" ]; then
  cd "$PORTADOC" && env PYTHONPATH="$EFTA_DIR" "$PYBIN" "$EFTA_DIR/rank_entities.py" \
    "$WORK/tagged.jsonl" "$WORK/rankings"
  cd "$PORTADOC" && env PYTHONPATH="$EFTA_DIR" "$PYBIN" "$EFTA_DIR/cooccur.py" \
    "$WORK/tagged.jsonl" "$WORK/cooccur"
else
  echo "  (no tagged.jsonl yet — run: $PYBIN $EFTA_DIR/ner_tag.py $WORK/corpus.hi.jsonl $WORK/tagged.jsonl)"
fi

echo
echo "===> [11/11] Export findings.json"
cd "$PORTADOC" && env PYTHONPATH="$EFTA_DIR" "$PYBIN" "$EFTA_DIR/export_findings.py" \
  "$WORK/rankings" "$WORK/findings.json" \
  --ngram-dir "$WORK/ngrams" \
  --grep-dir "$WORK/grep" \
  --dates-dir "$WORK/dates" \
  --tfidf-dir "$WORK/tfidf" \
  --cooccur-dir "$WORK/cooccur" \
  --quotes-dir "$WORK/quotes" \
  --threads-dir "$WORK/threads" \
  --imessages-dir "$WORK/imessages"

if [ -d "$SITE/efta" ]; then
  cp "$WORK/findings.json" "$SITE/efta/findings.json"
  echo
  echo "Copied $WORK/findings.json → $SITE/efta/findings.json"
  echo "Commit + push from $SITE when ready."
fi

echo
echo "DONE."
