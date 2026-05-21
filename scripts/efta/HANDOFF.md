# EFTA viewer — handoff for fresh-context agent

This file captures the state of the EFTA findings work so a fresh agent can
pick up without combing through the transcript.

## TL;DR

A static SPA at `https://2pizzaclub.com/efta/` showing statistical findings
extracted from the publicly-released DOJ Epstein Files Transparency Act
documents. Code lives in:

- **Pipeline**: `~/code/portadoc/scripts/efta/`
- **Viewer**: `~/code/2pizzaclub/efta/`
- **Working data**: `~/data/epstein-files/`

Two repos, both pushing to `main` on github.com/wabbazzar with auto-deploy.

## How to re-run end-to-end

```bash
# Full pipeline (everything that doesn't need NER):
cd ~/code/portadoc
.venv/bin/python scripts/efta/build_corpus.py ~/data/epstein-files/work/corpus.jsonl \
  ~/data/epstein-files/extracted ~/data/epstein-files/emails/txt \
  ~/data/epstein-files/raw/estate/pdfs ~/data/epstein-files/raw/estate/text_only \
  ~/data/epstein-files/raw/dataset_8 ~/data/epstein-files/raw/estate/dems

# Then re-run each analysis (each is independent, picks up new corpus rows):
.venv/bin/python scripts/efta/name_grep.py ~/data/epstein-files/work/corpus.jsonl ~/data/epstein-files/work/names
.venv/bin/python scripts/efta/topic_grep.py ~/data/epstein-files/work/corpus.jsonl ~/data/epstein-files/work/topics
.venv/bin/python scripts/efta/person_dossier.py ~/data/epstein-files/work/corpus.jsonl ~/data/epstein-files/work/persons.json
.venv/bin/python scripts/efta/verbatim_quotes.py ~/data/epstein-files/work/corpus.hi.jsonl ~/data/epstein-files/work/quotes
.venv/bin/python scripts/efta/grep_terms.py ~/data/epstein-files/work/corpus.hi.jsonl ~/data/epstein-files/work/grep
.venv/bin/python scripts/efta/dates.py ~/data/epstein-files/work/corpus.hi.jsonl ~/data/epstein-files/work/dates
.venv/bin/python scripts/efta/threads.py ~/data/epstein-files/work/corpus.hi.jsonl ~/data/epstein-files/work/threads
.venv/bin/python scripts/efta/imessages.py ~/data/epstein-files/work/corpus.jsonl ~/data/epstein-files/work/imessages
.venv/bin/python scripts/efta/ds10_financial.py ~/data/epstein-files/work/corpus.jsonl ~/data/epstein-files/work/ds10
.venv/bin/python scripts/efta/ngrams.py ~/data/epstein-files/work/corpus.emails.jsonl ~/data/epstein-files/work/ngrams
.venv/bin/python scripts/efta/tfidf.py  ~/data/epstein-files/work/corpus.emails.jsonl ~/data/epstein-files/work/tfidf

# NER (slow — 1-2 hrs on the ~900-doc text-rich subset):
.venv/bin/python scripts/efta/ner_tag.py ~/data/epstein-files/work/corpus.tagworthy.jsonl ~/data/epstein-files/work/tagged.jsonl --device cpu
PYTHONPATH=scripts/efta .venv/bin/python scripts/efta/rank_entities.py ~/data/epstein-files/work/tagged.jsonl ~/data/epstein-files/work/rankings
PYTHONPATH=scripts/efta .venv/bin/python scripts/efta/cooccur.py ~/data/epstein-files/work/tagged.jsonl ~/data/epstein-files/work/cooccur

# Export findings.json + push:
PYTHONPATH=scripts/efta .venv/bin/python scripts/efta/export_findings.py \
  ~/data/epstein-files/work/rankings ~/data/epstein-files/work/findings.json \
  --ngram-dir ~/data/epstein-files/work/ngrams --grep-dir ~/data/epstein-files/work/grep \
  --dates-dir ~/data/epstein-files/work/dates --tfidf-dir ~/data/epstein-files/work/tfidf \
  --cooccur-dir ~/data/epstein-files/work/cooccur --quotes-dir ~/data/epstein-files/work/quotes \
  --threads-dir ~/data/epstein-files/work/threads --imessages-dir ~/data/epstein-files/work/imessages \
  --ds10-dir ~/data/epstein-files/work/ds10 --names-dir ~/data/epstein-files/work/names \
  --topics-dir ~/data/epstein-files/work/topics --persons-json ~/data/epstein-files/work/persons.json

cp ~/data/epstein-files/work/findings.json ~/code/2pizzaclub/efta/findings.json
cd ~/code/2pizzaclub && git add efta/findings.json && git commit && git push
```

`run_pipeline.sh` is a wrapper but may be slightly out of date — verify against the explicit invocations above.

## What's live at 2pizzaclub.com/efta/

Static SPA with one HTML file + 2pizzaclub.css/efta.css + efta.js + findings.json + 4,300+ per-doc snippet JSONs.

**Page kinds** (16 total — see `KIND_GROUP` in efta.js):
1. `person_dossier` — 6 paginated cards w/ Wikipedia thumb + relation + sparkline + samples
2. `names_top20` — NER PERSON tags (Piiranha)
3. `names_grep` — curated alias grep across full 173K corpus
4. `topic_search` — 8 topics (WW3, Simulation, Pandemic, Antarctica, Reptilian, Sacrifice, Ritual, Pedophilia)
5. `verbatim_quote` — 20 journalist-cited phrases w/ samples + significance + URLs
6. `press_recreate` — counts only (NO URL — see Open Item #3)
7. `codeword_top` — code-language counts
8. `cooccur_pairs` — NER-tagged name pair co-occurrence
9. `email_threads` — top reconstructed threads, expandable
10. `imessages` — chronological reader, 20 pages × 60 msgs
11. `doc_dates_year` — yearly histogram of doc dates
12. `mention_dates_year` — yearly histogram of body-text dates
13. `tfidf` — TF-IDF n-grams (emails-only subset)
14. `ngram` — doc-spread n-grams (emails-only subset)
15. `label_top` — other NER labels (CITY/STREET/etc.)
16. `ds10_financial` — JPM/DB banker/entity/POA extracts

## Filter bar (top of every page)

5 chips, sticky-top: personal emails · estate dump (mixed) · court proceedings · DOJ Feb-2026 · DS10 financial (off by default).

**Filter is wired for these page kinds** (data has per-row dataset info):
person_dossier, names_top20, names_grep, topic_search, verbatim_quote,
press_recreate, codeword_top, cooccur_pairs, email_threads, imessages,
label_top, ds10_financial.

**Filter is NOT wired** (shows yellow "filter doesn't apply" banner):
ngram, tfidf, doc_dates_year, mention_dates_year. See Open Item #4.

## Open items (most → least important)

### 1. by_year_dataset for topic + person bars (subagents in flight)

JS in `efta.js` already expects a `by_year_dataset` field on topic_search pages and person_dossier rows — when present, the histograms / sparklines recompute based on enabled filter buckets. Two subagents were launched to add this field to:
- `topic_grep.py` (output: `~/data/epstein-files/work/topics/topic_search.json`)
- `person_dossier.py` (output: `~/data/epstein-files/work/persons.json`)

**Verify**: check if those scripts have `by_year_dataset` emit code and the output JSON files have the nested field. If not, edit the scripts to track a `Counter` per `(year, dataset)`, re-run, re-export.

### 2. press_recreate has no clickable URLs (lying explainer was fixed)

Verbatim quotes page (kind=`verbatim_quote`) has real significance paragraphs + clickable source URLs as of commit `726224f`. press_recreate (counts of journalist-cited names) still doesn't have URLs — the rows come from `grep_terms.py` which uses `journalist_grep.TERMS`. To add URLs there, convert `journalist_grep.TERMS` from 3-tuples `(term, category, note)` to 4-tuples or dicts with a `url` field. ~80 entries.

### 3. Aggregate pages still un-filterable

ngrams.py, tfidf.py, dates.py emit aggregate counts without per-dataset breakdown. To make them filter-aware: add per-(ngram, dataset) and per-(year, dataset) Counters in those scripts. JS already shows a clear yellow banner when filter is partial. Honest workaround; real fix is per-dataset counters.

### 4. Curated portraits.json is sparse

`/efta/portraits.json` has 3 entries (Leon Black, Darren Indyke, Matthew Hiltzik). Many other dossiers (Lesley Groff, Karyna Shuliak, Joe Recarey, Bruce Krischer, Sarah Kellen, Nadia Marcinkova, Adriana Ross, Jean-Luc Brunel, etc.) have no thumbnail because Wikipedia REST API doesn't return one. Add manual URL overrides as you find good Wikimedia Commons files. Pattern: `"Person Name": "https://commons.wikimedia.org/wiki/Special:FilePath/Filename.jpg?width=200"`.

### 5. Layout / UX

- Sticky-left TOC sidebar on ≥1080px ✓ live (commit `43cd450`).
- Doc-viewer side-panel hidden by default, opens on doc-id click ✓ live.
- ⓘ info button on every page with hidden explainer ✓ live.
- Person card thumbnails verified to match name (no more shared Jeffrey mugshot) ✓ live.

### 6. Things to expect that ARE working

- 4,333 per-doc snippet files at `efta/docs/<id>.json` — fetched lazily on doc-id click.
- Datasource filter respected on 12/16 page kinds (see "Filter bar" section above).
- Wikipedia thumb fetcher with title-verification (catches redirect → wrong-person).
- 5-min heartbeat monitor + GH Actions failure watch — re-arm if killed.

## Known fragilities

- **OCR fuzzy-merge**: not perfect. Some names still have OCR variants that don't merge (e.g., `jeev@cationagmail.com` vs `jeevacation@gmail.com`).
- **iMessage timestamps**: extracted from forensic dump format. Some go out of plausible bounds (1111, 2915, 2035) — JS now sanity-bounds 1990-2026 for display.
- **Aggregate counts include all sources** (DS10 dominates by mention volume). Filter banner warns.
- **Background bash processes get reaped** by the runtime sometimes. Heartbeat catches it. Use Bash `run_in_background: true` over `nohup ... &` when possible.

## CI / deployment

- GitHub Pages on `main` push → auto deploy ~3 min.
- `secret-scan.yml` workflow runs gitleaks. `efta/findings.json` and `efta/docs/*.json` are allowlisted (high-entropy article IDs trigger false positives). Config: `.gitleaks.toml`.
- DO NOT add `Co-Authored-By: Claude` to commits — explicit project policy.

## Recent commits worth knowing about (newest first)

```
726224f  verbatim quotes: real significance paragraphs + URLs
f5ef308  bar charts respect filter via by_year_dataset
e790947  split estate from personal filter bucket; add portraits.json
1395dbd  filter coverage: verbatim_quote + ds10 tagged
0224712  persons.json with relation one-liners
558511e  doc viewer hidden via .open class toggle
43cd450  sticky-left TOC sidebar
9207dcd  Phase 2: filter bar + clickable doc viewer
```
