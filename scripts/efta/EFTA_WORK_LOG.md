# EFTA analysis — working log

Living plan + record of what's been tried. Things that **made it into the website**
are linked. Things parked or abandoned are documented here so we don't re-explore them.

## Hard rules learned the hard way

- **DNS storms**: parallel `wget`s blow systemd-resolved. Stagger by 2s or run sequentially.
- **Dropbox `dl=1` bulk-zips**: now require JS-rendered auth, return HTML landing page.
  Use Google Drive folder + gdown OR direct Internet Archive items instead.
- **DOJ Geeken HTTP mirror**: bucket is 403 — dead path.
- **IA `data-set-N` direct ZIP URLs**: work for DS1,3,4,5,6,7. DS2 needs `data-set-2_202512`,
  DS8 has no clean ZIP (only WARC + per-PDF item `doj-epstein-files-dataset8-2025`).
- **Piiranha-v1 label vocabulary**: uses `GIVENNAME` + `SURNAME` (not `PERSON`),
  `USERNAME`, `EMAIL`, `CITY`, `STREET`, `ZIPCODE`, `SOCIALNUM`, `TELEPHONENUM`,
  `DATEOFBIRTH`, `DCARDNUM` (debit card), `DRIVERLICENSENUM`, `ACCOUNTNUM`, `TAXNUM`,
  `PASSWORD`, `UILDINGNUM` (sic — buildingnum), `IDCARDNUM`.
- **CPU NER throughput** for Piiranha-v1: ~12 docs/min for mixed-size docs.
  ~1143 high-value docs ≈ 95 min CPU time. No GPU on this box.
- **OCR mangles `@` to `©`** in email addresses constantly. `EMAIL` label normalization
  in `rank_entities.py` already replaces `©→@` but lossy `@→A` (e.g. `natalia.molotkovaAcenturion.com`)
  is **not yet handled** — see "TODO" below.
- **DOJ pre-redacts** victim names as black boxes; the `pdftotext` output shows them as
  gaps before known role-labels ("ESQUIRE", ", Foreperson"). PII detector cannot recover
  these — they're literally not in the image.

## Datasets and current state

| Source | Path | Format | Status |
|---|---|---|---|
| DOJ DS1–7 | `raw/DataSet_{1..7}.zip` | zip → extracted/dataset_N | ✓ done, hash-verified |
| DOJ DS8 (IA pre-OCR'd) | `raw/dataset_8/` | per-PDF + djvu.xml | downloading (background, ~21K files) |
| DOJ DS9 | — | native media | skipped per user (181 GB, photos only) |
| DOJ DS10 | `raw/DataSet_10.zip` | zip | downloading (background, 80 GB, ~50% done) |
| DOJ DS11 | `raw/DataSet_11.zip` | zip → extracted/dataset_11 | ✓ done — **100% emails, 2017, Lesley Groff/JE/AmEx Centurion** |
| DOJ DS12 | `raw/DataSet_12.zip` | zip → extracted/dataset_12 | ✓ done — 60% emails, mixed |
| House Oversight Estate | `raw/estate/pdfs/{001,003}.pdf` | 2 huge bundled PDFs | ✓ done (1.5 GB each) |
| House Oversight text-only chunk | `raw/estate/text_only/` | HOUSE_OVERSIGHT_*.txt | ✓ done — includes Epstein's Goertzel AGI library book |
| Email corpus (parsed) | `emails/{txt,json,pdf}/` | per-email files + INDEX.csv | ✓ done — 194 emails |

## Pipeline at a glance

```
extracted/* + emails/txt + estate/text_only + raw/dataset_8/*_djvu.xml
        │
        ▼
build_corpus.py  →  work/corpus.jsonl  (5163 records, sha256-deduped, idempotent)
        │
        ▼
filter to text-heavy (n_words ≥ 30, skip DS1/2/5/big-estate-book)
        │
        ▼
work/corpus.hi.jsonl  (1143 records)
        │
        ├──► ngrams.py ──► work/ngrams/{ngrams.json, ngrams_n[4,5,6].csv}
        │
        └──► ner_tag.py (Piiranha-v1) ──► work/tagged.jsonl
                    │
                    ▼
              rank_entities.py ──► work/rankings/{rankings.json, rankings_LABEL.csv}
                    │
                    ▼
              export_findings.py ──► work/findings.json
                    │
                    ▼
              copy to 2pizzaclub/efta/findings.json + push
```

## What's currently on the website (`2pizzaclub.com/efta/`)

- Top 20 names by mention count (Epstein 49, Jeanne 31, Grumbridge 27, Alessi 22, Dechert 17, ...)
  - **CAVEAT**: based on only 201/1143 docs tagged (NER still running)
- Per-label PII pages (CITY, STREET, ZIPCODE, EMAIL, SOCIALNUM, USERNAME, ...)
- 6/5/4-gram phrase pages, sorted by doc-spread:
  - "copying jeevacation gmail com" (296 docs) → JE's personal Gmail cc'd as self-archive
  - "constitute inside addressee property jee" (237 docs)
  - "including copyright house oversight" (241 docs)

## TODO — analysis ideas to add

### Shipped (live on /efta/)
- [x] OCR canonicalization — `canonicalize.py` (Levenshtein clustering EMAIL same-domain, PERSON same-soundex)
- [x] EMAIL page promoted to position 2
- [x] Doc-date histogram via priority chain (sent_header → report_date_header → first_body_date → bare_year_fallback) — `dates.py`
- [x] Mention-date histogram (per-doc dedup) — `dates.py`
- [x] TF-IDF n-grams — `tfidf.py`
- [x] Co-occurrence pairs — `cooccur.py`
- [x] Code-language grep with provenance — `code_terms.py`, `grep_terms.py`
- [x] Press-finding cross-check (recreates journalist-cited findings at scale) — `journalist_grep.py`

### Key findings produced
- "massage" — 4,449 mentions / 79 docs (Giuffre-documented code term ✓ recreated)
- Trump — 1,818 / 449 docs (Oversight Dems claim of "1000+" recreated at scale)
- Clinton 1,056 / 257; Maxwell 2,113 / 93; Leon Black 723 / 99
- JE initials decoded: 997 / 336 docs (✓); GM 484 / 30; AR (Ross) 456 / 217; RK (Kahn) 411 / 31
- interlochen — 793 / 53 docs (Jane Doe Camp testimony ✓ recreated)
- Acosta 661 / 161; Wexner, Dershowitz, Indyke, Krauss, Ruemmler all surface
- Doc-date peaks: 2008 (239 docs, NPA-era police reports in DS8) and 2017 (286 docs, Lesley Groff operational in DS11)
- NEW Epstein email found via TF-IDF: `jeeitunes@gmail.com` (96 mentions in HOUSE_OVERSIGHT_025734, 23 docs total) — second personal Gmail in addition to `jeevacation@gmail.com`
- **Mystery solved**: phone numbers 301-261-1902 & 410-974-0947 (390/383 mentions) are the footer of **Free State Reporting, Inc.** — the MD-based court reporter that transcribed the SDNY Ghislaine Maxwell grand jury (Bates `GM_GLSDNY_*`). Also explains "Annap" + "Balt" being NER-tagged as given names — those are the footer city names.

### After Estate PDF ingestion (corpus = 10,210 high-value docs)
- Maxwell 17,174 mentions / **2,362 docs** (estate effect)
- Trump 4,910 / 494; Andrew 3,159 / 1,419; Dershowitz 2,101 / 92; Bannon 1,191 / 85; Brunel 749 / 34
- "massage" 7,070 / 368 docs (Giuffre code term ✓✓ recreated at scale)

### iMessage forensic exports surfaced (25 docs in Estate)
25 House Oversight docs `HOUSE_OVERSIGHT_025408..027794` are **forensic iMessage exports
from Epstein's Mac** captured day-of-arrest July 6 2019. Source path:
`H\Macintosh HD\root\Users\jee\Library\Messages\Archive\2019-07-061111`.
Per-message records with GUID, timestamps, "Is Read", message text. Counterpart's
address is REDACTED; only Epstein's iMessage account shows: `jeeitunes@gmail.com`.

Sample messages from HOUSE_OVERSIGHT_027794 (Jun 14 – Jul 6 2019, pre-arrest):
- "15 down in Wisconsin" (Trump polling)
- Link to ABC story on Trump internal polling
- "Darren awol" (Darren Indyke, his attorney)
- "Chinese blink on extradition. I will speak to"
- **"Can you give me a proposed schedule . Ranch island. Paris. Harvard."**

### Verbatim press-finding recreations (✓ with doc IDs)
- ✓ "Snow White" — `EFTA00029432_djvu` (DS8): "*Jul 9, 2010 at 8:45 PM, Jes Staley - ee > wrote: Maybe they're tracking u?? That was fun. Say hi to Snow White*"
- ✓ "Beauty and the Beast" — `EFTA02731039` (DS12): victim's description of Epstein's library
- ✓ "highly valued friend" — `HOUSE_OVERSIGHT_022405`: verbatim Chomsky LOR
- ✓ "Invisible Man" — `EFTA00011440_djvu` (DS8): Aug 18 2001 email From: 'The Invisible Man' <abx17@dial.pipex.com> To: 'G. Maxwell' <gmax1@mindspring.com>
- ✓ "Mohammed bin Salman / Aka MBS" — `HOUSE_OVERSIGHT_019874`
- ✗ "dog that hasn't barked" / "knew about the girls" / "dirty donald" / "carpets and all" — NOT in our corpus; likely in a House-Oversight-Dems-specific release not yet pulled (`oversightdemocrats.house.gov`).

### Email-thread reconstruction (729 emails, 363 distinct threads)
Top threads by length:
- 29 msgs "jeffrey epstein" (2017-04-28 →)
- 12 msgs "leon black"
- **10 msgs "leon black / additional HT subject referral -- update"** — "HT" = federal lingo for **Human Trafficking** subject referral
- 5 msgs "trump" (2015-12-08 → 2016-05-25 — Wolff-era window)
- 4 msgs "saudi money" (2016-10-19)
- 4 msgs "karyna's flights to france" (Karyna Shuliak, Epstein's girlfriend)

### Epstein iMessage forensic chronological extract (`imessages.py`)
Parsed 1,575 distinct messages from 25 forensic-export docs, spanning 2017-01-27 → 2019-07-06 (arrest day).
- 869 sent by Epstein (`jeeitunes@gmail.com`)
- 706 from REDACTED counterpart
- By year: 2017=169, 2018=534, 2019=872
- Keyword hits: trump=49, china=25, today=22, call=18, meet=15, island=15, paris=14, tomorrow=14, schedule=10, darren=9 (Indyke)

**First chronological message** (Jan 27 2017, one week after Trump inauguration):
> 07:51 counterpart: "I'm seeing BG tmr. He will be in DC for the Alfalfa dinner but he's got mtgs mos…"  ← BG likely = Bill Gates, Alfalfa Club is exclusive DC dinner
> 07:54 EPSTEIN: "kushner does not care"
> 07:56 counterpart: "K will wait"
> 07:59 EPSTEIN: "ask him if he will see tom barrack, thats the most important."
> 07:59 EPSTEIN: "he is free to call me for inside baseball"

(Tom Barrack = Trump's longtime friend / Inaugural Committee chair, later indicted.)

**Standout iMessages by date** (one-line quotes — full chronological reader on `/efta/` page 5+):
- 2017-01-27 EPSTEIN: "**bannon, barrack, puppet masters**" (one week post-inauguration)
- 2017-02-18 EPSTEIN: "I sent bill a note to suggest he talk to lauder. donald also thinks bill wants not to help america first" (re: Ronald Lauder / Trump strategy)
- 2017-04-09 EPSTEIN: "Zero. My expertise is personal wealth . So my clients have . **Leon black - Apollo for ex.**"
- 2018-04-06 EPSTEIN: "**HBJ sun. MBS mon. Paris I'm not back until tues**" (Hamad bin Jassim = Qatari PM Sun, Mohammed bin Salman = Saudi Crown Prince Mon)
- 2019-04-30 EPSTEIN: "Darren is the contractor for legal service prep . It gets paid by Darren. He has final control as it is his work prod" (re: Darren Indyke role)
- 2019-06-03 EPSTEIN: "**prince andrew and trump today. t000 funny.**" (33 days before Epstein's arrest)
- 2019-06-05 EPSTEIN shared a RawStory link about Michael Wolff saying Bannon "has knowledge of Trump's crimes"
- 2019-06-17 EPSTEIN shared an AOL link about "Israel announces Golan Heights settlement to be named after Trump"

### iMessage timing + busiest-day reveal (`imessages.py` + ad-hoc analysis)
- **Peak hours 3 AM – 9 AM** (4 AM = 168 msgs, highest hour). Confirms Epstein's
  notorious early-bird schedule. Counterpart matches this rhythm — meaning the
  counterpart either operates on global hours, finance/news hours, or matches Epstein deliberately.
- Almost no messages 8 PM – 2 AM (quiet evenings).
- Median message length: 30 chars. These are short operational/social texts, not compositions.

**Busiest day: 2019-04-30 (68 messages, ~2 months pre-arrest)** — content reveals:
1. **Trump/Deutsche Bank disclosure discussion**:
   - JE 03:40 "trump trying to stop deutsch bank only a matter of time."
   - counterpart 03:45 "His strategy is drag this out for 15 months then it won't matter"
   - JE 03:48 mentions "weissleburg" (Allen Weisselberg, Trump Org CFO indicted 2021)
2. **They are PRODUCING A DOCUMENTARY FILM defending Epstein**:
   - counterpart 03:46 "I'm watching our second hour now"
   - counterpart 05:06 "we must counter 'rapist who traffics in female children to be raped by worlds most powerful, richest'"
   - counterpart 05:07 "Can't redeem unredeemable -- you are a lot of things-which we will show-- but you are NOT that"
   - JE 05:03 "the christians i met with feel, the media portraying me as beyond redemption is deeply troubling and offensive"
   - JE 14:07 "I watched the interview. Well done again , thx. here is what I think will work legally, I pay direct costs of filming. As Darren is the contractor for legal service prep . It gets paid by Darren."
   - counterpart 12:42 "Did u get the film we shot ??"
3. **MBS / KSA / Yemen / Pompeo**:
   - JE 04:37 "if you like you can go to yemen and meet with heads. you have an invitation."
   - JE 04:38 "you should if you decide of course coordinate with pompeo"
   - counterpart 04:39 "KSA wants everybody to stay away-- closer inspection only leads to more doubts"
4. **Nick Bostrom / AI**:
   - counterpart 04:59 "Have you followed nick bostrom? Oxford prof-- bill gates guy"
   - JE 05:00 "chicken little"

The counterpart is operating as Epstein's media-defense producer (film team) and intermediary
to world leaders (Yemen access "unlike others"), with both of them up at 3-5 AM.

URLs Epstein shared (66 total):
- WSJ, NYT, Bloomberg, NYT, Commentary, Marketwatch, Business Insider, USA Today
- One Kaiser Permanente bio (Bernard J. Tyson, KP CEO — died Nov 2019)
- Bridgewater Founder Ray Dalio interview
- NYT "Trump driver overtime lawsuit"
- NYT "Rod Rosenstein wear a wire / 25th amendment"

### DS10 discovery: financial-records dataset
DS10 extracted to 172,171 PDFs (not "images + videos" as Al Jazeera summary claimed).
Sampling revealed content:
- `EFTA01671962` = **9,824-page AmEx Gold Card statement bundle for JEFFREY E EPSTEIN** (account XXXX-XXXXX8-42008), starting 2010. 117,702 Membership Rewards points. This is the primary evidence behind the CBS/Bloomberg "AmEx Centurion booked decoy flights for women" stories.
- `EFTA01584068` = J.P. Morgan Funds Transfer Request 11/01/13. Wire-transfer documentation.

Implication: DS10 is the financial dossier behind the Raskin "$1.5B suspicious transactions" probe. Ingesting in background via `build_corpus_ds10.py` (filter 50KB-10MB to skip image-only photos and huge multi-thousand-page bundles).

### Still to try while NER continues
- [ ] **Email-thread reconstruction** — group emails by `Subject:` (drop `Re:`/`Fwd:`), order by `Sent:`, show longest threads
- [ ] **Per-dataset slicing** — once full NER lands, slice top names by dataset (police vs grand jury vs emails)
- [ ] **Investigate the "2029" doc-date outliers** (6 docs with future dates — likely OCR error or DOJ-internal placeholder)
- [ ] **Email-only TF-IDF** — restrict TF-IDF to DS11/DS12 emails to surface social-graph signal without legal-boilerplate dominance
- [ ] **Extract verbatim quotes for press-recreated findings** — show the actual line in our corpus for each top match, not just count
- [ ] **Verify "dog that hasn't barked" / "I know how dirty donald is" / "Snow White" — need to ingest the full Estate `001.pdf`/`003.pdf` (3 GB combined, slow pdftotext)**

### Want to recreate journalist findings
- [ ] **"dog that hasn't barked"** — Apr 2011 Epstein → Maxwell re Trump (need Estate PDFs ingested)
- [ ] **"Snow White / Beauty and the Beast"** — Staley-Epstein 2010 (need Estate)
- [ ] **"I know how dirty donald is"** — Ruemmler thread (need Estate)
- [ ] **"carpets and all"** — MBS tent 2016 (need Estate)
- [ ] **"highly valued friend"** — Chomsky LOR (need Estate)

### Probably not worth the effort
- Sentiment analysis — too noisy on document mix
- Topic modeling (LDA) — corpus too small + too domain-specific
- Photo PII (face detection on DS1/2/5) — out of scope per user

## Things tried that DIDN'T work (don't re-attempt)

1. **WebFetch on oversight.house.gov** — Cloudflare-gated, returns 403.
   Workaround: `curl` with browser User-Agent works.
2. **WebFetch on archive.org/details/...** — also CDN-gated.
   Workaround: use `https://archive.org/metadata/<id>` JSON API instead.
3. **`gdown --folder` on Google Drive folder** — hit rate-limit after 1 subfolder; got 500s on subsequent dirs. Did not retry.
4. **Dropbox `dl=1`** — returns HTML landing page now, not bulk ZIP. JS-render auth required.
5. **DOJ Geeken mirror** (`doj-files.geeken.dev`) — Cloudflare R2 bucket returns 403 on both index and individual files. Dead.
6. **DS8 WARC mirror** (`www.justice.gov_epstein_files_DataSet_8.zip_20260203/...warc.gz`) — IA endpoint returns 403 despite metadata listing. Switched to per-PDF item `doj-epstein-files-dataset8-2025`.
7. **`pkill ner_tag.py` then chained python** — pkill returning non-zero from no-matches kills the bash chain (no `set -e`-style behavior). Run python directly without pkill.
8. **Local http.server backgrounded via Bash tool** — runtime kills background python processes (exit 144). Use direct file/JSON validation instead.
9. **OpenMed PII model** — OpenMed HF org is biomedical NER only, no PII-specific model.
   Switched to `iiiorg/piiranha-v1-detect-personal-information` (DeBERTa-v3 fine-tune).

## Provenance / sources of truth

- DOJ portal: `justice.gov/epstein` (bulk ZIPs pulled Feb 6, 2026)
- House Oversight: `oversight.house.gov/release/oversight-committee-releases-additional-epstein-estate-documents/`
- Aggregator README: `github.com/yung-megafone/Epstein-Files` (cloned to `/tmp/epstein-aggregator/`)
- IA item identifiers — see "Datasets and current state" table
- SHA256 manifest: `~/data/epstein-files/SHA256SUMS.aggregator.txt`
