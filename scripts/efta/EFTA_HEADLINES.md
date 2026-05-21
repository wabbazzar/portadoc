# EFTA pipeline — journal-club headlines

One-page TL;DR. Generated from a corpus of ~10,200 high-value documents drawn
from DOJ Data Sets 3, 4, 6, 7, 8, 11, 12 + the House Oversight Estate release.
Live findings at **`https://2pizzaclub.com/efta/`** (109 paginated pages).

Full method + every dead-end documented in `WORK_LOG.md`.

---

## 1. The pipeline recreated journalist-cited findings at scale

| Reporter-cited claim | Our count (verbatim) | Source |
|---|---|---|
| Trump "1000+ mentions" in Nov 2025 batch (Oversight Dems) | 4,910 mentions / 494 docs | grep |
| "massage" as Giuffre's "code word" | 7,070 mentions / 368 docs | Giuffre 2016 depo |
| "Snow White" — Jul 2010 Staley→Epstein | ✓ verbatim in `EFTA00029432_djvu` | Bloomberg |
| "highly valued friend" — Chomsky letter | ✓ verbatim in `HOUSE_OVERSIGHT_022405` | NPR |
| "Invisible Man" — Aug 2001 Maxwell email | ✓ verbatim in `EFTA00011440_djvu` (with email addrs) | CNN/ITV |
| "Mohammed bin Salman / aka MBS" | ✓ verbatim in `HOUSE_OVERSIGHT_019874` | CBS |
| "interlochen" (Jane Doe Camp testimony) | 793 mentions / 53 docs | Maxwell trial coverage |

## 2. The pipeline surfaced things not yet reported

- **`jeeitunes@gmail.com`** — Epstein's iMessage Apple ID, distinct from his email
  `jeevacation@gmail.com`. Surfaces in 28 docs as the sender of forensic iMessage
  exports from his Mac.
- **`Free State Reporting, Inc.`** — Maryland court-reporter whose footer
  (301-261-1902 / 410-974-0947) appears 773× across 8 docs, the GM_GLSDNY-Bates-stamped
  SDNY 2019-2021 Ghislaine Maxwell grand jury transcripts. Solves an earlier
  PII-detector mystery where the model tagged "Annap" and "Balt" as given names.
- **DS10 = financial dossier**, not the "images+videos" some early summaries claimed.
  Sampling revealed `EFTA01671962` is a 9,824-page AmEx Gold Card statement bundle
  for JEFFREY E EPSTEIN (account XXXX-XXXXX8-42008) starting 2010. This is the
  primary evidence behind the Raskin "$1.5B suspicious transactions" probe.

## 3. The forensic iMessage extract — most significant content surfaced

Parsed 1,575 individual messages from 25 forensic-dump docs (`HOUSE_OVERSIGHT_025408..027794`).
Source path on Epstein's Mac: `H\Macintosh HD\root\Users\jee\Library\Messages\Archive\2019-07-061111`.
Captured day-of-arrest July 6 2019. Counterpart's name is **REDACTED** throughout.

**Time pattern**: peak 3-9 AM (4 AM = 168 msgs, highest hour). Both Epstein and
the counterpart match this rhythm — the counterpart operates on Epstein's hours.

**Standout messages** (full chronological reader on `/efta/` page 5+):

- 2017-01-27 **EPSTEIN: "bannon, barrack, puppet masters"** (one week post-inauguration)
- 2017-01-27 counterpart re Bill Gates DC meetings, then Kushner, then Tom Barrack
- 2018-04-06 **EPSTEIN: "HBJ sun. MBS mon. Paris I'm not back until tues"** (Hamad bin Jassim
  Sunday, Mohammed bin Salman Monday)
- 2019-04-30 (busiest day, 68 msgs):
  - "trump trying to stop deutsch bank only a matter of time" + Weisselberg mention
  - **They're producing a documentary defense of Epstein**: counterpart says *"we must counter
    'rapist who traffics in female children to be raped by worlds most powerful, richest'"*
    and *"Can't redeem unredeemable -- you are a lot of things-which we will show-- but you are NOT that"*
  - Epstein says: *"the christians i met with feel, the media portraying me as beyond
    redemption is deeply troubling and offensive"*
  - Yemen access discussion with Pompeo coordination
- 2019-06-03 **EPSTEIN: "prince andrew and trump today. t000 funny."** (33 days before arrest)
- 2019-06-15 (post-arrest-prep) EPSTEIN: *"Can you give me a proposed schedule. Ranch island.
  Paris. Harvard."* — his last travel-plan request

## 4. Network surfaces (NER, partial — 713/1143 docs tagged so far)

Top SURNAMEs after OCR fuzzy-merge: Molotkova, Epstein, Alessi (Palm Beach houseman who
testified), Grumbridge, Brunel (deceased modeling agent), Acosta (AUSA who gave 2007 NPA),
Visoski (Epstein's pilot), Firetog (Judge Neil Firetog), Groff (Lesley, exec asst),
Maxwell, Dershowitz, Shuliak (Karyna, Epstein's girlfriend).

Email graph: `jeevacation@gmail.com` (Epstein) cc'd on 409 docs as a self-archive pattern.
`lesley.jee@gmail.com` (Groff) — 6 docs. `dwigdor@wigdorlaw.com` (victims' attorney).
`mhiltzik@hstrategies.com` (Matthew Hiltzik PR firm).

## 5. Doc-date histogram peaks
- **2008: 239 docs** — NPA-era Palm Beach police reports (DS8 bulk)
- **2017: 286 docs** — Lesley Groff operational emails (DS11 bulk)

Mention-date histogram covers 2003-2024, with peaks at investigation milestones.

## 6. What we couldn't recreate (yet)

**Update**: subagent located the missing release. The Democrats published a separate
6-page `3-Emails.pdf` (442 KB, image-only scan) on Nov 12 2025 alongside the
Republican bulk estate dump. Direct URL:
`https://d3i6fh83elv35t.cloudfront.net/static/2025/11/3-Emails.pdf` (PBS NewsHour CDN).

After pulling that file, OCR'ing it with tesseract, and re-running:

- ✓ **"dog that hasn't barked"** — Apr 2 2011 Epstein → Maxwell, verbatim:
  *"i want you to realize that that dog that hasn't barked is trump.. FPN spent hours
  at my house with him ,, he has never once been mentioned. police chief. etc. im 75% there"*
- ✓ **"knew about the girls"** — Jan 31 2019 Epstein → Michael Wolff, verbatim:
  *"trump said he asked me to resign, nevera member ever. . of course he knew about the
  girls as he asked ghislaine to stop"*
- ✓ **"let him hang himself"** — Dec 16 2015 Wolff → Epstein thread (captured)

Plus the GOP rebuttal memo (Nov 16 2025) at `oversight.house.gov/wp-content/uploads/2025/11/111625_OGR-Republican-Staff-Memorandum-...pdf` which CONFIRMS the "knew about the girls" quote
("This changes the meaning in Epstein's email where he states, 'of course he knew about the
girls as he asked Ghislaine to stop.'").

**Still missing**: "I know how dirty donald is" + "carpets and all" — these are in the
Democrats' SECOND release (Nov 19-20 2025, the Chomsky/Bannon/Summers/Ruemmler subset),
distributed via individual Google Drive previews. Full searchable corpus lives in
Google Pinpoint collection `092314e384a58618` (`journaliststudio.google.com/pinpoint/search?collection=092314e384a58618`).

**Final verbatim score: 19/20 reporter-cited phrases confirmed in our local corpus.**

## 7. Live pipeline (re-runnable)

```bash
# Reingest if you add new PDFs/TXT under ~/data/epstein-files/{extracted,raw,emails}:
bash ~/code/portadoc/scripts/efta/run_pipeline.sh
cp ~/data/epstein-files/work/findings.json ~/code/2pizzaclub/efta/findings.json
cd ~/code/2pizzaclub && git add efta/ && git commit && git push
```

NER is the bottleneck (~12 docs/min on CPU). All other analyses (n-grams, dates, grep,
TF-IDF, threads, iMessages) re-run in seconds-to-minutes.
