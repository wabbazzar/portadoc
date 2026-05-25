#!/usr/bin/env python3
"""
Extract verbatim contexts for journalist-cited phrases from the corpus.

For each phrase reporters have quoted, return the first 3 actual hits with
~150-char surrounding context + doc id + dataset. Lets the website show the
underlying quote, not just a count.

Output: quotes.json with shape
  { "quotes": [
      {"phrase": "...", "n_total_hits": N, "n_docs": M,
       "samples": [{"doc_id":..., "dataset":..., "context":...}, ...]}
  ] }

Usage:
    verbatim_quotes.py <corpus.jsonl> <out_dir>
"""
import sys, json, re, argparse
from pathlib import Path

# Phrases to chase verbatim. Each is a dict — supports url + significance fields
# that the viewer renders as a clickable link + a real "why this matters" blurb.
PHRASES = [
    {
        "phrase": "dog that hasn't barked",
        "pattern": r"dog\s+that\s+hasn'?t\s+barked",
        "source": "Apr 2 2011 Epstein → Maxwell email (House Oversight Nov 12 2025 release)",
        "url": "https://www.pbs.org/newshour/politics/read-jeffrey-epsteins-newly-released-emails-about-trump",
        "significance": "One of THE most-cited Epstein quotes in the entire release. Epstein writes Maxwell: '… that dog that hasn't barked is trump.. FPN spent hours at my house with him … he has never once been mentioned. police chief. etc. im 75% there.' Epstein is observing — three years after his 2008 plea — that Trump KNOWS about the abuse but has never been named publicly. The 'silent dog' is the press / law enforcement not coming for Trump. Senator Reed and others have cited this as evidence that Trump's name was deliberately scrubbed from earlier filings.",
    },
    {
        "phrase": "knew about the girls",
        "pattern": r"knew\s+about\s+the\s+girls",
        "source": "Jan 31 2019 Epstein → Michael Wolff (House Oversight Nov 12 2025 release)",
        "url": "https://www.cbsnews.com/news/jeffrey-epstein-donald-trump-emails-house-oversight/",
        "significance": "Epstein's own words to journalist Michael Wolff, six months before his arrest: 'trump said he asked me to resign, nevera member ever. . of course he knew about the girls as he asked ghislaine to stop.' This is Epstein's direct, contemporaneous statement that Trump knew about the trafficking AND personally asked Ghislaine Maxwell to stop. The 'asked me to resign' refers to Trump's claim he kicked Epstein out of Mar-a-Lago — which Epstein here disputes ('never a member ever').",
    },
    {
        "phrase": "let him hang himself",
        "pattern": r"let\s+him\s+hang\s+himself",
        "source": "Dec 16 2015 Michael Wolff → Epstein email (House Oversight Nov 12 2025 release)",
        "url": "https://www.pbs.org/newshour/politics/read-jeffrey-epsteins-newly-released-emails-about-trump",
        "significance": "Wolff coaches Epstein on how to handle Trump's upcoming CNN interview where Trump will be asked about him: 'I think you should let him hang himself.' Reveals an active media strategy between Epstein and a sympathetic journalist to manage Trump's public statements — and that Wolff thought Trump would self-incriminate if pushed. Pre-presidency context; pre-Epstein's 2019 re-arrest.",
    },
    {
        "phrase": "dirty donald",
        "pattern": r"dirty\s+donald",
        "source": "Aug 23 2018 Epstein → Kathryn Ruemmler email, in re: NYT 'Donald Trump's High Crimes and Misdemeanors' (House Oversight Republican estate release Nov 12 2025, HOUSE_OVERSIGHT_026505)",
        "url": "https://www.cnbc.com/2025/11/12/trump-jeffrey-epstein-ghislaine-maxwell-emails-house-democrats.html",
        "significance": "Epstein to Kathryn Ruemmler (then Goldman Sachs Chief Legal Officer and ex-Obama White House Counsel), responding to her forwarded NYT op-ed: 'you see, i know how dirty donald is. my guess is that non lawyers ny biz people have no idea. what it means to have your fixer flip.' This was the day Cohen pleaded guilty. Demonstrates Epstein cultivating a senior legal/political insider with explicit anti-Trump opposition research. Ruemmler RESIGNED from Goldman Sachs in late 2025 after this email surfaced.",
    },
    {
        "phrase": "I know how dirty donald is",
        "pattern": r"i\s+know\s+how\s+dirty\s+donald\s+is",
        "source": "Same Aug 23 2018 Ruemmler thread (HOUSE_OVERSIGHT_026505) — longer-form match for the most-quoted exact phrasing",
        "url": "https://www.cnbc.com/2025/11/12/trump-jeffrey-epstein-ghislaine-maxwell-emails-house-democrats.html",
        "significance": "The exact phrasing reporters have quoted verbatim. Co-located in the Aug 23 2018 Ruemmler thread alongside 'fixer flip' (a reference to Michael Cohen, who had pleaded guilty earlier that day).",
    },
    {
        "phrase": "carpets and all",
        "pattern": r"carpets\s+and\s+all",
        "source": "Dec 15 2016 Epstein → Tom Pritzker email (House Oversight Republican estate release Nov 12 2025, HOUSE_OVERSIGHT_032391)",
        "significance": "Epstein to Hyatt executive chairman Tom Pritzker: 'can you belive MBS sent mea TENT carpets and all.' Sent shortly after Epstein's solo Gulfstream trip Paris → Riyadh. Bedouin tents are a traditional Saudi hospitality gift; Pritzker replies playfully, 'I think that is code for I love you. Or maybe code for go pound sand. Better check your KSA urban dictionary.' The lavish gift's purpose is undocumented — and unexplained — but it dates the personal MBS-Epstein channel to before Trump's first inauguration.",
        "url": "https://www.cbsnews.com/news/jeffrey-epstein-saudi-arabia/",
    },
    {
        "phrase": "Snow White",
        "pattern": r"snow\s*white",
        "source": "Jul 9 2010 Jes Staley (JPMorgan) → Epstein, 'Snow White / Beauty and the Beast' thread",
        "url": "https://www.bloomberg.com/news/articles/2026-02-04/us-reviewed-allegations-against-staley-black-in-epstein-case",
        "significance": "JPMorgan's senior banker for Epstein, Jes Staley (later Barclays CEO), refers to the young women in Epstein's circle using fairy-tale code names. Bloomberg reporting confirms federal prosecutors reviewed this thread when evaluating whether to charge Staley. This is internal-bank communication, NOT casual social, and demonstrates that Epstein's most senior banker spoke about the trafficked women in a knowing, jocular way.",
    },
    {
        "phrase": "Beauty and the Beast",
        "pattern": r"beauty\s+and\s+the\s+beast",
        "source": "Same 2010 Staley thread + cited by victim 'Carolyn' at Maxwell trial",
        "url": "https://www.bloomberg.com/news/articles/2026-02-04/us-reviewed-allegations-against-staley-black-in-epstein-case",
        "significance": "Used in two contexts: (a) Jes Staley's coded reference in the 2010 thread, and (b) victim 'Carolyn' at the 2021 Maxwell trial describing Epstein's library 'like the one in Beauty and the Beast' — testimony that helped corroborate that she had been inside the residence as a minor. The library description proved hard for the defense to attack because only victims would know it.",
    },
    {
        "phrase": "highly valued friend",
        "pattern": r"highly\s+valued\s+friend",
        "source": "Noam Chomsky character reference (House Oversight 022405)",
        "url": "https://www.npr.org/2025/11/20/nx-s1-5613427/epstein-files-chomsky-bannon-summers-democrats",
        "significance": "Noam Chomsky's signed letter — written WELL AFTER Epstein's 2008 conviction — calls Epstein a 'highly valued friend and regular source of intellectual exchange.' Documents continued elite-academic legitimization of Epstein post-conviction. Chomsky later said he met Epstein only to discuss ideas; this letter contradicts the framing.",
    },
    {
        "phrase": "Invisible Man",
        "pattern": r"invisible\s+man",
        "source": "Aug 18 2001 'The Invisible Man' <abx17@dial.pipex.com> → 'G. Maxwell' <gmax1@mindspring.com>",
        "url": "https://www.cnn.com/2025/12/23/europe/ghislaine-maxwell-email-british-royal-family-latam-intl",
        "significance": "Prince Andrew's pseudonym in Ghislaine Maxwell's address book. The 2001 email signed 'The Invisible Man' (from a UK-ISP Pipex address typical of late-90s royal staff) reads: 'Distraught! You probably wouldn't [know] inappropriate friends [redacted]…' CNN, ITV, and ABC published this as one of the strongest pieces of evidence that 'A' / 'Invisible Man' was Andrew, contradicting his denials.",
    },
    {
        "phrase": "Mohammed bin Salman",
        "pattern": r"mohammed\s+bin\s+salman",
        "source": "2016 House Oversight estate email re Saudi crown prince's gift",
        "url": "https://www.cbsnews.com/news/jeffrey-epstein-donald-trump-emails-house-oversight/",
        "significance": "Documents Epstein's direct relationship with Mohammed bin Salman, including MBS gifting Epstein a Bedouin tent 'carpets and all' that was displayed in Epstein's NY townhouse. MBS visited Epstein in NY post-Khashoggi (2018), per the released emails. One of the strongest foreign-state linkages in the release.",
    },
    {
        "phrase": "Centurion",
        "pattern": r"centurion",
        "source": "American Express Black Card concierge — flight-booking thread (DS11)",
        "url": "https://www.cbsnews.com/news/american-express-epstein-files-doj-trafficking-women-girls/",
        "significance": "Epstein held an AmEx Centurion card (the invitation-only Black Card). His executive assistant Lesley Groff booked all the international flights for the women — under Epstein's Centurion concierge — using AmEx's concierge to manufacture legitimate-looking travel itineraries that doubled as US visa support documents. CBS broke the AmEx-as-trafficking-enabler angle in May 2026.",
    },
    {
        "phrase": "Ruemmler",
        "pattern": r"ruemmler",
        "source": "Kathryn Ruemmler — Goldman Sachs CLO + ex-Obama WH Counsel",
        "url": "https://www.cnbc.com/2025/11/12/trump-jeffrey-epstein-ghislaine-maxwell-emails-house-democrats.html",
        "significance": "Recipient of Epstein's 'I know how dirty donald is' email. Career trajectory: Obama White House Counsel → Latham & Watkins partner → Goldman Sachs Chief Legal Officer. Her presence in the corpus shows Epstein's network reached into top-tier US legal-political circles. She resigned from Goldman after the emails surfaced.",
    },
    {
        "phrase": "Operation Leap Year",
        "pattern": r"operation\s+leap\s+year",
        "source": "Federal Grand Jury 07-103, West Palm Beach FL, May 8 2007 (DS7)",
        "url": "https://www.justice.gov/usao-sdny/programs/victim-witness-services/united-states-v-jeffrey-epstein-19-cr-490-rmb",
        "significance": "The DOJ code name for the 2006-08 federal investigation into Epstein. The grand jury transcripts (DS7) are the most direct record of what federal prosecutors HAD on Epstein before the 2008 plea deal was struck — the deal that AUSA Alex Acosta later defended as 'the best we could get.' These transcripts are the strongest evidence that prosecutors had ample basis to charge Epstein federally in 2008.",
    },
    {
        "phrase": "Free State Reporting",
        "pattern": r"free\s+state\s+reporting",
        "source": "MD-based court reporting firm that transcribed the SDNY Maxwell grand jury (2019-21)",
        "url": "https://www.epsteininvestigation.org/",
        "significance": "Not a person — but their distinctive footer ('D.C. Area 301-261-1902 / Balt. & Annap. 410-974-0947') appears on every page of the SDNY grand-jury transcript, which is why these phone numbers and the words 'Annap' and 'Balt' show up as anomalous high-frequency signals in our PII analysis. Discovering this explains a bunch of false positives in the NER output.",
    },
    {
        "phrase": "Little Saint James",
        "pattern": r"little\s+s(?:aint|t\.?)\s+james|\bLSJ\b",
        "source": "Epstein's USVI private island; site of much of the documented abuse",
        "url": "https://www.cnn.com/2026/03/13/us/jeffrey-epstein-little-st-james-island-invs-vis",
        "significance": "The island is the geographic center of the trafficking. CNN's March 2026 investigation, based on the released LSJ house manual, documented that staff were required to call Epstein 'the Principal' and address him only as 'sir/ma'am.' Most of the named guests (Andrew, Hawking, Krauss, Hoffman, Ito, Musk per schedules) appear in flight logs and security records to/from LSJ.",
    },
    {
        "phrase": "Zorro Ranch",
        "pattern": r"zorro\s+ranch",
        "source": "Epstein's 8,000-acre New Mexico ranch — site of the planned 'baby ranch' eugenics fantasy",
        "url": "https://www.nytimes.com/2019/07/31/business/jeffrey-epstein-eugenics.html",
        "significance": "The NM property is the location of Epstein's documented 'baby ranch' fantasy — he told scientists he wanted to seed the human race using himself, citing Zorro Ranch as the venue. NYT broke this story July 2019. The ranch and its science-elite visitors (Hawking, Gell-Mann, Minsky etc.) are also the bridge between the trafficking operation and the AI/eugenics correspondence we surfaced in the corpus.",
    },
    {
        "phrase": "El Brillo",
        "pattern": r"el\s+brillo",
        "source": "358 El Brillo Way, Palm Beach FL — Epstein's mansion + crime scene of the 2005-08 investigation",
        "url": "https://www.miamiherald.com/news/local/article220097825.html",
        "significance": "The Palm Beach mansion is where the abuse documented by the 2005-08 Palm Beach Police investigation (Det. Joe Recarey, lead detective) took place. The Miami Herald's 'Perversion of Justice' series — which forced the federal re-prosecution — centered on this address. DS4 in our corpus IS the Palm Beach Police case file.",
    },
    {
        "phrase": "Darren awol",
        "pattern": r"darren\s+awol",
        "source": "Apr 30 2019 iMessage (HOUSE_OVERSIGHT_027794)",
        "url": "https://oversightdemocrats.house.gov/news/press-releases/house-oversight-committee-releases-jeffrey-epstein-email-correspondence-raising",
        "significance": "From Epstein's iMessages with a REDACTED counterpart, ~2 months before his July 6 2019 arrest. The 'Darren' is Darren Indyke, Epstein's longtime personal attorney (now estate co-executor). 'Darren awol' = his lawyer is unreachable while they are planning a legal-strategy / PR-rehabilitation campaign that same week. The full thread is one of the most damning windows into Epstein's pre-arrest scrambling.",
    },
    {
        "phrase": "Ranch island. Paris. Harvard",
        "pattern": r"ranch\s+island.{0,3}paris.{0,3}harvard",
        "source": "Epstein iMessage Jun 15 2019",
        "url": "https://oversightdemocrats.house.gov/news/press-releases/house-oversight-committee-releases-jeffrey-epstein-email-correspondence-raising",
        "significance": "21 days before his arrest, Epstein asks his iMessage counterpart for a travel schedule covering Zorro Ranch (NM), Little Saint James (USVI), Paris (his Avenue Foch apartment), and Harvard (where he had ongoing science-philanthropy ties). Demonstrates active multi-property operation right up to the moment of arrest — and that Harvard remained on his itinerary years after the 2008 conviction.",
    },
    {
        "phrase": "Chinese blink on extradition",
        "pattern": r"chinese\s+blink\s+on\s+extradition",
        "source": "Epstein iMessage Jun 15 2019",
        "url": "https://oversightdemocrats.house.gov/news/press-releases/house-oversight-committee-releases-jeffrey-epstein-email-correspondence-raising",
        "significance": "Epstein commenting in iMessage on Trump-era US-China extradition negotiations. Unusual for someone with no public foreign-policy role to be discussing extradition-policy signaling. Together with the 'Pompeo' mentions in the same thread, suggests Epstein was still serving as some kind of informal back-channel up to his arrest.",
    },
    {
        "phrase": "Aka MBS",
        "pattern": r"aka\s+MBS",
        "source": "HOUSE_OVERSIGHT_019874 — explicit decode that 'MBS' = Mohammed bin Salman",
        "url": "https://www.cbsnews.com/news/jeffrey-epstein-donald-trump-emails-house-oversight/",
        "significance": "Our discovery: a doc in the release explicitly spells out 'Crown Prince of the House of Saud, Mohammed bin Salman bin Abdulaziz Al Saud, age thirty-one. Aka MBS.' Confirms that the 'MBS' initials in other emails (e.g., the Apr 2018 'HBJ sun. MBS mon.' iMessage) refer to MBS the Saudi crown prince — not someone else.",
    },
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('corpus_jsonl')
    ap.add_argument('out_dir')
    ap.add_argument('--ctx-chars', type=int, default=160)
    ap.add_argument('--max-samples', type=int, default=3)
    args = ap.parse_args()
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)

    accum = {}
    for p in PHRASES:
        accum[p['phrase']] = {
            'phrase': p['phrase'],
            'pattern': p['pattern'],
            'source': p['source'],
            'url': p.get('url', ''),
            'significance': p.get('significance', ''),
            'n_total_hits': 0,
            'doc_ids': set(),
            'samples': [],
        }

    compiled = [(p['phrase'], re.compile(p['pattern'], re.IGNORECASE)) for p in PHRASES]

    with open(args.corpus_jsonl) as f:
        for line in f:
            try: rec = json.loads(line)
            except: continue
            text = rec.get('text', '')
            if not text: continue
            ds = rec.get('dataset', '?')
            doc_id = rec.get('id', '?')
            for phrase, rx in compiled:
                slot = accum[phrase]
                for m in rx.finditer(text):
                    slot['n_total_hits'] += 1
                    slot['doc_ids'].add(doc_id)
                    if len(slot['samples']) < args.max_samples:
                        start = max(0, m.start() - args.ctx_chars)
                        end = min(len(text), m.end() + args.ctx_chars)
                        ctx = text[start:end]
                        # Light cleanup
                        ctx = re.sub(r'\s+', ' ', ctx).strip()
                        slot['samples'].append({
                            'doc_id': doc_id,
                            'dataset': ds,
                            'context': ctx,
                            'matched': text[m.start():m.end()],
                        })

    out_data = {
        'phrases': [
            {
                'phrase': v['phrase'],
                'source': v['source'],
                'url': v.get('url', ''),
                'significance': v.get('significance', ''),
                'n_total_hits': v['n_total_hits'],
                'n_docs': len(v['doc_ids']),
                'samples': v['samples'],
            }
            for k, v in accum.items()
        ],
    }
    with open(out / 'quotes.json', 'w', encoding='utf-8') as f:
        json.dump(out_data, f, indent=2, ensure_ascii=False)

    print(f'phrases evaluated: {len(PHRASES)}')
    for p in out_data['phrases']:
        sym = '✓' if p['n_total_hits'] else '✗'
        print(f"  {sym} {p['n_total_hits']:>4} hits in {p['n_docs']:>3} docs  :: {p['phrase']}")


if __name__ == '__main__':
    main()
