

from __future__ import annotations
import re, time, json, requests, feedparser
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime, timedelta
import pandas as pd
import yfinance as yf
try:
    from bs4 import BeautifulSoup
except Exception:
    BeautifulSoup = None

# =================== Config ===================

TRIAL_HINTS = [
    "first patient dosed", "first patient has been dosed", "initiates dosing", "doses first patient",
    "ind cleared", "ind accepted", "fda clears ind", "opens ind", "investigational new drug",
    "enrollment begins", "begin enrollment", "patient enrollment", "site initiated", "site activation",
    "topline", "top-line", "readout", "interim analysis",
    "fast track", "orphan drug", "breakthrough therapy",
    "collaboration", "licensing", "option deal", "co-development",
    "manufacturing agreement", "cdmo", "supply agreement",
    "enrolment begins", "begin enrolment", "patient enrolment"
]
TRIAL_HINTS_SET = {h.lower() for h in TRIAL_HINTS}

ACTIVE_CT_STATUSES = {
    "RECRUITING", "NOT_YET_RECRUITING", "ENROLLING_BY_INVITATION", "ACTIVE_NOT_RECRUITING"
}

PAT_ALPHANUM_CODE = re.compile(r"\b[A-Z]{2,}[A-Z0-9\-]{1,}[0-9]{1,}[A-Za-z0-9\-]*\b")
PAT_HYPHEN       = re.compile(r"\b[A-Za-z0-9]+(?:-[A-Za-z0-9]+)+\b")
PAT_CAPWORD      = re.compile(r"\b([A-Z][A-Za-z]{2,})\b")
COMMON_STOP = set("""
News Release Update Announces Limited Study Trial Safety Data Results Patient Cancer Solid
Immunotherapy Therapeutics Therapeutic Oncology Vaccine Viral Virus Company Limited Ltd Inc Plc
Investors Clinical Interim Poster Abstract Shares Announces Presents
""".split())
ALIAS_MIN_LEN = 3

INDUSTRY_MATCH = ["biotech", "biotechnology", "biotechnology & medical research", "life sciences"]

# =================== Time helpers ===================

def _iso(dt) -> Optional[str]:
    if not dt: return None
    if isinstance(dt, str): return dt
    try: return dt.isoformat()
    except Exception: return None

def _parse_dt(dstruct) -> Optional[datetime]:
    try: return datetime(*dstruct[:6])
    except Exception: return None

def _days_ago(iso_str: Optional[str]) -> Optional[int]:
    if not iso_str: return None
    try: return (datetime.utcnow() - datetime.fromisoformat(iso_str.replace("Z",""))).days
    except Exception: return None

# =================== Alias discovery ===================

def _ctgov_fetch_basic(company: str, page_size: int = 120) -> List[Dict[str, Any]]:
    url = "https://clinicaltrials.gov/api/v2/studies"
    expr = (
        f"AREA[LeadSponsorName]{company} OR "
        f"AREA[OrganizationFullName]{company} OR "
        f"AREA[BriefTitle]{company} OR "
        f"AREA[OfficialTitle]{company} OR "
        f"AREA[SponsorCollaboratorsModule.LeadSponsor.Name]{company} OR "
        f"AREA[SponsorCollaboratorsModule.Collaborators.Name]{company}"
    )
    body = {
        "query": {"expr": expr},
        "fields": [
            "NCTId","BriefTitle","OfficialTitle",
            "ArmsInterventionsModule.Interventions.Name",
            "SponsorCollaboratorsModule.LeadSponsor.Name",
            "SponsorCollaboratorsModule.Collaborators.Name"
        ],
        "pageSize": page_size
    }
    try:
        r = requests.post(url, json=body, timeout=25)
        r.raise_for_status()
        return r.json().get("studies", []) or []
    except Exception:
        return []

def _rss_fetch(company: str, ticker: str, days: int = 365, max_items: int = 80) -> List[Dict[str, Any]]:
    cutoff = datetime.utcnow() - timedelta(days=days)
    q = f'"{company}" OR {ticker} (trial OR study OR dosing OR enrollment OR topline OR readout OR FDA OR EMA OR TGA OR IND)'
    url = f"https://news.google.com/rss/search?q={requests.utils.quote(q)}&hl=en-US&gl=US&ceid=US:en"
    feed = feedparser.parse(url)
    out = []
    for e in feed.entries[: max_items]:
        dt = getattr(e, "published_parsed", None) or getattr(e, "updated_parsed", None)
        try: dt = datetime(*dt[:6]) if dt else None
        except Exception: dt = None
        if dt and dt < cutoff: continue
        out.append({"title": getattr(e, "title", ""), "link": getattr(e, "link", ""), "date": dt})
    return out

def _tokenize_aliases_from_text(title: str) -> List[str]:
    txt = (title or "").replace("–","-").replace("—","-")
    cand = set()
    cand.update(PAT_ALPHANUM_CODE.findall(txt))
    cand.update(PAT_HYPHEN.findall(txt))
    cand.update(PAT_CAPWORD.findall(txt))
    out = []
    for t in cand:
        if len(t) < ALIAS_MIN_LEN: continue
        if t.lower() in (w.lower() for w in COMMON_STOP): continue
        out.append(t)
    return out

def discover_aliases(company: str, ticker: str, days: int = 365) -> List[str]:
    studies = _ctgov_fetch_basic(company)
    ct_aliases = []
    for s in studies:
        p = s.get("protocolSection", {}) or {}
        ident = p.get("identificationModule", {}) or {}
        title = " ".join(filter(None, [ident.get("briefTitle") or "", ident.get("officialTitle") or ""]))
        intervs = ((p.get("armsInterventionsModule") or {}).get("interventions") or [])
        ct_aliases += [i.get("name","") for i in intervs if i.get("name")]
        ct_aliases += _tokenize_aliases_from_text(title)

    headlines = _rss_fetch(company, ticker, days=days, max_items=80)

    freq_ct, freq_news, hint_news = {}, {}, {}
    for a in ct_aliases:
        if a: freq_ct[a] = freq_ct.get(a, 0) + 1
    for h in headlines:
        title = (h.get("title") or "")
        toks = _tokenize_aliases_from_text(title)
        blob = title.lower()
        has_hint = any(k in blob for k in TRIAL_HINTS_SET)
        for t in toks:
            if company.lower() in t.lower():
                continue
            freq_news[t] = freq_news.get(t, 0) + 1
            if has_hint:
                hint_news[t] = hint_news.get(t, 0) + 1

    def is_code_like(term: str) -> bool:
        return bool(PAT_ALPHANUM_CODE.fullmatch(term) or PAT_HYPHEN.fullmatch(term))

    scored = []
    for k in set(freq_ct) | set(freq_news):
        if not is_code_like(k) and (freq_news.get(k, 0) < 2 or hint_news.get(k, 0) < 1):
            continue
        s = 1.5*freq_ct.get(k,0) + 1.0*freq_news.get(k,0) + 0.5*hint_news.get(k,0)
        scored.append((k, s))
    scored.sort(key=lambda x: x[1], reverse=True)

    aliases, seen = [], set()
    for term, _s in scored:
        tl = term.lower()
        if tl in seen: continue
        if company.lower().split()[0] in tl: continue
        seen.add(tl)
        aliases.append(term)
        if len(aliases) >= 12: break
    return aliases

# =================== Enhanced sources ===================
def asx_announcements(ticker: str, days: int = 365, max_items: int = 60) -> List[Dict[str, Any]]:
    """
    Fetch ASX company announcements for an .AX ticker (primary source).
    1) Try MarkitDigital JSON endpoint
    2) Fallback: scrape ASX announcements page (if BeautifulSoup available)
    Output schema: [{source_type:'asx_announcement', title, url, date}, ...]
    """
    if not ticker or not str(ticker).upper().endswith(".AX"):
        return []
    code = str(ticker).upper().replace(".AX", "")
    cutoff = datetime.utcnow() - timedelta(days=days)
    out: List[Dict[str, Any]] = []

    # Attempt 1: MarkitDigital JSON
    try:
        url = f"https://asx.api.markitdigital.com/asx-research/1.0/companies/{code}/announcements"
        # Some mirrors require a UA
        headers = {"User-Agent": "Mozilla/5.0"}
        r = requests.get(url, headers=headers, timeout=20)
        if r.ok:
            data = r.json()
            items = (data.get("data") or {}).get("items") or data.get("items") or []
            for it in items:
                # fields vary by mirror; be defensive
                title = it.get("headline") or it.get("title") or ""
                dtstr = it.get("published") or it.get("date") or it.get("releaseDate")
                link  = it.get("url") or it.get("pdfUrl") or it.get("webUrl")
                # Parse date if possible
                dt = None
                if isinstance(dtstr, str):
                    try:
                        # Common formats: "2024-05-01T23:59:00+10:00" or "2024-05-01"
                        dt = datetime.fromisoformat(dtstr.replace("Z","+00:00")) \
                             if "T" in dtstr else datetime.fromisoformat(dtstr + "T00:00:00")
                    except Exception:
                        dt = None
                if dt and dt < cutoff:
                    continue
                if not title:
                    continue
                out.append({
                    "source_type": "asx_announcement",
                    "title": title.strip(),
                    "url": (link or "").strip(),
                    "date": _iso(dt)
                })
                if len(out) >= max_items:
                    break
    except Exception:
        pass

    # Attempt 2: HTML fallback (scrape)
    if not out and BeautifulSoup is not None:
        try:
            # Classic announcements listing
            url = f"https://www.asx.com.au/asx/v2/statistics/announcements.do?by=asxCode&asxCode={code}"
            headers = {"User-Agent": "Mozilla/5.0"}
            r = requests.get(url, headers=headers, timeout=20)
            if r.ok:
                soup = BeautifulSoup(r.text, "html.parser")
                # This page is table-based; look for rows with links to PDFs/details
                rows = soup.find_all("tr")
                for tr in rows:
                    a = tr.find("a")
                    if not a or not a.text:
                        continue
                    title = a.text.strip()
                    # Date often in sibling <td> elements
                    tds = tr.find_all("td")
                    dt = None
                    if len(tds) >= 2:
                        # Try parse first/second TD as date
                        for td in tds[:2]:
                            s = (td.text or "").strip()
                            for fmt in ("%d/%m/%Y", "%d %b %Y", "%Y-%m-%d"):
                                try:
                                    dt = datetime.strptime(s, fmt)
                                    break
                                except Exception:
                                    continue
                            if dt:
                                break
                    if dt and dt < cutoff:
                        continue
                    href = a.get("href") or ""
                    link = href if href.startswith("http") else ("https://www.asx.com.au" + href)
                    out.append({
                        "source_type": "asx_announcement",
                        "title": title,
                        "url": link,
                        "date": _iso(dt)
                    })
                    if len(out) >= max_items:
                        break
        except Exception:
            pass

    return out


def _google_news_rss_url(query: str, region: str = "US") -> str:
    """
    Build a Google News RSS URL with regional bias.
    region: "US" or "AU" (extend as needed).
    """
    region = (region or "US").upper()
    if region == "AU":
        return f"https://news.google.com/rss/search?q={requests.utils.quote(query)}&hl=en-AU&gl=AU&ceid=AU:en"
    return f"https://news.google.com/rss/search?q={requests.utils.quote(query)}&hl=en-US&gl=US&ceid=US:en"


def ctgov_enriched(company: str, aliases: List[str], page_size: int = 180) -> List[Dict[str, Any]]:
    url = "https://clinicaltrials.gov/api/v2/studies"
    terms = []
    for a in {company, *aliases}:
        a = a.strip()
        terms += [
            f"AREA[LeadSponsorName]{a}",
            f"AREA[OrganizationFullName]{a}",
            f"AREA[SponsorCollaboratorsModule.LeadSponsor.Name]{a}",
            f"AREA[SponsorCollaboratorsModule.Collaborators.Name]{a}",
            f"AREA[InterventionName]{a}",
            f"AREA[BriefTitle]{a}",
            f"AREA[OfficialTitle]{a}"
        ]
    expr = " OR ".join(terms)
    body = {
        "query": {"expr": expr},
        "fields": [
            "NCTId","BriefTitle","OfficialTitle","OverallStatus","Phase",
            "StartDate","PrimaryCompletionDate","StudyType",
            "SponsorCollaboratorsModule.LeadSponsor.Name",
            "SponsorCollaboratorsModule.Collaborators.Name",
            "ArmsInterventionsModule.Interventions.Name",
            "ContactsLocationsModule.Locations.Facility"
        ],
        "pageSize": page_size
    }
    try:
        r = requests.post(url, json=body, timeout=25)
        r.raise_for_status()
        out = []
        for s in r.json().get("studies", []) or []:
            p = s.get("protocolSection", {}) or {}
            ident = p.get("identificationModule", {}) or {}
            status = p.get("statusModule", {}) or {}
            design = p.get("designModule", {}) or {}
            sponsor = p.get("sponsorCollaboratorsModule", {}) or {}
            intervs = ((p.get("armsInterventionsModule") or {}).get("interventions") or [])
            locs = ((p.get("contactsLocationsModule") or {}).get("locations") or [])
            phases = design.get("phases", [])
            out.append({
                "source_type": "ctgov",
                "NCTId": (ident.get("nctId")),
                "BriefTitle": ident.get("briefTitle"),
                "OfficialTitle": ident.get("officialTitle"),
                "OverallStatus": (status.get("overallStatus") or "").upper(),
                "Phase": ",".join(phases) if isinstance(phases, list) else (phases or ""),
                "StartDate": (status.get("startDateStruct") or {}).get("date"),
                "PrimaryCompletionDate": (status.get("primaryCompletionDateStruct") or {}).get("date"),
                "StudyType": (design.get("studyType") or "Interventional"),
                "LeadSponsorName": (sponsor.get("leadSponsor") or {}).get("name"),
                "Collaborators": [c.get("name") for c in (sponsor.get("collaborators") or [])],
                "Interventions": [i.get("name") for i in intervs if i.get("name")],
                "Locations": [loc.get("facility") for loc in locs if loc.get("facility")],
                "url": f"https://clinicaltrials.gov/study/{ident.get('nctId')}" if ident.get("nctId") else None,
                "date": (status.get("primaryCompletionDateStruct") or {}).get("date") \
                        or (status.get("startDateStruct") or {}).get("date")
            })
        return out
    except Exception:
        return []

def news_rss(company: str, ticker: str, aliases: List[str], days: int = 365, max_items: int = 60) -> List[Dict[str, Any]]:
    cutoff = datetime.utcnow() - timedelta(days=days)
    base_terms = [f'"{company}"', ticker] + [f'"{a}"' for a in (aliases or [])]
    q = " OR ".join([t for t in base_terms if t])
    q += " (trial OR study OR dosing OR dose OR enrollment OR enrolment OR topline OR top-line OR readout OR interim OR IND OR FDA OR EMA OR TGA OR \"first patient\")"

    region = "AU" if str(ticker).upper().endswith(".AX") else "US"
    url = _google_news_rss_url(q, region=region)

    feed = feedparser.parse(url)
    must_terms = [company.lower(), ticker.lower()] + [a.lower() for a in (aliases or [])]

    out = []
    for e in feed.entries[: max_items * 3]:
        dt = _parse_dt(getattr(e, "published_parsed", None)) or _parse_dt(getattr(e, "updated_parsed", None))
        if dt and dt < cutoff:
            continue
        title = (getattr(e, "title", "") or "").strip()
        if not title:
            continue
        tlow  = title.lower()
        # keep headlines that mention the company/ticker/alias somewhere
        if not any(term in tlow for term in must_terms):
            continue
        link  = (getattr(e, "link", "") or "").strip()
        stype = "press_release" if (company.lower().split()[0] in (link or "").lower()) else "news"
        out.append({
            "source_type": stype,
            "title": title,
            "url": link,
            "date": _iso(dt)
        })
        if len(out) >= max_items:
            break
    return out


# =================== Activity scoring ===================

def _headline_signals(title: str) -> List[str]:
    t = (title or "").lower()
    return [h for h in TRIAL_HINTS_SET if h in t]

def score_activity(company: str,
                   ctgov_items: List[Dict[str, Any]],
                   rss_items: List[Dict[str, Any]],
                   recency_days: int = 540) -> Tuple[float, List[str], List[str]]:
    reasons, score, titles = [], 0.0, []

    # CT.gov signals
    ct_active = ct_fresh = ct_newreg = ct_multisite = 0
    for e in ctgov_items:
        if e.get("source_type") != "ctgov": continue
        if (e.get("StudyType") or "Interventional").lower().startswith("inter"):
            status = (e.get("OverallStatus") or "").upper()
            if status in ACTIVE_CT_STATUSES: ct_active += 1
            d = e.get("date") or e.get("StartDate")
            if d:
                da = _days_ago(d)
                if da is not None and da <= recency_days:
                    ct_fresh += 1
                    if da <= 365: ct_newreg += 1
            sites = e.get("Locations") or []
            if isinstance(sites, list) and len(sites) >= 2: ct_multisite += 1

    if ct_active:   reasons.append(f"{ct_active} active interventional trial(s) on CT.gov"); score += min(0.35, 0.15 * ct_active)
    if ct_fresh:    reasons.append(f"{ct_fresh} trial(s) with recent dates (≤{recency_days}d)"); score += min(0.35, 0.12 * ct_fresh)
    if ct_newreg:   reasons.append(f"{ct_newreg} newly registered/updated trial(s) (≤365d)"); score += min(0.25, 0.10 * ct_newreg)
    if ct_multisite:reasons.append(f"{ct_multisite} multi-site trial(s)"); score += min(0.20, 0.05 * ct_multisite)

    # RSS signals
    buckets = dict(first_patient=0.0, ind=0.0, enrollment=0.0, readout=0.0, reg_design=0.0, partnership=0.0, mfg=0.0)
    rss_hits = 0
    for r in rss_items:
        title = r.get("title") or ""
        if not title: continue
        sigs = _headline_signals(title)
        if not sigs: continue
        rss_hits += 1; titles.append(title)
        tl = title.lower()
        if any(k in tl for k in ["first patient dosed", "doses first patient", "initiates dosing"]): buckets["first_patient"] += 0.25
        if any(k in tl for k in ["ind cleared", "ind accepted", "fda clears ind", "investigational new drug"]): buckets["ind"] += 0.25
        if any(k in tl for k in ["enrollment begins", "begin enrollment", "patient enrollment", "site activation", "site initiated"]): buckets["enrollment"] += 0.15
        if any(k in tl for k in ["topline", "top-line", "readout", "interim analysis"]): buckets["readout"] += 0.15
        if any(k in tl for k in ["fast track", "orphan drug", "breakthrough therapy"]): buckets["reg_design"] += 0.15
        if any(k in tl for k in ["collaboration", "licensing", "option deal", "co-development"]): buckets["partnership"] += 0.10
        if any(k in tl for k in ["manufacturing agreement", "cdmo", "supply agreement"]): buckets["mfg"] += 0.10

    caps = dict(first_patient=0.5, ind=0.5, enrollment=0.3, readout=0.3, reg_design=0.3, partnership=0.2, mfg=0.2)
    for k, v in buckets.items():
        if v > 0: score += min(v, caps[k])

    if rss_hits: reasons.append(f"{rss_hits} headline(s) with trial-related action")

    score = max(0.0, min(1.8, score))
    titles = titles[:6]
    return score, reasons, titles

def label_from_score(score: float) -> str:
    if score >= 0.9: return "High"
    if score >= 0.5: return "Medium"
    return "Low"

# =================== OpenAI blurb (optional) ===================

def openai_blurb(company: str,
                 ticker: str,
                 key_reasons: List[str],
                 ctgov_items: List[Dict[str, Any]],
                 rss_items: List[Dict[str, Any]],
                 api_key: Optional[str] = None,
                 model: str = "gpt-4o-mini",
                 max_titles: int = 8,
                 max_words: int = 140) -> Dict[str, Any]:
    """
    Generate a short blurb grounded in our evidence.
    Returns: { blurb: str, citations: [titles], confidence: float }
    """
    guess = False
    if not api_key:
        return {"blurb": "", "citations": [], "confidence": 0.0}

    # Build compact evidence lists
    ct_titles = []
    for e in ctgov_items:
        if e.get("source_type") != "ctgov": continue
        t = e.get("BriefTitle") or e.get("OfficialTitle")
        if t: ct_titles.append(t)
    ct_titles = ct_titles[: max(2, max_titles//2)]

    rss_titles = [r.get("title","") for r in rss_items if r.get("title")] \
                 [: max(2, max_titles - len(ct_titles))]

    evidence_titles = [t for t in (ct_titles + rss_titles) if t][:max_titles]
    if not evidence_titles:
        guess = True
        

    # Prompt
    if guess == False:
        prompt = (
            f"Company: {company} ({ticker})\n"
            f"Key signals: {', '.join(key_reasons) if key_reasons else '(none)'}\n\n"
            "Evidence titles (use these only, do not invent anything):\n"
            + "\n".join([f"- {t}" for t in evidence_titles]) + "\n\n"
            f"Task: Write a concise, neutral {max_words}-word blurb summarizing the company's CURRENT "
            "drug-development activity, focusing on *what happened* (trial starts/dosing/enrollment, IND, "
            "readouts, Fast Track/Orphan, partnerships). Avoid phases unless explicit in the titles. "
            "No hype, no forward-looking claims.\n\n"
            "Output STRICT JSON with keys: {\"blurb\": str, \"citations\": [titles you relied on], \"confidence\": number 0..1}."
        )
    else:
        prompt = (
            f"Company: {company} ({ticker})\n"
            f"Key signals: {', '.join(key_reasons) if key_reasons else '(none)'}\n\n"
            f"Task: Write a concise, neutral {max_words}-word blurb summarizing the company's CURRENT "
            "drug-development activity, focusing on *what happened* (trial starts/dosing/enrollment, IND, "
            "readouts, Fast Track/Orphan, partnerships)."
            "If you cannot find any reliable sources do not make anything up, please just state \"No available reliable evidence for current development process for (company)\""
            "No hype, no forward-looking claims and you are a guess so place at the end of your response (!GUESS!).\n\n"
            "Output STRICT JSON with keys: {\"blurb\": str, \"citations\": [sources you relied on], \"confidence\": number 0..1}."
        )
        
    try:
        from openai import OpenAI
        client = OpenAI(api_key=api_key)
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role":"user","content":prompt}],
            temperature=0.0,
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "Blurb",
                    "schema": {
                        "type":"object",
                        "properties":{
                            "blurb":{"type":"string"},
                            "citations":{"type":"array","items":{"type":"string"}},
                            "confidence":{"type":"number","minimum":0,"maximum":1}
                        },
                        "required":["blurb","citations","confidence"],
                        "additionalProperties": False
                    }
                }
            }
        )
        data = json.loads(resp.choices[0].message.content)
        print(f"OpenAI output {data}")
        # Guard: strip citations to only those we supplied
        supplied = set(evidence_titles)
        data["citations"] = [c for c in data.get("citations", []) if c in supplied]
        # Trim blurb length roughly
        words = data["blurb"].split()
        if len(words) > max_words:
            data["blurb"] = " ".join(words[:max_words]) + "…"
        return data
    except Exception as e:
        return {"blurb": "", "citations": [], "confidence": 0.0}

# =================== Company analysis ===================

def analyze_company_activity(company: str, ticker: str,
                             days: int = 365,
                             recency_days: int = 540,
                             api_key: Optional[str] = None,
                             gpt_model: str = "gpt-4o-mini",
                             want_blurb: bool = True) -> Dict[str, Any]:
    print(f"Analyzing {company} ({ticker})")
    aliases = discover_aliases(company, ticker, days=days)

    # Primary registry (CT.gov) — still useful for many US names; some AU names will be absent
    ct_items = ctgov_enriched(company, aliases, page_size=200)

    # Secondary sources
    rss_items = news_rss(company, ticker, aliases, days=days, max_items=60)

    # ASX primary announcements for .AX tickers
    asx_items = asx_announcements(ticker, days=days, max_items=60) if str(ticker).upper().endswith(".AX") else []

    # Merge (preserve fields your scorer expects)
    merged_rss = []
    seen = set()
    for lst in [asx_items, rss_items]:
        for it in lst:
            key = (it.get("title","").strip(), it.get("url","").strip())
            if key in seen:
                continue
            seen.add(key)
            # Normalize source_type for scoring compatibility: treat ASX announcements like 'press_release'
            norm = dict(it)
            if norm.get("source_type") == "asx_announcement":
                norm["source_type"] = "press_release"
            merged_rss.append(norm)

    # Score & label
    score, reasons, titles = score_activity(company, ct_items, merged_rss, recency_days=recency_days)
    label = label_from_score(score)

    # Blurb
    blurb_data = {"blurb":"", "citations":[], "confidence":0.0}
    if want_blurb:
        blurb_data = openai_blurb(
            company, ticker, reasons, ct_items, merged_rss,
            api_key=api_key, model=gpt_model, max_titles=8, max_words=140
        )

    return {
        "company": company,
        "ticker": ticker,
        "aliases": aliases,
        "activity_score": round(score, 3),
        "label": label,
        "key_reasons": reasons,
        "evidence_titles": titles,
        "blurb": blurb_data.get("blurb",""),
        "blurb_citations": blurb_data.get("citations", []),
        "blurb_confidence": blurb_data.get("confidence", 0.0),
        "ctgov_count": sum(1 for x in ct_items if x.get("source_type") == "ctgov"),
        "rss_count": len(merged_rss),
        "ctgov_items": ct_items,
        "rss_items": merged_rss,
    }


# =================== yfinance helpers ===================

def fetch_meta(tickers: List[str], sleep_between: float = 0.25) -> pd.DataFrame:
    rows = []
    for t in tickers:
        tk = yf.Ticker(t)
        info = {}
        try:
            info = tk.get_info()
        except Exception:
            try:
                info = tk.info
            except Exception:
                info = {}
        rows.append({
            "ticker": t,
            "name": info.get("shortName") or info.get("longName") or t,
            "industry": info.get("industry") or "",
            "sector": info.get("sector") or "",
            "market_cap": info.get("marketCap") or info.get("enterpriseValue")
        })
        if sleep_between:
            time.sleep(sleep_between)
    return pd.DataFrame(rows)

def _is_biotech(industry: str, sector: str) -> bool:
    s = f"{(industry or '').lower()} {(sector or '').lower()}"
    return any(k in s for k in INDUSTRY_MATCH)

def filter_biotech_cap(df: pd.DataFrame,
                       min_cap: Optional[float] = None,
                       max_cap: Optional[float] = None) -> pd.DataFrame:
    mask = df.apply(lambda r: _is_biotech(r.get("industry",""), r.get("sector","")), axis=1)
    out = df[mask].copy()
    if min_cap is not None:
        out = out[out["market_cap"].fillna(0) >= float(min_cap)]
    if max_cap is not None:
        out = out[out["market_cap"].fillna(0) <= float(max_cap)]
    return out.sort_values("market_cap", ascending=True)

# =================== Screening over a list ===================

def screen_activity_for_tickers(tickers: List[str],
                                min_cap: Optional[float] = None,
                                max_cap: Optional[float] = None,
                                days: int = 365,
                                recency_days: int = 540,
                                sleep_between: float = 0.0,
                                api_key: Optional[str] = None,
                                gpt_model: str = "gpt-4o-mini",
                                want_blurb: bool = True) -> pd.DataFrame:
    meta = fetch_meta(tickers)
    meta_f = filter_biotech_cap(meta, min_cap=min_cap, max_cap=max_cap)
    if meta_f.empty:
        return pd.DataFrame(columns=["ticker","name","market_cap","activity_score","label","blurb","blurb_citations"])

    records = []
    for _, r in meta_f.iterrows():
        name = r["name"]; tkr = r["ticker"]
        try:
            print(f"Trying analysis for {tkr}")
            res = analyze_company_activity(
                name, tkr, days=days, recency_days=recency_days,
                api_key=api_key, gpt_model=gpt_model, want_blurb=want_blurb
            )
        except Exception as e:
            print(f"Analysis failed for {tkr}")
            res = {
                "company": name, "ticker": tkr, "aliases": [],
                "activity_score": 0.0, "label": "Low",
                "key_reasons": [f"Error: {e}"], "evidence_titles": [],
                "blurb":"", "blurb_citations": [], "blurb_confidence": 0.0,
                "ctgov_count": 0, "rss_count": 0, "ctgov_items": [], "rss_items": []
            }

        records.append({
            "ticker": tkr,
            "name": name,
            "market_cap": r.get("market_cap"),
            "activity_score": res["activity_score"],
            "label": res["label"],
            "key_reasons": json.dumps(res["key_reasons"], ensure_ascii=False),
            "evidence_titles": json.dumps(res["evidence_titles"], ensure_ascii=False),
            "blurb": res["blurb"],
            "blurb_citations": json.dumps(res["blurb_citations"], ensure_ascii=False),
            "blurb_confidence": res["blurb_confidence"],
            "ctgov_count": res["ctgov_count"],
            "rss_count": res["rss_count"],
        })
        if sleep_between:
            time.sleep(sleep_between)

    cols = ["ticker","name","market_cap","activity_score","label",
            "blurb","blurb_citations","blurb_confidence",
            "key_reasons","evidence_titles","ctgov_count","rss_count"]
    return pd.DataFrame(records)[cols].sort_values("activity_score", ascending=False)