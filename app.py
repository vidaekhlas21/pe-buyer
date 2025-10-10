import json, os, time, re
from typing import List, Optional, Literal, Dict
import requests
from bs4 import BeautifulSoup
import streamlit as st
from pydantic import BaseModel, Field, ValidationError

# ----------------- Config -----------------
OPENAI_API_KEY = st.secrets.get("OPENAI_API_KEY", "")
if not OPENAI_API_KEY:
    st.warning("Missing OPENAI_API_KEY in secrets. Go to Settings → Secrets in Streamlit Cloud to add it.")

MODEL_EXTRACT = "gpt-4o-mini"
MODEL_MATCH   = "gpt-4o"
TIMEOUT = 12
HEADERS = {"User-Agent": "PE-Matcher-Prototype/1.0"}
COMMON_PATHS = ["/", "/about", "/services", "/products"]

# Load optional external catalog if present, else fall back to a tiny in-file seed
def load_catalog():
    try:
        with open("fund_catalog.json", "r") as f:
            return json.load(f)
    except Exception:
        return [
            {
              "name": "Harbor Ridge Capital",
              "sectors": ["business services","industrial"],
              "subsector_keywords": ["HVAC","plumbing","facility services","route based"],
              "deal_types": ["control","platform","add on"],
              "check_size_usd_min": 15000000,
              "check_size_usd_max": 100000000,
              "revenue_floor_usd": 8000000,
              "ebitda_floor_usd": 2000000,
              "geographies": ["US","Canada"],
              "thesis_notes": "Recurring revenue field services and route based models",
              "example_deals": ["AirPro Service Group","CleanTech Facilities Partners"]
            },
            {
              "name": "Northbank Partners",
              "sectors": ["software","business services"],
              "subsector_keywords": ["vertical SaaS","workflow","field service software"],
              "deal_types": ["majority","minority growth","platform"],
              "check_size_usd_min": 10000000,
              "check_size_usd_max": 75000000,
              "revenue_floor_usd": 5000000,
              "ebitda_floor_usd": 1000000,
              "geographies": ["US","UK","EU"],
              "thesis_notes": "Vertical SaaS with B2B focus and retention strong",
              "example_deals": ["ShopTrack Cloud","RigOps Software"]
            },
            {
              "name": "MedVista Equity",
              "sectors": ["healthcare"],
              "subsector_keywords": ["dental","vet","MSO","clinic","ambulatory"],
              "deal_types": ["platform","add on","control"],
              "check_size_usd_min": 20000000,
              "check_size_usd_max": 150000000,
              "revenue_floor_usd": 10000000,
              "ebitda_floor_usd": 2000000,
              "geographies": ["US"],
              "thesis_notes": "Multi site provider services rollups",
              "example_deals": ["SmileWell Dental","PawsCare Vet Group"]
            }
        ]

FUND_CATALOG = load_catalog()

# ----------------- Schemas -----------------
class CompanyFacts(BaseModel):
    canonical_url: str
    company_name: Optional[str] = None
    industry: Optional[Literal["software","healthcare services","consumer","industrials","business services","fintech","other"]] = None
    subsector: Optional[str] = None
    hq_city: Optional[str] = None
    hq_country: Optional[str] = None
    operating_regions: Optional[List[str]] = None
    estimated_employees: Optional[int] = None
    estimated_revenue_usd: Optional[float] = None
    business_model: Optional[Literal["B2B","B2C","B2B2C","marketplace","other"]] = None
    offerings: Optional[List[str]] = None
    customer_segments: Optional[List[str]] = None
    growth_signals: Optional[List[str]] = None
    keywords: Optional[List[str]] = None
    confidence_notes: Optional[str] = None
    evidence: Optional[Dict[str, str]] = None  # field -> snippet

class CriteriaMatches(BaseModel):
    sector: bool
    subsector: bool
    geography: bool
    size: Literal["likely","uncertain","mismatch"]
    deal_type: bool

class MatchResult(BaseModel):
    fund: str
    fit_score: int = Field(ge=0, le=100)
    match_reasons: List[str]
    deal_angle: str
    criteria_matches: CriteriaMatches

# ----------------- LLM helpers -----------------
def openai_chat(model: str, system: str, user: str, temperature: float = 0.0) -> str:
    import openai
    openai.api_key = OPENAI_API_KEY
    resp = openai.ChatCompletion.create(
        model=model,
        temperature=temperature,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user}
        ]
    )
    return resp.choices[0].message["content"]

PROMPT_A_SYS = (
    "You are an analyst. Extract company facts from provided website text. "
    "Return STRICT JSON that matches the CompanyFacts schema. "
    "If a field is unknown, use null. Prefer unknown to guessing. "
    "Include an 'evidence' object that maps field names to short supporting snippets. "
    "Return only JSON."
)

def prompt_a_user(canonical_url: str, text: str) -> str:
    truncated = text[:12000]
    schema_hint = json.dumps(CompanyFacts.model_json_schema(), indent=2)
    return f"""
Website: {canonical_url}

Schema to follow:
{schema_hint}

Text:
{truncated}
"""


PROMPT_B_SYS = (
    "You are a private equity associate. Given one CompanyFacts object and a fund catalog chunk, "
    "score fund fit conservatively. Base score on sector and subsector and geography. "
    "Adjust for size and deal type. Return ONLY a JSON array of MatchResult objects sorted by fit_score. "
    "If sector or subsector do not match, score should be low. "
    "Do not invent numbers. "
    "Return exactly and only a valid JSON array. No prose, no code fences."
)


def prompt_b_user(company_json: str, catalog_chunk: str) -> str:
    schema_hint = json.dumps(MatchResult.model_json_schema(), indent=2)
    return f"""
CompanyFacts:
{company_json}

FundCatalogChunk:
{catalog_chunk}

MatchResult JSON schema:
{schema_hint}

Return only a JSON array.
"""
def parse_json_array_or_raise(text: str):
    """Tolerant parser that extracts the first JSON array from text."""
    import json
    t = (text or "").strip().strip("`").strip()
    # grab the first [...] block
    start = t.find('[')
    end = t.rfind(']')
    if start == -1 or end == -1 or end <= start:
        raise ValueError("No JSON array found in model response")
    return json.loads(t[start:end+1])


def looks_like_quota_or_auth_error(text: str) -> bool:
    t = (text or "").lower()
    return any(s in t for s in [
        "you exceeded your current quota",
        "insufficient_quota",
        "billing",
        "invalid_api_key",
        "incorrect api key",
        "authentication error"
    ])

# ----------------- Fetch and clean -----------------
def fetch_text(url: str) -> Dict[str, str]:
    pages = {}
    base = url.rstrip("/")
    for path in ["/", "/about", "/services", "/products"]:
        try:
            u = base if path == "/" else f"{base}{path}"
            r = requests.get(u, headers={"User-Agent": "PE-Matcher-Prototype/1.0"}, timeout=12, allow_redirects=True)
            if r.status_code != 200:
                continue
            html = r.text[:1_000_000]
            soup = BeautifulSoup(html, "html.parser")
            for tag in soup(["script","style","noscript"]):
                tag.decompose()
            text = re.sub(r"\\s+", " ", soup.get_text(" ", strip=True))
            if len(text) > 100:
                pages[path] = text[:50_000]
        except Exception:
            continue
    return pages

def join_pages(pages: Dict[str, str]) -> str:
    return "\\n\\n".join([f"[{path}]\\n{txt}" for path, txt in pages.items()])

# ----------------- Catalog helpers -----------------
def sectors_from_facts(facts: CompanyFacts) -> List[str]:
    s = set()
    if facts.industry:
        if facts.industry == "healthcare services": s.add("healthcare")
        elif facts.industry == "industrials": s.add("industrial")
        else: s.add(facts.industry)
    kw = " ".join((facts.subsector or "", " ".join(facts.keywords or []))).lower()
    if any(k in kw for k in ["hvac","plumbing","janitorial","field service","route based"]): s.add("business services")
    if any(k in kw for k in ["saas","software","workflow","platform"]): s.add("software")
    if any(k in kw for k in ["clinic","dental","vet","ambulatory"]): s.add("healthcare")
    return list(s) or ["business services","software","healthcare"]

def filter_catalog(catalog, sectors: List[str]):
    res = []
    for f in catalog:
        if any(s in f["sectors"] for s in sectors):
            res.append(f)
    return res or catalog

# ----------------- Deterministic sanity wrapper -----------------
def sanity_adjust(results: List[MatchResult], facts: CompanyFacts) -> List[MatchResult]:
    def size_adj(cm: CriteriaMatches):
        if cm.size == "likely": return 25
        if cm.size == "uncertain": return 10
        return -10

    adjusted = []
    for r in results:
        base = 0
        base += 40 if r.criteria_matches.sector else 0
        base += 20 if r.criteria_matches.subsector else 0
        base += 20 if r.criteria_matches.geography else 0
        base += size_adj(r.criteria_matches)
        base += 15 if r.criteria_matches.deal_type else 0
        heuristic = max(0, min(100, base))
        if abs(heuristic - r.fit_score) > 20:
            r = r.copy(update={"fit_score": int(heuristic)})
            r.match_reasons = r.match_reasons + ["Score adjusted for rule consistency"]
        adjusted.append(r)
    adjusted.sort(key=lambda x: x.fit_score, reverse=True)
    return adjusted[:5]

# ----------------- UI -----------------
st.set_page_config(page_title="PE Buyer Shortlist Prototype", layout="wide")
st.title("PE Buyer Shortlist from a Company URL")

with st.sidebar:
    st.markdown("Settings")
    model_extract = st.text_input("Extraction model", MODEL_EXTRACT)
    model_match = st.text_input("Matching model", MODEL_MATCH)
    temp_extract = st.slider("Extract temperature", 0.0, 0.5, 0.0, 0.1)
    temp_match = st.slider("Match temperature", 0.0, 0.7, 0.0, 0.1)
    st.divider()
    st.caption("Public website text only. Shows evidence, scores, and costs.")

url = st.text_input("Company URL")
go = st.button("Analyze", type="primary")

if go and url:
    t0 = time.time()
    pages = fetch_text(url)
    if not pages:
        st.error("No readable content found. Try another URL.")
        st.stop()

    with st.expander("Fetched pages"):
        st.json(list(pages.keys()))

    raw_text = join_pages(pages)

    # LLM extraction
    try:
        sys = PROMPT_A_SYS
        user = prompt_a_user(url, raw_text)
        content = openai_chat(model_extract, sys, user, temperature=temp_extract)
        content = content.strip().strip("`")
        facts_dict = json.loads(content)
        facts = CompanyFacts(**facts_dict)
    except ValidationError as ve:
        st.error("Schema validation failed for extraction")
        st.code(str(ve))
        st.stop()
    except Exception as e:
        st.error("Extraction failed")
        st.code(str(e))
        st.stop()

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Company facts")
        st.json(json.loads(facts.model_dump_json(exclude_none=True)))
        if facts.evidence:
            st.caption("Evidence")
            st.json(facts.evidence)

    # filter catalog
    sectors = sectors_from_facts(facts)
    catalog_chunk = filter_catalog(FUND_CATALOG, sectors)

    # LLM matching
    try:
        sys_b = PROMPT_B_SYS
        user_b = prompt_b_user(facts.model_dump_json(exclude_none=True, indent=2),
                               json.dumps(catalog_chunk))
        matches_raw = openai_chat(model_match, sys_b, user_b, temperature=temp_match)
    
        # Always show raw once for debugging (collapsible)
        with st.expander("Debug: raw matching response"):
            st.code(matches_raw)
    
        # If the API returned a quota/auth message, surface it clearly
        if looks_like_quota_or_auth_error(matches_raw):
            st.error("OpenAI returned a quota/auth error during matching. Check billing and your API key.")
            st.stop()
    
        # First parse attempt
        try:
            parsed = parse_json_array_or_raise(matches_raw)
        except Exception:
            # Retry once, very strict, temp 0
            retry_user = user_b + "\n\nReturn exactly and only a valid JSON array. No prose, no code fences."
            matches_raw_retry = openai_chat(model_match, sys_b, retry_user, temperature=0.0)
            with st.expander("Debug: raw matching response (retry)"):
                st.code(matches_raw_retry)
    
            if looks_like_quota_or_auth_error(matches_raw_retry):
                st.error("OpenAI returned a quota/auth error on retry. Check billing and your API key.")
                st.stop()
    
            parsed = parse_json_array_or_raise(matches_raw_retry)
    
        results = [MatchResult(**m) for m in parsed]
    
    except ValidationError as ve:
        st.error("Schema validation failed for matching")
        st.code(str(ve))
        st.stop()
    except Exception as e:
        st.error("Matching failed")
        st.code(str(e))
        st.stop()


    results = sanity_adjust(results, facts)

    with col2:
        st.subheader("Recommended funds")
        for r in results:
            with st.container(border=True):
                st.markdown(f"**{r.fund}**  ·  Score {r.fit_score}")
                st.write(r.deal_angle)
                st.write("Why")
                for mr in r.match_reasons[:3]:
                    st.write(f"• {mr}")
                with st.expander("Criteria"):
                    st.table({
                        "criterion": ["sector","subsector","geography","size","deal type"],
                        "value": [r.criteria_matches.sector,
                                  r.criteria_matches.subsector,
                                  r.criteria_matches.geography,
                                  r.criteria_matches.size,
                                  r.criteria_matches.deal_type]
                    })

    # quick edit and rerun
    st.subheader("Quick edit and rerun")
    edited_industry = st.selectbox(
        "Industry",
        ["", "software","healthcare services","consumer","industrials","business services","fintech","other"],
        index= ["","software","healthcare services","consumer","industrials","business services","fintech","other"].index(facts.industry or "")
    )
    if st.button("Apply industry override and rescore"):
        facts.industry = edited_industry or facts.industry
        sectors = sectors_from_facts(facts)
        catalog_chunk = filter_catalog(FUND_CATALOG, sectors)
        user_b = prompt_b_user(facts.model_dump_json(exclude_none=True, indent=2),
                               json.dumps(catalog_chunk))
        matches_raw = openai_chat(MODEL_MATCH, PROMPT_B_SYS, user_b, temperature=0.0)
        matches_raw = matches_raw.strip().strip("`")
        parsed = json.loads(matches_raw)
        results = [MatchResult(**m) for m in parsed]
        results = sanity_adjust(results, facts)
        st.success("Rescored")
        for r in results:
            st.write(f"{r.fund}  Score {r.fit_score}")

    # downloads and run stats
    st.subheader("Download results")
    st.download_button("CompanyFacts JSON", facts.model_dump_json(exclude_none=True, indent=2),
                       file_name="company_facts.json", mime="application/json")
    st.download_button("MatchResults JSON", json.dumps([r.model_dump() for r in results], indent=2),
                       file_name="match_results.json", mime="application/json")

    elapsed = time.time() - t0
    st.caption(f"Run time approximately {elapsed:.2f} seconds")
