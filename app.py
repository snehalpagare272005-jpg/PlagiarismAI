"""

Pipeline
--------
1. Extract sentences from your text (academic-aware splitting).
2. Build several search-query types (exact phrase, distinctive phrase, keywords).
3. Search the web through a real search API (Brave or Tavily; optional unofficial ddgs).
4. Download the real pages / PDFs and extract readable text.
5. Compare: exact -> near-exact (difflib) -> semantic (Sentence-BERT + cosine).
6. Report one of three honest states:
     matches  /  no_matches  /  incomplete (search or downloads failed).

Semantic similarity is NOT proof of plagiarism. A human must review every match.
"""

import html
import io
import json
import math
import os
import re
import sqlite3
import sys
import time
import unicodedata
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from difflib import SequenceMatcher
from urllib.parse import parse_qs, urljoin, urlparse, urlunparse

import numpy as np
import plotly.graph_objects as go
import requests
import streamlit as st
from bs4 import BeautifulSoup
from docx import Document
from fpdf import FPDF
from pypdf import PdfReader
from sklearn.metrics.pairwise import cosine_similarity

# --------------------------------------------------------------------------
# Search-provider configuration.
# API keys are NEVER hard-coded. They are read from (in this order):
#   1. the sidebar key box (this browser session only, never written to disk)
#   2. environment variables, including a local ".env" file next to app.py
#   3. Streamlit secrets (.streamlit/secrets.toml)
# --------------------------------------------------------------------------
BRAVE_URL = "https://api.search.brave.com/res/v1/web/search"
TAVILY_URL = "https://api.tavily.com/search"


def _load_dotenv(path):
    """Tiny .env reader (no extra dependency). Real environment variables win."""
    try:
        with open(path, encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                name, value = line.split("=", 1)
                os.environ.setdefault(name.strip(), value.strip().strip('"').strip("'"))
    except OSError:
        pass


_load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env"))


def get_secret(name):
    """Return (value, source_description) for an API key, or ("", "") if not configured."""
    try:
        typed = st.session_state.get(f"session_key_{name}", "")
    except Exception:
        typed = ""
    if typed and typed.strip():
        return typed.strip(), "entered in the sidebar (this session only)"
    value = os.getenv(name, "").strip()
    if value:
        return value, "environment variable / .env file"
    try:
        value = str(st.secrets[name]).strip()
        if value:
            return value, "Streamlit secrets"
    except Exception:
        pass
    return "", ""


# --------------------------------------------------------------------------
# Constants
# --------------------------------------------------------------------------
APP_NAME = "PlagiarismAI"
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "plagiarism_history.db")

MIN_WORDS = 3                 # shortest sentence we keep at all
EXACT_MIN_WORDS = 4           # exact/near matching: 4+ words, or 3 with a technical term
SEMANTIC_MIN_WORDS = 5        # BERT matching needs enough context to be meaningful
MAX_SOURCE_SENTENCES = 1200   # per webpage (most relevant sentences are kept)
MAX_COMPARE_SENTENCES = 1500  # per document in document comparison
HTML_LIMIT = 3_000_000
PDF_LIMIT = 20_000_000

REQUEST_HEADERS = {
    "User-Agent": ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                   "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"),
    "Accept": "text/html,application/xhtml+xml,application/pdf;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
}
ALT_HEADERS = {
    "User-Agent": ("Mozilla/5.0 (X11; Linux x86_64; rv:124.0) Gecko/20100101 Firefox/124.0"),
    "Accept": "*/*",
}

EXACT = "Exact Match"
NEAR = "Near-Exact Match"
SEMANTIC = "Semantic Match"
TOPICAL = "Topical Similarity"          # similar meaning/topic, but no evidence of copying; NOT counted
NONE = "No Significant Match"
RANK = {NONE: 0, TOPICAL: 0, SEMANTIC: 1, NEAR: 2, EXACT: 3}
COUNTED_TYPES = (EXACT, NEAR, SEMANTIC)
COLORS = {EXACT: "#dc2626", NEAR: "#f97316", SEMANTIC: "#eab308", NONE: "#9ca3af"}
HIGHLIGHT_BG = {EXACT: "#fecaca", NEAR: "#fed7aa", SEMANTIC: "#fef08a"}

QUERY_EXACT, QUERY_PHRASE, QUERY_KEYWORDS = "Exact phrase", "Distinctive phrase", "Technical keywords"
QUERY_WEIGHT = {QUERY_EXACT: 3.0, QUERY_PHRASE: 2.0, QUERY_KEYWORDS: 1.0}

# --- Semantic evidence gate (starting values, NOT calibrated: tune with validate_real_model.py) ---
# A cosine >= threshold only makes a passage a *candidate*. It is counted as a Semantic Match
# only when independent evidence of reuse exists; otherwise it is reported as Topical Similarity.
CONTEXT_MIN_SIM = 0.60        # an aligned neighbour must be >= this similar AND be that neighbour's best match
DISTINCTIVE_DF = 0.10         # a word is "distinctive" if it occurs in <= 10% of the scan's sentences
LEXICAL_MIN_SHARED = 2        # distinctive words shared by the sentence and the passage
LEXICAL_MIN_COVERAGE = 0.25   # ...and they carry >= 25% of the sentence's distinctive-word weight
COMMON_SOURCE_MIN = 3         # candidates in >= 3 different sources = standard statement (lexical support vetoed)

STOPWORDS = set("""a about above after again all also an and any are as at be because been before being
between both but by can could did do does doing down during each few for from further had has have having
he her here hers him his how i if in into is it its itself just may me more most my no nor not of off on once
only or other our out over own same she should so some such than that the their theirs them then there these
they this those through to too under until up very was we were what when where which while who whom why will
with would you your thus however therefore which whose using used use based paper show shown results result
propose proposed present presented approach method methods""".split())

NOT_FOUND_MESSAGE = ("Web search could not be completed. "
                     "No reliable plagiarism conclusion can be generated.")


# --------------------------------------------------------------------------
# 1. Model
# --------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def load_bert_model():
    """Load Sentence-BERT once per server process (cached by Streamlit)."""
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(MODEL_NAME)


def encode_sentences(model, sentences):
    """Batch-encode sentences into unit-length BERT embeddings."""
    return model.encode(sentences, batch_size=64, convert_to_numpy=True,
                        normalize_embeddings=True, show_progress_bar=False)


# --------------------------------------------------------------------------
# 2-4. Text extraction from files
# --------------------------------------------------------------------------
def extract_text_from_pdf(data, max_pages=None):
    """Extract text from PDF bytes."""
    reader = PdfReader(io.BytesIO(data))
    if reader.is_encrypted:
        try:
            reader.decrypt("")
        except Exception:
            raise ValueError("This PDF is password-protected.")
    pages = reader.pages if max_pages is None else reader.pages[:max_pages]
    parts = []
    for page in pages:
        try:
            parts.append(page.extract_text() or "")
        except Exception:
            continue
    text = "\n\n".join(parts).strip()
    if not text:
        raise ValueError("No text found in the PDF (it may be a scanned image).")
    return text


def extract_text_from_docx(data):
    """Extract paragraph and table text from DOCX bytes."""
    doc = Document(io.BytesIO(data))
    parts = [p.text for p in doc.paragraphs if p.text.strip()]
    for table in doc.tables:
        for row in table.rows:
            for cell in row.cells:
                if cell.text.strip():
                    parts.append(cell.text.strip())
    text = "\n\n".join(parts).strip()
    if not text:
        raise ValueError("No text found in the DOCX file.")
    return text


def extract_text_from_txt(data):
    """Decode TXT bytes (UTF-8 first, then Latin-1 as fallback)."""
    for encoding in ("utf-8-sig", "utf-8", "latin-1"):
        try:
            return data.decode(encoding)
        except UnicodeDecodeError:
            continue
    return data.decode("utf-8", errors="ignore")


def extract_text_from_upload(uploaded_file):
    """Pick the right extractor for a Streamlit uploaded file."""
    name = uploaded_file.name.lower()
    data = uploaded_file.getvalue()
    if name.endswith(".pdf"):
        return extract_text_from_pdf(data)
    if name.endswith(".docx"):
        return extract_text_from_docx(data)
    if name.endswith(".txt"):
        return extract_text_from_txt(data)
    raise ValueError(f"Unsupported file type: {uploaded_file.name} (use PDF, DOCX or TXT).")


# --------------------------------------------------------------------------
# 5-6. Normalisation and sentences
# --------------------------------------------------------------------------
_CITATION = re.compile(r"\[\s*\d+(?:\s*[,\u2013\-]\s*\d+)*\s*\]")


def normalize_text(text):
    """Lowercase; unify quotes/dashes; drop citation marks like [12]; strip punctuation."""
    text = unicodedata.normalize("NFKC", text or "").replace("\u00ad", "").lower()
    text = _CITATION.sub(" ", text)
    text = re.sub(r"[\u2018\u2019\u201a\u201b]", "'", text)
    text = re.sub(r"[\u201c\u201d\u201e]", '"', text)
    text = re.sub(r"[\u2010-\u2015]", "-", text)
    text = re.sub(r"[^\w\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


_ABBREVIATIONS = ["et al.", "e.g.", "i.e.", "Fig.", "Figs.", "Eq.", "Eqs.", "vs.", "Sec.",
                  "Ref.", "Refs.", "Dr.", "Prof.", "approx.", "cf.", "No.", "Tab.", "Alg."]
_DOT = "\u2024"  # placeholder so abbreviation dots do not end a sentence
_SPLIT = re.compile(r"(?:(?<=[.!?])|(?<=[.!?][)\"'\]\u201d\u2019]))\s+(?=[A-Z0-9\"\u201c'(\[])")


def _is_technical_token(token):
    """CamelCase (FedProx), acronyms (IEEE), digits or hyphenated terms (non-IID)."""
    return bool(re.search(r"[A-Z].*[A-Z]|[a-z][A-Z]|\d|\w-\w", token))


def extract_sentences(text, min_words=MIN_WORDS, max_words=120):
    """Split text into unique sentences; keeps short technical sentences."""
    if not text:
        return []
    text = text.replace("\r", "\n")
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)           # PDF hyphenation
    text = re.sub(r"(?<!\n)\n(?!\n)", " ", text)           # soft line breaks
    for abbr in set(_ABBREVIATIONS + [a.lower() for a in _ABBREVIATIONS]):
        text = text.replace(abbr, abbr.replace(".", _DOT))
    sentences, seen = [], set()
    for paragraph in re.split(r"\n{2,}", text):
        for raw in _SPLIT.split(paragraph):
            sentence = re.sub(r"\s+", " ", raw.replace(_DOT, ".")).strip()
            words = sentence.split()
            if len(words) < min_words:
                continue
            for k in range(0, len(words), max_words):
                chunk = " ".join(words[k:k + max_words])
                key = normalize_text(chunk)
                if key and key not in seen:
                    seen.add(key)
                    sentences.append(chunk)
    return sentences


def _exact_eligible(sentence):
    words = sentence.split()
    return len(words) >= EXACT_MIN_WORDS or (
        len(words) >= MIN_WORDS and any(_is_technical_token(w) for w in words))


def _semantic_eligible(sentence):
    return len(sentence.split()) >= SEMANTIC_MIN_WORDS


# --------------------------------------------------------------------------
# 7a. Search-query generation (academic-friendly)
# --------------------------------------------------------------------------
def _term_weight(token):
    """How 'distinctive' a word is: technical terms and long words score higher."""
    t = token.strip(".,;:!?()[]{}\"'\u201c\u201d\u2018\u2019")
    if len(t) < 3 or t.lower() in STOPWORDS:
        return 0.0
    weight = 0.5
    if len(t) >= 7:
        weight += 1.0
    if len(t) >= 11:
        weight += 0.5
    if _is_technical_token(t):
        weight += 2.0
    return weight


def _clean_words(sentence):
    words = [w.strip(".,;:!?()[]{}\"'\u201c\u201d\u2018\u2019") for w in sentence.split()]
    return [w for w in words if w]


def distinctive_phrase(sentence, size=12):
    """The 8-15 word window of a sentence with the most distinctive terms."""
    words = _clean_words(sentence)
    if len(words) <= size:
        return " ".join(words)
    weights = [_term_weight(w) for w in words]
    start = max(range(len(words) - size + 1), key=lambda i: sum(weights[i:i + size]))
    return " ".join(words[start:start + size])


def top_terms(words, n):
    """The n most distinctive (weighted by frequency) terms, original spelling kept."""
    counter, spelling = Counter(), {}
    for word in words:
        weight = _term_weight(word)
        if weight > 0:
            counter[word.lower()] += weight
            spelling.setdefault(word.lower(), word)
    return [spelling[k] for k, _ in counter.most_common(n)]


def build_search_queries(sentences, max_queries):
    """
    Build up to `max_queries` queries of three kinds, interleaved so that even a
    small query budget uses all strategies:
      1. exact quoted phrase   2. distinctive 8-15 word phrase   3. technical keywords
    Sentences are picked from different parts of the text, best-scoring first.
    """
    if not sentences or max_queries < 1:
        return []
    scored = []
    for i, sentence in enumerate(sentences):
        words = _clean_words(sentence)
        if len(words) >= 4:
            scored.append((sum(_term_weight(w) for w in words) / math.sqrt(len(words)), i))
    if not scored:
        scored = [(1.0, i) for i in range(len(sentences))]

    n_pick = min(len(scored), max(max_queries, 3))
    segment = max(1, math.ceil(len(sentences) / n_pick))
    best_per_segment = {}
    for score, i in scored:
        key = i // segment
        if key not in best_per_segment or score > best_per_segment[key][0]:
            best_per_segment[key] = (score, i)
    chosen = [i for _, i in sorted(best_per_segment.values(), reverse=True)]

    exact_q = [{"query": f'"{distinctive_phrase(sentences[i])}"', "type": QUERY_EXACT} for i in chosen]
    phrase_q = [{"query": distinctive_phrase(sentences[i], size=14), "type": QUERY_PHRASE} for i in chosen]
    all_words = [w for s in sentences for w in _clean_words(s)]
    keyword_q = [{"query": " ".join(top_terms(all_words, 6)) + " paper", "type": QUERY_KEYWORDS}]
    for i in chosen:
        terms = top_terms(_clean_words(sentences[i]), 5)
        if len(terms) >= 2:
            keyword_q.append({"query": " ".join(terms), "type": QUERY_KEYWORDS})

    queries, seen = [], set()
    for group in zip_longest_lists(exact_q, keyword_q, phrase_q):
        key = group["query"].lower()
        if key not in seen and len(key) > 3:
            seen.add(key)
            group["query"] = group["query"][:250]
            queries.append(group)
        if len(queries) >= max_queries:
            break
    return queries


def zip_longest_lists(*lists):
    """Round-robin over lists, skipping exhausted ones."""
    for k in range(max((len(x) for x in lists), default=0)):
        for items in lists:
            if k < len(items):
                yield items[k]


# --------------------------------------------------------------------------
# 7b. Real web search
# --------------------------------------------------------------------------
def canonical_url(url):
    """Normalise a URL so duplicates can be detected."""
    parts = urlparse(url.strip())
    netloc = parts.netloc.lower()
    if netloc.startswith("www."):
        netloc = netloc[4:]
    return urlunparse((parts.scheme.lower(), netloc, parts.path.rstrip("/"), "", parts.query, ""))


class ProviderError(Exception):
    """A search provider failed. fatal=True means further queries are pointless (bad key etc.)."""

    def __init__(self, message, fatal=False):
        super().__init__(message)
        self.fatal = fatal


def _check_response(response, name):
    """Turn HTTP errors from a search API into clear messages."""
    code = response.status_code
    if code == 200:
        return
    body = response.text[:200].replace("\n", " ")
    if code in (401, 403):
        raise ProviderError(f"{name}: API key rejected (HTTP {code}). {body}", fatal=True)
    if code == 402:
        raise ProviderError(f"{name}: no active plan or credits (HTTP 402). {body}", fatal=True)
    if code == 429:
        raise ProviderError(f"{name}: rate limit or quota reached (HTTP 429). {body}")
    raise ProviderError(f"{name}: HTTP {code}. {body}")


def _search_brave(query, max_results, api_key, timeout=15):
    """Brave Search API: GET /res/v1/web/search, key in the X-Subscription-Token header."""
    response = requests.get(
        BRAVE_URL, timeout=timeout,
        headers={"Accept": "application/json", "X-Subscription-Token": api_key},
        params={"q": query[:400], "count": min(max_results, 20)})
    _check_response(response, "Brave")
    try:
        items = response.json().get("web", {}).get("results", [])
    except ValueError:
        raise ProviderError("Brave: response was not valid JSON")
    return [{"title": i.get("title", ""), "href": i.get("url"), "body": i.get("description", "")}
            for i in items]


def _search_tavily(query, max_results, api_key, timeout=20):
    """Tavily Search API: POST /search with a Bearer token and a JSON body."""
    response = requests.post(
        TAVILY_URL, timeout=timeout,
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        json={"query": query[:400], "max_results": min(max_results, 20), "search_depth": "basic"})
    _check_response(response, "Tavily")
    try:
        items = response.json().get("results", [])
    except ValueError:
        raise ProviderError("Tavily: response was not valid JSON")
    return [{"title": i.get("title", ""), "href": i.get("url"), "body": i.get("content", "")}
            for i in items]


def _search_ddgs(query, max_results, api_key="", timeout=15):
    """Optional, unofficial, key-less provider. One attempt only - no retries or backend cycling."""
    try:
        from ddgs import DDGS
    except Exception as exc:
        raise ProviderError(f"ddgs is not importable ({type(exc).__name__}: {exc}). Install it into "
                            f'this Python: "{sys.executable}" -m pip install -U ddgs', fatal=True)
    try:
        return DDGS(timeout=timeout).text(query, max_results=max_results) or []
    except Exception as exc:
        raise ProviderError(f"ddgs: {type(exc).__name__}: {exc}")


PROVIDERS = {
    "brave": {"label": "Brave Search API", "env": "BRAVE_API_KEY", "func": _search_brave,
              "interval": 1.1, "needs_key": True,
              "signup": "https://api-dashboard.search.brave.com/register"},
    "tavily": {"label": "Tavily Search API", "env": "TAVILY_API_KEY", "func": _search_tavily,
               "interval": 0.6, "needs_key": True, "signup": "https://app.tavily.com"},
    "ddgs": {"label": "DuckDuckGo via ddgs (unofficial, experimental)", "env": "",
             "func": _search_ddgs, "interval": 1.5, "needs_key": False, "signup": ""},
}
PROVIDER_CHOICES = ["auto", "brave", "tavily", "ddgs"]
SETUP_HELP = ("Set BRAVE_API_KEY or TAVILY_API_KEY (environment variable, .env file next to app.py, "
              "or .streamlit/secrets.toml), or paste a key in the sidebar under Search diagnostics. "
              "Tavily: https://app.tavily.com (free credits, no card). "
              "Brave: https://api-dashboard.search.brave.com/register (card required).")


def resolve_provider(choice="auto"):
    """Decide which provider to use. Returns id, label, key, source, configured, problem."""
    order = ["brave", "tavily"] if choice == "auto" else [choice]
    for provider_id in order:
        spec = PROVIDERS[provider_id]
        key, source = get_secret(spec["env"]) if spec["env"] else ("", "")
        if key or not spec["needs_key"]:
            return {"id": provider_id, "label": spec["label"], "key": key, "source": source,
                    "configured": True, "problem": ""}
    if choice == "auto":
        return {"id": None, "label": "none", "key": "", "source": "", "configured": False,
                "problem": "No search provider is configured. " + SETUP_HELP}
    spec = PROVIDERS[choice]
    return {"id": choice, "label": spec["label"], "key": "", "source": "", "configured": False,
            "problem": f"{spec['label']} has no API key ({spec['env']} is not set). " + SETUP_HELP}


def search_web_sources(queries, results_per_query=5, provider="auto"):
    """
    Run every query against the selected search API.
    Returns a dict:
      results        - genuine hits (title, url, snippet, found_by, best_rank), de-duplicated
      errors / notes - the real error messages and setup problems (never swallowed)
      queries        - per-query log (query, type, ok, hits, engine)
      provider / provider_label - what was used (provider is None when nothing is configured)
    Snippets are kept only for ranking. They are never treated as a verified comparison.
    """
    out = {"results": [], "errors": [], "notes": [], "queries": [], "engines": [],
           "provider": None, "provider_label": ""}
    info = resolve_provider(provider)
    out["provider_label"] = info["label"]
    if not info["configured"]:
        out["notes"].append(info["problem"])
        return out
    out["provider"] = info["id"]
    spec = PROVIDERS[info["id"]]
    by_url, error_counts, consecutive_failures = {}, Counter(), 0

    for position, q in enumerate(queries):
        entry = {"query": q["query"], "type": q["type"], "ok": False, "hits": 0,
                 "engine": info["label"]}
        raw, stop = [], False
        try:
            raw = spec["func"](q["query"], results_per_query, info["key"])
            entry["ok"] = True
            consecutive_failures = 0
            if info["label"] not in out["engines"]:
                out["engines"].append(info["label"])
        except ProviderError as exc:
            error_counts[str(exc)[:300]] += 1
            consecutive_failures += 1
            stop = exc.fatal
        except requests.exceptions.RequestException as exc:
            error_counts[f"{info['label']}: {type(exc).__name__}: {str(exc)[:200]}"] += 1
            consecutive_failures += 1
        except Exception as exc:
            error_counts[f"{info['label']}: {type(exc).__name__}: {str(exc)[:200]}"] += 1
            consecutive_failures += 1
        for rank, hit in enumerate(raw or []):
            url = hit.get("href") or hit.get("url")
            if not url or not str(url).startswith("http"):
                continue
            key = canonical_url(url)
            item = by_url.get(key)
            if item is None:
                item = {"title": (hit.get("title") or "").strip(), "url": url,
                        "snippet": (hit.get("body") or "").strip(), "found_by": [], "best_rank": rank}
                by_url[key] = item
                out["results"].append(item)
            if q["type"] not in item["found_by"]:
                item["found_by"].append(q["type"])
            item["best_rank"] = min(item["best_rank"], rank)
            entry["hits"] += 1
        out["queries"].append(entry)
        if stop or consecutive_failures >= 3:
            skipped = len(queries) - position - 1
            if skipped:
                out["notes"].append(f"{skipped} remaining search(es) were skipped because the provider "
                                    "keeps failing. Fix the error above and scan again.")
            break
        if position < len(queries) - 1:
            time.sleep(spec["interval"])  # respect the provider's rate limit

    out["errors"] = [f"{msg}" + (f"  (x{n})" if n > 1 else "") for msg, n in error_counts.items()]
    return out


def test_search_provider(choice="auto", query=None):
    """Independent provider test (used by the sidebar button and the command line)."""
    query = query or '"Federated Learning is a distributed learning paradigm with two key challenges"'
    info = resolve_provider(choice)
    started = time.time()
    out = search_web_sources([{"query": query, "type": QUERY_EXACT}], 5, choice)
    entry = out["queries"][0] if out["queries"] else {"ok": False, "hits": 0}
    return {"provider": info["label"], "configured": info["configured"],
            "key_source": info["source"], "connected": bool(entry["ok"]),
            "result_count": len(out["results"]), "results": out["results"],
            "errors": out["errors"] + out["notes"], "seconds": round(time.time() - started, 2),
            "query": query}


def prioritize_hits(hits, queries):
    """Rank sources: found by exact-phrase queries, high in results, snippet shares distinctive terms."""
    terms = set()
    for q in queries:
        for word in re.findall(r"[\w\-]+", q["query"]):
            if _term_weight(word) >= 1.5:
                terms.add(word.lower())
    ranked = []
    for hit in hits:
        score = sum(QUERY_WEIGHT.get(t, 1.0) for t in hit["found_by"]) + 1.0 / (1 + hit["best_rank"])
        text_tokens = set(re.findall(r"[\w\-]+", (hit["title"] + " " + hit["snippet"]).lower()))
        if terms:
            score += 2.0 * len(terms & text_tokens) / len(terms)
        ranked.append((score, hit))
    ranked.sort(key=lambda x: -x[0])
    return [h for _, h in ranked]


# --------------------------------------------------------------------------
# 8. Webpage retrieval and extraction
# --------------------------------------------------------------------------
class FetchError(Exception):
    pass


_NOISE_CLASS = re.compile(
    r"(^|[\s_-])(nav|navbar|navigation|menu|footer|sidebar|cookie|cookies|banner|advert|"
    r"advertisement|ads?|promo|share|social|breadcrumb|popup|modal|newsletter|toolbar)($|[\s_-])", re.I)
_BLOCK_CHILDREN = ["div", "p", "ul", "ol", "table", "section", "article", "h1", "h2", "h3", "h4"]


def _download(url, timeout):
    """Download bytes. Retries once with other headers when blocked. Returns (bytes, content_type)."""
    last_error = FetchError("Blocked")
    for headers in (REQUEST_HEADERS, ALT_HEADERS):
        response = None
        try:
            response = requests.get(url, headers=headers, timeout=timeout, stream=True)
            if response.status_code != 200:
                last_error = FetchError(f"HTTP {response.status_code} (page blocked or unavailable)")
                if response.status_code in (401, 403, 406, 429, 503):
                    continue
                raise last_error
            ctype = response.headers.get("Content-Type", "").lower()
            is_pdf = "pdf" in ctype or urlparse(url).path.lower().endswith(".pdf")
            limit = PDF_LIMIT if is_pdf else HTML_LIMIT
            declared = response.headers.get("Content-Length", "")
            if is_pdf and declared.isdigit() and int(declared) > limit:
                raise FetchError("PDF too large to download")
            chunks, size = [], 0
            for chunk in response.iter_content(65536):
                size += len(chunk)
                if size > limit:
                    if is_pdf:
                        raise FetchError("PDF too large to download")
                    break  # truncated HTML is still useful
                chunks.append(chunk)
            return b"".join(chunks), ctype
        finally:
            if response is not None:
                response.close()
    raise last_error


def _extract_html(raw):
    """Return (title, readable_text, pdf_link) from HTML bytes."""
    soup = BeautifulSoup(raw, "html.parser")

    def meta(*names):
        for name in names:
            tag = soup.find("meta", attrs={"name": re.compile(f"^{name}$", re.I)}) or \
                  soup.find("meta", attrs={"property": re.compile(f"^{name}$", re.I)})
            if tag and tag.get("content"):
                return tag["content"].strip()
        return ""

    title = meta("citation_title", "og:title") or (soup.title.get_text(strip=True) if soup.title else "")
    pdf_link = meta("citation_pdf_url")
    abstract = meta("citation_abstract", "dc.description", "og:description", "description")

    for tag in soup(["script", "style", "noscript", "nav", "footer", "header", "aside",
                     "form", "svg", "iframe", "button", "select"]):
        tag.decompose()
    noisy = []
    for tag in soup.find_all(True):
        attrs = getattr(tag, "attrs", None) or {}
        label = " ".join(attrs.get("class", [])) + " " + str(attrs.get("id", ""))
        if label.strip() and _NOISE_CLASS.search(label):
            noisy.append(tag)
    for tag in noisy:
        try:
            tag.decompose()
        except Exception:
            pass

    container = soup.find("article") or soup.find("main") or soup.find(attrs={"role": "main"})
    body = soup.body or soup
    if container is None or len(container.get_text(" ", strip=True)) < 400:
        container = body

    pieces = []
    if abstract and len(abstract.split()) >= 8:
        pieces.append(abstract)
    for element in container.find_all(["h1", "h2", "h3", "h4", "p", "li", "blockquote",
                                       "dd", "figcaption", "div"]):
        if element.name == "div" and element.find(_BLOCK_CHILDREN):
            continue  # only leaf divs (some sites put paragraphs in divs)
        piece = element.get_text(" ", strip=True)
        minimum = 8 if element.name == "div" else 3
        if len(piece.split()) >= minimum:
            pieces.append(piece)
    seen, unique = set(), []
    for piece in pieces:
        if piece not in seen:
            seen.add(piece)
            unique.append(piece)
    text = "\n\n".join(unique) if unique else container.get_text("\n\n", strip=True)
    return title, text, pdf_link


def fetch_webpage_content(url, timeout=12):
    """
    Download a real page or PDF and extract readable text.
    Returns {"ok", "text", "page_title", "reason", "kind", "pdf_url", "note"}.
    ok=False means the source is UNVERIFIED and must not be scored.
    """
    result = {"ok": False, "text": "", "page_title": "", "reason": "", "kind": "",
              "pdf_url": "", "note": ""}
    try:
        raw, ctype = _download(url, timeout)
        if "pdf" in ctype or urlparse(url).path.lower().endswith(".pdf"):
            result.update(ok=True, kind="pdf", text=extract_text_from_pdf(raw, max_pages=60)[:600000])
            return result
        if "html" not in ctype and "text/plain" not in ctype and "xml" not in ctype:
            result["reason"] = f"Unsupported content type: {ctype or 'unknown'}"
            return result
        if "text/plain" in ctype:
            text, title, pdf_link = extract_text_from_txt(raw), "", ""
        else:
            title, text, pdf_link = _extract_html(raw)
        result.update(page_title=title, kind="html")

        # Academic sites often show only an abstract; the page itself advertises the
        # full-text PDF in a standard <meta name="citation_pdf_url"> tag. Use it.
        if pdf_link:
            pdf_url = urljoin(url, pdf_link)
            if canonical_url(pdf_url) != canonical_url(url):
                try:
                    pdf_raw, pdf_type = _download(pdf_url, timeout + 8)
                    pdf_text = extract_text_from_pdf(pdf_raw, max_pages=60)
                    text = text + "\n\n" + pdf_text
                    result.update(kind="html+pdf", pdf_url=pdf_url)
                except Exception as exc:
                    result["note"] = f"Full-text PDF listed by the page could not be read: {exc}"
        if len(text) < 200:
            result["reason"] = "No readable text found on the page"
            return result
        result.update(ok=True, text=text[:600000])
        return result
    except requests.exceptions.Timeout:
        result["reason"] = "Timed out"
    except requests.exceptions.SSLError:
        result["reason"] = "SSL/certificate error"
    except requests.exceptions.ConnectionError as exc:
        result["reason"] = f"Connection error: {str(exc)[:120]}"
    except requests.exceptions.RequestException as exc:
        result["reason"] = f"Request failed: {str(exc)[:120]}"
    except FetchError as exc:
        result["reason"] = str(exc)
    except Exception as exc:
        result["reason"] = f"Could not read page: {exc}"
    return result


# --------------------------------------------------------------------------
# 9-12. Matching
# --------------------------------------------------------------------------
def _make_match(index, sentence, matched, score, match_type, level, title="", url=""):
    return {"index": int(index), "sentence": sentence, "matched": matched,
            "score": round(float(score), 2), "type": match_type, "level": level,
            "title": title, "url": url}


def detect_exact_matches(sentences, source_sentences, title="", url="", indices=None):
    """
    Exact match after normalisation. Also finds a sentence that sits verbatim inside a
    longer source sentence, or that the source split across two sentences.
    """
    source_norm = [(normalize_text(s), s) for s in source_sentences]
    lookup = {}
    for n, s in source_norm:
        if n:
            lookup.setdefault(n, s)
    for k in range(len(source_sentences) - 1):          # adjacent-sentence joins
        joined = source_sentences[k] + " " + source_sentences[k + 1]
        lookup.setdefault(normalize_text(joined), joined)

    matches = []
    for i in (range(len(sentences)) if indices is None else indices):
        if not _exact_eligible(sentences[i]):
            continue
        key = normalize_text(sentences[i])
        if not key:
            continue
        if key in lookup:
            matches.append(_make_match(i, sentences[i], lookup[key], 100.0, EXACT,
                                       "Identical wording", title, url))
            continue
        if len(key.split()) >= 6:                       # verbatim inside a longer passage
            for n, s in source_norm:
                if key in n:
                    matches.append(_make_match(i, sentences[i], s, 100.0, EXACT,
                                               "Found verbatim inside a longer passage", title, url))
                    break
    return matches


def detect_near_exact_matches(sentences, source_sentences, title="", url="",
                              threshold=0.90, indices=None):
    """Near-identical wording using difflib's character-level ratio (with cheap pre-filters)."""
    source_norm = []
    for s in source_sentences:
        n = normalize_text(s)
        if n:
            source_norm.append((n, s, set(n.split())))
    matches = []
    for i in (range(len(sentences)) if indices is None else indices):
        if not _exact_eligible(sentences[i]):
            continue
        a = normalize_text(sentences[i])
        if not a:
            continue
        a_tokens = set(a.split())
        matcher = SequenceMatcher(None, autojunk=False)
        matcher.set_seq2(a)
        best_ratio, best_sentence = 0.0, None
        for b, original, b_tokens in source_norm:
            union = len(a_tokens | b_tokens)
            if union and len(a_tokens & b_tokens) / union < 0.45:
                continue
            matcher.set_seq1(b)
            if matcher.real_quick_ratio() < threshold or matcher.quick_ratio() < threshold:
                continue
            ratio = matcher.ratio()
            if ratio > best_ratio:
                best_ratio, best_sentence = ratio, original
        if best_sentence is not None and best_ratio >= threshold:
            match_type = EXACT if best_ratio >= 1.0 else NEAR
            matches.append(_make_match(i, sentences[i], best_sentence, best_ratio * 100,
                                       match_type, "Nearly identical wording", title, url))
    return matches


def calculate_similarity(embeddings_a, embeddings_b):
    """Cosine similarity matrix (rows = A, columns = B) using scikit-learn."""
    return cosine_similarity(embeddings_a, embeddings_b)


def semantic_level(score, threshold):
    """Human-readable level for a cosine similarity score (0-1)."""
    if score >= 0.90:
        return "Very high semantic similarity"
    if score >= 0.80:
        return "High semantic similarity"
    if score >= 0.70:
        return "Moderate semantic similarity"
    return "Low similarity (custom threshold below 70%)"


def _stem(word):
    """Very light, language-agnostic suffix stripping so store/stores/storing share one stem."""
    for suffix, margin in (("ing", 3), ("ed", 3), ("es", 3), ("ly", 3), ("s", 2)):
        if len(word) > len(suffix) + margin and word.endswith(suffix):
            word = word[:-len(suffix)]
            break
    if len(word) > 4 and word.endswith("e"):
        word = word[:-1]
    return word


def content_tokens(sentence):
    """Stemmed content words (stop-words, short words and bare numbers removed)."""
    return [_stem(w) for w in normalize_text(sentence).split()
            if w not in STOPWORDS and len(w) >= 3 and not w.isdigit()]


def document_frequencies(sentence_pool):
    """How many sentences of the pool contain each content word (pool = your text + the source)."""
    df = Counter()
    for sentence in sentence_pool:
        df.update(set(content_tokens(sentence)))
    return df, max(len(sentence_pool), 1)


def lexical_evidence(sentence, passage, df, pool_size):
    """
    Distinctive wording shared by `sentence` and `passage`.

    A word is distinctive when few sentences of the scan contain it, so topic words that run
    through the whole text and the whole source ("database", "data" ...) do not count. This is
    relative to the scan's own sentences, so it needs no subject-specific word lists.
    Returns (shared distinctive words, share of the sentence's distinctive-word weight covered).
    """
    limit = max(2, DISTINCTIVE_DF * pool_size)
    idf = lambda w: math.log((pool_size + 1) / (df.get(w, 1) + 0.5))
    mine = {w for w in content_tokens(sentence) if df.get(w, 0) <= limit}
    shared = sorted(mine & set(content_tokens(passage)))
    total = sum(idf(w) for w in mine)
    coverage = sum(idf(w) for w in shared) / total if total else 0.0
    return shared, round(coverage, 3)


def detect_semantic_matches(model, sentences, source_sentences, title="", url="",
                            threshold=0.70, indices=None, user_embeddings=None,
                            source_embeddings=None):
    """
    BERT candidate search: best cosine passage in the source for each submitted sentence.

    Entries are returned as SEMANTIC *candidates* carrying evidence fields ("context",
    "shared_words", "coverage"). finalize_semantic_matches() decides, once all sources are
    known, whether a candidate is a real Semantic Match or only Topical Similarity.
    """
    indices = list(range(len(sentences))) if indices is None else list(indices)
    indices = [i for i in indices if _semantic_eligible(sentences[i])]
    if not indices or not source_sentences:
        return []
    if user_embeddings is None:
        user_embeddings = encode_sentences(model, sentences)
    if source_embeddings is None:
        source_embeddings = encode_sentences(model, source_sentences)
    similarity = calculate_similarity(user_embeddings, source_embeddings)   # all rows: context needs neighbours
    best_cols = np.argmax(similarity, axis=1)
    df, pool_size = document_frequencies(list(sentences) + list(source_sentences))
    matches = []
    for i in indices:
        best_col = int(best_cols[i])
        score = float(similarity[i][best_col])
        if score < threshold:
            continue
        # Context: the sentence after (and before) this one must have ITS best match in the source
        # exactly one sentence after (before) this match. Mere topical similarity of neighbours is not
        # enough: on a same-topic page unrelated neighbours often reach 0.6 by chance, but their best
        # match landing on the exactly adjacent sentence is unlikely unless the passage was reused.
        context = 0
        for di in (1, -1):
            ni, nj = i + di, best_col + di
            if (0 <= ni < len(sentences) and 0 <= nj < len(source_sentences)
                    and _semantic_eligible(sentences[ni]) and int(best_cols[ni]) == nj
                    and similarity[ni][nj] >= CONTEXT_MIN_SIM):
                context += 1
        shared, coverage = lexical_evidence(sentences[i], source_sentences[best_col], df, pool_size)
        m = _make_match(i, sentences[i], source_sentences[best_col], score * 100,
                        SEMANTIC, semantic_level(score, threshold), title, url)
        m.update(context=context, shared_words=shared, coverage=coverage)
        matches.append(m)
    return matches


def finalize_semantic_matches(matches):
    """
    Turn semantic candidates into Semantic Match (counted) or Topical Similarity (not counted).

    Supported when EITHER
      * context: neighbouring sentences also match the neighbouring source sentences, in order; OR
      * wording: the pair shares enough distinctive words AND the sentence is not a standard
        statement (candidates in COMMON_SOURCE_MIN or more different sources).
    Candidates are never merged or dropped: every one stays in `matches`, so one sentence with
    several sources is still several rows but is counted once by summarize_matches().
    """
    sources_per_sentence = {}
    for m in matches:
        if m["type"] == SEMANTIC:
            sources_per_sentence.setdefault(m["index"], set()).add(m.get("url") or m.get("title") or "")
    for m in matches:
        if m["type"] != SEMANTIC or "context" not in m:     # exact/near-exact, or already finalized
            continue
        common = len(sources_per_sentence[m["index"]]) >= COMMON_SOURCE_MIN
        lexical = (len(m["shared_words"]) >= LEXICAL_MIN_SHARED and m["coverage"] >= LEXICAL_MIN_COVERAGE)
        if m["context"] > 0:
            reason = "consecutive sentences also match in order"
        elif lexical and not common:
            reason = "shares distinctive wording: " + ", ".join(m["shared_words"][:6])
        else:
            m["type"] = TOPICAL
            if common:
                m["level"] = (f"Standard statement: similar passages in {len(sources_per_sentence[m['index']])} "
                              "sources, no copying evidence (not counted)")
            else:
                m["level"] = "Related topic only: no shared distinctive wording or matching context (not counted)"
            continue
        m["level"] = f"{m['level']} - {reason}"
    return matches


def match_against_source(model, sentences, user_embeddings, source_sentences,
                         title, url, semantic_threshold, near_threshold):
    """Cascade: exact -> near-exact -> semantic. Each sentence gets one result per source."""
    everything = list(range(len(sentences)))
    matches = detect_exact_matches(sentences, source_sentences, title, url, everything)
    done = {m["index"] for m in matches}
    remaining = [i for i in everything if i not in done]
    near = detect_near_exact_matches(sentences, source_sentences, title, url,
                                     near_threshold, remaining)
    matches += near
    done |= {m["index"] for m in near}
    remaining = [i for i in everything if i not in done]
    matches += detect_semantic_matches(model, sentences, source_sentences, title, url,
                                       semantic_threshold, remaining, user_embeddings)
    return matches


def rank_source_sentences(source_sentences, user_sentences, cap):
    """Keep the `cap` page sentences that share the most content words with the submitted text."""
    if len(source_sentences) <= cap:
        return source_sentences
    vocab = set()
    for s in user_sentences:
        vocab |= {w for w in normalize_text(s).split() if len(w) >= 4 and w not in STOPWORDS}
    scored = [(len(vocab & set(normalize_text(s).split())), k) for k, s in enumerate(source_sentences)]
    keep = sorted(k for _, k in sorted(scored, key=lambda x: -x[0])[:cap])
    return [source_sentences[k] for k in keep]


def semantic_breakdown(sentence_rows, matches):
    """
    Unique-SENTENCE view of semantic matches (never one count per source).

    `matches` holds one entry per (submitted sentence, source) pair, so a single sentence
    that resembles 3 sources appears 3 times. This counts each sentence once, and also
    records, for every sentence, how many distinct sources matched it semantically
    (row["semantic_sources"]). A sentence whose strongest match is exact/near-exact is
    still counted as exact/near-exact in the main statistics; if another source only
    resembles it in meaning, that is reported separately as `semantic_also`.
    """
    sources_by_sentence = {}
    for m in matches:
        if m["type"] == SEMANTIC:
            sources_by_sentence.setdefault(m["index"], set()).add(m.get("url") or m.get("title") or "")
    also = any_sentences = pairs = 0
    for i, row in enumerate(sentence_rows):
        n = len(sources_by_sentence.get(i, ()))
        row["semantic_sources"] = n
        if n:
            any_sentences += 1
            pairs += n
            if row["status"] != SEMANTIC:
                also += 1
    total = max(len(sentence_rows), 1)
    return {"semantic_also": also, "semantic_any": any_sentences, "semantic_pairs": pairs,
            "semantic_any_pct": round(any_sentences * 100 / total, 1)}


def summarize_matches(sentences, matches):
    """
    Pick the strongest match per sentence and compute overall statistics.

    Every sentence gets exactly ONE status (exact > near-exact > semantic > none), so
    exact + near_exact + semantic + none == total_sentences. `semantic` therefore means
    "semantic-ONLY" sentences. See semantic_breakdown() for the sentence-level count of
    semantic matches that are hidden behind a stronger match from another source.
    """
    best = {}
    for m in matches:
        if m["type"] not in COUNTED_TYPES:      # Topical Similarity is shown to reviewers, never counted
            continue
        key = (RANK[m["type"]], m["score"])
        if m["index"] not in best or key > best[m["index"]][0]:
            best[m["index"]] = (key, m)
    rows, counts = [], {EXACT: 0, NEAR: 0, SEMANTIC: 0, NONE: 0}
    for i, sentence in enumerate(sentences):
        if i in best:
            m = best[i][1]
            status, score = m["type"], m["score"]
        else:
            status, score = NONE, 0.0
        counts[status] += 1
        rows.append({"text": sentence, "status": status, "score": score})
    total = max(len(sentences), 1)
    stats = {
        "total_sentences": len(sentences),
        "exact": counts[EXACT], "near_exact": counts[NEAR],
        "semantic": counts[SEMANTIC], "none": counts[NONE],
        "exact_pct": round((counts[EXACT] + counts[NEAR]) * 100 / total, 1),
        "semantic_pct": round(counts[SEMANTIC] * 100 / total, 1),
        "overall_pct": round((total - counts[NONE]) * 100 / total, 1),
    }
    stats.update(semantic_breakdown(rows, matches))
    stats["topical_sentences"] = len({m["index"] for m in matches if m["type"] == TOPICAL})
    stats["topical_pairs"] = len({(m["index"], m.get("url") or m.get("title") or "")
                                  for m in matches if m["type"] == TOPICAL})
    return rows, stats


def report_stat_lines(stats):
    """The 'Overall statistics' wording of the PDF report (kept separate so it can be tested)."""
    lines = [
        f"Sentences scanned: {stats['total_sentences']}",
        f"Overall similarity: {stats['overall_pct']}% of sentences matched "
        "(each sentence counted once, by its strongest match)",
        f"Exact matches: {stats['exact']} sentence(s)   |   Near-exact matches: {stats['near_exact']} "
        f"sentence(s)   (exact + near-exact: {stats['exact_pct']}%)",
        f"Semantic-only matches: {stats['semantic']} sentence(s) ({stats['semantic_pct']}%) - "
        "no exact or near-exact match in any source",
    ]
    if stats.get("semantic_any", 0):
        lines.append(
            f"Sentences with any semantic match: {stats['semantic_any']} ({stats['semantic_any_pct']}%), "
            f"in {stats['semantic_pairs']} sentence-source pair(s); {stats['semantic_also']} of these are "
            "already counted as exact/near-exact because another source matches them more strongly")
    if stats.get("topical_pairs", 0):
        lines.append(
            f"Topical similarity (NOT counted above): {stats['topical_sentences']} sentence(s) resembled "
            f"source passages in meaning ({stats['topical_pairs']} sentence-source pair(s)) but showed no "
            "evidence of copying - no matching neighbouring sentences and no distinctive shared wording, "
            "or a statement found in many sources. Listed below for manual review.")
    return lines


# --------------------------------------------------------------------------
# Scan pipeline (web)
# --------------------------------------------------------------------------
def _summarize_reasons(sources, limit=4):
    counts = Counter(re.sub(r"\d{3,}", "N", s["note"]) for s in sources if not s["verified"] and s["note"])
    return "; ".join(f"{reason} (x{n})" if n > 1 else reason for reason, n in counts.most_common(limit))


def run_scan(model, text, document_name, settings, progress):
    """Full web scan. `progress(fraction, message)` updates the UI."""
    warnings = []
    sentences = extract_sentences(text)
    if not sentences:
        raise ValueError(f"No scannable sentences found (need at least {MIN_WORDS} words per sentence).")
    if len(sentences) > settings["max_sentences"]:
        warnings.append(f"Only the first {settings['max_sentences']} of {len(sentences)} "
                        "sentences were scanned (sidebar limit).")
        sentences = sentences[:settings["max_sentences"]]

    progress(0.05, "Encoding your sentences with BERT...")
    user_embeddings = encode_sentences(model, sentences)

    progress(0.12, "Building search queries...")
    queries = build_search_queries(sentences, settings["max_queries"])
    progress(0.15, f"Searching the web ({len(queries)} queries)...")
    search = search_web_sources(queries, settings["results_per_query"], settings.get("provider", "auto"))
    if search["provider"]:
        warnings += search["notes"]
    hits = prioritize_hits(search["results"], queries)
    results_retrieved = len(hits)
    selected = hits[:settings["max_pages"]]

    pages = {}
    if selected:
        with ThreadPoolExecutor(max_workers=6) as pool:
            futures = {pool.submit(fetch_webpage_content, h["url"]): h for h in selected}
            finished = 0
            for future in as_completed(futures):
                hit = futures[future]
                try:
                    pages[hit["url"]] = future.result()
                except Exception as exc:
                    pages[hit["url"]] = {"ok": False, "text": "", "page_title": "", "reason": str(exc),
                                         "kind": "", "pdf_url": "", "note": ""}
                finished += 1
                progress(0.20 + 0.40 * finished / len(selected),
                         f"Downloaded {finished}/{len(selected)} webpages...")

    all_matches, sources = [], []
    downloaded = compared = 0
    for k, hit in enumerate(selected):
        page = pages[hit["url"]]
        title = hit["title"] or page.get("page_title") or "(title unavailable)"
        source = {"title": title, "url": hit["url"], "verified": False, "downloaded": bool(page["ok"]),
                  "note": page.get("reason", ""), "matches": 0, "compared_sentences": 0,
                  "kind": page.get("kind", ""), "pdf_url": page.get("pdf_url", ""),
                  "found_by": hit["found_by"]}
        if page.get("note"):
            source["note"] = (source["note"] + " " + page["note"]).strip()
        if page["ok"]:
            downloaded += 1
            page_sentences = extract_sentences(page["text"], min_words=MIN_WORDS)
            page_sentences = rank_source_sentences(page_sentences, sentences, MAX_SOURCE_SENTENCES)
            if page_sentences:
                matches = match_against_source(model, sentences, user_embeddings, page_sentences,
                                               title, hit["url"], settings["semantic_threshold"],
                                               settings["near_threshold"])
                source.update(verified=True, matches=len(matches), compared_sentences=len(page_sentences))
                all_matches += matches
                compared += 1
            else:
                source["note"] = "No readable sentences found on the page"
        sources.append(source)
        progress(0.60 + 0.35 * (k + 1) / len(selected), f"Compared {k + 1}/{len(selected)} sources...")

    finalize_semantic_matches(all_matches)       # needs every source: "found in many sources" = standard statement
    counted_per_source = Counter(m["url"] for m in all_matches if m["type"] in COUNTED_TYPES)
    for source in sources:
        if source["verified"]:
            source["matches"] = counted_per_source[source["url"]]   # topical candidates are not matches
    sentence_rows, stats = summarize_matches(sentences, all_matches)
    with_matches = len([s for s in sources if s["verified"] and s["matches"] > 0])
    stats.update(verified_sources=compared, sources_with_matches=with_matches,
                 unverified_sources=len(sources) - compared)

    queries_ok = len([q for q in search["queries"] if q["ok"]])
    pipeline = {"queries_planned": len(queries), "queries_ok": queries_ok,
                "results_retrieved": results_retrieved, "pages_attempted": len(selected),
                "pages_downloaded": downloaded, "pages_compared": compared,
                "sources_with_matches": with_matches, "engine": search["provider_label"] or "none",
                "queries": search["queries"], "search_errors": search["errors"]}

    # --- the three honest states -------------------------------------------------
    status, reason = "matches", ""
    if compared == 0:
        status = "incomplete"
        if not search["provider"]:
            reason = search["notes"][0] if search["notes"] else "No search provider is configured. " + SETUP_HELP
        elif queries_ok == 0:
            reason = (f"The search provider ({search['provider_label']}) did not answer any query. "
                      + " | ".join(search["errors"][:4]))
        elif results_retrieved == 0:
            reason = ("The search provider answered but returned no results for any query, so there was "
                      "nothing to compare (insufficient coverage). Try more queries or different text.")
        else:
            reason = (f"{results_retrieved} search result(s) were retrieved, but no webpage could be "
                      f"downloaded and compared. Reasons: {_summarize_reasons(sources) or 'unknown'}")
    elif not any(m["type"] in COUNTED_TYPES for m in all_matches):
        status = "no_matches"
    if status != "incomplete":
        if queries_ok < len(queries):
            warnings.append(f"Only {queries_ok} of {len(queries)} searches succeeded, so coverage is "
                            "partial. " + " | ".join(search["errors"][:3]))
        if compared < len(selected):
            warnings.append(f"{len(selected) - compared} of {len(selected)} sources could not be "
                            "compared and are shown as unverified (not scored).")

    progress(1.0, "Done")
    return {
        "kind": "web_scan", "status": status, "incomplete_reason": reason,
        "document_name": document_name,
        "scanned_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "sentences": sentence_rows, "matches": all_matches, "sources": sources,
        "stats": stats, "pipeline": pipeline, "warnings": warnings,
        "settings": {k: settings[k] for k in settings},
    }


# --------------------------------------------------------------------------
# 13. Two-document comparison (no internet needed)
# --------------------------------------------------------------------------
def compare_documents(model, text_a, text_b, name_a, name_b,
                      semantic_threshold=0.70, near_threshold=0.90):
    """Compare every sentence of document A against document B."""
    sentences_a = extract_sentences(text_a)[:MAX_COMPARE_SENTENCES]
    sentences_b = extract_sentences(text_b)[:MAX_COMPARE_SENTENCES]
    if not sentences_a or not sentences_b:
        raise ValueError(f"Both documents must contain at least one sentence with {MIN_WORDS}+ words.")
    emb_a = encode_sentences(model, sentences_a)
    emb_b = encode_sentences(model, sentences_b)

    everything = list(range(len(sentences_a)))
    matches = detect_exact_matches(sentences_a, sentences_b, name_b, "", everything)
    done = {m["index"] for m in matches}
    remaining = [i for i in everything if i not in done]
    near = detect_near_exact_matches(sentences_a, sentences_b, name_b, "", near_threshold, remaining)
    matches += near
    done |= {m["index"] for m in near}
    remaining = [i for i in everything if i not in done]
    matches += detect_semantic_matches(model, sentences_a, sentences_b, name_b, "",
                                       semantic_threshold, remaining, emb_a, emb_b)
    finalize_semantic_matches(matches)

    similarity = calculate_similarity(emb_a, emb_b)
    best_scores = similarity.max(axis=1)
    rows, stats = summarize_matches(sentences_a, matches)
    doc_similarity = calculate_similarity(emb_a.mean(axis=0, keepdims=True),
                                          emb_b.mean(axis=0, keepdims=True))[0][0]
    stats.update({"sentences_b": len(sentences_b),
                  "mean_best_similarity": round(float(best_scores.mean()) * 100, 1),
                  "document_similarity": round(float(doc_similarity) * 100, 1)})
    return {"kind": "comparison", "name_a": name_a, "name_b": name_b, "sentences": rows,
            "matches": matches, "best_scores": [round(float(s) * 100, 2) for s in best_scores],
            "stats": stats}


# --------------------------------------------------------------------------
# 14-15. SQLite history
# --------------------------------------------------------------------------
def init_db():
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS scans (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                scan_date TEXT NOT NULL,
                document_name TEXT,
                total_sentences INTEGER,
                exact_matches INTEGER,
                near_exact_matches INTEGER,
                semantic_matches INTEGER,
                verified_sources INTEGER,
                sources_with_matches INTEGER,
                overall_similarity REAL,
                exact_pct REAL,
                semantic_pct REAL,
                details TEXT
            )""")
        columns = {row[1] for row in conn.execute("PRAGMA table_info(scans)")}
        if "status" not in columns:  # upgrade databases created by the first version
            conn.execute("ALTER TABLE scans ADD COLUMN status TEXT")


def save_scan_history(result):
    """Store a scan summary (plus full details as JSON). Returns the new scan ID."""
    s = result["stats"]
    incomplete = result.get("status") == "incomplete"
    with sqlite3.connect(DB_PATH) as conn:
        cursor = conn.execute(
            """INSERT INTO scans (scan_date, document_name, total_sentences, exact_matches,
               near_exact_matches, semantic_matches, verified_sources, sources_with_matches,
               overall_similarity, exact_pct, semantic_pct, details, status)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (result["scanned_at"], result["document_name"], s["total_sentences"], s["exact"],
             s["near_exact"], s["semantic"], s["verified_sources"], s["sources_with_matches"],
             None if incomplete else s["overall_pct"], None if incomplete else s["exact_pct"],
             None if incomplete else s["semantic_pct"], json.dumps(result), result.get("status")))
        return cursor.lastrowid


def get_scan_history(limit=200):
    """Return previous scans (summary columns only), newest first. Incomplete scans have no %."""
    legacy_incomplete = "(status IS NULL AND verified_sources = 0)"
    hide = f"(status = 'incomplete' OR {legacy_incomplete})"
    with sqlite3.connect(DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            f"""SELECT id AS "Scan ID", scan_date AS "Date/Time", document_name AS "Document",
               CASE WHEN status IS NOT NULL THEN status
                    WHEN verified_sources = 0 THEN 'incomplete (older version)'
                    ELSE 'completed (older version)' END AS "Status",
               total_sentences AS "Sentences", exact_matches AS "Exact",
               near_exact_matches AS "Near-exact", semantic_matches AS "Semantic",
               verified_sources AS "Verified sources", sources_with_matches AS "Sources with matches",
               CASE WHEN {hide} THEN NULL ELSE overall_similarity END AS "Overall %",
               CASE WHEN {hide} THEN NULL ELSE exact_pct END AS "Exact/near %",
               CASE WHEN {hide} THEN NULL ELSE semantic_pct END AS "Semantic %"
               FROM scans ORDER BY id DESC LIMIT ?""", (limit,)).fetchall()
    return [dict(r) for r in rows]


def get_scan_details(scan_id):
    with sqlite3.connect(DB_PATH) as conn:
        row = conn.execute("SELECT details FROM scans WHERE id = ?", (scan_id,)).fetchone()
    return json.loads(row[0]) if row and row[0] else None


def _normalize_result(result):
    """Fill in fields that results saved by the first version of the app do not have."""
    stats = result.setdefault("stats", {})
    for key in ("total_sentences", "exact", "near_exact", "semantic", "none", "verified_sources",
                "sources_with_matches", "unverified_sources"):
        stats.setdefault(key, 0)
    for key in ("exact_pct", "semantic_pct", "overall_pct"):
        stats.setdefault(key, 0.0)
    if "status" not in result:
        if stats["verified_sources"] == 0:
            result["status"] = "incomplete"
            result.setdefault("incomplete_reason",
                              "This scan was saved by an older version of the app and no webpage was "
                              "successfully compared, so its 0% was not a real result.")
        else:
            result["status"] = "matches" if stats["overall_pct"] > 0 else "no_matches"
    result.setdefault("incomplete_reason", "")
    result.setdefault("pipeline", {})
    for key in ("queries_planned", "queries_ok", "results_retrieved", "pages_attempted",
                "pages_downloaded", "pages_compared", "sources_with_matches"):
        result["pipeline"].setdefault(key, 0)
    result["pipeline"].setdefault("queries", [])
    result["pipeline"].setdefault("search_errors", [])
    result["pipeline"].setdefault("engine", "unknown")
    for key in ("warnings", "sources", "matches", "sentences"):
        result.setdefault(key, [])
    if "semantic_any" not in stats:                    # scans saved before the breakdown existed
        stats.update(semantic_breakdown(result["sentences"], result["matches"]))
    stats.setdefault("topical_sentences", len({m["index"] for m in result["matches"] if m["type"] == TOPICAL}))
    stats.setdefault("topical_pairs", len({(m["index"], m.get("url") or m.get("title") or "")
                                           for m in result["matches"] if m["type"] == TOPICAL}))
    for source in result["sources"]:
        source.setdefault("note", "")
        source.setdefault("compared_sentences", 0)
    return result


# --------------------------------------------------------------------------
# 16. PDF report
# --------------------------------------------------------------------------
class ReportPDF(FPDF):
    def footer(self):
        self.set_y(-15)
        self.set_font("Helvetica", "I", 8)
        self.cell(0, 10, f"{APP_NAME} report - page {self.page_no()}", align="C")


def _latin(text):
    """FPDF core fonts only support Latin-1; replace anything else safely."""
    text = text or ""
    for src, dst in {"\u2018": "'", "\u2019": "'", "\u201c": '"', "\u201d": '"',
                     "\u2013": "-", "\u2014": "-", "\u2026": "...", "\u00a0": " "}.items():
        text = text.replace(src, dst)
    text = text.encode("latin-1", "replace").decode("latin-1")
    return re.sub(r"(\S{70})(?=\S)", r"\1 ", text)


def _pdf_text(pdf, text, size=9, style="", height=4.5):
    pdf.set_font("Helvetica", style, size)
    pdf.multi_cell(0, height, _latin(text), new_x="LMARGIN", new_y="NEXT")


def _pdf_link(pdf, url):
    """Print a URL as clickable lines (each line links to the full URL)."""
    pdf.set_text_color(0, 0, 200)
    pdf.set_font("Helvetica", "U", 8)
    for k in range(0, len(url), 95):
        pdf.cell(0, 4, _latin(url[k:k + 95]), link=url, new_x="LMARGIN", new_y="NEXT")
    pdf.set_text_color(0, 0, 0)


def _pdf_match_block(pdf, m, counted_as=None):
    _pdf_text(pdf, f"{m['type']} - {m['score']:.1f}% ({m['level']})", 9, "B")
    if m["type"] == TOPICAL:
        _pdf_text(pdf, "(Not counted in the statistics: similar meaning only, no evidence of copying.)", 8, "I")
    elif counted_as and counted_as != m["type"]:
        _pdf_text(pdf, f"(This sentence is counted in the statistics as: {counted_as}, "
                       "because another source matches it more strongly.)", 8, "I")
    _pdf_text(pdf, "Your sentence: " + m["sentence"])
    _pdf_text(pdf, "Matching passage: " + m["matched"])
    _pdf_text(pdf, "Source: " + (m["title"] or "(title unavailable)"), 8, "I")
    if m.get("url"):
        _pdf_link(pdf, m["url"])
    pdf.ln(2)


def generate_pdf_report(result):
    """Build the PDF report and return it as bytes."""
    result = _normalize_result(result)
    stats, pipe = result["stats"], result["pipeline"]
    pdf = ReportPDF()
    pdf.set_auto_page_break(auto=True, margin=18)
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 18)
    pdf.cell(0, 10, f"{APP_NAME} - Plagiarism Scan Report", new_x="LMARGIN", new_y="NEXT")
    _pdf_text(pdf, f"Scan date: {result['scanned_at']}", 10)
    _pdf_text(pdf, f"Document: {result['document_name']}", 10)
    pdf.ln(2)

    pdf.set_font("Helvetica", "B", 12)
    pdf.cell(0, 8, "Scan status", new_x="LMARGIN", new_y="NEXT")
    if result["status"] == "incomplete":
        _pdf_text(pdf, NOT_FOUND_MESSAGE, 10, "B")
        _pdf_text(pdf, "Reason: " + (result["incomplete_reason"] or "unknown"), 9)
    elif result["status"] == "no_matches":
        _pdf_text(pdf, "Scan completed. No significant matches were found in the verified sources "
                       "that were compared. This does not prove the text is original.", 10)
    else:
        _pdf_text(pdf, "Scan completed. Matches were found (see below).", 10)
    pdf.ln(1)
    for line in (f"Searches succeeded: {pipe['queries_ok']}/{pipe['queries_planned']}   |   "
                 f"Search results retrieved: {pipe['results_retrieved']}",
                 f"Webpages downloaded: {pipe['pages_downloaded']}   |   "
                 f"Webpages compared: {pipe['pages_compared']}   |   "
                 f"Sources containing matches: {pipe['sources_with_matches']}"):
        _pdf_text(pdf, line, 9)
    pdf.ln(2)

    if result["status"] != "incomplete":
        pdf.set_font("Helvetica", "B", 12)
        pdf.cell(0, 8, "Overall statistics", new_x="LMARGIN", new_y="NEXT")
        for line in report_stat_lines(stats):
            _pdf_text(pdf, line, 10)
        _pdf_text(pdf, "Note: semantic scores are BERT cosine similarities. They show similar "
                       "meaning, not proof of plagiarism. Review every match manually.", 8, "I")
        pdf.ln(2)

    pdf.set_font("Helvetica", "B", 12)
    pdf.cell(0, 8, "Sources", new_x="LMARGIN", new_y="NEXT")
    if not result["sources"]:
        _pdf_text(pdf, "No sources were retrieved.")
    for s in result["sources"]:
        status = (f"Verified - {s['matches']} matching sentence(s)" if s["verified"]
                  else f"UNVERIFIED (not scored): {s['note']}")
        _pdf_text(pdf, s["title"], 9, "B")
        _pdf_link(pdf, s["url"])
        _pdf_text(pdf, status, 8, "I")
        pdf.ln(1)

    limit = 150
    by_type = {t: sorted([m for m in result["matches"] if m["type"] == t], key=lambda m: -m["score"])
               for t in (EXACT, NEAR, SEMANTIC, TOPICAL)}
    status_of = {i: row["status"] for i, row in enumerate(result["sentences"])}
    for heading, items in (("Exact matches", by_type[EXACT]),
                           ("Near-exact matches", by_type[NEAR]),
                           ("Semantic matches (BERT)", by_type[SEMANTIC]),
                           ("Topical similarity, not counted", by_type[TOPICAL])):
        unique_sentences = len({m["index"] for m in items})
        pdf.ln(2)
        pdf.set_font("Helvetica", "B", 12)
        pdf.cell(0, 8, f"{heading} ({len(items)} source match(es) for {unique_sentences} sentence(s))",
                 new_x="LMARGIN", new_y="NEXT")
        if not items:
            _pdf_text(pdf, "None found.")
        elif heading.startswith("Topical"):
            _pdf_text(pdf, "Related in meaning but with no evidence of copying (no matching neighbouring "
                           "sentences, no distinctive shared wording, or a statement found in many "
                           "sources). Not counted in the statistics; review manually.", 8, "I")
        for m in items[:limit]:
            _pdf_match_block(pdf, m, status_of.get(m["index"]))
        if len(items) > limit:
            _pdf_text(pdf, f"... {len(items) - limit} more not shown.", 8, "I")
    return bytes(pdf.output())


# --------------------------------------------------------------------------
# 17. Display
# --------------------------------------------------------------------------
def _highlight_html(sentence_rows):
    parts = []
    for row in sentence_rows:
        text = html.escape(row["text"])
        if row["status"] in HIGHLIGHT_BG:
            tip = f"{row['status']} - {row['score']:.1f}%"
            if row.get("semantic_sources") and row["status"] != SEMANTIC:
                tip += f" (also similar in meaning to {row['semantic_sources']} other source(s))"
            parts.append(f'<span title="{tip}" style="background:{HIGHLIGHT_BG[row["status"]]};'
                         f'color:#111;padding:2px 4px;border-radius:4px;'
                         f'border-bottom:2px solid {COLORS[row["status"]]};">{text}</span>')
        else:
            parts.append(text)
    return '<div style="line-height:2;font-size:0.95rem">' + " ".join(parts) + "</div>"


def _legend():
    st.markdown(
        f'<span style="background:{HIGHLIGHT_BG[EXACT]};color:#111;padding:2px 8px;border-radius:4px">'
        f'Exact</span> &nbsp; '
        f'<span style="background:{HIGHLIGHT_BG[NEAR]};color:#111;padding:2px 8px;border-radius:4px">'
        f'Near-exact</span> &nbsp; '
        f'<span style="background:{HIGHLIGHT_BG[SEMANTIC]};color:#111;padding:2px 8px;border-radius:4px">'
        f'Semantic (similar meaning)</span> &nbsp; Plain text = no significant match',
        unsafe_allow_html=True)


def _donut_data(stats):
    """One slice per sentence status; slices always add up to the number of sentences."""
    labels = ["Exact", "Near-exact", "Semantic only", "No significant match"]
    values = [stats["exact"], stats["near_exact"], stats["semantic"], stats["none"]]
    return labels, values


def _donut(stats, key):
    labels, values = _donut_data(stats)
    fig = go.Figure(go.Pie(labels=labels, values=values, hole=0.5, sort=False,
                           marker=dict(colors=[COLORS[EXACT], COLORS[NEAR], COLORS[SEMANTIC], COLORS[NONE]])))
    fig.update_layout(title="Sentence classification", height=340, margin=dict(t=50, b=10))
    st.plotly_chart(fig, key=key)


def _match_rows(matches):
    return [{"Match type": m["type"], "Similarity %": m["score"], "Your sentence": m["sentence"],
             "Matching passage": m["matched"], "Source title": m["title"],
             "Source URL": m["url"] or None}
            for m in sorted(matches, key=lambda m: (-RANK[m["type"]], -m["score"]))]


def _show_pipeline(pipe):
    cols = st.columns(5)
    cols[0].metric("Searches succeeded", f"{pipe['queries_ok']}/{pipe['queries_planned']}")
    cols[1].metric("Search results retrieved", pipe["results_retrieved"])
    cols[2].metric("Webpages downloaded", f"{pipe['pages_downloaded']}/{pipe['pages_attempted']}")
    cols[3].metric("Webpages compared", pipe["pages_compared"])
    cols[4].metric("Sources with matches", pipe["sources_with_matches"])
    if pipe["queries"]:
        with st.expander(f"Search queries used (engine: {pipe['engine']})"):
            for q in pipe["queries"]:
                state = f"{q['hits']} result(s) via {q['engine']}" if q["ok"] else "FAILED"
                st.markdown(f"- **{q['type']}** - `{q['query']}` -> {state}")


def _show_sources(result):
    st.markdown("#### Sources")
    if not result["sources"]:
        st.write("No sources were retrieved.")
    for s in result["sources"]:
        if s["verified"]:
            label = f"{s['matches']} matching sentence(s)" if s["matches"] else "no significant match"
            kind = {"html+pdf": "page + full-text PDF", "pdf": "PDF", "html": "webpage"}.get(s.get("kind"), "")
            st.markdown(f"- \u2705 **Verified** ({kind}) - [{s['title']}]({s['url']}) - {label} "
                        f"({s['compared_sentences']} sentences compared)")
        else:
            st.markdown(f"- \u26a0\ufe0f **Unverified (not scored)** - [{s['title']}]({s['url']}) - {s['note']}")


def display_results(result, key_prefix="scan"):
    """Render a web-scan result (new scans and history). Handles all three scan states."""
    result = _normalize_result(result)
    stats, pipe = result["stats"], result["pipeline"]
    st.subheader(f"Results: {result['document_name']}")
    st.caption(f"Scanned on {result['scanned_at']}")

    if result["status"] == "incomplete":
        st.error(NOT_FOUND_MESSAGE)
        st.markdown(f"**Reason:** {result['incomplete_reason'] or 'unknown'}")
        st.info("Similarity percentages are intentionally not shown: nothing was compared, so a "
                "0% would be misleading. Open *Search diagnostics* in the sidebar to test the "
                "connection, then scan again.")
        for warning in result["warnings"]:
            st.warning(warning)
        _show_pipeline(pipe)
        _show_sources(result)
    else:
        for warning in result["warnings"]:
            st.warning(warning)
        _show_pipeline(pipe)
        if result["status"] == "no_matches":
            st.success(f"Scan completed: no significant matches were found in the "
                       f"{stats['verified_sources']} verified source(s) that were compared. "
                       "This is not proof of originality; only the pages that were retrieved were checked.")
        cols = st.columns(4)
        cols[0].metric("Overall similarity", f"{stats['overall_pct']}%")
        cols[1].metric("Exact / near-exact", f"{stats['exact_pct']}%")
        cols[2].metric("Semantic only", f"{stats['semantic_pct']}%")
        cols[3].metric("Verified sources compared", stats["verified_sources"])
        st.caption("Percentages = share of scanned sentences. Each sentence is counted once, by its "
                   "strongest match. Semantic scores are BERT cosine similarities and are **not** "
                   "proof of plagiarism.")
        if stats.get("semantic_also", 0):
            st.caption(f"{stats['semantic_any']} sentence(s) have a semantic match in at least one source; "
                       f"{stats['semantic_also']} of them are already counted as exact/near-exact "
                       "because another source matches them more strongly.")

        left, right = st.columns(2)
        with left:
            _donut(stats, f"{key_prefix}_donut")
        with right:
            per_source = [(s["title"][:40] or s["url"][:40], s["matches"])
                          for s in result["sources"] if s["verified"] and s["matches"] > 0]
            if per_source:
                fig = go.Figure(go.Bar(x=[c for _, c in per_source], y=[t for t, _ in per_source],
                                       orientation="h", marker_color="#4f46e5"))
                fig.update_layout(title="Matching sentences per source", height=340,
                                  margin=dict(t=50, b=10), yaxis=dict(autorange="reversed"))
                st.plotly_chart(fig, key=f"{key_prefix}_bar")
            else:
                st.info("No verified source contained a significant match.")

        st.markdown("#### Highlighted text")
        _legend()
        st.markdown(_highlight_html(result["sentences"]), unsafe_allow_html=True)
        _show_sources(result)

        st.markdown("#### Detailed comparison")
        rows = _match_rows(result["matches"])
        if rows:
            st.caption(f"{len(rows)} source match(es) for {len({m['index'] for m in result['matches']})} "
                       "sentence(s): one row per sentence and source, so a sentence found in several "
                       "sources appears several times but is counted once in the statistics."
                       + (f" {stats['topical_pairs']} row(s) are Topical Similarity: related in meaning but "
                          "with no evidence of copying, so they are not counted." if stats.get("topical_pairs") else ""))
            st.dataframe(rows, column_config={
                "Source URL": st.column_config.LinkColumn("Source URL"),
                "Similarity %": st.column_config.NumberColumn(format="%.1f")})
        else:
            st.write("No matches found in the verified sources.")

    if result.get("settings"):
        with st.expander("Settings used for this scan"):
            st.json(result["settings"])
    try:
        st.download_button("\u2b07\ufe0f Download PDF report", data=generate_pdf_report(result),
                           file_name=f"plagiarismai_report_{result['scanned_at'][:10]}.pdf",
                           mime="application/pdf", key=f"{key_prefix}_pdf")
    except Exception as exc:
        st.error(f"Could not build the PDF report: {exc}")


def display_comparison(result):
    stats = result["stats"]
    st.subheader(f"{result['name_a']}  vs  {result['name_b']}")
    cols = st.columns(5)
    cols[0].metric("Sentences in A matched in B", f"{stats['overall_pct']}%")
    cols[1].metric("Exact / near-exact", f"{stats['exact_pct']}%")
    cols[2].metric("Semantic only", f"{stats['semantic_pct']}%")
    cols[3].metric("Mean best similarity", f"{stats['mean_best_similarity']}%")
    cols[4].metric("Whole-document similarity", f"{stats['document_similarity']}%")
    st.caption(f"A has {stats['total_sentences']} sentences, B has {stats['sentences_b']}. "
               "Similarity measures meaning overlap, not proof of copying.")
    left, right = st.columns(2)
    with left:
        _donut(stats, "cmp_donut")
    with right:
        fig = go.Figure(go.Histogram(x=result["best_scores"], nbinsx=20, marker_color="#4f46e5"))
        fig.update_layout(title="Best cosine similarity per sentence of A (%)", height=340,
                          margin=dict(t=50, b=10), xaxis_title="Similarity %", yaxis_title="Sentences")
        st.plotly_chart(fig, key="cmp_hist")
    st.markdown("#### Highlighted text of document A")
    _legend()
    st.markdown(_highlight_html(result["sentences"]), unsafe_allow_html=True)
    st.markdown("#### Matching sentences")
    rows = [{k: v for k, v in r.items() if k not in ("Source title", "Source URL")}
            for r in _match_rows(result["matches"])]
    st.dataframe(rows) if rows else st.write("No matching sentences found.")


# --------------------------------------------------------------------------
# 18. Main app
# --------------------------------------------------------------------------
def _get_model():
    try:
        with st.spinner("Loading BERT model (the first run downloads about 90 MB)..."):
            return load_bert_model()
    except Exception as exc:
        st.error(f"Could not load the BERT model: {exc}\n\nThe first run needs internet to download "
                 f"`{MODEL_NAME}`. Check your connection and that `sentence-transformers` and "
                 "`torch` are installed.")
        return None


def _provider_status_line(pid):
    spec = PROVIDERS[pid]
    key, source = get_secret(spec["env"])
    if key:
        return f"\u2705 **{spec['label']}** - key configured ({source})"
    return f"\u274c **{spec['label']}** - `{spec['env']}` not set"


def _search_diagnostics(provider_choice):
    """Sidebar panel: provider status, key status, and an independent connection + search test."""
    with st.sidebar.expander("Search diagnostics"):
        info = resolve_provider(provider_choice)
        st.markdown(f"**Active provider:** {info['label']}")
        if info["configured"]:
            st.success("Provider is configured" + (f" - key from {info['source']}" if info["source"] else ""))
        else:
            st.error(info["problem"])
        st.markdown(_provider_status_line("brave"))
        st.markdown(_provider_status_line("tavily"))
        st.text_input("Brave API key (this session only)", type="password", key="session_key_BRAVE_API_KEY")
        st.text_input("Tavily API key (this session only)", type="password", key="session_key_TAVILY_API_KEY")
        st.caption("Keys typed here are kept in memory for this session only and are never saved. "
                   "For permanent setup use BRAVE_API_KEY / TAVILY_API_KEY in a .env file.")
        query = st.text_input("Test query", value='"Federated Learning is a distributed learning '
                                                  'paradigm with two key challenges"', key="diag_query")
        if st.button("Test provider connection and search", key="diag_btn"):
            with st.spinner("Contacting the search provider..."):
                test = test_search_provider(provider_choice, query)
            if not test["configured"]:
                st.error("Not configured: " + (test["errors"][0] if test["errors"] else ""))
            elif test["connected"]:
                st.success(f"Connection OK ({test['seconds']} s)")
                st.write(f"Results retrieved: **{test['result_count']}**")
                for r in test["results"]:
                    st.markdown(f"- [{r['title'] or r['url']}]({r['url']})")
                if test["result_count"] == 0:
                    st.warning("The provider answered but returned zero results for this query.")
            else:
                st.error("Provider connection or search FAILED")
            for message in test["errors"] if test["configured"] else []:
                st.caption(message)


def _sidebar_settings():
    st.sidebar.header("Settings")
    labels = {"auto": "Auto (first configured: Brave, then Tavily)", "brave": "Brave Search API",
              "tavily": "Tavily Search API", "ddgs": "DuckDuckGo via ddgs (unofficial)"}
    provider = st.sidebar.selectbox("Search provider", PROVIDER_CHOICES, format_func=labels.get,
                                    key="provider_choice")
    status = resolve_provider(provider)
    if status["configured"]:
        st.sidebar.caption(f"\u2705 Using {status['label']}")
    else:
        st.sidebar.error("Search provider not configured. Open *Search diagnostics* below.")
    semantic_pct = st.sidebar.slider("Semantic similarity threshold (%)", 50, 95, 70,
                                     help="Default 70%. 90+ very high, 80-89 high, 70-79 moderate.")
    near_pct = st.sidebar.slider("Near-exact threshold (%)", 80, 99, 90)
    max_queries = st.sidebar.slider("Search queries", 1, 15, 8,
                                    help="Mix of exact-phrase, distinctive-phrase and keyword queries.")
    results_per_query = st.sidebar.slider("Results per query", 1, 10, 5)
    max_pages = st.sidebar.slider("Max webpages to download", 1, 30, 12)
    max_sentences = st.sidebar.slider("Max sentences scanned", 20, 500, 200)
    st.sidebar.caption("More queries/pages = better coverage but slower scans.")
    _search_diagnostics(provider)
    return {"provider": provider, "semantic_threshold": semantic_pct / 100, "near_threshold": near_pct / 100,
            "max_queries": max_queries, "results_per_query": results_per_query,
            "max_pages": max_pages, "max_sentences": max_sentences}


def _scan_tab(settings):
    st.write("Paste text or upload a PDF, DOCX or TXT file. The app searches the web, downloads the "
             "real pages it finds, and compares them with your text using exact matching and BERT.")
    pasted = st.text_area("Paste your text", height=220, key="scan_text")
    upload = st.file_uploader("...or upload a document", type=["pdf", "docx", "txt"], key="scan_file")
    st.caption("If a file is uploaded, it is used instead of the pasted text.")

    if st.button("Start Plagiarism Scan", type="primary", key="scan_btn"):
        try:
            if upload is not None:
                text, name = extract_text_from_upload(upload), upload.name
            else:
                text, name = pasted, "Pasted text"
        except Exception as exc:
            st.error(f"Could not read the file: {exc}")
            return
        if not text or not text.strip():
            st.error("Please paste some text or upload a document first.")
            return
        model = _get_model()
        if model is None:
            return
        bar = st.progress(0.0, text="Starting...")
        try:
            result = run_scan(model, text, name, settings,
                              lambda f, m: bar.progress(min(max(f, 0.0), 1.0), text=m))
            result["scan_id"] = save_scan_history(result)
            st.session_state["scan_result"] = result
        except Exception as exc:
            st.error(f"Scan failed: {exc}")
        finally:
            bar.empty()
    if "scan_result" in st.session_state:
        display_results(st.session_state["scan_result"], "scan")


def _compare_tab(settings):
    st.write("Upload two documents to compare them directly. This works offline "
             "(after the model has been downloaded once).")
    col_a, col_b = st.columns(2)
    file_a = col_a.file_uploader("Document A (checked against B)", type=["pdf", "docx", "txt"], key="cmp_a")
    file_b = col_b.file_uploader("Document B (reference)", type=["pdf", "docx", "txt"], key="cmp_b")
    if st.button("Compare documents", type="primary", key="cmp_btn"):
        if file_a is None or file_b is None:
            st.error("Please upload both documents.")
            return
        try:
            text_a = extract_text_from_upload(file_a)
            text_b = extract_text_from_upload(file_b)
        except Exception as exc:
            st.error(f"Could not read a file: {exc}")
            return
        model = _get_model()
        if model is None:
            return
        try:
            with st.spinner("Comparing with BERT..."):
                st.session_state["cmp_result"] = compare_documents(
                    model, text_a, text_b, file_a.name, file_b.name,
                    settings["semantic_threshold"], settings["near_threshold"])
        except Exception as exc:
            st.error(f"Comparison failed: {exc}")
    if "cmp_result" in st.session_state:
        display_comparison(st.session_state["cmp_result"])


def _history_tab():
    rows = get_scan_history()
    if not rows:
        st.info("No scans saved yet.")
        return
    st.dataframe(rows)
    options = {f"#{r['Scan ID']} - {r['Document']} - {r['Date/Time']} - {r['Status']}": r["Scan ID"]
               for r in rows}
    choice = st.selectbox("Open a previous scan", list(options.keys()), key="hist_select")
    details = get_scan_details(options[choice])
    if details:
        display_results(details, f"hist_{options[choice]}")
    else:
        st.warning("Details for this scan are not available.")


def _about_tab():
    st.markdown(f"""
**Scan states**

| State | Meaning |
|---|---|
| Matches found | At least one verified source contained exact, near-exact or semantic matches |
| No matches | Pages were downloaded and compared; nothing significant was found |
| Incomplete | Search or downloads failed. **No conclusion is given and no 0% is shown** |

**How matches are classified**

| Type | Meaning |
|---|---|
| Exact Match | Identical after lowercasing, punctuation, citation-mark and whitespace cleanup |
| Near-Exact Match | Character-level similarity at or above the near-exact threshold (default 90%) |
| Semantic Match | BERT cosine similarity at or above the threshold (default 70%) with different wording |

**Semantic bands:** 90-100% very high, 80-89% high, 70-79% moderate.

**Sources:** only pages that were really downloaded and compared are scored. Blocked or unreadable
pages are listed as *unverified*. When a page advertises its full-text PDF (standard
`citation_pdf_url` tag), that PDF is downloaded too.

**Search providers:** Brave Search API and Tavily Search API (API key required; Bing's search API was retired by Microsoft in August 2025). Keys are read from environment variables, a `.env` file or Streamlit secrets and are never stored by this app. Search snippets are used only for ranking; a source is scored only after its page or PDF was downloaded and compared.

**Limits:** search only finds what the search engine indexes; paywalled papers and pages that block
bots cannot be compared. Model: `{MODEL_NAME}`.
""")


def main():
    st.set_page_config(page_title=f"{APP_NAME} - Semantic Plagiarism Detection",
                       page_icon="\U0001F50E", layout="wide")
    init_db()
    st.title("\U0001F50E PlagiarismAI")
    st.caption("BERT-based semantic plagiarism detection with real web sources")
    settings = _sidebar_settings()
    tabs = st.tabs(["Plagiarism Scan", "Compare Documents", "Scan History", "About"])
    with tabs[0]:
        _scan_tab(settings)
    with tabs[1]:
        _compare_tab(settings)
    with tabs[2]:
        _history_tab()
    with tabs[3]:
        _about_tab()


def run_cli_search_test(query=None):
    """python app.py --test-search ["query"]   -> real provider connection + search test."""
    test = test_search_provider("auto", query)
    print(f"Provider        : {test['provider']}")
    print(f"Configured      : {test['configured']} {('(' + test['key_source'] + ')') if test['key_source'] else ''}")
    print(f"Query           : {test['query']}")
    print(f"Connection OK   : {test['connected']}  ({test['seconds']} s)")
    print(f"Results retrieved: {test['result_count']}")
    for r in test["results"]:
        print(f"  - {r['title']}\n    {r['url']}")
    for message in test["errors"]:
        print("ERROR/NOTE:", message)
    return 0 if test["result_count"] > 0 else 1


def run_cli_scan(path):
    """python app.py --test-scan passage.txt   -> full real scan (BERT + web) printed to the terminal."""
    text = open(path, encoding="utf-8").read()
    model = load_bert_model()
    settings = {"provider": "auto", "semantic_threshold": 0.70, "near_threshold": 0.90, "max_queries": 8,
                "results_per_query": 5, "max_pages": 12, "max_sentences": 200}
    result = run_scan(model, text, os.path.basename(path), settings,
                      lambda fraction, message: print(f"[{fraction:4.0%}] {message}"))
    pipe, stats = result["pipeline"], result["stats"]
    print(f"\nSTATUS: {result['status'].upper()}  {result['incomplete_reason']}")
    print(f"Provider: {pipe['engine']} | searches ok {pipe['queries_ok']}/{pipe['queries_planned']} | "
          f"results {pipe['results_retrieved']} | downloaded {pipe['pages_downloaded']} | "
          f"compared {pipe['pages_compared']} | sources with matches {pipe['sources_with_matches']}")
    if result["status"] != "incomplete":
        print(f"Overall {stats['overall_pct']}% | exact/near {stats['exact_pct']}% | "
              f"semantic-only {stats['semantic_pct']}% | any semantic {stats['semantic_any_pct']}%")
    for source in result["sources"]:
        state = "VERIFIED" if source["verified"] else "UNVERIFIED: " + source["note"]
        print(f"  [{state}] {source['title']} - {source['url']} ({source['matches']} matches)")
    return 0 if result["status"] != "incomplete" else 1


if __name__ == "__main__" and "--test-search" in sys.argv:
    _args = sys.argv[sys.argv.index("--test-search") + 1:]
    sys.exit(run_cli_search_test(_args[0] if _args else None))
elif __name__ == "__main__" and "--test-scan" in sys.argv:
    sys.exit(run_cli_scan(sys.argv[sys.argv.index("--test-scan") + 1]))
elif __name__ == "__main__":
    main()
