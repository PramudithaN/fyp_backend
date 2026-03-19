"""
News fetching and sentiment computation service.

Uses NewsAPI to fetch oil-related news and computes sentiment features.
Uses custom ProsusAI/finbert model for sentiment analysis (matching Colab training).

SENTIMENT COMPUTATION (matching Colab exactly):
1. Each article gets a sentiment score using finbert_sentiment_continuous()
2. Daily sentiment = simple mean of all article scores (NO within-day decay!)
3. Cross-day decay is applied later in sentiment_service.py
"""

import os
import math
import re
import hashlib
import numpy as np
import logging
import yake
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Tuple
from urllib.parse import urlsplit, urlunsplit

from app.config import (
    NEWSAPI_KEY as CONFIG_NEWSAPI_KEY,
    SENTIMENT_MODE,
    PEXELS_API_KEY,
    PEXELS_PER_PAGE,
    PEXELS_TIMEOUT_SECONDS,
)

logger = logging.getLogger(__name__)

# NewsAPI configuration
NEWSAPI_BASE_URL = "https://newsapi.org/v2/everything"
NEWSAPI_KEY = CONFIG_NEWSAPI_KEY  # From config.py

# Oil-related search terms (simplified for better results)
OIL_SEARCH_QUERY = "oil price OR crude oil OR brent OR OPEC"
PEXELS_SEARCH_URL = "https://api.pexels.com/v1/search"
MAX_PEXELS_LOOKUPS_PER_RUN = 200   # raised: 200 Pexels lookups per scrape run (free tier allows 200/hour)
DEFAULT_IMAGE_QUERY = "energy infrastructure"
DEFAULT_OIL_QUERY = "oil industry"
DEFAULT_REFINERY_QUERY = "oil refinery"
DEFAULT_CRUDE_QUERY = "crude oil"
DEFAULT_ARTICLE_IMAGE_URL = "https://images.pexels.com/photos/257700/pexels-photo-257700.jpeg"

HEADLINE_STOP_WORDS = {
    "the",
    "and",
    "are",
    "why",
    "who",
    "what",
    "when",
    "where",
    "which",
    "while",
    "than",
    "been",
    "being",
    "into",
    "for",
    "with",
    "from",
    "that",
    "this",
    "will",
    "have",
    "after",
    "amid",
    "tests",
    "test",
    "says",
    "said",
    "say",
    "rise",
    "rises",
    "fall",
    "falls",
    "new",
    "latest",
    "more",
    "over",
    "under",
    "near",
    "company",
    "companies",
    "group",
    "groups",
    "firm",
    "firms",
    "embracing",
    "business",
    "looks",
    "look",
    "post",
    "opportunity",
    "opportunities",
}

# Rule-first visual overrides for recurring headline domains.
DOMAIN_VISUAL_MAP = [
    (
        {"interest rate", "inflation", "fed", "federal reserve", "gdp", "recession"},
        "stock market trading floor",
        ["financial charts", "economy money"],
    ),
    (
        {"wildfire", "fire", "blaze", "burn", "forest fire"},
        "wildfire smoke aerial",
        ["forest fire flames", "fire emergency"],
    ),
    (
        {"flood", "flooding", "hurricane", "tornado", "earthquake", "storm", "tsunami"},
        "natural disaster aerial view",
        ["flood damage", "storm destruction"],
    ),
    (
        {"election", "vote", "ballot", "candidate", "polling", "democrat", "republican"},
        "voting booth ballot box",
        ["election campaign rally", "democracy vote"],
    ),
    (
        {"war", "military", "troops", "soldier", "attack", "conflict", "invasion", "airstrike"},
        "military soldiers deployment",
        ["conflict zone", "army troops"],
    ),
    (
        {"protest", "rally", "demonstration", "march", "riot", "demonstrators"},
        "protest crowd street",
        ["demonstration rally", "civil protest"],
    ),
    (
        {"vaccine", "covid", "pandemic", "virus", "outbreak", "disease", "infection"},
        "medical laboratory research",
        ["vaccine syringe", "hospital healthcare"],
    ),
    (
        {"ai", "artificial intelligence", "machine learning", "robot", "automation", "chatgpt", "openai"},
        "artificial intelligence technology",
        ["robot automation", "computer technology"],
    ),
    (
        {"semiconductor", "chip", "microchip", "processor", "nvidia", "intel", "apple silicon"},
        "semiconductor microchip closeup",
        ["computer processor", "technology chip"],
    ),
    (
        {"climate", "global warming", "carbon", "emissions", "renewable", "solar", "wind energy"},
        "climate change renewable energy",
        ["solar panels wind turbines", "green energy"],
    ),
    (
        {"oil", "gas", "energy", "opec", "fuel", "pipeline", "petroleum"},
        "oil refinery pipeline",
        ["energy fuel industry", "petroleum gas"],
    ),
    (
        {"trade", "tariff", "sanction", "export", "import", "supply chain"},
        "cargo shipping port containers",
        ["international trade", "logistics supply chain"],
    ),
    (
        {"space", "nasa", "rocket", "astronaut", "satellite", "moon", "mars", "launch"},
        "rocket launch space",
        ["astronaut space", "nasa mission"],
    ),
    (
        {"crypto", "bitcoin", "blockchain", "ethereum", "nft"},
        "cryptocurrency bitcoin digital",
        ["blockchain technology", "digital currency"],
    ),
    (
        {"merger", "acquisition", "ipo", "startup", "deal", "buyout"},
        "business handshake deal",
        ["corporate meeting boardroom", "business merger"],
    ),
    (
        {"hospital", "surgery", "cancer", "drug", "medicine", "treatment", "patient", "health"},
        "hospital medical care",
        ["doctor patient healthcare", "medical treatment"],
    ),
    (
        {"school", "university", "education", "student", "campus", "tuition"},
        "students classroom university",
        ["education learning", "school campus"],
    ),
    (
        {"immigration", "border", "migrant", "refugee", "asylum"},
        "border crossing migration",
        ["refugee camp", "immigration border"],
    ),
    (
        {"housing", "real estate", "mortgage", "rent", "property", "home price"},
        "real estate house neighborhood",
        ["housing market property", "home mortgage"],
    ),
    (
        {"food", "agriculture", "farm", "crop", "drought", "harvest", "famine"},
        "farm agriculture harvest",
        ["food crops field", "agriculture farming"],
    ),
    (
        {"cybersecurity", "hack", "data breach", "ransomware", "phishing"},
        "cybersecurity hacker dark",
        ["data security computer", "cyber attack"],
    ),
    (
        {"car", "electric vehicle", "ev", "tesla", "automotive", "autonomous"},
        "electric vehicle charging station",
        ["automobile car technology", "ev car"],
    ),
    (
        {"bank", "banking", "loan", "credit", "debt", "finance"},
        "bank building finance",
        ["banking money finance", "loan credit"],
    ),
]

ORIENTATION_HINTS = {
    "landscape": {"aerial", "panorama", "skyline", "field", "crowd", "city"},
    "portrait": {"person", "leader", "president", "ceo", "doctor", "soldier", "protester"},
}

yake_extractor = yake.KeywordExtractor(
    lan="en",
    n=3,
    dedupLim=0.7,
    top=10,
    features=None,
)

STOP_WORDS = {
    "say", "says", "said", "new", "first", "last", "year", "years",
    "week", "month", "day", "time", "amid", "after", "report", "reports",
    "billion", "million", "percent", "according", "could", "would", "may",
    "make", "get", "use", "take", "give", "come", "go", "set", "call",
    "show", "us", "u.s", "uk", "world", "official", "plan", "plans", "move",
}

KEYWORD_SYNONYMS = {
    "petrol": "oil",
    "gasoline": "oil",
    "diesel": "oil",
    "crude": "oil",
    "brent": "oil",
    "opec": "oil",
    "renewables": "renewable_energy",
    "renewable": "renewable_energy",
    "venezuelan": "venezuela",
    "iranian": "iran",
    "russian": "russia",
    "ukrainian": "ukraine",
    "saudi": "saudi_arabia",
    "emirati": "uae",
    "qatari": "qatar",
    "iraqi": "iraq",
    "american": "usa",
    "european": "europe",
    "chinese": "china",
    "indian": "india",
    "sanctions": "sanction",
    "wars": "war",
    "prices": "price",
    "markets": "market",
    "refinery": "oil_refinery",
    "refineries": "oil_refinery",
    "tankers": "oil_tanker",
    "tanker": "oil_tanker",
    "pipelines": "gas_pipeline",
    "pipeline": "gas_pipeline",
    "lng": "lng_terminal",
}

KEY_PHRASES = [
    ("natural gas", "natural_gas"),
    ("renewable energy", "renewable_energy"),
    ("energy sector", "energy_sector"),
    ("oil industry", "oil_industry"),
    ("oil refinery", "oil_refinery"),
    ("oil price", "oil_price"),
    ("fuel price", "fuel_price"),
    ("lng terminal", "lng_terminal"),
    ("shipping route", "shipping_route"),
    ("red sea", "red_sea"),
    ("strait of hormuz", "strait_of_hormuz"),
    ("gulf of mexico", "gulf_of_mexico"),
    ("north sea", "north_sea"),
    ("middle east", "middle_east"),
    ("supply cut", "supply_cut"),
    ("production cut", "production_cut"),
]

LOCATION_TERMS = {
    "venezuela",
    "iran",
    "russia",
    "ukraine",
    "middle_east",
    "saudi_arabia",
    "uae",
    "iraq",
    "qatar",
    "china",
    "india",
    "europe",
    "usa",
    "red_sea",
    "strait_of_hormuz",
    "gulf_of_mexico",
    "north_sea",
}

THEME_KEYWORDS = {
    "natural_gas": {"natural_gas", "gas_pipeline", "lng_terminal"},
    "renewable_energy": {"renewable_energy", "solar", "wind", "power_grid"},
    "shipping": {"shipping_route", "oil_tanker", "port", "terminal"},
    "oil": {
        "oil",
        "oil_price",
        "fuel_price",
        "oil_industry",
        "oil_refinery",
        "market",
        "supply_cut",
        "production_cut",
    },
}

GEOPOLITICAL_TERMS = {
    "war",
    "conflict",
    "sanction",
    "election",
    "government",
    "policy",
    "tariff",
    "trade",
}

THEME_QUERY_TEMPLATES = {
    "natural_gas": [
        "natural gas infrastructure",
        "gas pipeline",
        "lng terminal",
    ],
    "renewable_energy": [
        "renewable energy infrastructure",
        "solar wind farm",
        "energy grid",
    ],
    "shipping": [
        "oil tanker shipping",
        "energy port terminal",
        "crude oil tanker",
    ],
    "oil": [
        DEFAULT_OIL_QUERY,
        DEFAULT_REFINERY_QUERY,
        DEFAULT_CRUDE_QUERY,
        "offshore oil platform",
        "oil pumpjack",
    ],
}

BROAD_FALLBACK_IMAGE_QUERIES = [
    DEFAULT_REFINERY_QUERY,
    DEFAULT_CRUDE_QUERY,
    DEFAULT_IMAGE_QUERY,
]


def _dedupe_preserve_order(values: List[str]) -> List[str]:
    deduped: List[str] = []
    seen = set()
    for value in values:
        if not value or value in seen:
            continue
        seen.add(value)
        deduped.append(value)
    return deduped


def _clean_text(text: str) -> str:
    return re.sub(r"[\"'(){}\[\]]", "", text).strip().lower()


def _domain_match(headline_lower: str) -> Optional[Tuple[str, List[str]]]:
    for triggers, primary, fallbacks in DOMAIN_VISUAL_MAP:
        if any(trigger in headline_lower for trigger in triggers):
            return primary, fallbacks
    return None


def _yake_keywords(headline: str) -> List[str]:
    raw = yake_extractor.extract_keywords(headline)
    result: List[str] = []
    for keyword, score in sorted(raw, key=lambda item: item[1]):
        _ = score
        cleaned = _clean_text(keyword)
        words = cleaned.split()
        if cleaned and not all(word in STOP_WORDS for word in words):
            result.append(cleaned)
    return _dedupe_preserve_order(result)


def _infer_orientation(headline_lower: str) -> str:
    for orientation, hints in ORIENTATION_HINTS.items():
        if any(hint in headline_lower for hint in hints):
            return orientation
    return "landscape"


def _build_query_from_terms(terms: List[str], max_words: int = 6) -> str:
    seen: set[str] = set()
    parts: List[str] = []
    for term in terms:
        for word in term.split():
            if word in seen or word in STOP_WORDS:
                continue
            seen.add(word)
            parts.append(word)
            if len(parts) >= max_words:
                return " ".join(parts)
    return " ".join(parts)


def _headline_specific_terms(title: str, keywords: List[str], max_terms: int = 5) -> List[str]:
    """Extract additional non-generic terms to specialize image queries per headline."""
    generic_terms = {
        "oil",
        "energy",
        "industry",
        "market",
        "price",
        "prices",
        "business",
        "sector",
        "company",
        "companies",
        "group",
        "groups",
    }

    seen = set()
    terms: List[str] = []

    for keyword in keywords:
        if keyword in generic_terms:
            continue
        if keyword in seen:
            continue
        seen.add(keyword)
        terms.append(keyword)
        if len(terms) >= max_terms:
            return terms

    raw_words = re.findall(r"[a-zA-Z]{3,}", (title or "").lower())
    for raw in raw_words:
        normalized = KEYWORD_SYNONYMS.get(raw, raw)
        if normalized in generic_terms:
            continue
        if normalized in STOP_WORDS or normalized in HEADLINE_STOP_WORDS:
            continue
        if normalized in seen:
            continue
        seen.add(normalized)
        terms.append(normalized)
        if len(terms) >= max_terms:
            break

    return terms


def _build_headline_specific_query_variants(
    title: str,
    keywords: List[str],
    base_queries: List[str],
    max_variants: int = 4,
) -> List[str]:
    """Create query variants by appending headline-specific terms to strong base templates."""
    if not base_queries:
        return []

    specific_terms = _headline_specific_terms(title, keywords, max_terms=5)
    if not specific_terms:
        return []

    variants: List[str] = []
    for base_query in base_queries[:2]:
        base_words = set(base_query.split())
        for term in specific_terms:
            term_words = term.replace("_", " ").split()
            if all(word in base_words for word in term_words):
                continue
            variants.append(f"{base_query} {term.replace('_', ' ')}".strip())
            if len(variants) >= max_variants:
                return _dedupe_preserve_order(variants)

    return _dedupe_preserve_order(variants)


def _format_query_tokens(tokens: List[str]) -> str:
    return " ".join(token.replace("_", " ") for token in _dedupe_preserve_order(tokens)).strip()


def _ordered_phrase_matches(clean_title: str) -> List[str]:
    phrase_hits: List[Tuple[int, str]] = []
    for phrase, canonical in KEY_PHRASES:
        idx = clean_title.find(phrase)
        if idx >= 0:
            phrase_hits.append((idx, canonical))

    phrase_hits.sort(key=lambda item: item[0])
    return [phrase for _, phrase in phrase_hits]


def _matched_phrase_words(ordered_phrases: List[str]) -> set[str]:
    return {
        token
        for phrase, canonical in KEY_PHRASES
        if canonical in ordered_phrases
        for token in phrase.split()
    }


def _keyword_theme(keyword: str) -> Optional[str]:
    for theme, terms in THEME_KEYWORDS.items():
        if keyword in terms:
            return theme
    return None


def _rank_visual_themes(keywords: List[str]) -> List[str]:
    themes: List[str] = []
    for keyword in keywords:
        theme = _keyword_theme(keyword)
        if theme and theme not in themes:
            themes.append(theme)

    if not themes and (set(keywords) & GEOPOLITICAL_TERMS or any(k in LOCATION_TERMS for k in keywords)):
        themes.append("oil")

    if not themes:
        themes.append("oil")

    return themes


def _expand_query_templates(location: str, templates: List[str]) -> List[str]:
    prefixed = [f"{location.replace('_', ' ')} {template}" for template in templates] if location else []
    return [*prefixed, *templates]


def _build_structured_image_queries(keywords: List[str]) -> List[str]:
    """Build energy-specific image queries from normalized headline intent."""
    location = next((keyword for keyword in keywords if keyword in LOCATION_TERMS), "")
    themes = _rank_visual_themes(keywords)

    queries: List[str] = []
    for index, theme in enumerate(themes):
        templates = THEME_QUERY_TEMPLATES.get(theme, [])
        if not templates:
            continue

        include_location = bool(location) and (index == 0 or theme == "oil")
        queries.extend(_expand_query_templates(location if include_location else "", templates))

    if location and "oil" not in themes:
        queries.extend(_expand_query_templates(location, THEME_QUERY_TEMPLATES["oil"]))

    queries.extend([DEFAULT_IMAGE_QUERY, DEFAULT_OIL_QUERY])
    return _dedupe_preserve_order([query for query in queries if query])


def _normalize_image_url(url: str) -> str:
    """Trim tracking params to reduce storage and keep canonical image URLs."""
    if not url:
        return ""
    try:
        parsed = urlsplit(url.strip())
        return urlunsplit((parsed.scheme, parsed.netloc, parsed.path, "", ""))
    except Exception:
        return url.strip()


def _extract_headline_keywords(title: str) -> List[str]:
    """Extract key terms from a headline for image search intent."""
    clean_title = (title or "").lower().replace("'s", "")
    words = re.findall(r"[a-zA-Z]{3,}", clean_title)
    ordered_phrases = _ordered_phrase_matches(clean_title)
    matched_phrase_words = _matched_phrase_words(ordered_phrases)

    seen = set()
    keywords: List[str] = []

    for phrase in ordered_phrases:
        if phrase in seen:
            continue
        seen.add(phrase)
        keywords.append(phrase)

    # Keep unique words in order so repeated terms don't dominate.
    for raw_word in words:
        if raw_word in matched_phrase_words:
            continue

        if raw_word in HEADLINE_STOP_WORDS:
            continue

        word = KEYWORD_SYNONYMS.get(raw_word, raw_word)
        if word in seen:
            continue

        seen.add(word)
        keywords.append(word)
        if len(keywords) >= 8:
            break

    return keywords


def _build_image_search_query(title: str) -> str:
    """Build a Pexels query from headline keywords and domain context."""
    keywords = _extract_headline_keywords(title)
    queries = _build_structured_image_queries(keywords)
    if not queries:
        return DEFAULT_IMAGE_QUERY
    return queries[0]


def _build_fallback_image_queries(title: str) -> List[str]:
    """Build fallback Pexels queries from specific to broad."""
    headline_lower = (title or "").lower()
    keywords = _extract_headline_keywords(title)
    structured_queries = _build_structured_image_queries(keywords)
    specific_variants = _build_headline_specific_query_variants(
        title,
        keywords,
        structured_queries,
    )

    domain_match = _domain_match(headline_lower)
    yake_terms = _yake_keywords(title)

    if domain_match:
        primary, fallbacks = domain_match
        domain_candidates: List[str] = []
        if structured_queries:
            domain_candidates.append(structured_queries[0])

        domain_candidates.append(primary)
        domain_candidates.extend(specific_variants)
        domain_candidates.extend(fallbacks)
        domain_candidates.extend(structured_queries[1:4])

        # Keep a YAKE-enriched query, but only after domain- and rule-based intent.
        enriched_primary = _build_query_from_terms([primary, *yake_terms], max_words=7)
        if enriched_primary:
            domain_candidates.append(enriched_primary)

        if DEFAULT_OIL_QUERY not in domain_candidates:
            domain_candidates.append(DEFAULT_OIL_QUERY)

        domain_candidates.extend(BROAD_FALLBACK_IMAGE_QUERIES)
        return _dedupe_preserve_order([
            candidate.strip().lower()
            for candidate in domain_candidates
            if candidate and candidate.strip()
        ])

    candidates = list(structured_queries)
    if candidates:
        candidates = [candidates[0], *specific_variants, *candidates[1:]]
    else:
        candidates.extend(specific_variants)

    # YAKE terms improve long-tail uniqueness when rule/theme terms are too broad.
    yake_query = _build_query_from_terms(yake_terms, max_words=5)
    if yake_query:
        candidates.append(yake_query)

    # Broad fallbacks for hard-to-match headlines.
    candidates.extend(BROAD_FALLBACK_IMAGE_QUERIES)
    return _dedupe_preserve_order([candidate.strip().lower() for candidate in candidates if candidate.strip()])


def _get_cached_image_query_result(query: str, cache: Optional[Dict[str, str]]) -> Optional[str]:
    if cache is None or query not in cache:
        return None
    return cache[query]


def _lookup_limit_reached(
    max_new_lookups: Optional[int],
    lookup_counter: Optional[Dict[str, int]],
) -> bool:
    return (
        max_new_lookups is not None
        and lookup_counter is not None
        and lookup_counter.get("count", 0) >= max_new_lookups
    )


def _stable_page_for_title(title: str, max_page: int = 5) -> int:
    """Pick a deterministic Pexels page for a title to diversify repeated queries."""
    if max_page <= 1:
        return 1

    seed = (title or "").strip().lower()
    if not seed:
        return 1

    digest = hashlib.sha1(seed.encode("utf-8")).hexdigest()
    return (int(digest[:8], 16) % max_page) + 1


def _build_image_cache_key(query: str, orientation: str, page: int) -> str:
    """Cache key includes query + orientation + page to avoid one-image-for-all behavior."""
    return f"{query.strip().lower()}|{orientation}|p{max(1, int(page))}"


def _resolve_image_url_from_headline(
    title: str,
    cache: Optional[Dict[str, str]] = None,
    max_new_lookups: Optional[int] = None,
    lookup_counter: Optional[Dict[str, int]] = None,
) -> str:
    """Resolve an image URL by trying multiple keyword queries for one headline."""
    queries = _build_fallback_image_queries(title)
    orientation = _infer_orientation((title or "").lower())
    page = _stable_page_for_title(title)

    for query in queries:
        cache_key = _build_image_cache_key(query, orientation, page)
        cached_image_url = _get_cached_image_query_result(cache_key, cache)
        if cached_image_url is not None:
            if cached_image_url:
                return cached_image_url
            continue

        if _lookup_limit_reached(max_new_lookups, lookup_counter):
            break

        image_url = _fetch_pexels_image_url(
            query,
            orientation=orientation,
            page=page,
        )

        if lookup_counter is not None:
            lookup_counter["count"] = lookup_counter.get("count", 0) + 1

        if cache is not None:
            cache[cache_key] = image_url

        if image_url:
            return image_url

    return ""


def _fetch_pexels_image_url(
    query: str,
    orientation: str = "landscape",
    page: int = 1,
) -> str:
    """Fetch a relevant free-stock image URL from Pexels."""
    if not PEXELS_API_KEY:
        logger.debug("PEXELS_API_KEY not set — skipping image lookup for query: %s", query)
        return ""

    import requests

    headers = {"Authorization": PEXELS_API_KEY}
    params = {
        "query": query,
        "per_page": max(1, PEXELS_PER_PAGE),
        "page": max(1, int(page)),
        "orientation": orientation,
        "size": "medium",
    }

    try:
        response = requests.get(
            PEXELS_SEARCH_URL,
            headers=headers,
            params=params,
            timeout=PEXELS_TIMEOUT_SECONDS,
        )
        response.raise_for_status()
        photos = response.json().get("photos", [])
        if not photos:
            return ""

        src = photos[0].get("src", {})
        # Prefer medium-sized image for card usage.
        return _normalize_image_url(
            src.get("medium") or src.get("large") or src.get("original") or ""
        )
    except Exception as e:
        logger.debug(f"Pexels lookup failed for query '{query}': {e}")
        return ""


def fetch_oil_news(
    date: str = None, api_key: str = None, page_size: int = 100
) -> List[Dict[str, Any]]:
    """
    Fetch oil-related news articles from NewsAPI.

    Args:
        date: Date to fetch news for (YYYY-MM-DD). Default: yesterday.
        api_key: NewsAPI key. Falls back to NEWSAPI_KEY from config.
        page_size: Number of articles to fetch (max 100).

    Returns:
        List of article dictionaries with title, description, publishedAt, source.

    Raises:
        ValueError: If no API key is provided.
    """
    import requests

    key = api_key or NEWSAPI_KEY
    if not key:
        raise ValueError(
            "NewsAPI key required. Set NEWSAPI_KEY in config.py or pass api_key parameter."
        )

    # Default to yesterday's date
    if date is None:
        date = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")

    # First try date-specific fetch
    articles = _fetch_news_for_date(date, key, page_size)

    # If no articles found, try fetching recent news without date filter
    if not articles:
        logger.warning(f"No articles for {date}, fetching recent news instead")
        articles = _fetch_recent_news(key, page_size)

    return articles


def _fetch_news_for_date(
    date: str, api_key: str, page_size: int
) -> List[Dict[str, Any]]:
    """Fetch news for a specific date."""
    import requests

    target_date = datetime.strptime(date, "%Y-%m-%d")
    from_date = target_date.strftime("%Y-%m-%d")
    to_date = (target_date + timedelta(days=1)).strftime("%Y-%m-%d")

    params = {
        "q": OIL_SEARCH_QUERY,
        "from": from_date,
        "to": to_date,
        "language": "en",
        "sortBy": "publishedAt",
        "pageSize": page_size,
        "apiKey": api_key,
    }

    try:
        response = requests.get(NEWSAPI_BASE_URL, params=params, timeout=30)
        response.raise_for_status()
        data = response.json()

        if data.get("status") != "ok":
            logger.error(f"NewsAPI error: {data.get('message')}")
            return []

        articles = data.get("articles", [])
        logger.info(f"Fetched {len(articles)} articles for {date}")
        return articles

    except Exception as e:
        logger.error(f"Error fetching news: {e}")
        return []


def _fetch_recent_news(api_key: str, page_size: int = 100) -> List[Dict[str, Any]]:
    """Fetch recent oil news without date filter (for NewsAPI free tier compatibility)."""
    import requests

    params = {
        "q": OIL_SEARCH_QUERY,
        "language": "en",
        "sortBy": "publishedAt",
        "pageSize": page_size,
        "apiKey": api_key,
    }

    try:
        response = requests.get(NEWSAPI_BASE_URL, params=params, timeout=30)
        response.raise_for_status()
        data = response.json()

        if data.get("status") != "ok":
            logger.error(f"NewsAPI error: {data.get('message')}")
            return []

        articles = data.get("articles", [])
        logger.info(f"Fetched {len(articles)} recent articles")
        return articles

    except Exception as e:
        logger.error(f"Error fetching recent news: {e}")
        return []


def analyze_sentiment_simple(text: str) -> float:
    """
    Simple keyword-based sentiment analysis for oil news.
    Fallback when custom model is not available.

    Returns:
        Sentiment score between -1 (negative) and 1 (positive)
    """
    if not text:
        return 0.0

    text_lower = text.lower()

    # Positive keywords for oil market
    positive_words = [
        "surge",
        "soar",
        "rally",
        "gain",
        "rise",
        "jump",
        "increase",
        "boost",
        "growth",
        "recovery",
        "demand",
        "bullish",
        "optimism",
        "supply cut",
        "production cut",
        "higher",
        "up",
    ]

    # Negative keywords for oil market
    negative_words = [
        "fall",
        "drop",
        "decline",
        "crash",
        "plunge",
        "slump",
        "tumble",
        "bearish",
        "oversupply",
        "glut",
        "recession",
        "weak",
        "lower",
        "down",
        "loss",
        "concern",
        "fear",
        "crisis",
    ]

    pos_count = sum(1 for word in positive_words if word in text_lower)
    neg_count = sum(1 for word in negative_words if word in text_lower)

    total = pos_count + neg_count
    if total == 0:
        return 0.0

    return (pos_count - neg_count) / total


def analyze_sentiment(text: str, mode: str = None) -> float:
    """
    Analyze sentiment of text using the specified mode.

    Args:
        text: Text to analyze
        mode: 'simple' or 'finbert'. If None, uses SENTIMENT_MODE from config.

    Returns:
        Sentiment score between -1 (negative) and 1 (positive)
    """
    if mode is None:
        mode = SENTIMENT_MODE

    if mode == "finbert":
        try:
            from app.services.finbert_analyzer import (
                analyze_sentiment_finbert,
                is_finbert_available,
            )

            if is_finbert_available():
                return analyze_sentiment_finbert(text)
            else:
                logger.warning(
                    "Custom FinBERT model not available, falling back to simple sentiment"
                )
                return analyze_sentiment_simple(text)
        except Exception as e:
            logger.warning(
                f"FinBERT analysis failed: {e}, falling back to simple sentiment"
            )
            return analyze_sentiment_simple(text)
    else:
        return analyze_sentiment_simple(text)


def _extract_texts_from_articles(articles: List[Dict[str, Any]]) -> List[str]:
    """Extract and combine text fields from articles for sentiment analysis."""
    texts = []
    for article in articles:
        title = article.get("title", "") or ""
        description = article.get("description", "") or ""
        content = article.get("content", "") or ""
        # Combine title + description (matching Colab)
        text = f"{title}. {description}"
        if content and len(content) > len(description):
            text = f"{title}. {content}"
        texts.append(text)
    return texts


def _analyze_with_finbert(texts: List[str]) -> Optional[List[float]]:
    """
    Attempt to analyze sentiments using FinBERT batch processing.
    
    Returns:
        List of sentiment scores if successful, None if unavailable/failed.
    """
    try:
        from app.services.finbert_analyzer import (
            analyze_batch_finbert,
            is_finbert_available,
        )

        if not is_finbert_available():
            logger.warning("Custom FinBERT not available, using simple sentiment")
            return None

        import time
        logger.info(f"Analyzing {len(texts)} articles with custom FinBERT...")
        t_sent = time.time()
        sentiments = analyze_batch_finbert(texts)
        elapsed_sent = time.time() - t_sent
        logger.info(
            f"FinBERT analysis complete in {elapsed_sent:.1f}s. "
            f"Mean sentiment: {np.mean(sentiments):.4f}"
        )
        return sentiments
    except Exception as e:
        logger.warning(f"FinBERT batch analysis failed: {e}, using simple sentiment")
        return None


def _analyze_with_simple_sentiment(articles: List[Dict[str, Any]]) -> List[float]:
    """Fallback sentiment analysis using simple keyword-based method."""
    sentiments = []
    for article in articles:
        title = article.get("title", "") or ""
        description = article.get("description", "") or ""
        text = f"{title} {description}"
        sentiment = analyze_sentiment_simple(text)
        sentiments.append(sentiment)
    return sentiments


def _compute_sentiment_dict(sentiments: List[float], news_volume: int) -> Dict[str, Any]:
    """Compute sentiment feature dictionary from sentiment scores and volume."""
    log_news_volume = math.log(news_volume + 1)
    daily_sentiment = float(np.mean(sentiments)) if sentiments else 0.0
    decayed_news_volume = float(news_volume) * 0.5  # Approximate
    high_news_regime = 1 if news_volume > 30 else 0

    logger.info(
        f"Computed sentiment: {daily_sentiment:.4f} from {news_volume} articles"
    )

    return {
        "daily_sentiment_decay": round(daily_sentiment, 6),
        "news_volume": news_volume,
        "log_news_volume": round(log_news_volume, 6),
        "decayed_news_volume": round(decayed_news_volume, 6),
        "high_news_regime": high_news_regime,
    }


def compute_sentiment_features(
    articles: List[Dict[str, Any]], sentiment_mode: str = None
) -> Dict[str, Any]:
    """
    Compute sentiment features from a list of articles.

    IMPORTANT: This matches Colab training preprocessing exactly:
    - daily_sentiment: Simple mean of all article sentiments (NO within-day decay!)
    - news_volume: Number of articles
    - log_news_volume: log(news_volume + 1)
    - decayed_news_volume: EWM of volume (for feature matching)
    - high_news_regime: Binary flag (1 if volume > 30 articles)

    Cross-day decay formula (s[t] + exp(-0.3) * s[t-1]) is applied
    LATER in sentiment_service.py, NOT here!

    Args:
        articles: List of article dictionaries from NewsAPI
        sentiment_mode: 'simple' or 'finbert' (default: from config)

    Returns:
        Dictionary with sentiment features
    """
    if not articles:
        return {
            "daily_sentiment_decay": 0.0,
            "news_volume": 0,
            "log_news_volume": 0.0,
            "decayed_news_volume": 0.0,
            "high_news_regime": 0,
        }

    # Determine which sentiment analysis mode to use
    use_finbert = (
        (sentiment_mode == "finbert")
        if sentiment_mode
        else (SENTIMENT_MODE == "finbert")
    )

    # Extract texts from articles
    texts = _extract_texts_from_articles(articles)

    # Try FinBERT if requested
    all_sentiments = None
    if use_finbert:
        all_sentiments = _analyze_with_finbert(texts)

    # Fallback to simple sentiment if needed
    if not all_sentiments:
        all_sentiments = _analyze_with_simple_sentiment(articles)

    # Compute and return features
    return _compute_sentiment_dict(all_sentiments, len(articles))



def compute_sentiment_features_with_articles(
    articles: List[Dict[str, Any]], sentiment_mode: str = None
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """
    Same as compute_sentiment_features but also returns per-article details.

    Returns:
        (features_dict, enriched_articles)
        where each item in enriched_articles is the original article dict with
        an additional 'sentiment_score' key.
    """
    if not articles:
        return (
            {
                "daily_sentiment_decay": 0.0,
                "news_volume": 0,
                "log_news_volume": 0.0,
                "decayed_news_volume": 0.0,
                "high_news_regime": 0,
            },
            [],
        )

    use_finbert = (
        (sentiment_mode == "finbert") if sentiment_mode else (SENTIMENT_MODE == "finbert")
    )

    def _compute_all_sentiments() -> List[float]:
        texts = _extract_texts_from_articles(articles)
        sentiments = None
        if use_finbert:
            sentiments = _analyze_with_finbert(texts)
        if not sentiments:
            sentiments = _analyze_with_simple_sentiment(articles)
        return sentiments

    def _resolve_all_images() -> List[str]:
        image_cache: Dict[str, str] = {}
        lookup_counter: Dict[str, int] = {"count": 0}
        urls: List[str] = []
        for article in articles:
            resolved = _resolve_image_url_from_headline(
                title=article.get("title", ""),
                cache=image_cache,
                max_new_lookups=MAX_PEXELS_LOOKUPS_PER_RUN,
                lookup_counter=lookup_counter,
            )
            # Always persist a non-empty image URL for frontend cards.
            urls.append(resolved or DEFAULT_ARTICLE_IMAGE_URL)
        return urls

    with ThreadPoolExecutor(max_workers=2) as executor:
        sentiments_future = executor.submit(_compute_all_sentiments)
        images_future = executor.submit(_resolve_all_images)

        all_sentiments = sentiments_future.result()
        image_urls = images_future.result()

    enriched = []
    for idx, article in enumerate(articles):
        score = all_sentiments[idx] if idx < len(all_sentiments) else 0.0
        image_url = image_urls[idx] if idx < len(image_urls) else DEFAULT_ARTICLE_IMAGE_URL

        enriched.append(
            {
                "title": article.get("title", ""),
                "description": article.get("description", ""),
                "url": article.get("url"),
                "image_url": image_url,
                "source": (
                    article["source"]["name"]
                    if isinstance(article.get("source"), dict)
                    else str(article.get("source", ""))
                ),
                "published_at": article.get("publishedAt", ""),
                "sentiment_score": round(float(score), 6),
            }
        )

    return _compute_sentiment_dict(all_sentiments, len(articles)), enriched


def fetch_oil_news_combined(
    date: str = None, api_key: str = None, page_size: int = 100
) -> List[Dict[str, Any]]:
    """
    Fetch oil news from web scrapers first, then fall back to NewsAPI.

    Priority:
    1. Web scraping (OilPrice, EconomyNext, BOE Report, FT)
    2. NewsAPI (if scraping yields fewer than 3 articles)

    Args:
        date: Date to fetch news for (YYYY-MM-DD). Default: yesterday.
        api_key: NewsAPI key (for fallback).
        page_size: Number of articles for NewsAPI fallback.

    Returns:
        Combined list of article dictionaries.
    """
    if date is None:
        date = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")

    articles = []

    # Try web scraping first
    try:
        from app.services.news_scraper import scrape_all_sources

        scraped = scrape_all_sources(target_date=date)
        articles.extend(scraped)
        logger.info(f"Web scraping returned {len(scraped)} articles for {date}")
    except Exception as e:
        logger.warning(f"Web scraping failed: {e}")

    # Fall back to NewsAPI if scraping yielded few results
    if len(articles) < 3:
        try:
            newsapi_articles = fetch_oil_news(
                date=date, api_key=api_key, page_size=page_size
            )
            articles.extend(newsapi_articles)
            logger.info(f"NewsAPI fallback added {len(newsapi_articles)} articles")
        except Exception as e:
            logger.warning(f"NewsAPI fallback also failed: {e}")

    return articles


def fetch_and_compute_sentiment(
    date: str = None, api_key: str = None, sentiment_mode: str = None
) -> Dict[str, Any]:
    """
    Convenience function: Fetch news and compute sentiment features.

    Uses web scraping as primary source with NewsAPI fallback.

    Args:
        date: Date to analyze (YYYY-MM-DD). Default: yesterday.
        api_key: NewsAPI key.
        sentiment_mode: 'simple' or 'finbert'. Default: from config.

    Returns:
        Dictionary with date and all sentiment features.
    """
    import time

    if date is None:
        date = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")

    t_total = time.time()

    t0 = time.time()
    articles = fetch_oil_news_combined(date=date, api_key=api_key)
    logger.info(
        f"[TIMING] News fetching took {time.time() - t0:.1f}s ({len(articles)} articles)"
    )

    t1 = time.time()
    features = compute_sentiment_features(articles, sentiment_mode=sentiment_mode)
    logger.info(f"[TIMING] Sentiment computation took {time.time() - t1:.1f}s")

    logger.info(
        f"[TIMING] Total fetch_and_compute_sentiment: {time.time() - t_total:.1f}s"
    )

    return {"date": date, **features}


# For CLI usage
if __name__ == "__main__":
    import sys
    import json

    # Check for API key
    if not NEWSAPI_KEY:
        print("Warning: NEWSAPI_KEY not set, using web scraping only")

    # Get date from command line or use yesterday
    date = sys.argv[1] if len(sys.argv) > 1 else None

    result = fetch_and_compute_sentiment(date=date)
    print(json.dumps(result, indent=2))
