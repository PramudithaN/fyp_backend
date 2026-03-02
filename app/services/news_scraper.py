"""
Web scraper for crude oil news from multiple sources.

Scrapes articles from:
1. OilPrice.com - /Latest-Energy-News/World-News/
2. EconomyNext - /petroleum/
3. BOE Report  - /category/oil-and-gas-news-headlines/
4. FT.com      - /oil?page=1

Each scraper returns articles in the same dict format as NewsAPI:
    {"title", "description", "publishedAt", "source", "url"}

so the downstream compute_sentiment_features() pipeline works unchanged.
"""

import base64
import logging
import re
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
from difflib import SequenceMatcher

import requests
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)

# Shared HTTP config
DEFAULT_TIMEOUT = 30
MAX_RETRIES = 2
USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/122.0.0.0 Safari/537.36"
)

# Source name constants
SOURCE_OILPRICE = "OilPrice.com"
SOURCE_BOE_REPORT = "BOE Report"
SOURCE_FT = "FT.com"

HEADERS = {
    "User-Agent": USER_AGENT,
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
    "Accept-Encoding": "gzip, deflate",
    "Connection": "keep-alive",
}


# ---------------------------------------------------------------------------
# HTTP helpers
# ---------------------------------------------------------------------------


def _fetch_page(
    url: str, timeout: int = DEFAULT_TIMEOUT, extra_headers: dict = None
) -> Optional[BeautifulSoup]:
    """Fetch a page and return parsed BeautifulSoup, with retries."""
    headers = {**HEADERS}
    if extra_headers:
        headers.update(extra_headers)
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = requests.get(url, headers=headers, timeout=timeout)
            resp.raise_for_status()
            # Ensure text decoding works
            resp.encoding = resp.apparent_encoding or "utf-8"
            return BeautifulSoup(resp.text, "lxml")
        except requests.RequestException as e:
            logger.warning(f"Attempt {attempt}/{MAX_RETRIES} failed for {url}: {e}")
            if attempt == MAX_RETRIES:
                logger.error(f"All attempts failed for {url}")
                return None


def _parse_date_flexible(text: str) -> Optional[datetime]:
    """Try multiple date formats to parse a date string."""
    text = text.strip()
    formats = [
        "%b %d, %Y at %H:%M",  # "Mar 02, 2026 at 03:39"
        "%b %d, %Y",  # "Mar 02, 2026"
        "%B %d, %Y",  # "March 2, 2026"
        "%Y-%m-%d",  # "2026-03-02"
        "%Y-%m-%dT%H:%M:%S%z",  # ISO with timezone
    ]
    for fmt in formats:
        try:
            return datetime.strptime(text, fmt)
        except ValueError:
            continue
    return None


def _matches_target_date(article_date: Optional[datetime], target_date: str) -> bool:
    """Check whether an article's date matches the target YYYY-MM-DD."""
    if article_date is None:
        return True  # If we can't parse the date, include it anyway
    return article_date.strftime("%Y-%m-%d") == target_date


def _dedup_articles(
    articles: List[Dict[str, Any]], threshold: float = 0.85
) -> List[Dict[str, Any]]:
    """Remove near-duplicate articles by title similarity."""
    if not articles:
        return articles

    unique: List[Dict[str, Any]] = []
    for article in articles:
        title = (article.get("title") or "").lower().strip()
        is_dup = False
        for existing in unique:
            existing_title = (existing.get("title") or "").lower().strip()
            if SequenceMatcher(None, title, existing_title).ratio() >= threshold:
                is_dup = True
                break
        if not is_dup:
            unique.append(article)

    removed = len(articles) - len(unique)
    if removed > 0:
        logger.info(f"Deduplication removed {removed} near-duplicate articles")
    return unique


# ---------------------------------------------------------------------------
# Helper functions for parsing
# ---------------------------------------------------------------------------


def _extract_title_and_url_oilprice(card) -> tuple:
    """Extract title and URL from OilPrice article card."""
    title_link = card.select_one("a:has(h2.categoryArticle__title)")
    if title_link is None:
        h2 = card.select_one("h2.categoryArticle__title")
        if h2 is None:
            return None, None
        title = h2.get_text(strip=True)
        parent_a = h2.find_parent("a")
        article_url = parent_a["href"] if parent_a else ""
    else:
        title = title_link.get_text(strip=True)
        article_url = title_link.get("href", "")
    
    if article_url and not article_url.startswith("http"):
        article_url = "https://oilprice.com" + article_url
    
    return title, article_url


def _parse_date_from_time_element(time_el) -> Optional[datetime]:
    """Parse date from HTML time element with datetime attribute."""
    if not time_el:
        return None
    
    datetime_attr = time_el.get("datetime")
    if datetime_attr:
        try:
            return datetime.fromisoformat(datetime_attr)
        except (ValueError, TypeError):
            return _parse_date_flexible(time_el.get_text(strip=True))
    return None


def _parse_date_ft_article(teaser) -> Optional[datetime]:
    """Parse date from FT article teaser with multiple fallback strategies."""
    time_el = teaser.select_one("time.o-teaser__timestamp-date")
    if time_el and time_el.get("datetime"):
        try:
            return datetime.fromisoformat(
                time_el["datetime"].replace("+0000", "+00:00")
            )
        except (ValueError, TypeError):
            pass
    
    # Fallback: look for parent date header
    parent = teaser.find_parent("div")
    if parent:
        date_header = parent.find_previous("time", class_="o-date")
        if date_header:
            return _parse_date_flexible(date_header.get_text(strip=True))
    
    return None


def _clean_description_text(content_el) -> str:
    """Remove links from description and return cleaned text."""
    if not content_el:
        return ""
    
    for a_tag in content_el.find_all("a"):
        a_tag.decompose()
    
    return content_el.get_text(strip=True)


def _create_article_dict(title: str, description: str, article_date: Optional[datetime], 
                         source_name: str, url: str) -> Dict[str, Any]:
    """Create standardized article dictionary."""
    return {
        "title": title,
        "description": description,
        "publishedAt": article_date.isoformat() if article_date else "",
        "source": {"name": source_name},
        "url": url,
        "_parsed_date": article_date,
    }


# ---------------------------------------------------------------------------
# Internal: parse articles from a single page of each site
# (These are the building blocks; the public API adds pagination on top.)
# ---------------------------------------------------------------------------


def _parse_oilprice_page(soup: BeautifulSoup) -> List[Dict[str, Any]]:
    """Parse article cards from a single OilPrice.com listing page."""
    articles = []
    for card in soup.select("div.categoryArticle"):
        try:
            title, article_url = _extract_title_and_url_oilprice(card)
            if not title:
                continue

            meta_el = card.select_one("p.categoryArticle__meta")
            raw_meta = meta_el.get_text(strip=True) if meta_el else ""
            date_part = raw_meta.split("|")[0].strip() if raw_meta else ""
            article_date = _parse_date_flexible(date_part)

            excerpt_el = card.select_one("p.categoryArticle__excerpt")
            description = excerpt_el.get_text(strip=True) if excerpt_el else ""

            articles.append(_create_article_dict(
                title, description, article_date, SOURCE_OILPRICE, article_url
            ))
        except Exception as e:
            logger.debug(f"Error parsing OilPrice article: {e}")
    return articles


def _parse_boereport_page(soup: BeautifulSoup) -> List[Dict[str, Any]]:
    """Parse article entries from a single BOE Report listing page."""
    articles = []
    for entry in soup.select("article.entry"):
        try:
            title_a = entry.select_one("h2.entry-title a.entry-title-link")
            if title_a is None:
                continue
            title = title_a.get_text(strip=True)
            article_url = title_a.get("href", "")

            time_el = entry.select_one("time.entry-time")
            article_date = _parse_date_from_time_element(time_el)

            content_el = entry.select_one("div.entry-content p")
            description = _clean_description_text(content_el)

            articles.append(_create_article_dict(
                title, description, article_date, SOURCE_BOE_REPORT, article_url
            ))
        except Exception as e:
            logger.debug(f"Error parsing BOE Report article: {e}")
    return articles


def _parse_ft_page(soup: BeautifulSoup) -> List[Dict[str, Any]]:
    """Parse teaser blocks from a single FT.com listing page."""
    articles = []
    for teaser in soup.select("div.o-teaser"):
        try:
            heading_link = teaser.select_one("a.js-teaser-heading-link")
            if heading_link is None:
                continue
            
            title = heading_link.get_text(strip=True)
            article_url = heading_link.get("href", "")
            if article_url and not article_url.startswith("http"):
                article_url = "https://www.ft.com" + article_url

            standfirst = teaser.select_one("a.js-teaser-standfirst-link")
            description = standfirst.get_text(strip=True) if standfirst else ""

            article_date = _parse_date_ft_article(teaser)

            articles.append(_create_article_dict(
                title, description, article_date, "Financial Times", article_url
            ))
        except Exception as e:
            logger.debug(f"Error parsing FT article: {e}")
    return articles


def _parse_economynext_page(soup: BeautifulSoup) -> List[Dict[str, Any]]:
    """Parse articles from a single EconomyNext listing page."""
    articles = []
    for card in soup.select("div.top-story, div.top-story-2"):
        try:
            title_a = card.select_one("h3.recent-top-header a")
            if title_a is None:
                continue
            title = title_a.get_text(strip=True)
            article_url = title_a.get("href", "")
            if not article_url.startswith("http"):
                article_url = "https://economynext.com" + article_url

            date_el = card.select_one("span.article-publish-date")
            raw_date = date_el.get_text(strip=True) if date_el else ""
            article_date = _parse_date_flexible(raw_date)

            desc_el = card.select_one("div.top-story-desc p, div.top-story-desc-2 p")
            description = desc_el.get_text(strip=True) if desc_el else ""

            articles.append(
                {
                    "title": title,
                    "description": description,
                    "publishedAt": article_date.isoformat() if article_date else "",
                    "source": {"name": "EconomyNext"},
                    "url": article_url,
                    "_parsed_date": article_date,
                }
            )
        except Exception as e:
            logger.debug(f"Error parsing EconomyNext article: {e}")
    return articles


# ---------------------------------------------------------------------------
# Paginated scrapers — crawl multiple pages to reach older articles
# ---------------------------------------------------------------------------


def _process_article_for_date(art: Dict[str, Any], target_date: str, target_dt: datetime) -> tuple:
    """Process article and determine if it matches target date.
    
    Returns: (should_include, is_future, article) tuple
    """
    pd = art.pop("_parsed_date", None)
    if pd is None:
        # Can't determine date — include it
        art["publishedAt"] = target_date
        return True, False, art
    
    if pd.strftime("%Y-%m-%d") == target_date:
        return True, False, art
    
    if pd.date() > target_dt.date():
        return False, True, art  # Future article, don't include but not older
    
    return False, False, art  # Older article


def _scrape_paginated(
    site_name: str,
    base_url_fn,  # callable(page_num) -> url string
    parse_fn,  # callable(soup) -> list of raw article dicts
    target_date: str,
    max_pages: int = 10,
) -> List[Dict[str, Any]]:
    """Generic paginated scraper.

    Crawls pages until:
      - All articles on a page are older than target_date, OR
      - max_pages reached, OR
      - An empty page is returned.
    """
    target_dt = datetime.strptime(target_date, "%Y-%m-%d")
    matched: List[Dict[str, Any]] = []

    for page_num in range(1, max_pages + 1):
        url = base_url_fn(page_num)
        soup = _fetch_page(url)
        if soup is None:
            break

        raw = parse_fn(soup)
        if not raw:
            logger.debug(f"{site_name} page {page_num}: no articles found, stopping")
            break

        page_matched = 0
        all_older = True
        
        for art in raw:
            should_include, is_future, processed_art = _process_article_for_date(art, target_date, target_dt)
            if should_include:
                matched.append(processed_art)
                page_matched += 1
                all_older = False
            elif is_future:
                all_older = False

        logger.debug(
            f"{site_name} page {page_num}: {page_matched} matched, {len(raw)} total"
        )

        if all_older:
            logger.debug(
                f"{site_name} page {page_num}: all articles older than {base64.b64encode(target_date.encode('utf-8')).decode('utf-8')}, stopping"
            )
            break

    logger.info(
        f"{site_name}: found {len(matched)} articles for {base64.b64encode(target_date.encode('utf-8')).decode('utf-8')} (paginated)"
    )
    return matched


def scrape_oilprice(target_date: str, max_pages: int = 1) -> List[Dict[str, Any]]:
    """Scrape OilPrice.com with optional pagination."""
    return _scrape_paginated(
        site_name=SOURCE_OILPRICE,
        base_url_fn=lambda p: (
            "https://oilprice.com/Latest-Energy-News/World-News/"
            if p == 1
            else f"https://oilprice.com/Latest-Energy-News/World-News/Page-{p}.html"
        ),
        parse_fn=_parse_oilprice_page,
        target_date=target_date,
        max_pages=max_pages,
    )


def scrape_economynext(target_date: str, max_pages: int = 1) -> List[Dict[str, Any]]:
    """Scrape EconomyNext with optional pagination."""
    return _scrape_paginated(
        site_name="EconomyNext",
        base_url_fn=lambda p: (
            "https://economynext.com/petroleum/"
            if p == 1
            else f"https://economynext.com/petroleum/page/{p}/"
        ),
        parse_fn=_parse_economynext_page,
        target_date=target_date,
        max_pages=max_pages,
    )


def scrape_boereport(target_date: str, max_pages: int = 1) -> List[Dict[str, Any]]:
    """Scrape BOE Report with optional pagination."""
    return _scrape_paginated(
        site_name=SOURCE_BOE_REPORT,
        base_url_fn=lambda p: (
            "https://boereport.com/category/oil-and-gas-news-headlines/"
            if p == 1
            else f"https://boereport.com/category/oil-and-gas-news-headlines/page/{p}/"
        ),
        parse_fn=_parse_boereport_page,
        target_date=target_date,
        max_pages=max_pages,
    )


def scrape_ft(target_date: str, max_pages: int = 1) -> List[Dict[str, Any]]:
    """Scrape FT.com with optional pagination."""
    return _scrape_paginated(
        site_name=SOURCE_FT,
        base_url_fn=lambda p: f"https://www.ft.com/oil?page={p}",
        parse_fn=_parse_ft_page,
        target_date=target_date,
        max_pages=max_pages,
    )


# ---------------------------------------------------------------------------
# Combined scraper (single day)
# ---------------------------------------------------------------------------

SCRAPERS = [
    (SOURCE_OILPRICE, scrape_oilprice),
    ("EconomyNext", scrape_economynext),
    (SOURCE_BOE_REPORT, scrape_boereport),
    (SOURCE_FT, scrape_ft),
]


def scrape_all_sources(
    target_date: str = None, max_pages: int = 1
) -> List[Dict[str, Any]]:
    """
    Run all scrapers for a given date with pagination and return deduplicated articles.

    Args:
        target_date: Date in YYYY-MM-DD format. Defaults to yesterday.
        max_pages:   Max pages to crawl per site (1 = listing page only).

    Returns:
        Combined, deduplicated list of article dicts.
    """
    import time as _time
    from concurrent.futures import ThreadPoolExecutor, as_completed

    if target_date is None:
        target_date = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")

    logger.info(
        f"=== Scraping {base64.b64encode(target_date.encode('utf-8')).decode('utf-8')} (max_pages={max_pages}) across {len(SCRAPERS)} sources ==="
    )
    t0 = _time.time()

    all_articles: List[Dict[str, Any]] = []

    with ThreadPoolExecutor(max_workers=len(SCRAPERS)) as pool:
        futures = {
            pool.submit(fn, target_date, max_pages): name for name, fn in SCRAPERS
        }
        for future in as_completed(futures):
            name = futures[future]
            try:
                all_articles.extend(future.result())
            except Exception as e:
                logger.error(f"Scraper {name} failed: {e}")

    unique = _dedup_articles(all_articles)
    elapsed = _time.time() - t0
    logger.info(
        f"=== {len(unique)} unique articles in {elapsed:.1f}s for {base64.b64encode(target_date.encode('utf-8')).decode('utf-8')} ==="
    )
    return unique


# ---------------------------------------------------------------------------
# Multi-day backfill — fill a range of dates (for the 30-day rolling window)
# ---------------------------------------------------------------------------


def _crawl_site_pages(site_name: str, base_url_fn, parse_fn, max_pages: int, cutoff_date) -> List[Dict[str, Any]]:
    """Crawl multiple pages from a single site, returning all articles."""
    site_articles = []
    for pg in range(1, max_pages + 1):
        url = base_url_fn(pg)
        soup = _fetch_page(url)
        if soup is None:
            break
        
        raw = parse_fn(soup)
        if not raw:
            break
        
        site_articles.extend(raw)
        
        # Check if the oldest article on this page is older than our window
        if _should_stop_pagination(raw, cutoff_date, site_name):
            break
    
    return site_articles


def _should_stop_pagination(raw_articles: List[Dict[str, Any]], cutoff_date, site_name: str) -> bool:
    """Determine if pagination should stop based on article dates."""
    dates_on_page = [
        a.get("_parsed_date") for a in raw_articles if a.get("_parsed_date")
    ]
    if not dates_on_page:
        return False
    
    oldest = min(
        d.date() if hasattr(d, "date") else d for d in dates_on_page
    )
    
    if oldest < cutoff_date:
        logger.debug(
            f"{site_name}: reached articles from {oldest}, stopping pagination"
        )
        return True
    
    return False


def _get_site_configs() -> List[tuple]:
    """Return list of site configurations for backfill."""
    return [
        (
            "OilPrice.com",
            lambda p: (
                "https://oilprice.com/Latest-Energy-News/World-News/"
                if p == 1
                else f"https://oilprice.com/Latest-Energy-News/World-News/Page-{p}.html"
            ),
            _parse_oilprice_page,
        ),
        (
            "EconomyNext",
            lambda p: (
                "https://economynext.com/petroleum/"
                if p == 1
                else f"https://economynext.com/petroleum/page/{p}/"
            ),
            _parse_economynext_page,
        ),
        (
            "BOE Report",
            lambda p: (
                "https://boereport.com/category/oil-and-gas-news-headlines/"
                if p == 1
                else f"https://boereport.com/category/oil-and-gas-news-headlines/page/{p}/"
            ),
            _parse_boereport_page,
        ),
        ("FT.com", lambda p: f"https://www.ft.com/oil?page={p}", _parse_ft_page),
    ]


def _filter_and_group_articles(all_raw: List[Dict[str, Any]], target_dates: set) -> Dict[str, List[Dict[str, Any]]]:
    """Filter articles by date range and group by date."""
    by_date: Dict[str, List[Dict[str, Any]]] = {}
    
    for art in all_raw:
        pd_date = art.pop("_parsed_date", None)
        if pd_date is None:
            continue
        
        date_str = pd_date.strftime("%Y-%m-%d")
        if date_str not in target_dates:
            continue
        
        art["publishedAt"] = pd_date.isoformat()
        by_date.setdefault(date_str, []).append(art)
    
    # Deduplicate per day
    for date_str in by_date:
        by_date[date_str] = _dedup_articles(by_date[date_str])
    
    return by_date


def scrape_all_sources_multiday(
    days_back: int = 30,
    max_pages_per_site: int = 15,
) -> Dict[str, List[Dict[str, Any]]]:
    """Backfill articles for the last N days by paginating through site archives.

    Instead of scraping each day individually (slow), this function:
    1. Crawls up to max_pages_per_site pages from each site
    2. Collects ALL articles found across those pages
    3. Groups them by date (YYYY-MM-DD)
    4. Returns a dict of {date_str: [articles]}

    Args:
        days_back: Number of days to look back (default 30).
        max_pages_per_site: Max pages to crawl per site (default 15).

    Returns:
        Dict mapping date strings to lists of article dicts.
    """
    import time as _time
    from concurrent.futures import ThreadPoolExecutor, as_completed

    logger.info(f"=== BACKFILL: collecting articles for last {days_back} days ===")
    t0 = _time.time()

    today = datetime.now().date()
    target_dates = {
        (today - timedelta(days=i)).strftime("%Y-%m-%d")
        for i in range(1, days_back + 1)
    }
    cutoff = today - timedelta(days=days_back + 2)

    # Scrape many pages from each site in parallel, collecting ALL articles
    all_raw: List[Dict[str, Any]] = []
    site_configs = _get_site_configs()

    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = {
            pool.submit(_crawl_site_pages, name, url_fn, parse_fn, max_pages_per_site, cutoff): name
            for name, url_fn, parse_fn in site_configs
        }
        for future in as_completed(futures):
            name = futures[future]
            try:
                site_articles = future.result()
                all_raw.extend(site_articles)
                logger.info(
                    f"Backfill {name}: collected {len(site_articles)} raw articles"
                )
            except Exception as e:
                logger.error(f"Backfill {name} failed: {e}")

    # Group by date and filter to our target window
    by_date = _filter_and_group_articles(all_raw, target_dates)

    elapsed = _time.time() - t0
    total_articles = sum(len(v) for v in by_date.values())
    days_filled = len(by_date)
    logger.info(
        f"=== BACKFILL complete in {elapsed:.1f}s: {total_articles} articles "
        f"across {days_filled}/{days_back} days ==="
    )
    return by_date


# ---------------------------------------------------------------------------
# CLI entry point for testing
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import json
    import sys

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    date = sys.argv[1] if len(sys.argv) > 1 else None
    articles = scrape_all_sources(target_date=date)

    print(f"\n{'='*60}")
    print(f"Total unique articles: {len(articles)}")
    print(f"{'='*60}")

    for i, art in enumerate(articles, 1):
        source = art.get("source", {}).get("name", "Unknown")
        print(f"\n[{i}] ({source}) {art['title']}")
        print(f"    Date: {art['publishedAt']}")
        print(f"    URL:  {art['url']}")
        if art.get("description"):
            desc = (
                art["description"][:120] + "..."
                if len(art.get("description", "")) > 120
                else art["description"]
            )
            print(f"    Desc: {desc}")
