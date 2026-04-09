"""
══════════════════════════════════════════════════════════════
  Web Search + Scraping Tool
  Primary: Serper.dev API (Google search results)
  Fallback: DuckDuckGo (no API key)
  Enhancement: Scrapes top URLs for full evidence content
══════════════════════════════════════════════════════════════
"""
import logging
import re
import requests
from duckduckgo_search import DDGS
from bs4 import BeautifulSoup
from config import SERPER_API_KEY, MAX_SEARCH_RESULTS, MAX_SCRAPED_TEXT

logger = logging.getLogger(__name__)

SCRAPE_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    )
}


# ══════════════════════════════════════════════════════════════
#  PUBLIC API
# ══════════════════════════════════════════════════════════════


def search_web(query: str, max_results: int = MAX_SEARCH_RESULTS) -> list[dict]:
    """
    Search the web for evidence snippets.
    Uses Serper.dev (Google) as primary, DuckDuckGo as fallback.

    Returns:
        [{"title": str, "snippet": str, "url": str}, ...]
    """
    results = []

    # Primary: Serper.dev (faster, Google-quality results)
    if SERPER_API_KEY:
        results = _search_serper(query, max_results)

    # Fallback: DuckDuckGo
    if not results:
        logger.info("Serper returned nothing — falling back to DuckDuckGo")
        results = _search_duckduckgo(query, max_results)

    if not results:
        logger.warning(f"No search results for: {query[:60]}")

    return results


def search_and_scrape(query: str, claim: str, max_results: int = 3, scrape_top: int = 2) -> list[dict]:
    """
    Search the web AND scrape top URLs for full evidence text.
    Extracts the most relevant paragraphs from each scraped page.

    Args:
        query: Search query string
        claim: The original claim (used for relevance ranking)
        max_results: Max search results to fetch
        scrape_top: How many top URLs to actually scrape

    Returns:
        List of enriched evidence dicts:
        [{
            "title": str,
            "snippet": str,          # Original search snippet
            "url": str,
            "scraped_text": str,      # Full relevant text from scraping
            "scraped": bool,          # Whether scraping succeeded
        }, ...]
    """
    # Step 1: Search
    results = search_web(query, max_results)

    # Step 2: Scrape top URLs for full content
    for i, r in enumerate(results):
        if i < scrape_top:
            scraped = _scrape_and_extract(r["url"], claim)
            r["scraped_text"] = scraped["text"]
            r["scraped"] = scraped["success"]
            if scraped["success"]:
                logger.info(
                    f"Scraped {r['url'][:50]} — extracted {len(scraped['text'])} chars"
                )
        else:
            r["scraped_text"] = r["snippet"]  # Use snippet for non-scraped results
            r["scraped"] = False

    return results


# ══════════════════════════════════════════════════════════════
#  SEARCH ENGINES
# ══════════════════════════════════════════════════════════════


def _search_serper(query: str, max_results: int) -> list[dict]:
    """Search using Serper.dev API (Google results, 100 free/day)."""
    try:
        response = requests.post(
            "https://google.serper.dev/search",
            json={"q": query, "num": max_results},
            headers={
                "X-API-KEY": SERPER_API_KEY,
                "Content-Type": "application/json",
            },
            timeout=10,
        )
        response.raise_for_status()
        data = response.json()

        results = []
        for r in data.get("organic", [])[:max_results]:
            results.append({
                "title": r.get("title", ""),
                "snippet": r.get("snippet", ""),
                "url": r.get("link", ""),
            })

        logger.info(f"Serper returned {len(results)} results for: {query[:50]}")
        return results

    except Exception as e:
        logger.warning(f"Serper search failed: {e}")
        return []


def _search_duckduckgo(query: str, max_results: int) -> list[dict]:
    """Search using DuckDuckGo (no API key, unlimited but slower)."""
    try:
        with DDGS() as ddgs:
            raw = list(ddgs.text(query, max_results=max_results))

        results = [
            {
                "title": r.get("title", ""),
                "snippet": r.get("body", ""),
                "url": r.get("href", ""),
            }
            for r in raw
            if r.get("body")
        ]
        logger.info(f"DuckDuckGo returned {len(results)} results for: {query[:50]}")
        return results

    except Exception as e:
        logger.warning(f"DuckDuckGo search failed: {e}")
        return []


# ══════════════════════════════════════════════════════════════
#  WEB SCRAPING + RELEVANCE EXTRACTION
# ══════════════════════════════════════════════════════════════


def _scrape_and_extract(url: str, claim: str) -> dict:
    """
    Scrape a URL and extract the most relevant paragraphs to the claim.

    Returns:
        {"text": str, "success": bool}
    """
    try:
        response = requests.get(url, headers=SCRAPE_HEADERS, timeout=10)
        response.raise_for_status()
        response.encoding = response.apparent_encoding

        soup = BeautifulSoup(response.text, "html.parser")

        # Remove noise
        for tag in soup(["script", "style", "nav", "header", "footer",
                         "aside", "iframe", "form", "noscript"]):
            tag.decompose()

        # Extract all paragraphs
        paragraphs = []
        for p in soup.find_all("p"):
            text = p.get_text(strip=True)
            if len(text) > 40:  # Skip tiny paragraphs
                paragraphs.append(text)

        if not paragraphs:
            # Fallback: get text from divs
            for div in soup.find_all("div"):
                text = div.get_text(strip=True)
                if 50 < len(text) < 500:
                    paragraphs.append(text)

        if not paragraphs:
            return {"text": "", "success": False}

        # Rank paragraphs by relevance to the claim
        relevant = _rank_by_relevance(paragraphs, claim, top_k=3)

        # Combine top relevant paragraphs
        combined = "\n\n".join(relevant)

        # Cap at MAX_SCRAPED_TEXT chars to keep stance detection focused
        if len(combined) > MAX_SCRAPED_TEXT:
            combined = combined[:MAX_SCRAPED_TEXT]

        return {"text": combined, "success": True}

    except requests.exceptions.Timeout:
        logger.debug(f"Scraping timed out: {url[:50]}")
        return {"text": "", "success": False}
    except Exception as e:
        logger.debug(f"Scraping failed for {url[:50]}: {e}")
        return {"text": "", "success": False}


def _rank_by_relevance(paragraphs: list[str], claim: str, top_k: int = 3) -> list[str]:
    """
    Rank paragraphs by keyword overlap with the claim.
    Simple but fast — no LLM call needed.
    """
    # Extract keywords from the claim (words > 3 chars, lowered)
    claim_words = set(
        w.lower() for w in re.findall(r'\b\w+\b', claim)
        if len(w) > 3
    )

    if not claim_words:
        return paragraphs[:top_k]

    scored = []
    for p in paragraphs:
        p_words = set(w.lower() for w in re.findall(r'\b\w+\b', p))
        # Jaccard-like overlap score
        overlap = len(claim_words & p_words)
        # Bonus for longer paragraphs (more context)
        length_bonus = min(len(p) / 500, 0.5)
        score = overlap + length_bonus
        scored.append((score, p))

    # Sort by score descending
    scored.sort(key=lambda x: x[0], reverse=True)

    return [p for _, p in scored[:top_k]]
