"""
══════════════════════════════════════════════════════════════
  Article Scraper
  Extracts clean article text from URLs using requests + BS4.
  Handles common edge cases (paywalls, encoding, empty pages).
══════════════════════════════════════════════════════════════
"""
import logging
import requests
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    )
}


def scrape_article(url: str) -> dict:
    """
    Extract article title and body text from a URL.

    Args:
        url: The article URL to scrape

    Returns:
        {
            "title": str,
            "text": str,
            "success": bool,
            "error": str | None
        }
    """
    try:
        response = requests.get(url, headers=HEADERS, timeout=15)
        response.raise_for_status()
        response.encoding = response.apparent_encoding  # Handle encoding

        soup = BeautifulSoup(response.text, "html.parser")

        # Remove noise elements
        for tag in soup(["script", "style", "nav", "header", "footer",
                         "aside", "iframe", "form", "noscript", "ad"]):
            tag.decompose()

        # ── Extract title ────────────────────────────────
        title = _extract_title(soup)

        # ── Extract body text ────────────────────────────
        text = _extract_body(soup)

        if len(text) < 100:
            return {
                "title": title,
                "text": text,
                "success": False,
                "error": "Could not extract sufficient article text (< 100 chars)",
            }

        logger.info(f"Scraped article: {title[:60]} ({len(text)} chars)")
        return {"title": title, "text": text, "success": True, "error": None}

    except requests.exceptions.RequestException as e:
        logger.error(f"HTTP request failed for {url}: {e}")
        return {"title": "", "text": "", "success": False, "error": str(e)}
    except Exception as e:
        logger.error(f"Scraping error for {url}: {e}")
        return {"title": "", "text": "", "success": False, "error": str(e)}


def _extract_title(soup: BeautifulSoup) -> str:
    """Extract the best available title from the page."""
    # Priority: og:title → h1 → <title>
    og = soup.find("meta", property="og:title")
    if og and og.get("content"):
        return og["content"].strip()

    h1 = soup.find("h1")
    if h1:
        return h1.get_text(strip=True)

    title_tag = soup.find("title")
    if title_tag:
        return title_tag.get_text(strip=True)

    return "Untitled"


def _extract_body(soup: BeautifulSoup) -> str:
    """Extract the main article body text."""
    # Strategy 1: Look for <article> tag
    article = soup.find("article")
    if article:
        paragraphs = article.find_all("p")
        text = "\n\n".join(
            p.get_text(strip=True) for p in paragraphs
            if len(p.get_text(strip=True)) > 30
        )
        if len(text) > 200:
            return text

    # Strategy 2: Look for common content containers
    for selector in [
        {"class_": "article-body"},
        {"class_": "story-body"},
        {"class_": "post-content"},
        {"class_": "entry-content"},
        {"id": "article-body"},
        {"id": "content"},
    ]:
        container = soup.find("div", **selector)
        if container:
            paragraphs = container.find_all("p")
            text = "\n\n".join(
                p.get_text(strip=True) for p in paragraphs
                if len(p.get_text(strip=True)) > 30
            )
            if len(text) > 200:
                return text

    # Strategy 3: Fallback — all paragraphs on the page
    paragraphs = soup.find_all("p")
    text = "\n\n".join(
        p.get_text(strip=True) for p in paragraphs
        if len(p.get_text(strip=True)) > 30
    )
    if text:
        return text

    # Strategy 4: Last resort — raw text
    raw = soup.get_text(separator="\n", strip=True)
    lines = [line.strip() for line in raw.split("\n") if len(line.strip()) > 20]
    return "\n".join(lines[:100])  # Cap at 100 lines
