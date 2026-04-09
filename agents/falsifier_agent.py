"""
══════════════════════════════════════════════════════════════
  Falsifier Agent — Anti-Truth Agent
  Searches for evidence that CONTRADICTS each claim.
  Builds a counter-argument using Groq LLM.
══════════════════════════════════════════════════════════════
"""
import logging
from concurrent.futures import ThreadPoolExecutor

from tools.groq_client import groq_chat, groq_chat_json
from tools.web_search import search_and_scrape
from tools.stance_detector import detect_stance

logger = logging.getLogger(__name__)



def falsify_claims(claims: list[dict], progress_callback=None) -> dict:
    """
    For each claim, search for contradicting evidence and build a counter-case.

    Args:
        claims: List of claim dicts from claim_extractor
        progress_callback: Optional callable(claim_index, total, status_text)

    Returns:
        {
            "claim_reports": [ ... per-claim report ... ],
            "overall_assessment": str
        }
    """
    reports = []

    for i, claim_data in enumerate(claims):
        claim = claim_data["claim"]
        logger.info(f"[FALSIFIER] Claim {i + 1}/{len(claims)}: {claim[:80]}")

        if progress_callback:
            progress_callback(i, len(claims), f"Falsifying: {claim[:60]}...")

        report = _falsify_single_claim(claim)
        reports.append(report)

    overall = _build_overall(reports)

    return {"claim_reports": reports, "overall_assessment": overall}


# ── Single-claim falsification ───────────────────────────────


def _falsify_single_claim(claim: str) -> dict:
    """Search for contradicting evidence for one claim, scrape top URLs, and build counter-argument."""

    # 1. Generate adversarial search queries
    queries = _make_queries(claim)

    # 2. Search AND scrape top URLs for full evidence
    all_hits = []
    for q in queries:
        hits = search_and_scrape(q, claim=claim, max_results=3, scrape_top=2)
        all_hits.extend(hits)

    # Deduplicate by URL
    seen_urls = set()
    unique_hits = []
    for h in all_hits:
        if h["url"] not in seen_urls:
            seen_urls.add(h["url"])
            unique_hits.append(h)

    # 3. Score using scraped full text — parallelized for speed
    def _score(h):
        evidence_text = h.get("scraped_text") or h["snippet"]
        stance = detect_stance(claim, evidence_text)
        return {
            "title": h["title"],
            "snippet": h["snippet"],
            "full_text": evidence_text,
            "url": h["url"],
            "scraped": h.get("scraped", False),
            "stance": stance["stance"],
            "confidence": stance["confidence"],
        }

    with ThreadPoolExecutor(max_workers=3) as pool:
        evidence = list(pool.map(_score, unique_hits[:5]))

    # 4. Rank contradicting evidence
    contradicting = sorted(
        [e for e in evidence if e["stance"] == "CONTRADICT"],
        key=lambda x: x["confidence"],
        reverse=True,
    )

    # 5. Build counter-argument using full scraped evidence
    argument = _build_argument(claim, contradicting[:3], evidence)

    avg_conf = (
        sum(e["confidence"] for e in contradicting) / len(contradicting)
        if contradicting
        else 0.0
    )

    return {
        "claim": claim,
        "search_queries": queries,
        "evidence": evidence,
        "contradicting_evidence": contradicting[:3],
        "argument": argument,
        "confidence": round(avg_conf, 3),
        "evidence_count": len(contradicting),
    }


# ── Query generation ─────────────────────────────────────────


def _make_queries(claim: str) -> list[str]:
    """Generate 2-3 web search queries to find CONTRADICTING evidence."""
    try:
        data = groq_chat_json(
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Generate 2 concise web search queries to find evidence that "
                        "CONTRADICTS, DISPROVES, or CASTS DOUBT on the following claim. "
                        "Look for fact-checks, corrections, and conflicting data.\n"
                        '{"queries": ["q1", "q2"]}'
                    ),
                },
                {"role": "user", "content": f"Claim: {claim}"},
            ],
            temperature=0.3,
            max_tokens=200,
        )
        return data.get("queries", [f"is it true that {claim}"])[:2]
    except Exception as e:
        logger.warning(f"Query generation failed: {e}")
        return [f"{claim} debunked", f"{claim} fact check"]


# ── Argument builder ─────────────────────────────────────────


def _build_argument(claim: str, contradicting: list, all_evidence: list) -> str:
    """Build a persuasive falsification argument using top contradicting evidence."""
    if contradicting:
        ev_block = "\n".join(
            f"- [{e['title']}]: {e.get('full_text', e['snippet'])[:300]} "
            f"(stance: {e['stance']}, score: {e['confidence']:.0%}, scraped: {e.get('scraped', False)})"
            for e in contradicting[:3]
        )
    else:
        ev_block = "No strong contradicting evidence was found."

    try:
        return groq_chat(
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a FALSIFIER agent in a fact-checking debate. "
                        "Your role is to argue AGAINST the truth of the given claim "
                        "using the evidence provided.\n"
                        "- Cite specific evidence and sources\n"
                        "- Point out inconsistencies, missing context, or exaggerations\n"
                        "- Be critical but fair — don't fabricate objections\n"
                        "- If evidence is weak, acknowledge it but note what's suspicious\n"
                        "- Keep your argument to 2-3 paragraphs"
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        f"CLAIM: {claim}\n\n"
                        f"CONTRADICTING EVIDENCE:\n{ev_block}\n\n"
                        f"Total snippets found: {len(all_evidence)}, "
                        f"contradicting: {len(contradicting)}\n\n"
                        f"Build your falsification argument."
                    ),
                },
            ],
            temperature=0.4,
            max_tokens=800,
        )
    except Exception as e:
        logger.error(f"Argument generation failed: {e}")
        return (
            f"[Argument generation failed] "
            f"Found {len(contradicting)} contradicting evidence snippets."
        )


# ── Overall assessment ───────────────────────────────────────


def _build_overall(reports: list) -> str:
    """Summarize all per-claim falsification results."""
    summary = "\n".join(
        f"• Claim {i + 1}: {r['claim'][:80]} → "
        f"{r['evidence_count']} contradicting, confidence {r['confidence']:.0%}"
        for i, r in enumerate(reports)
    )
    try:
        return groq_chat(
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a falsification agent. Summarize your overall "
                        "counter-assessment of the article's claims in 1-2 paragraphs. "
                        "Highlight the weakest claims and any red flags."
                    ),
                },
                {"role": "user", "content": f"FALSIFICATION RESULTS:\n{summary}"},
            ],
            temperature=0.3,
            max_tokens=500,
        )
    except Exception as e:
        logger.error(f"Overall assessment failed: {e}")
        return "Unable to generate overall counter-assessment."
