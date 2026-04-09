"""
══════════════════════════════════════════════════════════════
  Verifier Agent — Pro-Truth Agent
  Searches for evidence that SUPPORTS each claim.
  Builds a verification case using Groq LLM.
══════════════════════════════════════════════════════════════
"""
import logging
from concurrent.futures import ThreadPoolExecutor

from tools.groq_client import groq_chat, groq_chat_json
from tools.web_search import search_and_scrape
from tools.stance_detector import detect_stance

logger = logging.getLogger(__name__)



def verify_claims(claims: list[dict], progress_callback=None) -> dict:
    """
    For each claim, search for supporting evidence and build a verification case.

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
        logger.info(f"[VERIFIER] Claim {i + 1}/{len(claims)}: {claim[:80]}")

        if progress_callback:
            progress_callback(i, len(claims), f"Verifying: {claim[:60]}...")

        report = _verify_single_claim(claim)
        reports.append(report)

    overall = _build_overall(reports)

    return {"claim_reports": reports, "overall_assessment": overall}


# ── Single-claim verification ────────────────────────────────


def _verify_single_claim(claim: str) -> dict:
    """Search for supporting evidence for one claim, scrape top URLs, and build argument."""

    # 1. Generate pro-truth search queries
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

    # 4. Rank supporting evidence
    supporting = sorted(
        [e for e in evidence if e["stance"] == "SUPPORT"],
        key=lambda x: x["confidence"],
        reverse=True,
    )

    # 5. Build argument using full scraped evidence
    argument = _build_argument(claim, supporting[:3], evidence)

    avg_conf = (
        sum(e["confidence"] for e in supporting) / len(supporting)
        if supporting
        else 0.0
    )

    return {
        "claim": claim,
        "search_queries": queries,
        "evidence": evidence,
        "supporting_evidence": supporting[:3],
        "argument": argument,
        "confidence": round(avg_conf, 3),
        "evidence_count": len(supporting),
    }


# ── Query generation ─────────────────────────────────────────


def _make_queries(claim: str) -> list[str]:
    """Generate 2-3 web search queries to find SUPPORTING evidence."""
    try:
        data = groq_chat_json(
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Generate 2 concise web search queries to find evidence that "
                        "SUPPORTS and CONFIRMS the following claim. "
                        "Target official sources, reputable news, and public records.\n"
                        '{"queries": ["q1", "q2"]}'
                    ),
                },
                {"role": "user", "content": f"Claim: {claim}"},
            ],
            temperature=0.3,
            max_tokens=200,
        )
        return data.get("queries", [claim])[:2]
    except Exception as e:
        logger.warning(f"Query generation failed: {e}")
        return [claim]


# ── Argument builder ─────────────────────────────────────────


def _build_argument(claim: str, supporting: list, all_evidence: list) -> str:
    """Build a persuasive verification argument using top supporting evidence."""
    if supporting:
        ev_block = "\n".join(
            f"- [{e['title']}]: {e.get('full_text', e['snippet'])[:300]} "
            f"(stance: {e['stance']}, score: {e['confidence']:.0%}, scraped: {e.get('scraped', False)})"
            for e in supporting[:3]
        )
    else:
        ev_block = "No strong supporting evidence was found."

    try:
        return groq_chat(
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a VERIFIER agent in a fact-checking debate. "
                        "Your role is to argue FOR the truth of the given claim "
                        "using the evidence provided.\n"
                        "- Cite specific evidence and sources\n"
                        "- Be persuasive but honest\n"
                        "- If evidence is weak, acknowledge it\n"
                        "- Keep your argument to 2-3 paragraphs"
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        f"CLAIM: {claim}\n\n"
                        f"SUPPORTING EVIDENCE:\n{ev_block}\n\n"
                        f"Total snippets found: {len(all_evidence)}, "
                        f"supporting: {len(supporting)}\n\n"
                        f"Build your verification argument."
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
            f"Found {len(supporting)} supporting evidence snippets."
        )


# ── Overall assessment ───────────────────────────────────────


def _build_overall(reports: list) -> str:
    """Summarize all per-claim verification results."""
    summary = "\n".join(
        f"• Claim {i + 1}: {r['claim'][:80]} → "
        f"{r['evidence_count']} supporting, confidence {r['confidence']:.0%}"
        for i, r in enumerate(reports)
    )
    try:
        return groq_chat(
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a verification agent. Summarize your overall "
                        "assessment of the article's truthfulness in 1-2 paragraphs."
                    ),
                },
                {"role": "user", "content": f"VERIFICATION RESULTS:\n{summary}"},
            ],
            temperature=0.3,
            max_tokens=500,
        )
    except Exception as e:
        logger.error(f"Overall assessment failed: {e}")
        return "Unable to generate overall assessment."
