"""
Judge Agent
Weighs evidence from both Verifier and Falsifier agents.
Produces per-claim verdicts and an overall article verdict.
"""
import json
import logging

from tools.groq_client import groq_chat, groq_chat_json

logger = logging.getLogger(__name__)



def judge_debate(verifier_report: dict, falsifier_report: dict, progress_callback=None) -> dict:
    """
    Judge the debate between Verifier and Falsifier.

    Args:
        verifier_report: Full report from verify_claims()
        falsifier_report: Full report from falsify_claims()
        progress_callback: Optional callable(step, total, status)

    Returns:
        {
            "claim_verdicts": [
                {
                    "claim": str,
                    "verdict": "SUPPORTED" | "REFUTED" | "UNVERIFIABLE",
                    "confidence": float,
                    "reasoning": str
                }
            ],
            "overall_verdict": "REAL" | "FAKE" | "MISLEADING",
            "overall_confidence": float,
            "reasoning": str,
            "summary": str
        }
    """
    ver_claims = verifier_report["claim_reports"]
    fal_claims = falsifier_report["claim_reports"]

    verdicts = []
    total = len(ver_claims)

    for i in range(total):
        ver = ver_claims[i]
        fal = fal_claims[i] if i < len(fal_claims) else None

        if progress_callback:
            progress_callback(i, total, f"Judging claim {i + 1}: {ver['claim'][:50]}...")

        verdicts.append(_judge_single(ver, fal))

    if progress_callback:
        progress_callback(total, total, "Determining final verdict...")

    overall = _overall_verdict(
        verdicts,
        verifier_report["overall_assessment"],
        falsifier_report["overall_assessment"],
    )

    return {"claim_verdicts": verdicts, **overall}


def _judge_single(ver: dict, fal: dict | None) -> dict:
    """Judge one claim by weighing both agents' evidence and arguments."""
    claim = ver["claim"]

    ver_arg = ver["argument"]
    ver_conf = ver["confidence"]
    ver_n = ver["evidence_count"]
    ver_evidence_block = _format_evidence_block(ver.get("supporting_evidence", []))

    fal_arg = fal["argument"] if fal else "No falsification attempted."
    fal_conf = fal.get("confidence", 0) if fal else 0
    fal_n = fal.get("evidence_count", 0) if fal else 0
    fal_evidence_block = _format_evidence_block(
        fal.get("contradicting_evidence", []) if fal else []
    )

    try:
        data = groq_chat_json(
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are an IMPARTIAL JUDGE in a fact-checking debate.\n"
                        "You must weigh evidence from both sides and deliver a verdict.\n\n"
                        "RULES:\n"
                        "1. Prioritize the raw evidence over each agent's rhetoric\n"
                        "2. Consider evidence quantity AND quality\n"
                        "3. Official / institutional sources carry more weight\n"
                        "4. If evidence is inconclusive from both sides, return UNVERIFIABLE\n"
                        "5. Be specific about which evidence influenced your decision\n\n"
                        "Respond in JSON:\n"
                        "{\n"
                        '  "verdict": "SUPPORTED" | "REFUTED" | "UNVERIFIABLE",\n'
                        '  "confidence": 0.0 to 1.0,\n'
                        '  "reasoning": "Your detailed reasoning"\n'
                        "}"
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        f"CLAIM: {claim}\n\n"
                        f"{'=' * 40}\n"
                        f"VERIFIER'S CASE (Pro-Truth)\n"
                        f"{'=' * 40}\n"
                        f"Supporting evidence found: {ver_n}\n"
                        f"Average confidence: {ver_conf:.0%}\n"
                        f"Top supporting evidence:\n{ver_evidence_block}\n\n"
                        f"Argument:\n{ver_arg}\n\n"
                        f"{'=' * 40}\n"
                        f"FALSIFIER'S CASE (Counter-Truth)\n"
                        f"{'=' * 40}\n"
                        f"Contradicting evidence found: {fal_n}\n"
                        f"Average confidence: {fal_conf:.0%}\n"
                        f"Top contradicting evidence:\n{fal_evidence_block}\n\n"
                        f"Argument:\n{fal_arg}\n\n"
                        "Deliver your verdict."
                    ),
                },
            ],
            temperature=0.2,
            max_tokens=600,
        )

        return {
            "claim": claim,
            "verdict": data.get("verdict", "UNVERIFIABLE"),
            "confidence": data.get("confidence", 0.5),
            "reasoning": data.get("reasoning", ""),
        }

    except Exception as e:
        logger.error(f"Judging failed for claim: {e}")
        return {
            "claim": claim,
            "verdict": "UNVERIFIABLE",
            "confidence": 0.0,
            "reasoning": "Error during judgement; could not reach a verdict.",
        }


def _overall_verdict(verdicts: list, ver_assessment: str, fal_assessment: str) -> dict:
    """Determine the overall article verdict from all per-claim results."""
    verdict_summary = "\n".join(
        f"  Claim: {v['claim'][:80]}\n"
        f"  Verdict: {v['verdict']} ({v['confidence']:.0%})\n"
        f"  Reason: {v['reasoning'][:120]}\n"
        for v in verdicts
    )

    try:
        data = groq_chat_json(
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are the FINAL JUDGE in a fact-checking system.\n"
                        "Deliver the overall verdict for the news article.\n\n"
                        "CRITERIA:\n"
                        "- REAL: Most claims are well-supported by evidence\n"
                        "- FAKE: Most claims are refuted by evidence\n"
                        "- MISLEADING: Mix of supported/refuted, or mostly unverifiable\n\n"
                        "Respond in JSON:\n"
                        "{\n"
                        '  "overall_verdict": "REAL" | "FAKE" | "MISLEADING",\n'
                        '  "overall_confidence": 0.0 to 1.0,\n'
                        '  "reasoning": "Multi-sentence explanation",\n'
                        '  "summary": "One-sentence summary for display"\n'
                        "}"
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        f"PER-CLAIM VERDICTS:\n{verdict_summary}\n\n"
                        f"VERIFIER'S OVERALL VIEW:\n{ver_assessment}\n\n"
                        f"FALSIFIER'S OVERALL VIEW:\n{fal_assessment}\n\n"
                        "Deliver the FINAL verdict."
                    ),
                },
            ],
            temperature=0.2,
            max_tokens=600,
        )

        return {
            "overall_verdict": data.get("overall_verdict", "MISLEADING"),
            "overall_confidence": data.get("overall_confidence", 0.5),
            "reasoning": data.get("reasoning", ""),
            "summary": data.get("summary", ""),
        }

    except Exception as e:
        logger.error(f"Overall verdict generation failed: {e}")
        return _fallback_verdict(verdicts)


def _fallback_verdict(verdicts: list) -> dict:
    """Heuristic fallback when LLM verdict fails."""
    supported = sum(1 for v in verdicts if v["verdict"] == "SUPPORTED")
    refuted = sum(1 for v in verdicts if v["verdict"] == "REFUTED")
    total = len(verdicts)

    if supported > refuted and supported >= total / 2:
        verdict = "REAL"
    elif refuted > supported and refuted >= total / 2:
        verdict = "FAKE"
    else:
        verdict = "MISLEADING"

    return {
        "overall_verdict": verdict,
        "overall_confidence": 0.5,
        "reasoning": (
            f"Fallback verdict: {supported} claims supported, "
            f"{refuted} refuted, {total - supported - refuted} unverifiable."
        ),
        "summary": f"Article appears to be {verdict.lower()} based on claim analysis.",
    }


def _format_evidence_block(evidence: list[dict], top_k: int = 3) -> str:
    """Format the strongest evidence snippets for the judge prompt."""
    if not evidence:
        return "No direct evidence provided."

    lines = []
    ranked = sorted(evidence, key=lambda item: item.get("confidence", 0), reverse=True)
    for item in ranked[:top_k]:
        excerpt = (item.get("full_text") or item.get("snippet") or "").replace("\n", " ").strip()
        excerpt = excerpt[:280] if excerpt else "No excerpt available."
        lines.append(
            f"- {item.get('title', 'Untitled source')} | "
            f"{item.get('stance', 'UNKNOWN')} {item.get('confidence', 0):.0%} | "
            f"{item.get('url', 'No URL')}\n"
            f"  {excerpt}"
        )
    return "\n".join(lines)
