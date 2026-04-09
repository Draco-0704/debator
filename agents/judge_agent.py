"""
Judge Agent
Weighs evidence from both Verifier and Falsifier agents.
Produces per-claim verdicts and an overall article verdict.
"""
import logging

from tools.groq_client import groq_chat_json

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
    metrics = _score_overall_verdict(verdicts)
    overall = metrics["overall_verdict"]
    confidence = metrics["overall_confidence"]
    scores = metrics["confidence_metrics"]

    supported = sum(1 for v in verdicts if v.get("verdict") == "SUPPORTED")
    refuted = sum(1 for v in verdicts if v.get("verdict") == "REFUTED")
    unverifiable = sum(1 for v in verdicts if v.get("verdict") == "UNVERIFIABLE")

    reasoning = (
        f"Overall verdict is based on claim-level outcomes rather than a final free-form model guess. "
        f"Supported claims contributed most to REAL ({scores['REAL']:.0%}), "
        f"refuted claims contributed most to FAKE ({scores['FAKE']:.0%}), and "
        f"mixed or unverifiable claims contributed most to MISLEADING ({scores['MISLEADING']:.0%}). "
        f"Claim counts: {supported} supported, {refuted} refuted, {unverifiable} unverifiable. "
        f"Verifier summary: {ver_assessment[:180] or 'No verifier summary.'} "
        f"Falsifier summary: {fal_assessment[:180] or 'No falsifier summary.'}"
    )

    summary_map = {
        "REAL": "Most claims are supported strongly enough for the article to read as real.",
        "FAKE": "Refuted claims outweigh supported ones, so the article reads as fake.",
        "MISLEADING": "The article mixes weak, conflicting, or unverifiable claims, so it reads as misleading.",
    }

    return {
        "overall_verdict": overall,
        "overall_confidence": confidence,
        "reasoning": reasoning,
        "summary": summary_map[overall],
        "confidence_metrics": scores,
    }


def _score_overall_verdict(verdicts: list[dict]) -> dict:
    """
    Convert claim-level verdicts into stable article-level scores.

    The old implementation asked an LLM for the final article label, which made
    the headline verdict drift toward "MISLEADING" even when the claim outcomes
    clearly leaned real or fake. Here we deterministically score three article
    buckets and choose the strongest one.
    """
    if not verdicts:
        return {
            "overall_verdict": "MISLEADING",
            "overall_confidence": 0.0,
            "confidence_metrics": {"REAL": 0.0, "FAKE": 0.0, "MISLEADING": 1.0},
        }

    real_points = 0.0
    fake_points = 0.0
    misleading_points = 0.0

    for verdict in verdicts:
        label = verdict.get("verdict", "UNVERIFIABLE")
        confidence = _clamp(verdict.get("confidence", 0.5))

        if label == "SUPPORTED":
            real_points += confidence
            misleading_points += (1.0 - confidence) * 0.25
        elif label == "REFUTED":
            fake_points += confidence
            misleading_points += (1.0 - confidence) * 0.25
        else:
            misleading_points += 0.45 + (confidence * 0.35)

    # Conflicting support and refutation should visibly increase the
    # misleading bucket even when both sides look strong.
    misleading_points += min(real_points, fake_points) * 0.9

    total = real_points + fake_points + misleading_points
    if total <= 0:
        return {
            "overall_verdict": "MISLEADING",
            "overall_confidence": 0.0,
            "confidence_metrics": {"REAL": 0.0, "FAKE": 0.0, "MISLEADING": 1.0},
        }

    scores = {
        "REAL": round(real_points / total, 3),
        "FAKE": round(fake_points / total, 3),
        "MISLEADING": round(misleading_points / total, 3),
    }

    ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)
    overall_verdict, top_score = ranked[0]
    runner_up = ranked[1][1] if len(ranked) > 1 else 0.0

    # If real and fake are very close, prefer misleading even if it is not the
    # single largest bucket, because the article is genuinely mixed.
    if abs(scores["REAL"] - scores["FAKE"]) <= 0.08 and max(scores["REAL"], scores["FAKE"]) >= 0.3:
        overall_verdict = "MISLEADING"
        top_score = scores["MISLEADING"]
        runner_up = max(scores["REAL"], scores["FAKE"])

    confidence = round(max(top_score, top_score - (runner_up * 0.15)), 3)

    return {
        "overall_verdict": overall_verdict,
        "overall_confidence": confidence,
        "confidence_metrics": scores,
    }


def _fallback_verdict(verdicts: list) -> dict:
    """Heuristic fallback when LLM verdict fails."""
    metrics = _score_overall_verdict(verdicts)
    supported = sum(1 for v in verdicts if v["verdict"] == "SUPPORTED")
    refuted = sum(1 for v in verdicts if v["verdict"] == "REFUTED")
    total = len(verdicts)

    return {
        "overall_verdict": metrics["overall_verdict"],
        "overall_confidence": metrics["overall_confidence"],
        "reasoning": (
            f"Fallback verdict: {supported} claims supported, "
            f"{refuted} refuted, {total - supported - refuted} unverifiable."
        ),
        "summary": f"Article appears to be {metrics['overall_verdict'].lower()} based on claim analysis.",
        "confidence_metrics": metrics["confidence_metrics"],
    }


def _clamp(value: float | int) -> float:
    """Clamp a confidence-like value to the 0-1 range."""
    try:
        return max(0.0, min(float(value), 1.0))
    except (TypeError, ValueError):
        return 0.0


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
