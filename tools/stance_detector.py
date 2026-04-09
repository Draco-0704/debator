"""
Stance Detector
Uses Groq to determine whether evidence SUPPORTS, CONTRADICTS,
or is NEUTRAL to a claim.
"""
import logging
import time

from tools.groq_client import groq_chat_json

logger = logging.getLogger(__name__)

STANCE_LABEL_MAP = {
    "SUPPORT": "SUPPORT",
    "SUPPORTED": "SUPPORT",
    "ENTAILMENT": "SUPPORT",
    "CONTRADICT": "CONTRADICT",
    "CONTRADICTION": "CONTRADICT",
    "REFUTE": "CONTRADICT",
    "REFUTED": "CONTRADICT",
    "NEUTRAL": "NEUTRAL",
    "UNCLEAR": "NEUTRAL",
    "UNVERIFIABLE": "NEUTRAL",
}

SYSTEM_PROMPT = """You are a stance classification system for fact-checking.
Given a factual claim and an evidence passage, decide whether the evidence:
- SUPPORTS the claim
- CONTRADICTS the claim
- is NEUTRAL / insufficient

Rules:
1. Use SUPPORT only when the evidence directly backs the claim.
2. Use CONTRADICT only when the evidence directly disputes the claim.
3. Use NEUTRAL when the evidence is unrelated, ambiguous, or insufficient.
4. Confidence must be a number from 0.0 to 1.0.

Respond in JSON only:
{
  "stance": "SUPPORT" | "CONTRADICT" | "NEUTRAL",
  "confidence": 0.0,
  "reasoning": "short explanation"
}"""


def detect_stance(claim: str, evidence: str, max_retries: int = 3) -> dict:
    """
    Detect whether evidence supports or contradicts a claim using Groq.

    Args:
        claim: The factual claim to check
        evidence: The evidence text to evaluate
        max_retries: Retries if the model response is empty or malformed

    Returns:
        {"stance": "SUPPORT"|"CONTRADICT"|"NEUTRAL", "confidence": 0.0-1.0}
    """
    if not evidence or not claim:
        return {"stance": "NEUTRAL", "confidence": 0.0}

    for attempt in range(max_retries):
        try:
            data = groq_chat_json(
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {
                        "role": "user",
                        "content": (
                            f"CLAIM:\n{claim[:500]}\n\n"
                            f"EVIDENCE:\n{evidence[:1400]}\n\n"
                            "Return the stance classification."
                        ),
                    },
                ],
                temperature=0.0,
                max_tokens=150,
                max_retries=2,
            )

            if not data:
                raise ValueError("Groq returned empty or invalid JSON.")

            parsed = _parse_result(data)
            if parsed["confidence"] == 0.0 and not str(data.get("stance", "")).strip():
                raise ValueError(f"Missing stance label in response: {data}")

            return parsed

        except Exception as e:
            logger.warning(f"Stance detection attempt {attempt + 1} failed: {e}")
            if attempt < max_retries - 1:
                time.sleep(1.5)

    logger.error("All stance detection attempts failed — returning NEUTRAL")
    return {"stance": "NEUTRAL", "confidence": 0.0}


def batch_detect_stance(claim: str, evidence_list: list[str]) -> list[dict]:
    """
    Score multiple evidence texts against a single claim.
    Includes small delays between calls.
    """
    results = []
    for evidence in evidence_list:
        result = detect_stance(claim, evidence)
        results.append(result)
        time.sleep(0.1)
    return results


def _parse_result(result) -> dict:
    """
    Normalize stance outputs from Groq.

    This parser also tolerates the old Hugging Face-style list output so the
    rest of the project and tests stay robust during the transition.
    """
    try:
        if isinstance(result, dict):
            raw_label = str(result.get("stance") or result.get("label") or "NEUTRAL").strip().upper()
            confidence = _clamp(result.get("confidence", result.get("score", 0.0)))
            return {
                "stance": STANCE_LABEL_MAP.get(raw_label, "NEUTRAL"),
                "confidence": round(confidence, 3),
            }

        scores = result
        if isinstance(result, list) and result:
            if isinstance(result[0], list):
                scores = result[0]
            else:
                scores = result

        if isinstance(scores, list) and scores:
            best = max(scores, key=lambda item: item.get("score", 0))
            raw_label = str(best.get("label", "NEUTRAL")).strip().upper()
            return {
                "stance": STANCE_LABEL_MAP.get(raw_label, "NEUTRAL"),
                "confidence": round(_clamp(best.get("score", 0.0)), 3),
            }

    except Exception as e:
        logger.warning(f"Failed to parse stance result: {e} — raw: {result}")

    return {"stance": "NEUTRAL", "confidence": 0.0}


def _clamp(value: float | int) -> float:
    """Clamp a confidence-like value to the 0-1 range."""
    try:
        return max(0.0, min(float(value), 1.0))
    except (TypeError, ValueError):
        return 0.0
