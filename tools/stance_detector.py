"""
══════════════════════════════════════════════════════════════
  Stance Detector
  Uses HuggingFace Inference API with NLI model to determine
  if evidence SUPPORTS, CONTRADICTS, or is NEUTRAL to a claim.
══════════════════════════════════════════════════════════════
"""
import logging
import time
import requests
from config import HF_API_TOKEN, NLI_MODEL_URL

logger = logging.getLogger(__name__)

# Label mapping: NLI model labels → our stance labels
# cross-encoder/nli-deberta-v3-small label order: contradiction(0), entailment(1), neutral(2)
NLI_LABEL_MAP = {
    # Standard text labels (most HF API responses)
    "entailment": "SUPPORT",
    "contradiction": "CONTRADICT",
    "neutral": "NEUTRAL",
    # Uppercase variants
    "ENTAILMENT": "SUPPORT",
    "CONTRADICTION": "CONTRADICT",
    "NEUTRAL": "NEUTRAL",
    # Numeric labels (some API versions)
    "LABEL_0": "CONTRADICT",   # contradiction
    "LABEL_1": "SUPPORT",      # entailment
    "LABEL_2": "NEUTRAL",      # neutral
}


def detect_stance(claim: str, evidence: str, max_retries: int = 3) -> dict:
    """
    Detect whether evidence supports or contradicts a claim.

    Args:
        claim: The factual claim to check
        evidence: The evidence text to evaluate
        max_retries: Retries for model cold-start delays

    Returns:
        {"stance": "SUPPORT"|"CONTRADICT"|"NEUTRAL", "confidence": 0.0-1.0}
    """
    if not evidence or not claim:
        return {"stance": "NEUTRAL", "confidence": 0.0}

    if not HF_API_TOKEN:
        logger.warning("HF_API_TOKEN is not configured; returning NEUTRAL stance")
        return {"stance": "NEUTRAL", "confidence": 0.0}

    headers = {"Authorization": f"Bearer {HF_API_TOKEN}"}

    # The inference API supports text-pair classification inputs.
    # Treat the evidence as the premise and the claim as the hypothesis.
    payload = {
        "inputs": {
            "text": evidence[:1200],
            "text_pair": claim[:500],
        },
        "parameters": {"top_k": 3},
    }

    for attempt in range(max_retries):
        try:
            response = requests.post(
                NLI_MODEL_URL,
                headers=headers,
                json=payload,
                timeout=60,
            )

            # Handle model cold-start (503 = model loading)
            if response.status_code == 503:
                try:
                    body = response.json()
                except ValueError:
                    body = {}
                wait = min(body.get("estimated_time", 20), 30)
                logger.info(f"NLI model loading — waiting {wait:.0f}s (attempt {attempt + 1})")
                time.sleep(wait)
                continue

            response.raise_for_status()
            result = response.json()
            return _parse_result(result)

        except Exception as e:
            logger.warning(f"Stance detection attempt {attempt + 1} failed: {e}")
            if attempt < max_retries - 1:
                time.sleep(2)

    logger.error("All stance detection attempts failed — returning NEUTRAL")
    return {"stance": "NEUTRAL", "confidence": 0.0}


def batch_detect_stance(claim: str, evidence_list: list[str]) -> list[dict]:
    """
    Score multiple evidence texts against a single claim.
    Includes rate-limit delays between calls.
    """
    results = []
    for evidence in evidence_list:
        result = detect_stance(claim, evidence)
        results.append(result)
        time.sleep(0.1)  # Minimal rate limiting
    return results


def _parse_result(result) -> dict:
    """Parse the HuggingFace text-classification API response."""
    try:
        # API returns: [[{"label": "...", "score": 0.xx}, ...]]
        # or: [{"label": "...", "score": 0.xx}, ...]
        scores = result
        if isinstance(result, list) and len(result) > 0:
            if isinstance(result[0], list):
                scores = result[0]
            else:
                scores = result

        if isinstance(scores, list) and len(scores) > 0:
            # Find the label with highest score
            best = max(scores, key=lambda x: x.get("score", 0))
            raw_label = best.get("label", "neutral")
            stance = NLI_LABEL_MAP.get(raw_label, NLI_LABEL_MAP.get(raw_label.upper(), "NEUTRAL"))

            return {
                "stance": stance,
                "confidence": round(best.get("score", 0.0), 3),
            }

    except Exception as e:
        logger.warning(f"Failed to parse NLI result: {e} — raw: {result}")

    return {"stance": "NEUTRAL", "confidence": 0.0}
