"""
══════════════════════════════════════════════════════════════
  Claim Extractor Agent
  Uses spaCy NER + Groq LLM to extract verifiable factual
  claims from news articles.
══════════════════════════════════════════════════════════════
"""
import json
import logging
import spacy

from config import SPACY_MODEL, MAX_CLAIMS
from tools.groq_client import groq_chat_json

logger = logging.getLogger(__name__)

# ── Load spaCy ───────────────────────────────────────────────
try:
    nlp = spacy.load(SPACY_MODEL)
    logger.info(f"spaCy model loaded: {SPACY_MODEL}")
except OSError:
    try:
        nlp = spacy.load("en_core_web_sm")
        logger.warning("Fell back to en_core_web_sm")
    except OSError:
        nlp = None
        logger.error(
            "No spaCy model found! Run: python -m spacy download en_core_web_sm"
        )



# ── Entity label display names ───────────────────────────────
LABEL_NAMES = {
    "PERSON": "People",
    "ORG": "Organizations",
    "GPE": "Locations",
    "LOC": "Locations",
    "DATE": "Dates",
    "TIME": "Times",
    "MONEY": "Money",
    "PERCENT": "Percentages",
    "EVENT": "Events",
    "NORP": "Groups/Nationalities",
    "PRODUCT": "Products",
    "WORK_OF_ART": "Works",
    "LAW": "Laws/Regulations",
    "QUANTITY": "Quantities",
}

SYSTEM_PROMPT = """You are an expert fact-checker and claim extraction specialist.
Your task is to extract specific, verifiable FACTUAL claims from news articles.

RULES:
1. Extract exactly 3 to 5 claims that can be independently verified via web search
2. Each claim MUST be a specific factual statement — NOT an opinion or subjective assessment
3. Each claim should reference specific people, organizations, places, dates, or numbers
4. Claims must be self-contained — understandable without reading the full article
5. Prioritize the most important, consequential, and checkable claims
6. Avoid vague claims like "the economy is bad" — prefer "GDP dropped 2% in Q3 2025"

RESPOND IN THIS EXACT JSON FORMAT ONLY:
{
    "claims": [
        {
            "claim": "The specific factual claim statement",
            "entities": ["Entity1", "Entity2"],
            "importance": "high"
        }
    ]
}"""


def extract_entities(text: str) -> dict[str, list[str]]:
    """
    Extract named entities from text using spaCy NER.

    Returns:
        Dict mapping entity labels to lists of entity texts.
        Example: {"PERSON": ["Joe Biden"], "ORG": ["WHO"]}
    """
    if not nlp:
        return {}

    doc = nlp(text[:10000])  # Cap text length for spaCy

    entities: dict[str, list[str]] = {}
    for ent in doc.ents:
        label = ent.label_
        if label not in entities:
            entities[label] = []
        if ent.text.strip() not in entities[label]:
            entities[label].append(ent.text.strip())

    logger.info(
        f"NER extracted {sum(len(v) for v in entities.values())} entities "
        f"across {len(entities)} categories"
    )
    return entities


def extract_claims(article_text: str) -> list[dict]:
    """
    Extract 3-5 verifiable factual claims from a news article.

    Pipeline:
        1. spaCy NER  →  extract entities (who, where, what, when)
        2. Groq LLM   →  generate structured claims using entities as hints

    Args:
        article_text: Full text of the news article

    Returns:
        List of claim dicts:
        [{"claim": str, "entities": [str], "importance": str}, ...]
    """
    # Step 1: NER
    entities = extract_entities(article_text)
    entity_block = _format_entities(entities)

    # Step 2: LLM claim extraction
    user_prompt = f"""ARTICLE TEXT:
{article_text[:4000]}

DETECTED ENTITIES:
{entity_block}

Extract 3–5 verifiable factual claims from this article."""

    try:
        data = groq_chat_json(
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.2,
            max_tokens=1500,
        )

        claims = data.get("claims", [])[:MAX_CLAIMS]

        logger.info(f"Extracted {len(claims)} claims via LLM")
        return claims

    except json.JSONDecodeError as e:
        logger.error(f"LLM response was not valid JSON: {e}")
        return _fallback_extraction(article_text)
    except Exception as e:
        logger.error(f"Claim extraction failed: {e}")
        return _fallback_extraction(article_text)


# ── Helpers ──────────────────────────────────────────────────


def _format_entities(entities: dict[str, list[str]]) -> str:
    """Format the NER output into a readable block for the LLM prompt."""
    if not entities:
        return "  (no entities detected)"

    lines = []
    for label, items in entities.items():
        display = LABEL_NAMES.get(label, label)
        lines.append(f"  {display}: {', '.join(items[:6])}")
    return "\n".join(lines)


def _fallback_extraction(article_text: str) -> list[dict]:
    """
    Simple heuristic fallback when LLM extraction fails.
    Picks sentences that contain numbers (likely factual).
    """
    logger.info("Using heuristic fallback for claim extraction")

    sentences = [s.strip() for s in article_text.split(".") if s.strip()]
    claims = []

    for sent in sentences[:20]:
        # Heuristic: sentences with numbers are more likely to be factual
        if len(sent) > 40 and any(c.isdigit() for c in sent):
            claims.append(
                {"claim": sent.rstrip(".") + ".", "entities": [], "importance": "medium"}
            )
        if len(claims) >= 3:
            break

    # If still nothing, just take the first substantive sentence
    if not claims and sentences:
        claims.append(
            {"claim": sentences[0].rstrip(".") + ".", "entities": [], "importance": "medium"}
        )

    return claims
