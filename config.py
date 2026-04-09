"""
══════════════════════════════════════════════════════════════
  FAKE NEWS DEBATER — Central Configuration
══════════════════════════════════════════════════════════════
"""
import os
import logging
from dotenv import load_dotenv

load_dotenv()

# ── Logging ──────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
)

# ── API Keys ─────────────────────────────────────────────────
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
SERPER_API_KEY = os.getenv("SERPER_API_KEY")
HF_API_TOKEN = os.getenv("HF_API_TOKEN")

# ── Model Configuration ─────────────────────────────────────
GROQ_MODEL = "llama-3.3-70b-versatile"
SPACY_MODEL = "en_core_web_sm"  # Using small model for speed; switch to en_core_web_md for accuracy
NLI_MODEL_URL = "https://router.huggingface.co/hf-inference/models/cross-encoder/nli-deberta-v3-small"

# ── Pipeline Limits ──────────────────────────────────────────
MAX_CLAIMS = 5
MAX_SEARCH_RESULTS = 5
API_CALL_DELAY = 0.5  # seconds between Groq API calls (rate-limit protection)
MIN_ARTICLE_LENGTH = 50  # Minimum article text length (chars)
MAX_ARTICLE_LENGTH = 50_000  # Maximum article text length (chars)
MAX_SCRAPED_TEXT = 1500  # Max chars extracted from a scraped page for stance detection

# ── Validation ───────────────────────────────────────────────
def validate_config():
    """Check that all required API keys are set."""
    missing = []
    if not GROQ_API_KEY:
        missing.append("GROQ_API_KEY")
    if not HF_API_TOKEN:
        missing.append("HF_API_TOKEN")
    if missing:
        raise EnvironmentError(
            f"Missing required API keys: {', '.join(missing)}. "
            f"Please set them in your .env file."
        )
    # SERPER is optional (fallback only)
    if not SERPER_API_KEY:
        logging.getLogger(__name__).warning(
            "SERPER_API_KEY not set — DuckDuckGo will be the only search source."
        )
