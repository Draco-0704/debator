"""
Fake News Debater central configuration.
"""
import logging
import os

from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
)

# API keys
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
SERPER_API_KEY = os.getenv("SERPER_API_KEY")
HF_API_TOKEN = os.getenv("HF_API_TOKEN")

# Model configuration
GROQ_MODEL = "llama-3.3-70b-versatile"
SPACY_MODEL = "en_core_web_sm"  # Switch to en_core_web_md for higher accuracy

# Pipeline limits
MAX_CLAIMS = 5
MAX_SEARCH_RESULTS = 5
API_CALL_DELAY = 0.5  # seconds between Groq API calls
MIN_ARTICLE_LENGTH = 50
MAX_ARTICLE_LENGTH = 50_000
MAX_SCRAPED_TEXT = 1500


def validate_config():
    """Check that all required API keys are set."""
    missing = []
    if not GROQ_API_KEY:
        missing.append("GROQ_API_KEY")

    if missing:
        raise EnvironmentError(
            f"Missing required API keys: {', '.join(missing)}. "
            "Please set them in your .env file."
        )

    if not SERPER_API_KEY:
        logging.getLogger(__name__).warning(
            "SERPER_API_KEY not set - DuckDuckGo will be the only search source."
        )

    if HF_API_TOKEN:
        logging.getLogger(__name__).info(
            "HF_API_TOKEN is set but no longer required for stance detection."
        )
