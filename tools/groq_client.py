"""
══════════════════════════════════════════════════════════════
  Shared Groq Client
  Centralized LLM client with retry logic, rate-limit handling,
  and exponential backoff for 429/503 errors.
══════════════════════════════════════════════════════════════
"""
import json
import logging
import time
import threading
from functools import lru_cache

from groq import Groq

from config import GROQ_API_KEY, GROQ_MODEL, API_CALL_DELAY

logger = logging.getLogger(__name__)

_client_lock = threading.Lock()


@lru_cache(maxsize=1)
def get_groq_client() -> Groq:
    """Create the Groq client lazily — thread-safe singleton."""
    with _client_lock:
        if not GROQ_API_KEY:
            raise ValueError("GROQ_API_KEY is not configured.")
        return Groq(api_key=GROQ_API_KEY)


def groq_chat(
    messages: list[dict],
    *,
    temperature: float = 0.3,
    max_tokens: int = 800,
    json_mode: bool = False,
    max_retries: int = 3,
    model: str | None = None,
) -> str:
    """
    Call Groq chat completion with automatic retry on rate-limit (429)
    and server errors (500/502/503).

    Args:
        messages: Chat messages list
        temperature: Sampling temperature
        max_tokens: Max response tokens
        json_mode: If True, request JSON response format
        max_retries: Number of retry attempts
        model: Override model (defaults to config GROQ_MODEL)

    Returns:
        The response content string

    Raises:
        Exception: After all retries exhausted
    """
    client = get_groq_client()
    target_model = model or GROQ_MODEL

    kwargs = {
        "model": target_model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    if json_mode:
        kwargs["response_format"] = {"type": "json_object"}
        # Groq requires the word "json" in messages when using json_object mode.
        # Auto-inject if not already present.
        has_json_word = any(
            "json" in (m.get("content") or "").lower() for m in messages
        )
        if not has_json_word:
            # Append instruction to the last system message, or add one
            patched = False
            for m in kwargs["messages"]:
                if m["role"] == "system":
                    m["content"] += "\nRespond in JSON format."
                    patched = True
                    break
            if not patched:
                kwargs["messages"].insert(0, {"role": "system", "content": "Respond in JSON format."})

    last_error = None

    for attempt in range(max_retries):
        try:
            time.sleep(API_CALL_DELAY)
            response = client.chat.completions.create(**kwargs)
            return response.choices[0].message.content

        except Exception as e:
            last_error = e
            error_str = str(e).lower()

            # Rate limit (429) or server error (500/502/503)
            is_retryable = any(
                code in error_str
                for code in ["429", "rate_limit", "too many requests", "500", "502", "503"]
            )

            if is_retryable and attempt < max_retries - 1:
                wait = min(2 ** (attempt + 1), 30)  # 2s, 4s, 8s... max 30s
                logger.warning(
                    f"Groq API error (attempt {attempt + 1}/{max_retries}): {e} "
                    f"— retrying in {wait}s"
                )
                time.sleep(wait)
            elif not is_retryable:
                # Non-retryable error — fail immediately
                raise
            else:
                logger.error(
                    f"Groq API failed after {max_retries} attempts: {e}"
                )

    raise last_error  # type: ignore[misc]


def groq_chat_json(
    messages: list[dict],
    *,
    temperature: float = 0.3,
    max_tokens: int = 800,
    max_retries: int = 3,
) -> dict:
    """
    Call Groq and parse the response as JSON.
    Falls back to empty dict on parse failure.
    """
    text = groq_chat(
        messages,
        temperature=temperature,
        max_tokens=max_tokens,
        json_mode=True,
        max_retries=max_retries,
    )
    try:
        return json.loads(text)
    except json.JSONDecodeError as e:
        logger.error(f"Groq response was not valid JSON: {e}")
        return {}
