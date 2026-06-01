"""
llm_client.py — Thin wrapper around Ollama for all LLM calls.
Single point of control for model, temperature, and error handling.
"""

import ollama
from config import OLLAMA_MODEL


def llm_call(
    system: str,
    user: str,
    num_predict: int = 512,
    temperature: float = 0,
) -> str:
    """
    Execute a single Ollama chat call.

    Args:
        system:      System prompt string.
        user:        User message string.
        num_predict: Max tokens to generate (default 512).
        temperature: Sampling temperature (default 0 = deterministic).

    Returns the model's reply string, or an ERROR: ... string on failure.
    """
    try:
        resp = ollama.chat(
            model=OLLAMA_MODEL,
            messages=[
                {"role": "system", "content": system},
                {"role": "user",   "content": user},
            ],
            options={"temperature": temperature, "num_predict": num_predict},
        )
        return resp["message"]["content"].strip()
    except ollama.ResponseError as e:
        return (
            f"ERROR: Model '{OLLAMA_MODEL}' returned an error: {e}. "
            "Check that the model is pulled — run: `ollama pull qwen2.5:7b`"
        )
    except Exception as e:
        return f"ERROR: Unexpected error calling Ollama: {e}"