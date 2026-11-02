"""Small Ollama HTTP client used by the STOV-ATOLL RAG application."""

from __future__ import annotations

from functools import lru_cache
from typing import Any

import requests

# Every generation call uses the same window so Ollama never reloads the model between calls.
CONTEXT_TOKENS = 8192

class OllamaError(RuntimeError):
    """Raised when Ollama cannot complete a request."""


def _post(base_url: str, endpoint: str, payload: dict[str, Any], timeout: int) -> dict[str, Any]:
    try:
        response = requests.post(
            f"{base_url.rstrip('/')}{endpoint}",
            json=payload,
            timeout=timeout,
        )
        response.raise_for_status()
        return response.json()
    except requests.HTTPError as exc:
        raise OllamaError(
            f"{base_url}{endpoint} returned {exc.response.status_code}: {exc.response.text[:500]}"
        ) from exc
    except requests.RequestException as exc:
        raise OllamaError(
            f"Cannot reach Ollama at {base_url}. Start Ollama and verify the model is installed."
        ) from exc
    except ValueError as exc:
        raise OllamaError("Ollama returned an invalid JSON response.") from exc


@lru_cache(maxsize=None)
def _slot_tokens(base_url: str) -> int:
    response = requests.get(f"{base_url.rstrip('/')}/props", timeout=30)
    response.raise_for_status()
    return response.json()["default_generation_settings"]["n_ctx"]


def _split_text(text: str, base_url: str, limit: int) -> list[str]:
    tokens = _post(base_url, "/tokenize", {"content": text}, timeout=60)["tokens"]
    if len(tokens) <= limit or len(text) < 2:
        return [text]
    middle = len(text) // 2
    cut = text.rfind(" ", 0, middle)
    cut = cut if cut > 0 else middle
    return [
        piece
        for half in (text[:cut].strip(), text[cut:].strip())
        if half
        for piece in _split_text(half, base_url, limit)
    ]


def split_to_context(texts: list[str], base_url: str) -> list[list[str]]:
    """Split each text into pieces that fit a llama-server slot, keeping every character."""
    # Leave room for the [CLS]/[SEP] tokens the server adds when embedding.
    limit = _slot_tokens(base_url) - 4
    return [_split_text(text, base_url, limit) for text in texts]


def embed(
    texts: list[str],
    model: str,
    base_url: str = "http://127.0.0.1:11434",
    api: str = "ollama",
) -> list[list[float]]:
    """Create embeddings through Ollama or an OpenAI-compatible server such as llama-server."""
    if api == "openai":
        result = _post(base_url, "/v1/embeddings", {"model": model, "input": texts}, timeout=300)
        data = sorted(result.get("data") or [], key=lambda item: item["index"])
        if len(data) != len(texts):
            raise OllamaError("Embedding server returned an unexpected number of embeddings.")
        return [item["embedding"] for item in data]

    result = _post(
        base_url,
        "/api/embed",
        {"model": model, "input": texts, "keep_alive": -1},
        timeout=300,
    )
    embeddings = result.get("embeddings")
    if not embeddings:
        raise OllamaError("Ollama returned no embeddings.")
    return embeddings


def generate(
    prompt: str,
    model: str,
    base_url: str = "http://127.0.0.1:11434",
    temperature: float = 0.1,
    json_output: bool = False,
    max_tokens: int | None = None,
) -> str:
    """Generate a response without streaming; json_output constrains it to valid JSON."""
    options: dict[str, Any] = {"temperature": temperature, "num_ctx": CONTEXT_TOKENS}
    if max_tokens:
        options["num_predict"] = max_tokens
    payload: dict[str, Any] = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "keep_alive": -1,
        "options": options,
    }
    if json_output:
        payload["format"] = "json"
    result = _post(base_url, "/api/generate", payload, timeout=300)
    answer = result.get("response", "").strip()
    if not answer:
        raise OllamaError("Ollama returned an empty answer.")
    return answer


def chat(
    messages: list[dict[str, Any]],
    model: str,
    base_url: str = "http://127.0.0.1:11434",
    tools: list[dict[str, Any]] | None = None,
    temperature: float = 0.1,
) -> dict[str, Any]:
    """Send a chat turn and return the assistant message, which may contain tool calls."""
    payload: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "stream": False,
        "keep_alive": -1,
        "options": {"temperature": temperature, "num_ctx": CONTEXT_TOKENS},
    }
    if tools:
        payload["tools"] = tools
    message = _post(base_url, "/api/chat", payload, timeout=300).get("message") or {}
    if not message.get("content", "").strip() and not message.get("tool_calls"):
        raise OllamaError("Ollama returned an empty answer.")
    return message
