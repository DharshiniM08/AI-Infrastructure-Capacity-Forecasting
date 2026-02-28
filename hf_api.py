from __future__ import annotations

import logging
import os
from typing import Any

import requests

from config import get_settings, hf_model_id, hf_token


logger = logging.getLogger(__name__)

API_URL = "https://router.huggingface.co/featherless-ai/v1/completions"


def _get_token() -> str:
    """
    Use env var HF_TOKEN primarily, with a fallback to HF_API_TOKEN.
    Mirrors the user's desired `os.environ['HF_TOKEN']` style while staying robust.
    """
    s = get_settings()
    # Preferred: HF_TOKEN
    token = os.environ.get("HF_TOKEN") or hf_token()
    if not token:
        raise EnvironmentError(
            f"Missing HuggingFace token. Set environment variable `{s.HF_TOKEN_ENV}`."
        )
    return token


# Global headers object, per user template.
headers = {
    "Authorization": f"Bearer {_get_token()}",
}


def query(payload: dict[str, Any]) -> dict[str, Any]:
    """
    User-required shape:

    ```python
    response = requests.post(API_URL, headers=headers, json=payload)
    return response.json()
    ```
    """
    s = get_settings()
    try:
        response = requests.post(API_URL, headers=headers, json=payload, timeout=s.HF_TIMEOUT_SECONDS)
    except requests.RequestException as e:
        raise RuntimeError(f"HuggingFace request failed: {e}") from e

    if response.status_code >= 400:
        try:
            detail = response.json()
        except Exception:  # noqa: BLE001
            detail = response.text
        raise RuntimeError(f"HuggingFace API error ({response.status_code}): {detail}")

    data = response.json()
    if not isinstance(data, dict):
        return {"raw": data}
    return data


def _build_prompt(
    ticket_text: str,
    predicted_category: str | None = None,
    confidence: float | None = None,
    context: dict[str, Any] | None = None,
) -> str:
    context = context or {}
    lines = [
        "You are an expert cloud reliability engineer and customer support agent.",
        "Write a helpful, concise, professional response to the customer ticket.",
        "Use bullet points when appropriate. Include next steps and questions to unblock troubleshooting.",
        "Avoid revealing internal chain-of-thought. Do not mention you are an AI unless asked.",
        "",
        "## Ticket",
        ticket_text.strip(),
        "",
    ]
    if predicted_category:
        lines.extend(
            [
                "## Model context",
                f"- Predicted category: {predicted_category}",
                f"- Confidence: {confidence:.2f}" if confidence is not None else "- Confidence: n/a",
                "",
            ]
        )
    if context:
        lines.append("## Additional context (telemetry / metadata)")
        for k, v in context.items():
            lines.append(f"- {k}: {v}")
        lines.append("")

    lines.extend(
        [
            "## Response requirements",
            "- Start with a brief empathetic acknowledgement.",
            "- Provide a clear diagnosis hypothesis based on the ticket.",
            "- Provide 3-6 concrete next steps.",
            "- End with 1-3 clarifying questions.",
        ]
    )
    return "\n".join(lines)


def generate_hf_llm_response(
    ticket_text: str,
    predicted_category: str | None = None,
    confidence: float | None = None,
    context: dict[str, Any] | None = None,
    model_id: str | None = None,
) -> str:
    """
    Calls HuggingFace Router (featherless-ai) via requests.post (no local transformers).
    """
    model_id = model_id or hf_model_id()
    prompt = _build_prompt(
        ticket_text=ticket_text,
        predicted_category=predicted_category,
        confidence=confidence,
        context=context,
    )

    # Per user snippet: call `query({...})` with an `"inputs"` field.
    # We also pass model + generation settings in the same payload.
    payload: dict[str, Any] = {
        "inputs": prompt,
        "model": model_id,
        "parameters": {
            "max_new_tokens": 280,
            "temperature": 0.5,
            "top_p": 0.9,
        },
    }

    data = query(payload)

    # OpenAI-compatible completion response: {"choices":[{"text":"..."}], ...}
    if isinstance(data, dict) and "choices" in data and isinstance(data["choices"], list) and data["choices"]:
        choice0 = data["choices"][0]
        if isinstance(choice0, dict) and "text" in choice0:
            return str(choice0["text"]).strip()

    # Fallbacks (in case provider changes formats)
    if isinstance(data, dict) and "generated_text" in data:
        return str(data["generated_text"]).strip()

    logger.warning("Unexpected HF response format keys: %s", list(data.keys()) if isinstance(data, dict) else type(data))
    return str(data).strip()

