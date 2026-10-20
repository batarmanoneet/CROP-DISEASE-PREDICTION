from __future__ import annotations

import base64
import os
from io import BytesIO
from typing import List, Dict, Optional

from PIL import Image


def get_setting(name: str, default: str = "") -> str:
    """Read configuration from environment variables or Streamlit secrets.

    Environment variables take precedence, which keeps local Windows setup simple.
    Missing Streamlit secrets are normal when running without an API key.
    """
    value = os.getenv(name)
    if value:
        return value
    try:
        import streamlit as st
        secret_value = st.secrets.get(name, default)
        return str(secret_value) if secret_value is not None else default
    except Exception:
        return default


def is_chat_api_configured() -> bool:
    return bool(get_setting("OPENAI_API_KEY").strip())


def _fallback_answer(question: str, context: str) -> str:
    q = question.lower()
    if context:
        lead = "Based on the latest image screening, here is the cautious interpretation:\n\n" + context + "\n\n"
    else:
        lead = "I can help with crop-disease questions, but I need more context to be specific.\n\n"

    if any(x in q for x in ["what should i check", "next", "verify", "confirm"]):
        return lead + "Next, check the crop, plant part affected, how quickly the symptoms appeared, whether spots are dry or water-soaked, and whether nearby plants show the same symptoms. A close-up photo of both the affected area and a healthy area is useful."
    if any(x in q for x in ["prevent", "prevention", "spread"]):
        return lead + "Start with sanitation, remove badly affected plant material where appropriate, avoid prolonged leaf wetness, improve airflow, and avoid moving contaminated plant material between plots. Use crop-specific extension guidance for any treatment decision."
    return lead + "I can explain the predicted disease, compare it with the top alternatives, or help you decide what additional photo or field information to collect."


def _jpeg_data_url(image_bytes: bytes) -> str:
    """Normalize uploaded JPG/PNG/WEBP content into a valid JPEG data URL."""
    with Image.open(BytesIO(image_bytes)) as image:
        rgb_image = image.convert("RGB")
        buffer = BytesIO()
        rgb_image.save(buffer, format="JPEG", quality=85, optimize=True)
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/jpeg;base64,{encoded}"


def generate_chat_response(
    question: str,
    conversation: List[Dict],
    context: str,
    image_bytes: Optional[bytes] = None,
) -> str:
    api_key = get_setting("OPENAI_API_KEY").strip()
    if not api_key:
        return _fallback_answer(question, context)

    from openai import OpenAI

    client = OpenAI(api_key=api_key)
    model = get_setting("OPENAI_MODEL", "gpt-6-luna").strip() or "gpt-6-luna"

    system = (
        "You are Crop Doctor, a cautious agricultural assistant. "
        "The dedicated image classifier is the primary disease-screening signal; never present its prediction as certain. "
        "Use the supplied knowledge-base context when present. Ask for missing crop/field context when needed. "
        "Do not invent pesticide names, dosages, waiting periods, or legal recommendations. "
        "Prefer integrated pest management and general, non-prescriptive steps. "
        "For low-confidence or contradictory evidence, say so clearly and recommend confirmation by a local agriculture professional or extension service. "
        "Answer in simple language suitable for farmers."
    )

    history = list(conversation or [])
    # The Streamlit UI may already have appended the current question to history.
    if history and history[-1].get("role") == "user" and history[-1].get("content") == question:
        history = history[:-1]
    recent = history[-8:]
    history_text = "\n".join(f"{m.get('role', 'user')}: {m.get('content', '')}" for m in recent)
    user_text = (
        f"Knowledge/image context:\n{context or 'None'}\n\n"
        f"Conversation so far:\n{history_text or '(none)'}\n\n"
        f"Current question:\n{question}"
    )

    content = [{"type": "input_text", "text": user_text}]
    if image_bytes:
        try:
            content.append({"type": "input_image", "image_url": _jpeg_data_url(image_bytes), "detail": "low"})
        except Exception:
            # Text chat should still work if the uploaded image can't be normalized.
            pass

    try:
        response = client.responses.create(
            model=model,
            instructions=system,
            input=[{"role": "user", "content": content}],
        )
        answer = (response.output_text or "").strip()
        return answer or _fallback_answer(question, context)
    except Exception:
        return (
            "I couldn't reach the optional AI chat service just now. Check the API key, configured model, "
            "internet connection, and API account billing. Here is the built-in response instead:\n\n"
            + _fallback_answer(question, context)
        )
