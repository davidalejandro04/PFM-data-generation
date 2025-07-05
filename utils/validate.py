"""
Funciones de robustecimiento – intentar extraer JSON válido y reintentar.
"""
import json
import re
from typing import Optional


def extract_json(text: str) -> Optional[str]:
    """
    Extrae la primera cadena que empiece por '{' y acabe por '}'
    (ingenua pero suficiente con Gemma).
    """
    m = re.search(r"\{.*\}", text, re.S)
    return m.group(0) if m else None


def parse_json_or_retry(llm, prompt: str, max_attempts: int = 3) -> dict:
    """
    Invoca el modelo hasta devolver un JSON parseable
    """
    for attempt in range(max_attempts):
        raw = llm.invoke(prompt)
        jtxt = extract_json(raw) or raw
        try:
            return json.loads(jtxt)
        except Exception:
            prompt = ("‼️ DEVUELVE SOLO UN OBJETO JSON, sin texto extra:\n" +
                      jtxt)
    raise ValueError("No JSON válido tras reintentos")
