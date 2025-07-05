# utils/validate.py
import json, re, logging
from typing import Optional
from rubrics import EVAL_CODES, ACTION_CODES, STATE_CODES

log = logging.getLogger(__name__)

# ---------- helpers ------------------------------------------------------
def _extract_json(text: str) -> Optional[str]:
    m = re.search(r"\{.*\}", text, re.S)
    return m.group(0) if m else None

def _codes_valid(j: dict) -> bool:
    """Comprueba que los tres códigos estén dentro de las tablas oficiales."""
    return (j.get("Eval of Student Response") in EVAL_CODES and
            j.get("Action Based on Eval")    in ACTION_CODES and
            j.get("Subproblem State")        in STATE_CODES)

# ---------- entry-point --------------------------------------------------
def parse_json_or_retry(llm, prompt: str,
                        max_attempts: int = 5) -> Optional[dict]:
    """
    Intenta que el modelo devuelva un JSON 100 % válido y consistente:
    - Estructura JSON parseable
    - Códigos dentro de las tablas oficiales
    Devuelve None si falla tras max_attempts.
    """
    for attempt in range(1, max_attempts + 1):
        raw = llm.invoke(prompt)
        jtxt = _extract_json(raw) or raw
        try:
            data = json.loads(jtxt)
            if _codes_valid(data):
                return data
        except Exception:
            pass
        # reforzar instrucción
        prompt = ("❗ Devuelve SOLO el objeto JSON, sin texto extra, "
                  "corrigiendo cualquier error:")
    log.warning("⚠️  JSON inválido tras %d intentos", max_attempts)
    return None