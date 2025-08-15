#!/usr/bin/env python3
"""
Traduce todos los valores de un JSONL al español y añade clasificación pedagógica.
NO altera las claves ni el formato original.
"""

import argparse
import json
import re
from typing import Dict, Any
import os
from tqdm import tqdm
#!/usr/bin/env python3
import argparse
import json
import re
from typing import Dict, Any

from tqdm import tqdm

AREAS_OBJETIVOS_CONTEXT = """
Clasifica el problema y su solución (ya traducidos) según:

ÁREAS DE CONOCIMIENTO
- AC01: Pensamiento numérico y sistemas numéricos
- AC02: Pensamiento espacial y sistemas geométricos
- AC03: Medición y sistemas métricos
- AC04: Análisis de datos y probabilidad
- AC05: Álgebra y relaciones

OBJETIVOS PEDAGÓGICOS
- OP01: Reconocer y describir regularidades y patrones en distintos contextos.
- OP02: Describir y representar situaciones de variación (diagramas, tablas, etc.).
- OP03: Construir igualdades/desigualdades numéricas.
- OP04: Resolver situaciones aditivas y multiplicativas.
- OP05: Resolver problemas con fracciones, decimales y porcentajes.
- OP06: Interpretar y representar datos (gráficos, tablas).
- OP07: Aplicar geometría para describir figuras/cuerpos.
- OP08: Utilizar unidades de medida (longitud, área, volumen, tiempo).
- OP09: Desarrollar pensamiento lógico-crítico.
- OP10: Fomentar la comunicación matemática.

**Responde ÚNICAMENTE** con un JSON EXACTO, sin texto adicional, con este formato:

{{"area_conocimiento":"ACxx","objetivo_pedagogico":"OPxx"}}
"""

def _init_ollama(model: str):
    from langchain_ollama import OllamaLLM
    from langchain_core.prompts import ChatPromptTemplate

    llm = OllamaLLM(model=model, temperature=0)

    # Prompt de traducción
    translate_prompt = ChatPromptTemplate.from_messages([
        ("system", "Traduce al español exactamente el texto que se te da. No añadas ni quites nada. IMPORTANTE: SOLO GENERA TEXTO EN ESPAÑOL"),
        ("user", "{input}")
    ])
    translate_chain = translate_prompt | llm

    # Prompt de clasificación (llaves escapadas con doble {{ }} en Python)
    classify_prompt = ChatPromptTemplate.from_messages([
        ("system", AREAS_OBJETIVOS_CONTEXT + "\nResponde SOLO el JSON citado arriba. La clasificación debe ser estricta y precisa."),
        ("user", "{input}")
    ])
    classify_chain = classify_prompt | llm

    summarize_prompt = ChatPromptTemplate.from_messages([
        ("system",
         "Resume brevemente en español los pasos clave de la solución dada. "
         "Sé claro y conciso, sin cambiar la idea esencial."),
        ("user", "{input}")
    ])
    summarize_chain = summarize_prompt | llm


    number_regex = re.compile(r"^\s*\d+(\.\d+)?\s*$")

    def translate(text: str) -> str:
        # No traducir números puros
        if number_regex.match(text):
            return text.strip()
        out = translate_chain.invoke({"input": text}).strip()
        # Eliminar prefijos tipo "Human:" o "Assistant:"
        out = re.sub(r"^(Human|Assistant)\s*:\s*", "", out, flags=re.I)
        return out or text

    def classify(problem_es: str, solution_es: str) -> Dict[str, str]:
        prompt_input = f"Problema: {problem_es}\n\nSolución: {solution_es}"
        raw = classify_chain.invoke({"input": prompt_input}).strip()

        # Aislar JSON
        if not raw.startswith("{"):
            m = re.search(r"(\{.*\})", raw, re.DOTALL)
            raw = m.group(1) if m else ""

        try:
            parsed = json.loads(raw)
            return {
                "area_conocimiento": parsed.get("area_conocimiento", "AC01"),
                "objetivo_pedagogico": parsed.get("objetivo_pedagogico", "OP09")
            }
        except Exception as e:
            print(f"⚠️ No se pudo parsear JSON de clasificación: {e}")
            return {"area_conocimiento": "AC01", "objetivo_pedagogico": "OP09"}

    def summarize(solution_es: str) -> str:
        if not solution_es.strip():
            return ""
        return summarize_chain.invoke({"input": solution_es}).strip()


    return translate, classify,summarize

def translate_classify_and_summarize(
        obj: Dict[str, Any],
        translate_fn,
        classify_fn,
        summarize_fn) -> Dict[str, Any]:
    """Procesa un objeto: traduce, clasifica y añade el resumen."""
    new_obj: Dict[str, Any] = {}
    # 1) Traducir todos los valores string
    for k, v in obj.items():
        new_obj[k] = translate_fn(v) if isinstance(v, str) else v

    # 2) Clasificar usando los campos traducidos
    tags = classify_fn(
        new_obj.get("problem", ""),
        new_obj.get("generated_solution", "")
    )
    new_obj.update(tags)

    # 3) Resumir la solución traducida
    new_obj["generated_secondary_answer"] = summarize_fn(
        new_obj.get("generated_solution", "")
    )
    return new_obj



def process_file(input_path: str, output_path: str,
                 translate_fn, classify_fn, summarize_fn,
                 skip_lines: int = 0):
    total = sum(1 for _ in open(input_path, encoding="utf-8"))
    with open(input_path, encoding="utf-8") as fin, \
         open(output_path, "a", encoding="utf-8") as fout:  # append
        for idx, line in enumerate(tqdm(fin, total=total, desc="Procesando")):
            if idx < skip_lines or not line.strip():
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                print("⚠️ Línea ignorada: JSON inválido.")
                continue
            if not isinstance(obj, dict):
                continue
            out_obj = translate_classify_and_summarize(
                obj, translate_fn, classify_fn, summarize_fn
            )
            fout.write(json.dumps(out_obj, ensure_ascii=False) + "\n")


