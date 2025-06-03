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
        ("system", "Traduce al español exactamente el texto que se te da. No añadas ni quites nada."),
        ("user", "{input}")
    ])
    translate_chain = translate_prompt | llm

    # Prompt de clasificación (llaves escapadas con doble {{ }} en Python)
    classify_prompt = ChatPromptTemplate.from_messages([
        ("system", AREAS_OBJETIVOS_CONTEXT + "\nResponde SOLO el JSON citado arriba."),
        ("user", "{input}")
    ])
    classify_chain = classify_prompt | llm

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

    return translate, classify

def translate_and_classify(obj: Dict[str, Any],
                           translate_fn,
                           classify_fn) -> Dict[str, Any]:
    # Traduce TODOS los valores string
    new_obj: Dict[str, Any] = {}
    for k, v in obj.items():
        if isinstance(v, str):
            new_obj[k] = translate_fn(v)
        else:
            new_obj[k] = v

    # Clasifica usando los campos traducidos
    tags = classify_fn(
        new_obj.get("problem", ""),
        new_obj.get("generated_solution", "")
    )
    new_obj.update(tags)
    return new_obj

def process_file(input_path: str, output_path: str,
                 translate_fn, classify_fn, skip_lines: int = 0):
    # Count total lines in input for progress bar
    total = sum(1 for _ in open(input_path, encoding="utf-8"))

    with open(input_path, encoding="utf-8") as fin, \
         open(output_path, "a", encoding="utf-8") as fout:  # Append mode
        for idx, line in enumerate(tqdm(fin, total=total, desc="Procesando")):
            if idx < skip_lines:
                continue
            if not line.strip():
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                print("⚠️ Línea ignorada: JSON inválido.")
                continue
            if not isinstance(obj, dict):
                continue

            out_obj = translate_and_classify(obj, translate_fn, classify_fn)
            fout.write(json.dumps(out_obj, ensure_ascii=False) + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="Traduce TODOS los campos de un JSONL y añade 'area_conocimiento' + 'objetivo_pedagogico'.")
    parser.add_argument("input", help="Archivo de entrada (.jsonl)")
    parser.add_argument("output", help="Archivo de salida (.jsonl)")
    parser.add_argument("--model", default="gemma3:4b",
                        help="Modelo Ollama a usar (p.ej. gemma3:4b)")
    parser.add_argument("--skip", type=int, default=0,
                        help="Cantidad de líneas iniciales a omitir del archivo de entrada")
    args = parser.parse_args()


    if args.skip == 0 and os.path.exists(args.output):
        args.skip = sum(1 for _ in open(args.output, encoding="utf-8"))
        print(f"📌 Detección automática: omitiendo las primeras {args.skip} líneas ya presentes en el archivo de salida.")

    translate_fn, classify_fn = _init_ollama(args.model)
    process_file(args.input, args.output, translate_fn, classify_fn, args.skip)
    print(f"\n✅ Traducción y clasificación completadas. Resultados agregados a {args.output}")

if __name__ == "__main__":
    main()
