#!/usr/bin/env python3
"""
build_dp.py – contexto = resumen del estado actual del problema.
"""

import json, argparse, collections, textwrap
from pathlib import Path
from tqdm import tqdm

from llms import build_llm
from utils.context import build_context        # para concatenar sin numerar


# --------------------------- funciones auxiliares ------------------------
def summarized_context(llm, history):
    """
    history: list[{"student": str, "tutor": str}]
    Devuelve un resumen en ≤2 frases del progreso del ejercicio.
    """
    if not history:
        return ""   # primer turno -> contexto vacío

    raw_dialogue = build_context(history)  # "Alumno:...\nTutor:...\n..."
    prompt = textwrap.dedent(f"""
        Resume en no más de dos frases, en español,
        el progreso del alumno y lo que ha explicado el tutor hasta ahora.
        No agregues pasos nuevos ni reveles la solución final.

        DIÁLOGO:
        {raw_dialogue}

        Resumen:
    """)
    return llm.invoke(prompt).strip()


def generate_response(llm, role_prompt, ctx):
    prompt = role_prompt.format(contexto=ctx)
    return llm.invoke(prompt).strip()

# --------------------------- prompts fijos -------------------------------
PROMPT_CHOSEN = textwrap.dedent("""
    Eres un tutor de matemáticas paciente.
    Basándote en el RESUMEN a continuación, responde
    resaltando lo que el alumno ha hecho bien y haz UNA pregunta
    que lo ayude a avanzar. Finaliza tu mensaje con un signo de interrogación.

    RESUMEN:
    {contexto}

    Tu respuesta:
""")

PROMPT_REJECTED = textwrap.dedent("""
    Eres un tutor que entrega directamente el resultado del paso actual
    sin fomentar el diálogo. Basándote en el RESUMEN a continuación,
    explica brevemente el cálculo o valor requerido y no hagas preguntas.

    RESUMEN:
    {contexto}

    Tu respuesta:
""")

# -------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dc", required=True)
    ap.add_argument("--dp", required=True)
    ap.add_argument("--model", default="gemma3:4b")
    args = ap.parse_args()

    llm = build_llm(args.model, temperature=0.3, timeout=600)

    # 1) Agrupar Dc por conversación
    convs = collections.defaultdict(list)
    with open(args.dc, encoding="utf-8") as fdc:
        for line in fdc:
            row = json.loads(line)
            convs[row["conversation_id"]].append(row)

    # 2) Procesar
    out_path = Path(args.dp)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with open(out_path, "w", encoding="utf-8") as fdp:
        for cid, turns in tqdm(convs.items(), desc="Generando Dp"):
            turns.sort(key=lambda r: r["turn_idx"])
            history = []

            for turn in turns:
                # ---- construir resumen del contexto -----------------------
                ctx_summary = summarized_context(llm, history)

                # ---- generar chosen & rejected ----------------------------
                chosen   = generate_response(llm, PROMPT_CHOSEN, ctx_summary)
                rejected = generate_response(llm, PROMPT_REJECTED, ctx_summary)

                # ---- registro ---------------------------------------------
                rec = {
                    "context": ctx_summary,
                    "chosen":  chosen,
                    "rejected": rejected,
                    "preference": True
                }
                fdp.write(json.dumps(rec, ensure_ascii=False) + "\n")

                # ---- actualizar historial con la respuesta chosen ---------
                history.append({"student": turn["student"], "tutor": chosen})


if __name__ == "__main__":
    main()
