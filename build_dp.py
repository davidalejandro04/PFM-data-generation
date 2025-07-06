#!/usr/bin/env python3
"""
build_dp.py
Genera un Preference Dataset Dp a partir de Dc **creando** nuevas respuestas
(chosen / rejected) con un único modelo local (gemma3) conforme a la rúbrica.

Uso:
  python build_dp.py --dc data/dc.jsonl --dp data/dp.jsonl
Opcional:
  --model gemma3:4b
"""

import json, argparse, collections, textwrap
from pathlib import Path
from tqdm import tqdm

from llms import build_llm
from utils.context import build_context   # re-usa para limpiar saltos

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

# ---------- función segura para formatear + llamar al modelo -------------
def safe_generate(llm, role_prompt: str, ctx: str) -> str | None:
    """
    Devuelve la respuesta del modelo o None si algo falla.
    • Solo se aplica .format() si la cadena contiene el marcador '{contexto}'.
    • Si se lanza cualquier excepción se captura y se retorna None.
    """
    try:
        if "{contexto}" in role_prompt:
            prompt = role_prompt.format(contexto=ctx)
        else:
            prompt = role_prompt
        return llm.invoke(prompt).strip()
    except Exception as e:
        print(f"⚠️  Se omitió un turno por error: {e}")
        return None

def load_processed_conversations(dp_path: Path) -> set[str]:
    """
    Devuelve un set de conversation_id ya presentes en el DP.
    Si el archivo no existe aún, devuelve set vacío.
    """
    processed = set()
    if dp_path.exists():
        with open(dp_path, encoding="utf-8") as f:
            for line in f:
                try:
                    processed.add(json.loads(line)["conversation_id"])
                except Exception:
                    pass
    return processed
# ---------------------------------------------------------------- helper
def numbered_context(history):
    """
    history: list[{"student": str, "tutor": str}]
    Devuelve string numerado:
       1. Alumno: ...
          Tutor : ...
    """
    lines = []
    for i, pair in enumerate(history, 1):
        a = pair["student"].replace("\n", " ")
        t = pair["tutor"].replace("\n", " ")
        lines.append(f"{i}. Alumno: {a}\n   Tutor : {t}")
    return "\n".join(lines)

# --------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dc", required=True)
    ap.add_argument("--dp", required=True)
    ap.add_argument("--model", default="gemma3:4b",
                    help="Modelo Ollama para generar chosen / rejected")
    args = ap.parse_args()

    llm = build_llm(args.model, temperature=0.3, timeout=600)

    # 1) agrupar Dc
    convs = collections.defaultdict(list)
    with open(args.dc, encoding="utf-8") as fdc:
        for line in fdc:
            row = json.loads(line)
            convs[row["conversation_id"]].append(row)

    out_path = Path(args.dp)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    processed_cids = load_processed_conversations(out_path)

    # 2) procesar conversación
    with open(out_path, "a", encoding="utf-8") as fdp:   # append mode
        for cid, turns in tqdm(convs.items(), desc="Generando Dp"):
            if cid in processed_cids:
                continue  # 🔹  ya se procesó esta conversación            turns.sort(key=lambda r: r["turn_idx"])
            turns.sort(key=lambda r: r["turn_idx"])
            history = []


            for turn in turns:


                # ---- construir contexto numerado incluyendo alumno actual
                ctx = numbered_context(
                    history + [{"student": turn["student"], "tutor": ""}]
                )

                # ---- generar CHOSEN (andamiaje)
                prompt_chosen = textwrap.dedent(f"""
                Eres un tutor de matemáticas paciente. Basándote en el
                diálogo a continuación, responde refiriéndote al razonamiento
                del alumno, resaltando lo que haya hecho bien y formulando
                UNA pregunta que lo ayude a avanzar al siguiente paso. 
                En caso de que el estudiante haya cometdo un error, señala el error
                y ofrece una pista para corregirlo.
                
                DIÁLOGO:
                {ctx}
                
                Tu respuesta:
                """)

                chosen = safe_generate(llm, prompt_chosen, ctx)

                # ---- generar REJECTED (respuesta directa)
                PROMPT_REJECTED = textwrap.dedent(f"""
                Eres un tutor que proporciona la respuesta del paso actual
                sin fomentar diálogo. Lee el contexto y ofrece el resultado
                directamente, sin preguntas ni elogios adicionales.
                
                DIÁLOGO:
                {ctx}
                
                Tu respuesta:
                """)
                rejected = safe_generate(llm, PROMPT_REJECTED, ctx)

                if chosen is None or rejected is None:
                    continue  # pasa al siguiente turno sin detener el programa
                # ---- registro -------------------------------------------------

                if turn==0:
                    # primer turno, no hay contexto previo
                    ctx_out = turn["student"]
                else:
                    ctx_out = summarized_context(llm, history)

                if ctx_out=="":
                    ctx_out = turn["student"]

                rec = {
                    "conversation_id": cid,        #  ← NUEVO
                    "context": ctx_out,
                    "chosen":  chosen,
                    "rejected": rejected,
                    "preference": True
                }
                fdp.write(json.dumps(rec, ensure_ascii=False) + "\n")

                
                # ---- actualizar historial (con chosen) -----------------------
                history.append({"student": turn["student"], "tutor": chosen})


if __name__ == "__main__":
    main()
