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

    # 2) procesar conversación
    with open(out_path, "w", encoding="utf-8") as fdp:
        for cid, turns in tqdm(convs.items(), desc="Generando Dp"):
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
                del alumno, elogiando lo que haya hecho bien y formulando
                UNA pregunta que lo ayude a avanzar al siguiente paso.
                
                DIÁLOGO:
                {ctx}
                
                Tu respuesta:
                """)

                chosen = llm.invoke(prompt_chosen).strip()

                # ---- generar REJECTED (respuesta directa)
                prompt_rejected = textwrap.dedent(f"""
                Eres un tutor que proporciona la respuesta del paso actual
                sin fomentar diálogo. Lee el contexto y ofrece el resultado
                directamente, sin preguntas ni elogios adicionales.
                
                DIÁLOGO:
                {ctx}
                
                Tu respuesta:
                """)
                rejected = llm.invoke(prompt_rejected).strip()

                # ---- registro -------------------------------------------------
                rec = {
                    "context": ctx,
                    "chosen":  chosen,
                    "rejected": rejected,
                    "preference": True   # garantizado por diseño
                }
                fdp.write(json.dumps(rec, ensure_ascii=False) + "\n")

                # ---- actualizar historial (con chosen) -----------------------
                history.append({"student": turn["student"], "tutor": chosen})


if __name__ == "__main__":
    main()
