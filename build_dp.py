#!/usr/bin/env python3
"""
Construye el Preference Dataset Dp a partir de Dc.
   Uso:
     python build_dp.py --dc data/dc.jsonl --dp data/dp.jsonl
"""
import json, argparse
from pathlib import Path
from tqdm import tqdm
from translate_lib import get_translators
from utils.context import build_context

def has_divergence(at: dict, as_: dict) -> bool:
    """
    Detecta divergencia pedagógica (Eval, Action o Subproblem State distintos)
    """
    keys = ["Eval of Student Response",
            "Action Based on Eval",
            "Subproblem State"]
    return any(at.get(k) != as_.get(k) for k in keys)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dc", required=True)
    ap.add_argument("--dp", required=True)
    ap.add_argument("--translator_model", default="gemma3:4b")
    args = ap.parse_args()

    translate, classify, summarize = get_translators(args.translator_model)

    out_path = Path(args.dp)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with open(args.dc, encoding="utf-8") as fdc, \
         open(out_path, "w", encoding="utf-8") as fdp:

        # Como solo tenemos turn_idx 0 no necesitamos agrupar,
        # pero dejamos la lógica lista para diálogos futuros.
        for line in tqdm(fdc, desc="Construyendo Dp"):
            turn = json.loads(line)
            at, as_ = turn["tutor_AT"], turn["tutor_AS"]

            if not has_divergence(at, as_):
                continue  # no hay preferencia clara

            context_txt = build_context(
                [{"student": turn["student"],
                  "tutor": at["Tutorbot"]}], trim_last_k=None)

            # Clasificación pedagógica con la respuesta elegida
            tags = classify(turn["problem"] if "problem" in turn else "",
                            at["Tutorbot"])

            record = {
                "context": translate(context_txt),   # por si futuro multi-inglés
                "chosen":  at["Tutorbot"],
                "rejected": as_["Tutorbot"],
                # copiamos los tres campos informativos (en español)
                "Eval of Student Response": at["Eval of Student Response"],
                "Action Based on Eval": at["Action Based on Eval"],
                "Subproblem State": at["Subproblem State"],
                # etiquetas pedagógicas
                **tags,
                # resumen del chosen
                "generated_solution": at["Tutorbot"],
                "generated_secondary_answer": summarize(at["Tutorbot"])
            }
            fdp.write(json.dumps(record, ensure_ascii=False) + "\n")

if __name__ == "__main__":
    main()
