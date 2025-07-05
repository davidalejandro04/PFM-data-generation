#!/usr/bin/env python3
"""
Genera el Conversational Dataset Dc con tres agentes locales Gemma-3.
   Uso:
     python generate_dc.py --questions data/question_bank.sample.jsonl \
                           --output data/dc.jsonl
"""
import json, uuid, argparse, time
from pathlib import Path
from tqdm import tqdm
from llms import build_llm
from translate_lib import get_translators
from utils.validate import parse_json_or_retry
from utils.context import build_context

# Prompt “Tabla 1” traducido (claves en inglés, contenido en español)
TUTOR_PROMPT_ES = """Eres un tutor de matemáticas experto.
Responde SOLO un objeto JSON con las claves EXACTAS:

Eval of Student Response / Códigos de evaluación (a–g): 
Action Based on Eval / Acciones pedagógicas (1–12): 1 = Señalar error y dar pista; 2 = Revelar solución tras varios intentos; 3 = Reforzar respuesta correcta; 4 = Reconocer acierto parcial y guiar el resto; 5 = Dividir el problema en pasos más pequeños; 6 = Plantear pregunta guía; 7 = Proporcionar solución parcial; 8 = Motivar y animar a continuar; 9 = Resumir progreso hasta el momento; 10 = Explicar solución completa; 11 = Verificar comprensión del alumno; 12 = Cerrar subproblema / pasar al siguiente
Subproblem State / Estados de subproblema (w–z): 

{{
  "Eval of Student Response": "a",
  "Action Based on Eval": "1",
  "Subproblem State": "w",
  "Subproblem": "…",
  "Tutorbot": "…explicación paso a paso en español…"
}}

• ‘Eval of Student Response’ debe ser una letra a–g (a = Respuesta incorrecta; b = Respuesta correcta; c = Parcialmente correcta; d = Ambigua / muy breve; e = Fuera de tema; f = Pregunta del estudiante; g = Sin evaluación (N/A))
• ‘Action Based on Eval’ un número 1–12 (1 = Señalar error y dar pista; 2 = Revelar solución tras varios intentos; 3 = Reforzar respuesta correcta; 4 = Reconocer acierto parcial y guiar el resto; 5 = Dividir el problema en pasos más pequeños; 6 = Plantear pregunta guía; 7 = Proporcionar solución parcial; 8 = Motivar y animar a continuar; 9 = Resumir progreso hasta el momento; 10 = Explicar solución completa; 11 = Verificar comprensión del alumno; 12 = Cerrar subproblema / pasar al siguiente)
• ‘Subproblem State’ una letra (w = Nuevo subproblema; x = Subproblema en curso; y = Subproblema resuelto; z = Problema principal resuelto)
• ‘Subproblem’ describe la sub-tarea actual
• ‘Tutorbot’ es la respuesta completa al alumno
NO añadas comentarios ni texto fuera del JSON.
CONVERSACIÓN HASTA AHORA (si existe):
{contexto}

PREGUNTA / RESPUESTA ACTUAL DEL ALUMNO:
{student}
"""

print("Iniciando con: ", TUTOR_PROMPT_ES)


# --- Prompt para el agente-alumno (AG) en los turnos 1-3 -----------------
STUDENT_FOLLOWUP_PROMPT = """Eres un alumno de primaria que
intenta resolver el ejercicio de matemáticas. Basado en la
explicación previa del tutor, responde de una de estas dos maneras:
1) Explica brevemente qué entendiste e intenta avanzar un paso, o
2) Formula una pregunta corta pidiendo aclaración.

Tu respuesta debe ser breve (máx 30 palabras) y en español.
CONVERSACIÓN HASTA AHORA:
{contexto}
"""



# ------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--questions", required=True)
    ap.add_argument("--output",    required=True)
    ap.add_argument("--model_student", default="gemma3:4b")
    ap.add_argument("--model_tutor_at", default="gemma3:4b")
    ap.add_argument("--model_tutor_as", default="gemma3:4b")
    ap.add_argument("--translator_model", default="gemma3:4b")
    ap.add_argument("--max_turns", type=int, default=10)   # 4 turnos
    args = ap.parse_args()

    # Instancias LLM
    AG = build_llm(args.model_student,   temperature=0.7)
    AT = build_llm(args.model_tutor_at,  temperature=0.3)
    AS = build_llm(args.model_tutor_as,  temperature=0.7)

    # Traductores
    translate, _, _ = get_translators(args.translator_model)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with open(args.questions, encoding="utf-8") as fq, \
         open(out_path, "w", encoding="utf-8") as fout:

        for line in tqdm(fq, desc="Generando Dc"):
            q = json.loads(line)
            original_problem = q["problem"].strip()

            # ↳ traducimos la pregunta al español (por si estuviera en otro idioma)
            problem_es = translate(original_problem)

            conv_id      = str(uuid.uuid4())
            conversation = []          # acumula turnos para el contexto

            for t_idx in range(args.max_turns):

                # -------- ALUMNO (AG) ------------------------------------
                if t_idx == 0:
                    student_utt = problem_es
                else:
                    ctx_txt = build_context(conversation)
                    student_prompt = STUDENT_FOLLOWUP_PROMPT.format(
                        contexto=ctx_txt)
                    student_utt = AG.invoke(student_prompt).strip()
                    student_utt = translate(student_utt)  # garantiza español

                # ------- CONTEXTO PARA EL TUTOR --------------------------
                ctx_txt = build_context(
                    conversation + [{"student": student_utt, "tutor": ""}]
                )

                tutor_prompt_filled = TUTOR_PROMPT_ES.format(
                    contexto=ctx_txt or "—", student=student_utt)

                # --------- RESPUESTAS DE TUTORES -------------------------
                at_json = parse_json_or_retry(AT, tutor_prompt_filled)
                as_json = parse_json_or_retry(AS, tutor_prompt_filled)

                # Si cualquiera falló → avisamos y pasamos al siguiente problema
                if at_json is None or as_json is None:
                    print(f"⚠️  Saltando pregunta (id={conv_id}) por JSON inválido")
                    break   # abandona los 4 turnos de esta pregunta y continúa con la siguiente
                at_json_es = {k: translate(v) for k, v in at_json.items()}
                as_json_es = {k: translate(v) for k, v in as_json.items()}

                # --------- REGISTRO --------------------------------------
                record = {
                    "conversation_id": conv_id,
                    "turn_idx": t_idx,
                    "student": student_utt,
                    "tutor_AT": at_json_es,
                    "tutor_AS": as_json_es
                }
                fout.write(json.dumps(record, ensure_ascii=False) + "\n")

                # --------- Actualiza historial para turnos siguientes ----
                conversation.append({
                    "student": student_utt,
                    "tutor": at_json_es["Tutorbot"]   # elegimos el AT
                })

                time.sleep(0.1)   # no saturar Ollama

if __name__ == "__main__":
    main()