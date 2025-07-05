# rubrics.py
"""
Tablas maestras (tomadas y condensadas de la Sección 3 del paper)
para codificar la retro-alimentación pedagógica.
"""

# ------------- 1) EVAL OF STUDENT RESPONSE (a–g) -------------------------
EVAL_CODES = {
    "a": "Respuesta incorrecta",
    "b": "Respuesta correcta",
    "c": "Parcialmente correcta",
    "d": "Ambigua / muy breve",
    "e": "Fuera de tema",
    "f": "Pregunta del estudiante",
    "g": "Sin evaluación (N/A)"
}

# ------------- 2) ACTION BASED ON EVAL (1–12) ---------------------------
ACTION_CODES = {
    "1": "Señalar error y dar pista",
    "2": "Revelar solución tras varios intentos",
    "3": "Reforzar respuesta correcta",
    "4": "Reconocer acierto parcial y guiar el resto",
    "5": "Dividir el problema en pasos más pequeños",
    "6": "Plantear pregunta guía",
    "7": "Proporcionar solución parcial",
    "8": "Motivar y animar a continuar",
    "9": "Resumir progreso hasta el momento",
    "10": "Explicar solución completa",
    "11": "Verificar comprensión del alumno",
    "12": "Cerrar subproblema / pasar al siguiente"
}

# ------------- 3) SUBPROBLEM STATE (w–z) --------------------------------
STATE_CODES = {
    "w": "Nuevo subproblema",
    "x": "Subproblema en curso",
    "y": "Subproblema resuelto",
    "z": "Problema principal resuelto"
}
