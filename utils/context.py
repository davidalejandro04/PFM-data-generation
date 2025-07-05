"""
Construye el contexto C_t (concat de turnos previos).
Por simplicidad solo retorna el turno cero, pero está preparada
para diálogos multi-turn si amplías generate_dc.py.
"""
from typing import List, Dict


def build_context(turns: List[Dict], trim_last_k: int = None) -> str:
    """
    turns: lista de dict con claves 'student' y 'tutor'
    trim_last_k: recorta a los k últimos turnos si no es None
    """
    if trim_last_k is not None:
        turns = turns[-trim_last_k:]
    parts = []
    for tr in turns:
        parts.append(f"Alumno: {tr['student']}")
        parts.append(f"Tutor: {tr['tutor']}")
    return "\n".join(parts)
