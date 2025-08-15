"""
translate_lib.py – Envuelve tu lógica original de traducción + clasificación.
Mantiene intacta tu función _init_ollama pero la expone limpiamente.
"""
from typing import Callable, Tuple

# Copia aquí tu implementación completa de _init_ollama
# (recortada para brevedad; pega la versión final que ya usamos).
from util_translate_summarize import _init_ollama   # ← Pega o mueve el código real


def get_translators(model: str = "gemma3:4b") -> Tuple[
        Callable[[str], str],          # translate
        Callable[[str, str], dict],    # classify
        Callable[[str], str]]:         # summarize
    """
    Devuelve tres callables: translate, classify, summarize
    """
    return _init_ollama(model)
