"""
llms.py – Factory minimal de modelos Ollama (solo Gemma3 por defecto).
"""
from typing import Optional
from langchain_ollama import OllamaLLM


def build_llm(model_name: str = "gemma3:4b",
              temperature: float = 0.7,
              timeout: int = 600) -> OllamaLLM:
    """
    Devuelve un objeto LangChain LLM apuntando al modelo local en Ollama.
    """
    return OllamaLLM(model=model_name,
                     temperature=temperature,
                     timeout=timeout)  # puerto por defecto
