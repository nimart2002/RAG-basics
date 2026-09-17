"""Embeddings con Gemini (Parte 2 / Día 3).

Referencia: referencia/01_embeddings.py

Regla de oro: usar el MISMO modelo (config.EMBEDDING_MODEL) para indexar
los chunks y para embeddear la pregunta del usuario en retrieve.py.
"""

from google.genai import types

from config import EMBEDDING_MODEL
from src.client import get_client


def embeddear_textos(textos: list[str]) -> list[list[float]]:
    """Un vector por texto (no un único vector agregado)."""
    if not textos:
        return []

    client = get_client()
    contents = [types.Content(parts=[types.Part(text=t)]) for t in textos]
    result = client.models.embed_content(model=EMBEDDING_MODEL, contents=contents)
    return [list(e.values) for e in result.embeddings]


def embeddear_consulta(pregunta: str) -> list[float]:
    """Embeddea una única pregunta del usuario (mismo modelo que al indexar)."""
    return embeddear_textos([pregunta])[0]