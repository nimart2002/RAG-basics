"""Embeddings con Gemini (Parte 2 / Día 3).

Referencia: referencia/01_embeddings.py

Regla de oro: usar el MISMO modelo (config.EMBEDDING_MODEL) para indexar
los chunks y para embeddear la pregunta del usuario en retrieve.py.
"""

from google.genai import types
import time

from config import EMBEDDING_MODEL, BATCH_SIZE
from src.client import get_client


def embeddear_textos(textos: list[str]) -> list[list[float]]:
    if not textos:
        return []

    client = get_client()
    vectores = []

    for inicio in range(0, len(textos), BATCH_SIZE):
        lote = textos[inicio:inicio + BATCH_SIZE]
        contents = [types.Content(parts=[types.Part(text=t)]) for t in lote]
        result = client.models.embed_content(model=EMBEDDING_MODEL, contents=contents)
        vectores.extend(list(e.values) for e in result.embeddings)
        time.sleep(60)  # margen entre lotes para no chocar con el límite por minuto

    return vectores

def embeddear_consulta(pregunta: str) -> list[float]:
    """Embeddea una única pregunta del usuario (mismo modelo que al indexar)."""
    return embeddear_textos([pregunta])[0]