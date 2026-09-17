"""Retrieval en memoria: pregunta -> top-K chunks + contexto.

Versión SIN base de datos vectorial: el "índice" es una lista de dicts
en RAM. Cuando se añada ChromaDB, esta misma interfaz (recuperar,
formatear_contexto, fuentes_desde_chunks) se mantiene igual; solo cambia
cómo se guarda y busca el índice por dentro.
"""

import math
from pathlib import Path

from langchain_core.documents import Document

from config import TOP_K
from src.embed import embeddear_consulta, embeddear_textos


def similitud_coseno(v1: list[float], v2: list[float]) -> float:
    """Coseno del ángulo entre dos vectores: 1 = idénticos en dirección."""
    producto_punto = sum(a * b for a, b in zip(v1, v2))
    norma1 = math.sqrt(sum(a * a for a in v1))
    norma2 = math.sqrt(sum(b * b for b in v2))
    if norma1 == 0 or norma2 == 0:
        return 0.0
    return producto_punto / (norma1 * norma2)


def construir_indice(chunks: list[Document]) -> list[dict]:
    """Embeddea los chunks y los guarda en una lista en memoria.

    Cada entrada: {id, text, metadata, vector}. Esto sustituye a
    src/index.py (ChromaDB) mientras trabajemos sin base de datos.
    """
    textos = [c.page_content for c in chunks]
    vectores = embeddear_textos(textos)

    indice = []
    for i, (chunk, vector) in enumerate(zip(chunks, vectores)):
        indice.append({
            "id": f"chunk_{i}",
            "text": chunk.page_content,
            "metadata": dict(chunk.metadata),
            "vector": vector,
        })
    return indice


def recuperar(pregunta: str, indice: list[dict], top_k: int = TOP_K) -> list[dict]:
    """Embeddea la pregunta y devuelve los top_k chunks más similares.

    Cada resultado incluye 'distance' (menor = más cercano), igual
    convención que usará ChromaDB más adelante.
    """
    vector_pregunta = embeddear_consulta(pregunta)

    candidatos = []
    for item in indice:
        similitud = similitud_coseno(vector_pregunta, item["vector"])
        candidatos.append({
            "id": item["id"],
            "text": item["text"],
            "metadata": item["metadata"],
            "distance": 1 - similitud,
        })

    candidatos.sort(key=lambda c: c["distance"])  # menor distancia primero
    return candidatos[:top_k]


def formatear_contexto(chunks: list[dict]) -> str:
    """Chunks -> bloque de texto legible con delimitadores y fuente citada.
    Es lo que va a recibir el LLM en el prompt.

    Cada fragmento es asociado con su distancia a la pregunta original del usuario.
    """
    if not chunks:
        return "(sin resultados)"

    partes = []
    for i, c in enumerate(chunks, start=1):
        fuente = Path(str(c["metadata"].get("source", "?"))).name
        partes.append(
            f"--- Fragmento {i} (dist={c['distance']:.4f}) ---\n"
            f"Fuente: {fuente}\n"
            f"{c['text']}"
        )
    return "\n\n".join(partes)


def fuentes_desde_chunks(chunks: list[dict]) -> list[str]:
    """Nombres de archivo únicos a partir de la metadata recuperada.
    Construye una lista corta de nombres de archivo. Será usada en la construcción del Streamlit.

    """
    fuentes = []
    for c in chunks:
        nombre = Path(str(c["metadata"].get("source", "?"))).name
        if nombre not in fuentes:
            fuentes.append(nombre)
    return fuentes