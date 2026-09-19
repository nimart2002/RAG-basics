"""Retrieval: pregunta -> top-K chunks + contexto, vía ChromaDB.

Referencia:
- referencia/01_implementar_retriever.py
- referencia/01_evaluar_y_ajustar_retrieval.py
"""

from pathlib import Path

from config import TOP_K, UMBRAL_DISTANCIA
from src.embed import embeddear_consulta


def recuperar(pregunta: str, coleccion, top_k: int = TOP_K) -> list[dict]:
    """Embeddea la pregunta, consulta la colección de Chroma y devuelve top_k chunks."""
    vector = embeddear_consulta(pregunta)
    n = min(top_k, coleccion.count())
    if n == 0:
        return []

    resultados = coleccion.query(
        query_embeddings=[vector],
        n_results=n,
        include=["documents", "metadatas", "distances"],
    )

    chunks = []
    for i, doc_id in enumerate(resultados["ids"][0]):
        chunks.append({
            "id": doc_id,
            "text": resultados["documents"][0][i],
            "metadata": resultados["metadatas"][0][i],
            "distance": resultados["distances"][0][i],
        })
    return chunks


def filtrar_por_umbral(chunks: list[dict], umbral: float = UMBRAL_DISTANCIA) -> list[dict]:
    """Descarta chunks cuya distancia supere el umbral (poco relevantes)."""
    return [c for c in chunks if c["distance"] <= umbral]


def formatear_contexto(chunks: list[dict]) -> str:
    """Chunks -> bloque de texto legible con delimitadores y fuente citada."""
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
    """Nombres de archivo únicos a partir de la metadata recuperada."""
    fuentes = []
    for c in chunks:
        nombre = Path(str(c["metadata"].get("source", "?"))).name
        if nombre not in fuentes:
            fuentes.append(nombre)
    return fuentes