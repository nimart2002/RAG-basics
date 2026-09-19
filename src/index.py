"""Indexación persistente en ChromaDB.

Referencia: referencia/01_crear_base_vectorial_chromadb.py
"""

import chromadb
from langchain_core.documents import Document

from config import CHROMA_DIR, COLLECTION_NAME


def sanitizar_metadata(metadata: dict) -> dict:
    """Chroma solo acepta metadatos simples: str, int, float o bool."""
    limpia = {}
    for clave, valor in metadata.items():
        if valor is None:
            continue
        if isinstance(valor, (str, int, float, bool)):
            limpia[clave] = valor
        else:
            limpia[clave] = str(valor)
    return limpia


def construir_indice(chunks: list[Document], vectores: list[list[float]],
                      recreate: bool = True) -> None:
    """Crea/recrea la colección de Chroma e indexa ids/embeddings/documents/metadatas."""
    CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    cliente = chromadb.PersistentClient(path=str(CHROMA_DIR))

    if recreate:
        try:
            cliente.delete_collection(COLLECTION_NAME)
        except Exception:
            pass  # no había colección previa, no pasa nada

    coleccion = cliente.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )

    ids, embeddings, documents, metadatas = [], [], [], []
    for i, (chunk, vector) in enumerate(zip(chunks, vectores)):
        meta = sanitizar_metadata(chunk.metadata)
        ids.append(f"chunk_{meta.get('chunk_index', i)}")
        embeddings.append(vector)
        documents.append(chunk.page_content)
        metadatas.append(meta)

    coleccion.add(ids=ids, embeddings=embeddings, documents=documents, metadatas=metadatas)


def obtener_coleccion():
    """Abre la colección persistida para poder hacer retrieval (retrieve.py).

    Lanza una excepción si todavía no existe (no se ha ejecutado --index).
    """
    cliente = chromadb.PersistentClient(path=str(CHROMA_DIR))
    return cliente.get_collection(COLLECTION_NAME)