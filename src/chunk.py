"""Chunking del corpus ya cargado y limpio 
Referencia: referencia/01_carga_documentos_y_chunking.py
"""

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from config import CHUNK_OVERLAP, CHUNK_SIZE
from src.load import normalizar_texto


def limpiar_documentos(documentos: list[Document]) -> list[Document]:
    """Aplica normalizar_texto() y descarta documentos vacíos tras limpiar."""
    limpios = []
    for doc in documentos:
        texto = normalizar_texto(doc.page_content)
        if not texto:
            continue
        limpios.append(Document(page_content=texto, metadata=dict(doc.metadata)))
    return limpios


def trocear(documentos: list[Document], chunk_size: int = CHUNK_SIZE,
            chunk_overlap: int = CHUNK_OVERLAP) -> list[Document]:
    """Limpia y trocea los documentos; numera cada chunk (chunk_index)."""
    limpios = limpiar_documentos(documentos)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    chunks = splitter.split_documents(limpios)

    #Añadimos el índice del chunk en la metadata:
    
    for i, chunk in enumerate(chunks):
        chunk.metadata["chunk_index"] = i

    return chunks