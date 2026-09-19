"""Carga y limpieza del corpus.

Referencia: referencia/01_carga_documentos_y_chunking.py
"""

import re
from pathlib import Path

import pandas as pd
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_core.documents import Document


def normalizar_texto(texto: str) -> str:
    """Limpieza mínima antes de trocear: saltos de línea y espacios repetidos."""
    if not texto:
        return ""
    t = texto.replace("\r\n", "\n").replace("\r", "\n")
    t = re.sub(r"\n{3,}", "\n\n", t)     # no más de 2 saltos seguidos
    t = re.sub(r"[ \t]+", " ", t)         # espacios/tabs repetidos -> uno
    return "\n".join(linea.strip() for linea in t.split("\n")).strip()


def _leer_csv(ruta: Path) -> pd.DataFrame:
    """Lee un CSV probando encodings y separadores habituales en open data ES."""
    for encoding in ("utf-8", "latin-1"):
        try:
            return pd.read_csv(ruta, sep=None, engine="python", encoding=encoding)
        except UnicodeDecodeError:
            continue
    raise ValueError(f"No se pudo leer {ruta} con utf-8 ni latin-1")


def fila_a_texto(fila: pd.Series) -> str | None:
    """Convierte una fila de CSV en texto legible.

    TODO (cuando tengáis el dataset real): sustituir por una versión que
    elija columnas concretas y las etiquete en lenguaje natural, como en
    referencia/01_carga_documentos_y_chunking.py (fila_a_texto).
    """
    lineas = []
    for columna, valor in fila.items():
        if pd.isna(valor):
            continue
        texto = str(valor).strip()
        if texto:
            lineas.append(f"{columna}: {texto}")
    return "\n".join(lineas) if lineas else None


def _cargar_csv(ruta: Path) -> list[Document]:
    df = _leer_csv(ruta)
    documentos = []
    for idx, fila in df.iterrows():
        texto = fila_a_texto(fila)
        if texto is None:
            continue
        documentos.append(
            Document(
                page_content=texto,
                metadata={"source": str(ruta), "row": int(idx)},
            )
        )
    return documentos


def cargar_corpus(data_dir: Path) -> list[Document]:
    """Lee todos los archivos soportados de data_dir y devuelve Document."""
    documentos: list[Document] = []

    for ruta in sorted(data_dir.iterdir()):
        if not ruta.is_file() or ruta.name.startswith("."):
            continue

        suf = ruta.suffix.lower()
        antes = len(documentos)

        if suf in {".txt", ".md"}:
            documentos.extend(TextLoader(str(ruta), encoding="utf-8").load())
        elif suf == ".pdf":
            documentos.extend(PyPDFLoader(str(ruta)).load())
        elif suf == ".csv":
            documentos.extend(_cargar_csv(ruta))
        else:
            continue  # formato no soportado, lo ignoramos

        print(f"  {ruta.name}: +{len(documentos) - antes} documento(s)")

    return documentos