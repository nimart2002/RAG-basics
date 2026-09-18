"""CLI del sistema RAG.

Uso:
  python main.py --index                    # offline: load -> chunk -> embed -> ChromaDB
  python main.py --index --recreate-index    # ídem, forzando borrar la colección anterior
  python main.py --query "pregunta"          # online: solo retrieval + contexto
  python main.py --ask "pregunta"            # online: RAG completo (retrieval + generación)

App Streamlit: streamlit run app.py
"""

import argparse
import sys

from config import DATA_DIR, TOP_K
from src.chunk import trocear
from src.embed import embeddear_textos
from src.generate import responder
from src.index import construir_indice, obtener_coleccion
from src.load import cargar_corpus
from src.retrieve import formatear_contexto, recuperar


def cmd_index(recreate: bool) -> None:
    """Pipeline offline completo: load -> chunk -> embed -> ChromaDB."""
    documentos = cargar_corpus(DATA_DIR)
    chunks = trocear(documentos)
    textos = [c.page_content for c in chunks]
    vectores = embeddear_textos(textos)
    construir_indice(chunks, vectores, recreate=recreate)
    print(f"Índice construido en ChromaDB: {len(chunks)} chunks (desde {len(documentos)} documentos).")


def _obtener_coleccion_o_salir():
    """Abre la colección de Chroma, o corta la ejecución con un mensaje claro."""
    try:
        return obtener_coleccion()
    except Exception:
        print("No hay índice construido. Ejecuta primero: python main.py --index")
        sys.exit(1)


def cmd_query(pregunta: str, top_k: int) -> None:
    """Solo retrieval: imprime los chunks recuperados y sus fuentes."""
    coleccion = _obtener_coleccion_o_salir()
    chunks = recuperar(pregunta, coleccion, top_k=top_k)
    print(formatear_contexto(chunks))


def cmd_ask(pregunta: str, top_k: int) -> None:
    """RAG completo: imprime respuesta + fuentes (usa src.generate.responder)."""
    coleccion = _obtener_coleccion_o_salir()
    resultado = responder(pregunta, coleccion=coleccion, top_k=top_k)

    if resultado["error"]:
        print("[SIN RESPUESTA]", resultado["error"])
        if resultado["contexto"] and resultado["contexto"] != "(sin resultados)":
            print("\n--- Contexto recuperado (para depurar) ---")
            print(resultado["contexto"])
        return

    print(resultado["respuesta"])
    print("\nFuentes:", ", ".join(resultado["fuentes"]) or "(ninguna)")


def main() -> None:
    parser = argparse.ArgumentParser(description="Sistema RAG")
    parser.add_argument("--index", action="store_true", help="Reconstruye el índice desde data/")
    parser.add_argument("--recreate-index", action="store_true",
                         help="Con --index: borra la colección anterior antes de reindexar")
    parser.add_argument("--query", type=str, help="Solo retrieval: muestra el contexto recuperado")
    parser.add_argument("--ask", type=str, help="RAG completo: pregunta y genera respuesta")
    parser.add_argument("--top-k", type=int, default=TOP_K, help=f"Nº de chunks a recuperar (default {TOP_K})")
    args = parser.parse_args()

    if not any([args.index, args.query, args.ask]):
        parser.print_help()
        return

    if args.index:
        cmd_index(recreate=args.recreate_index)
    if args.query:
        cmd_query(args.query, args.top_k)
    if args.ask:
        cmd_ask(args.ask, args.top_k)


if __name__ == "__main__":
    main()