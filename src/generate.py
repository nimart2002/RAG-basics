"""Generación anclada al contexto + API interna reutilizable.

Contrato clave del proyecto (lo reutilizará el Project Break de Agentes):
    responder(pregunta) -> dict   # respuesta, contexto, fuentes, error
    rag_ask(consulta) -> str      # envoltorio simple sobre responder()
"""

from config import (
    GENERATION_MODEL,
    GENERATION_TEMPERATURE,
    INSTRUCCIONES_RAG,
    TOP_K,
    UMBRAL_DISTANCIA,
)
from src.client import get_client
from src.index import obtener_coleccion
from src.retrieve import (
    filtrar_por_umbral,
    formatear_contexto,
    fuentes_desde_chunks,
    recuperar,
)


def build_rag_prompt(contexto: str, pregunta: str) -> str:
    """Prompt con secciones delimitadas: instrucciones + CONTEXTO + PREGUNTA."""
    return (
        f"{INSTRUCCIONES_RAG.strip()}\n\n"
        f"--- CONTEXTO RECUPERADO ---\n"
        f"{contexto.strip()}\n\n"
        f"--- PREGUNTA ---\n"
        f"{pregunta.strip()}\n\n"
        f"--- RESPUESTA ---"
    )


def generar_respuesta(prompt: str) -> str:
    """Llama al LLM de generación con el prompt ya construido."""
    client = get_client()
    response = client.models.generate_content(
        model=GENERATION_MODEL,
        contents=prompt,
        config={"temperature": GENERATION_TEMPERATURE},
    )
    return (response.text or "").strip()


def responder(pregunta: str, coleccion=None, top_k: int = TOP_K,
              umbral_distancia: float = UMBRAL_DISTANCIA) -> dict:
    pregunta = (pregunta or "").strip()
    if not pregunta:
        return {"respuesta": "", "contexto": "", "fuentes": [], "n_chunks": 0,
                 "error": "La pregunta no puede estar vacía."}

    coleccion = coleccion if coleccion is not None else obtener_coleccion()
    chunks = recuperar(pregunta, coleccion, top_k=top_k)
    chunks = filtrar_por_umbral(chunks, umbral_distancia)
    contexto = formatear_contexto(chunks)

    if not chunks:
        return {"respuesta": "", "contexto": contexto, "fuentes": [], "n_chunks": 0,  
                 "error": "Sin evidencia suficientemente relevante en el corpus."}

    prompt = build_rag_prompt(contexto, pregunta)
    respuesta = generar_respuesta(prompt)
    fuentes = fuentes_desde_chunks(chunks)

    return {"respuesta": respuesta, "contexto": contexto,
             "fuentes": fuentes, "n_chunks": len(chunks), "error": None} 

             
def rag_ask(consulta: str) -> str:
    """Envoltorio simple sobre responder(), pensado para el proyecto de Agentes."""
    resultado = responder(consulta)
    return resultado["respuesta"] or (resultado["error"] or "")