"""Interfaz Streamlit — chat + historial, streaming (simulado), sidebar.
"""

import time
from collections.abc import Iterator

import streamlit as st

from config import GENERATION_MODEL, TOP_K
from src.generate import responder
from src.index import obtener_coleccion
from src.logging_utils import log_query


def stream_palabras(texto: str, delay: float = 0.02) -> Iterator[str]:
    """Genera el texto palabra a palabra (efecto 'máquina de escribir').

    No cambia el contenido: la respuesta ya viene completa de responder().
    Esto solo controla el ritmo de aparición en pantalla.
    """
    for palabra in texto.split():
        yield palabra + " "
        time.sleep(delay)


def mensaje_bienvenida(nombre: str) -> dict:
    """Primer mensaje del asistente al abrir (o al limpiar) el chat."""
    return {
        "role": "assistant",
        "content": (
            f"Hola — soy **{nombre}**. Pregúntame sobre la bibliografía del "
            "TFM (categorización basada en esquemas, memorias auto-asociativas, "
            "modelos generativos)."
        ),
        "fuentes": [],
        "contexto": "",
        "error": False,
    }


def render_mensaje(message: dict) -> None:
    """Pinta un mensaje del historial (usuario o asistente), con fuentes/contexto."""
    with st.chat_message(message["role"]):
        if message.get("error"):
            st.error(message["content"])
        else:
            st.markdown(message["content"])

        if message.get("fuentes"):
            st.markdown("**Fuentes**")
            for fuente in message["fuentes"]:
                st.write(f"- `{fuente}`")

        if message.get("contexto"):
            with st.expander("Contexto recuperado (debug)"):
                st.text(message["contexto"])


@st.cache_resource
def cargar_coleccion():
    """Abre la colección de Chroma una sola vez por sesión, no en cada rerun."""
    return obtener_coleccion()


st.set_page_config(
    page_title="TFM Categorización y Esquemas — RAG",
    page_icon="🧠",
    layout="centered",
)

with st.sidebar:
    st.header("Configuración")
    nombre_bot = st.text_input("Nombre del bot", value="Asistente TFM")
    top_k = st.slider("Top-K (chunks)", min_value=1, max_value=10, value=TOP_K)
    st.caption("Índice: `output/chroma_db/` (ejecuta `python main.py --index`).")
    if st.button("Limpiar chat", use_container_width=True):
        st.session_state.messages = [mensaje_bienvenida(nombre_bot)]
        st.rerun()

st.title(nombre_bot)
st.caption("RAG sobre la bibliografía del TFM · Demo Web")

try:
    coleccion = cargar_coleccion()
except Exception:
    st.error("No hay índice construido. Ejecuta primero: `python main.py --index`")
    st.stop()

if "messages" not in st.session_state:
    st.session_state.messages = [mensaje_bienvenida(nombre_bot)]

for message in st.session_state.messages:
    render_mensaje(message)

if prompt := st.chat_input("Tu pregunta sobre la bibliografía..."):
    st.session_state.messages.append(
        {"role": "user", "content": prompt, "fuentes": [], "contexto": "", "error": False}
    )
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.status("Consultando el corpus...", expanded=False) as status:
            inicio = time.perf_counter()
            resultado = responder(prompt, coleccion=coleccion, top_k=top_k)
            tiempo_s = time.perf_counter() - inicio
            status.update(label="Listo", state="complete")

        log_query(prompt, top_k, len(resultado["chunks"]), tiempo_s, GENERATION_MODEL,
                  abstuvo=resultado["error"] is not None)

        if resultado["error"]:
            st.error(resultado["error"])
            st.session_state.messages.append({
                "role": "assistant",
                "content": resultado["error"],
                "fuentes": resultado["fuentes"],
                "contexto": resultado["contexto"],
                "error": True,
            })
        else:
            escrito = st.write_stream(stream_palabras(resultado["respuesta"]))
            contenido = escrito if isinstance(escrito, str) else resultado["respuesta"]

            if resultado["fuentes"]:
                st.markdown("**Fuentes**")
                for fuente in resultado["fuentes"]:
                    st.write(f"- `{fuente}`")

            if resultado["contexto"]:
                with st.expander("Contexto recuperado (debug)"):
                    st.text(resultado["contexto"])

            st.session_state.messages.append({
                "role": "assistant",
                "content": contenido,
                "fuentes": resultado["fuentes"],
                "contexto": resultado["contexto"],
                "error": False,
            })