"""Configuración centralizada del sistema RAG.

Todos los módulos importan sus parámetros de aquí en lugar de tener
valores "mágicos" repartidos por el código. Cambiar un hiperparámetro
(chunking, K, modelo, prompt...) se hace en un único sitio.
"""

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# --- Claves de API ---
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# --- Rutas ---
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
OUTPUT_DIR = BASE_DIR / "output"
CHROMA_DIR = OUTPUT_DIR / "chroma_db"
QUERIES_DIR = BASE_DIR / "queries"

# --- Chunking ---
CHUNK_SIZE = 800
CHUNK_OVERLAP = 100

# --- Embeddings / índice ---
EMBEDDING_MODEL = "gemini-embedding-2"
COLLECTION_NAME = "corpus_rag"  # TODO: renombrar según la temática elegida
MAX_CHUNKS = None  # límite de chunks a indexar; None = corpus completo

# --- Retrieval ---
TOP_K = 3
UMBRAL_DISTANCIA = 0.6  # TODO: ajustar empíricamente con el corpus real

# --- Generación ---
GENERATION_MODEL = "gemini-3.1-flash-lite"
GENERATION_TEMPERATURE = 0.2

# --- Prompt RAG ---
INSTRUCCIONES_RAG = """Eres un asistente que ayuda a analizar oportunidades \
de mercado sobre universidades latinoamericanas.

Reglas:
- Responde ÚNICAMENTE con la información del contexto proporcionado.
- Si el contexto no contiene información suficiente, indícalo explícitamente
  en vez de inventar.
- Cuando cites un hecho, menciona la fuente si aparece en el contexto.
- No inventes universidades, cifras ni datos que no estén en el contexto.
"""