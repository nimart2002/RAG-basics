"""Configuración centralizada del sistema RAG.

Todos los módulos importan sus parámetros de aquí en lugar de tener
valores repartidos por el código.
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
COLLECTION_NAME = "tfm_categorizacion_esquemas" 
MAX_CHUNKS = None  # límite de chunks a indexar; None = corpus completo

# --- Retrieval ---
TOP_K = 3
UMBRAL_DISTANCIA = 0.6  

# --- Generación ---
GENERATION_MODEL = "gemini-3.1-flash-lite"
GENERATION_TEMPERATURE = 0.2
BATCH_SIZE = 100

# --- Prompt RAG ---
INSTRUCCIONES_RAG = """Eres un asistente de investigación que ayuda a consultar \
la bibliografía de un TFM sobre categorización basada en esquemas, IA y física \
teórica (memorias auto-asociativas, modelos generativos, paisajes de energía).

Reglas:
- Responde ÚNICAMENTE con la información del contexto proporcionado.
- Si el contexto no contiene información suficiente, indícalo explícitamente
  en vez de inventar.
- Cuando menciones un resultado o afirmación, indica de qué paper/fuente
  proviene, tal como aparece en el contexto.
- No atribuyas afirmaciones a un autor o paper que no aparezca en el contexto.
"""