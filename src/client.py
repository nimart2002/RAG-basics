"""Cliente compartido de la API de Gemini.

Un único cliente para todo el proyecto (embeddings y generación lo
comparten), en vez de que cada módulo cree el suyo por separado.
"""

from google import genai

from config import GEMINI_API_KEY

_client: genai.Client | None = None


def get_client() -> genai.Client:
    """Cliente Gemini perezoso: se crea una sola vez en todo el proceso."""
    global _client
    if _client is None:
        if not GEMINI_API_KEY:
            raise RuntimeError(
                "Falta GEMINI_API_KEY. Copia .env.example a .env y rellénala."
            )
        _client = genai.Client()
    return _client