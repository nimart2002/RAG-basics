"""Logging básico de cada consulta.

Registra pregunta, k, nº de chunks, tiempo y modelo.
"""

import json
from datetime import datetime, timezone

from config import OUTPUT_DIR

LOG_FILE = OUTPUT_DIR / "logs.jsonl"


def log_query(pregunta: str, top_k: int, chunks: int, tiempo_s: float,
              modelo: str, abstuvo: bool = False) -> None:
    """Registra una consulta: una línea legible en consola + una línea JSON en disco."""
    registro = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "pregunta": pregunta,
        "top_k": top_k,
        "chunks": chunks,
        "tiempo_s": round(tiempo_s, 3),
        "modelo": modelo,
        "abstuvo": abstuvo,
    }

    print(f"[LOG] k={top_k} chunks={chunks} tiempo={tiempo_s:.2f}s "
          f"modelo={modelo} abstuvo={abstuvo} | {pregunta[:60]}")

    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(registro, ensure_ascii=False) + "\n")