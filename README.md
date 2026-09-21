# Project Break RAG — Asistente de bibliografía TFM

Asistente RAG que responde preguntas sobre la bibliografía de un Trabajo de
Fin de Máster sobre **categorización basada en esquemas, IA y física
teórica** (memorias auto-asociativas, modelos generativos, paisajes de
energía aplicados a cognición).

## Tema y corpus

- **Temática:** RAG sobre bibliografía académica de investigación (IA,
  neurociencia computacional, física teórica), pensado como herramienta de
  apoyo real para el TFM del autor.
- **Fuentes:**
  - Kingma, D. P. y Welling, M. (2014). *Auto-Encoding Variational Bayes*.
    ICLR. (`data/Kingma(2022).pdf`)
  - Parga, N., Serrano-Fernández, L. y Falco-Roget, J. (2023). *Emergent
    computations in trained artificial neural networks and real brains*.
    Journal of Instrumentation 18, C02060. (`data/Parga(2022).pdf`)
  - Spens, E. y Burgess, N. (2024). *A generative model of memory
    construction and consolidation*. Nature Human Behaviour 8, 526-543.
    (`data/Spens(2023).pdf`)
  - Glosario de conceptos propio, elaborado a partir de la terminología del
    TFM (`data/glosario_conceptos.md`)
- **Formatos:** PDF (3 papers académicos) + Markdown (glosario)
- **Preguntas de ejemplo:**
  1. ¿Qué es el ELBO (Evidence Lower BOund) en un VAE?
  2. ¿Cómo se modela la memoria episódica antes de la consolidación según Spens y Burgess?
  3. ¿Qué es un esquema (schema) y en qué se diferencia de una simple categoría de clasificación?
  4. ¿Qué relación establecen Parga et al. entre las computaciones emergentes de redes neuronales entrenadas y el cerebro real?
  5. ¿Cuál es la capital de Francia? *(fuera de corpus — debe abstenerse)*
- Los materiales utilizados son papers públicos/con acceso institucional y de uso educativo.

## Instalación

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env        # rellenar GEMINI_API_KEY
```

## Indexar el corpus (offline)
Reconstruye el índice desde cero en `output/chroma_db/` (gitignored,
regenerable en cualquier momento con este comando). Usa `--recreate-index`
si cambias el modelo de embeddings o los límites de chunking.

```bash
python main.py --index
```

## Preguntar (online)

Ambos aceptan `--top-k N` para cambiar cuántos chunks se recuperan (por
defecto, el valor de `TOP_K` en `config.py`).

```bash
python main.py --query "tu pregunta"   # solo retrieval, muestra el contexto recuperado
python main.py --ask "tu pregunta"     # RAG completo: retrieval + respuesta generada
```

## Interfaz Streamlit

```
streamlit run app.py
```


## Estructura del proyecto

```
.
├── config.py            # hiperparámetros centralizados (chunking, K, modelos, prompt)
├── main.py              # CLI: --index / --query / --ask
├── app.py                # interfaz Streamlit
├── src/
│   ├── load.py            # carga de documentos (data/ -> Document)
│   ├── chunk.py           # troceado en chunks
│   ├── client.py          # cliente Gemini compartido (embeddings + generación)
│   ├── embed.py           # embeddings, con reintento por límite de cuota
│   ├── index.py           # indexación y apertura de la colección ChromaDB
│   ├── retrieve.py        # retrieval top-K + formateo de contexto
│   ├── generate.py        # prompt + generación + responder()/rag_ask()
│   └── logging_utils.py   # logging básico por consulta (output/logs.jsonl)
├── data/                  # corpus: 3 papers PDF + glosario Markdown
├── queries/
│   └── preguntas_eval.json  # 13 preguntas de evaluación (11 in-corpus + 2 fuera)
├── entregables/
│   └── informe_decisiones.md
└── referencia/             # notebooks de clase (Sprints 8-10), material de apoyo
```

## Decisiones clave

- CHUNK_SIZE=800, CHUNK_OVERLAP=100 — ver experimento en entregables/informe_decisiones.md.
- TOP_K=3, UMBRAL_DISTANCIA=0.6 — filtro determinista de relevancia antes de generar.
- Embeddings: gemini-embedding-2. Generación: gemini-3.1-flash-lite.
- Proveedor: Google Gemini (API de Google AI Studio).