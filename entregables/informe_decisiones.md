# Informe de decisiones — RAG sobre bibliografía de TFM

## 1. Chunking

Configuración usada en el sistema: `CHUNK_SIZE=800`, `CHUNK_OVERLAP=100`.

Experimento comparando dos configuraciones sobre el mismo corpus (3 papers
académicos + glosario de conceptos):

| chunk_size | overlap | Nº de chunks |
|---|---|---|
| 400 | 50 | 696 |
| 800 | 100 | 362 |

Reducir el tamaño a la mitad prácticamente duplica el número de chunks
(696 vs 362), como cabía esperar. Con chunks más pequeños (400) cada
fragmento es más autocontenido pero puede perder contexto de la idea
completa; con chunks más grandes (800) hay menos fragmentos, pero cada uno
puede mezclar varias ideas dentro de un mismo párrafo largo — algo habitual
en papers académicos, donde una explicación técnica se extiende varias
frases. Nos quedamos con 800 por defecto porque el corpus son papers
largos y densos (no fragmentos cortos tipo FAQ), y el retrieval observado
(ver sección 2) recupera fragmentos coherentes y relevantes con este
tamaño.

## 2. Retrieval

Pregunta de prueba: *"¿Cómo se modela la memoria episódica antes de la
consolidación según Spens y Burgess?"*

| K | Fuente(s) | Distancia (rango) | Observación |
|---|---|---|---|
| 1 | Spens(2023).pdf | 0.2348 | Fragmento muy relevante (hipocampo, consolidación, esquemas) |
| 5 | Spens(2023).pdf (los 5) | 0.2348 – 0.2599 | El top-1 es idéntico al de K=1; los 5 fragmentos vienen del mismo paper y cubren subtemas relacionados (consolidación de sistemas, redes generativas, semantización, memoria episódica vs. semántica) |

**Observación clave:** con K=5, el fragmento #1 no cambia (misma distancia,
mismo texto) respecto a K=1 — señal de que el retrieval es estable para
esta pregunta. Además, ampliar K **no introdujo ruido**: los 5 fragmentos
recuperados son relevantes y coherentes entre sí, todos profundizando en
distintos aspectos de la misma idea central. Esto sugiere que, para
preguntas muy específicas sobre un paper concreto y denso en el tema, un
K más alto aporta más *cobertura* (más matices para una respuesta más
completa) sin penalizar la precisión — al contrario de lo que se esperaría
con un corpus más heterogéneo, donde K alto suele arrastrar contenido de
otras fuentes menos relacionadas.

**Nota sobre calidad del texto:** el Fragmento 2 incluye metadatos de
cabecera del paper (título de la revista, DOI, nombres de autores)
mezclados con el inicio del abstract — el mismo problema de extracción de
PDF académico documentado en la sección de fallos (punto 4.1).

## 3. Generación

**Acierto in-corpus:**
- Pregunta: *"¿Qué es una memoria auto-asociativa?"*
- Respuesta: el sistema explicó función y rol (red que involucra al
  hipocampo para enlazar conceptos), implementación técnica (red de
  Hopfield moderna, MHN), y mecanismo (codificación dispersa, redes
  generativas entrenadas con representaciones "replayed"), citando
  `Spens(2023).pdf` y el fragmento exacto en cada afirmación.

**Abstención fuera de corpus:**
- Pregunta: *"¿Cuál es la capital de Francia?"*
- Respuesta: *"El contexto proporcionado no contiene información
  suficiente para responder a esa pregunta."*
- Es interesante en este caso hacer notar que en la fase de retrieval, se recuperó un chunk (de `Spens(2023).pdf`) que pasó el filtro determinista de `UMBRAL_DISTANCIA=0.6`. Es decir, la abstención no vino del filtro
  determinista, sino de la instrucción del prompt. Confirma
  que ambas capas de defensa son necesarias: el umbral no detecta todo,
  el prompt es la última línea de defensa.

## 4. Fallos conocidos

1. **Ruido en la extracción de texto de PDFs académicos.** `PyPDFLoader`
   extrae el texto de forma literal, sin entender la estructura visual del
   documento — en papers con columnas y figuras, las etiquetas de
   subfigura ("a", "b", "c"...) y pies de figura se mezclan con el cuerpo
   del texto, produciendo chunks con palabras partidas y fragmentos fuera
   de orden. Posibles soluciones (?): probar una librería de extracción más
   consciente de la estructura de los papers académicos o filtrar
   líneas muy cortas para evitar que mezcle con texto real con posibles leyendas de gráficas.
    **Ejemplo real** (para ejemplificar lo señalado anteriormente, véase el chunk recuperado de `Spens(2023).pdf` al preguntar
   *"¿Qué es una memoria auto-asociativa?"*):

   ```text
   Gener ativ e network: A ut oassociativ e network:
   An episode, for example, encountering an unfamiliar animal in a forest
   a
   b
   c d
   Encoding Recall Decomposed
   event
   Autoassociative
   network output
   Original
   event
   Noisy input Decomposed
   input Generative
   network output
   Recombined
   output
   Feature

2. **Latencia alta y persistente (~60s por consulta).** Tanto `--query`
   como `--ask` tardan sistemáticamente decenas de segundos, incluso en
   consultas que solo embeddean una pregunta. Es coherente con estar en el
   tier gratuito de la API de Gemini (100 requests/minuto), que fuerza
   reintentos con espera.

3. **El umbral de distancia no detecta todas las preguntas fuera de
   dominio.** Como se vio en la prueba de abstención, una pregunta
   claramente ajena al corpus puede recuperar un chunk con distancia por
   debajo de `UMBRAL_DISTANCIA=0.6`. La abstención final dependió
   enteramente de la instrucción del prompt, no del filtro determinista.
   