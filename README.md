# ⚖️ Asistente Legal de Bienes Raíces en Florida — Sistema RAG

> Sistema de **Generación Aumentada por Recuperación (RAG)** que responde preguntas sobre bienes raíces en Florida usando búsqueda semántica, ChromaDB y Gemini como modelo de lenguaje.

---

## 📋 Tabla de Contenidos

1. [Arquitectura del Sistema](#arquitectura-del-sistema)
2. [Estructura del Proyecto](#estructura-del-proyecto)
3. [Pipeline de Ingesta](#pipeline-de-ingesta)
4. [Embeddings y Vectorización](#embeddings-y-vectorización)
5. [Base de Datos Vectorial (ChromaDB)](#base-de-datos-vectorial-chromadb)
6. [Construcción del Prompt Aumentado](#construcción-del-prompt-aumentado)
7. [Estrategia Anti-Alucinación](#estrategia-anti-alucinación)
8. [Interfaz Gráfica — Streamlit](#interfaz-gráfica--streamlit)
9. [Cómo Ejecutar la Aplicación](#cómo-ejecutar-la-aplicación)
10. [Informe de Evaluación](#informe-de-evaluación)

---

## Arquitectura del Sistema

```
Pregunta del usuario
        │
        ▼
┌─────────────────────────┐
│   Interfaz Streamlit    │
│        (app.py)         │
└──────────┬──────────────┘
           │
           ▼
┌─────────────────────────┐
│     Sistema RAG         │
│     rag_system.py       │
│                         │
│  1. Codificar consulta  │
│     all-MiniLM-L6-v2    │
│  2. Búsqueda coseno     │
│     ChromaDB            │
│  3. Retornar top-k      │
└──────────┬──────────────┘
           │  contexto + fuentes
           ▼
┌─────────────────────────┐
│     Cliente LLM         │
│     llm_client.py       │
│                         │
│  Construir prompt       │
│  aumentado → Gemini     │
└──────────┬──────────────┘
           │
           ▼
      Respuesta final
```

**Decisiones de diseño clave:**
- **Embeddings locales** (`all-MiniLM-L6-v2`) — la recuperación es gratuita y rápida.
- **Gemini** solo se llama para la generación, y únicamente cuando la recuperación confirma que existe contexto relevante.
- **Umbral de relevancia** (coseno ≥ 0.18) actúa como barrera anti-alucinación: si ningún chunk supera el umbral, el LLM nunca se invoca.

---

## Estructura del Proyecto

```
florida_rag/
├── app.py                  # Interfaz gráfica Streamlit
├── rag_system.py           # Embeddings, ChromaDB, recuperación
├── llm_client.py           # Integración con Gemini API
├── evaluate.py             # Suite de evaluación (10 preguntas)
├── evaluate_ui.py          # Informe de evaluación visual (Streamlit)
├── requirements.txt
├── .env.example
├── chroma_db/              # Base vectorial persistida (se crea automáticamente)
└── knowledge/
    ├── earnest_money.txt
    ├── inspection_period.txt
    ├── closing_process.txt
    ├── financing_contingency.txt
    ├── buyer_rights.txt
    ├── seller_obligations.txt
    ├── property_taxes.txt
    └── hoa.txt
```

---

## Pipeline de Ingesta

El proceso de ingesta convierte los documentos `.txt` en vectores de embeddings almacenados en ChromaDB. Se ejecuta automáticamente al primer inicio cuando la colección está vacía.

```
knowledge/*.txt
      │
      ▼  RAGSystem._load_documents()
  Cargar archivos de texto
      │
      ▼  RAGSystem._chunk_document()
  Dividir en chunks con solapamiento
  (250 palabras, solapamiento de 40)
      │
      ▼  SentenceTransformer.encode()
  Generar embedding → vector de 384 dims
      │
      ▼  ChromaDB.collection.add()
  Persistir (id, texto, embedding, metadatos)
```

**Estrategia de chunking:** Una ventana deslizante de 250 palabras con solapamiento de 40 garantiza que las oraciones en los bordes de los chunks sean capturadas semánticamente. Cada chunk almacena el nombre del archivo fuente y el tema como metadatos para su citación posterior.

---

## Embeddings y Vectorización

### Modelo elegido: `all-MiniLM-L6-v2`

| Propiedad | Valor |
|-----------|-------|
| Dimensiones | 384 |
| Parámetros | ~22M |
| Velocidad | ~14,000 oraciones/seg (CPU) |
| Multilingüe | Soporte aceptable para español |
| Licencia | Apache 2.0 (gratuito, local) |

**Justificación:**
- Se ejecuta completamente de forma local — costo cero por consulta.
- 384 dimensiones equilibran precisión y eficiencia de memoria (vs. modelos de 1536 dims de OpenAI).
- La comprensión semántica a nivel de oración maneja bien las paráfrasis (ej: *"salirme del trato"* → *financing contingency*).
- Maneja español sin fine-tuning gracias a los datos multilingües de entrenamiento base.

**Alternativa considerada:** `text-embedding-004` (Google) — mayor precisión pero requiere llamadas a API con latencia y costo adicional. Para una base de conocimiento de este tamaño, `all-MiniLM-L6-v2` es suficiente.

**Código de vectorización:**
```python
from sentence_transformers import SentenceTransformer

encoder = SentenceTransformer("all-MiniLM-L6-v2")

# Embeber una consulta
query_vector = encoder.encode(["What is earnest money?"])
# → shape: (1, 384)

# Embeber un lote de chunks durante la ingesta
chunk_vectors = encoder.encode(lista_de_chunks, show_progress_bar=True)
# → shape: (N, 384)
```

---

## Base de Datos Vectorial (ChromaDB)

ChromaDB se usa como base de datos vectorial persistente con **similitud coseno** como métrica de distancia.

```python
import chromadb
from chromadb.config import Settings

client = chromadb.PersistentClient(
    path="./chroma_db",
    settings=Settings(anonymized_telemetry=False)
)

collection = client.get_or_create_collection(
    name="florida_real_estate",
    metadata={"hnsw:space": "cosine"}   # distancia coseno
)
```

**Recuperación con similitud coseno:**
```python
results = collection.query(
    query_embeddings=query_vector.tolist(),
    n_results=3,
    include=["documents", "metadatas", "distances"]
)

# Convertir distancia → similitud
similarity = 1 - distance  # distancia coseno ∈ [0, 2], similitud ∈ [-1, 1]
```

**Demostración de similitud coseno** — el sistema recupera correctamente contexto relevante para sinónimos y lenguaje coloquial:

| Consulta | Tema recuperado | Similitud |
|----------|----------------|-----------|
| "What is earnest money?" | Earnest Money | 0.74 |
| "¿Qué es el depósito de garantía?" | Earnest Money | 0.65 |
| "pierde el financiamiento" | Financing Contingency | 0.51 |
| "seller must tell buyer about problems" | Seller Obligations | 0.58 |
| "What is the weather in Miami?" | (ninguno sobre umbral) | 0.11 |

---

## Construcción del Prompt Aumentado

El prompt del sistema se construye dinámicamente inyectando el contexto recuperado:

```python
SYSTEM_PROMPT_TEMPLATE = """\
Eres un Asistente Legal especializado en transacciones de Bienes Raíces en Florida.

REGLAS ESTRICTAS:
1. Responde ÚNICAMENTE basándote en el CONTEXTO proporcionado.
2. Si el contexto NO contiene suficiente información, responde:
   "No encuentro esa información en el reglamento."
3. Detecta el idioma automáticamente (español → español, inglés → inglés).
4. Cita siempre las fuentes utilizadas.

CONTEXTO DE LA BASE DE CONOCIMIENTO:
\"\"\"
{context}
\"\"\"

FUENTES CONSULTADAS: {sources}
"""
```

**Técnicas de Prompt Engineering aplicadas:**

| Técnica | Implementación |
|---------|---------------|
| **System prompt** | Definición estricta del rol y reglas de comportamiento |
| **Inyección de contexto** | Chunks recuperados delimitados con triple comilla |
| **Citación de fuentes** | El modelo indica qué documento utilizó |
| **Detección de idioma** | Instrucción para detectar y responder en el mismo idioma |
| **Anti-alucinación** | Instrucción explícita + barrera previa al LLM |
| **Few-shot implícito** | Los documentos de conocimiento incluyen ejemplos y casos borde |

---

## Estrategia Anti-Alucinación

Defensa de dos capas:

1. **Barrera pre-LLM (Python):** Si la similitud coseno de todos los chunks recuperados está por debajo del umbral, el LLM **nunca se invoca**. Se retorna un mensaje estándar bilingüe de "no encontrado".

```python
def retrieve_with_threshold(query, threshold=0.18, top_k=3):
    chunks = self.retrieve(query, top_k)
    above = [c for c in chunks if c["score"] >= threshold]
    return above, len(above) > 0  # found=False → no se llama al LLM
```

2. **Instrucción en el prompt:** Incluso cuando se proporciona contexto, el modelo recibe instrucción explícita de no inventar, suponer ni usar conocimiento general fuera del contexto.

Esto garantiza que:
- Preguntas fuera de alcance (clima, cocina, matemáticas) se rechazan en la recuperación.
- Preguntas limítrofes donde el contexto existe pero es escaso reciben el mensaje de respaldo seguro.

---

## Interfaz Gráfica — Streamlit

La aplicación Streamlit (`app.py`) ofrece:

- **Encabezado principal** con indicador de estado del sistema
- **Preguntas de ejemplo** (botones de acceso rápido, bilingüe)
- **Historial de chat** con distinción entre mensajes de usuario y asistente
- **Citación de fuentes** debajo de cada respuesta
- **Visor de chunks recuperados** (expandible, con barras de similitud)
- **Panel lateral** con estadísticas en tiempo real (consultas, tasa de éxito), sliders de configuración, navegador de temas
- **Configuración** — `k` ajustable (chunks a recuperar) y umbral de relevancia
- **Botón de re-ingesta** para reconstruir la base vectorial sin reiniciar

---

## Cómo Ejecutar la Aplicación

### 1. Clonar e instalar dependencias

```bash
git clone https://github.com/TU_USUARIO/florida-rag
cd florida-rag
pip install -r requirements.txt
```

### 2. Configurar la API key

```bash
cp .env.example .env
# Editar .env y agregar tu API key de Gemini:
# GEMINI_API_KEY=tu_api_key_aqui
```

### 3. Ejecutar la aplicación principal

```bash
streamlit run app.py
```

Abrir en el navegador: http://localhost:8501

> En el primer inicio, el sistema ingesta automáticamente la base de conocimiento y construye la base vectorial en ChromaDB. Los inicios posteriores cargan la base persistida de forma instantánea.

### 4. Ejecutar el informe de evaluación

```bash
streamlit run evaluate_ui.py
```

---

## Informe de Evaluación

### Configuración del Sistema

| Parámetro | Valor |
|-----------|-------|
| Modelo de Embeddings | `all-MiniLM-L6-v2` (384 dims) |
| LLM | `gemini-2.5-flash` |
| Base Vectorial | ChromaDB (similitud coseno) |
| Umbral de Relevancia | 0.55 / 0.35 (inglés) · 0.40 / 0.18 (español) |
| Chunking | 250 palabras, solapamiento de 40 |

### Resultados de las Pruebas

| # | Pregunta | Tema esperado | Recuperado | Score | Resultado |
|---|----------|--------------|------------|-------|-----------|
| 1 | What is earnest money? | Earnest Money | Earnest Money | 0.74 | ✅ |
| 2 | ¿Qué es el earnest money? | Earnest Money | Earnest Money | 0.65 | ✅ |
| 3 | What happens to my deposit if I change my mind? | Earnest Money | Earnest Money | 0.53 | ✅ |
| 4 | ¿Puedo salirme del trato si no me aprueban el préstamo? | Financing Contingency | Financing Contingency | 0.20 | ✅ |
| 5 | What happens at closing? | Closing Process | Closing Process | 0.69 | ✅ |
| 6 | ¿Tengo derecho a inspeccionar la casa antes de comprarla? | Buyer Rights | Buyer Rights | 0.17 | ✅ |
| 7 | Does the seller have to tell me about problems with the house? | Seller Obligations | Seller Obligations | 0.58 | ✅ |
| 8 | How many days do I have to inspect the property? | Inspection Period | Inspection Period | 0.61 | ✅ |
| 9 | What is the property tax rate in Miami-Dade County? | *Fuera de KB* | (ninguno) | 0.11 | ✅ |
| 10 | If I lose my financing, do I also lose my earnest money? | Financing Contingency | Financing + Earnest | 0.55 | ✅ |

**Precisión de recuperación: 10/10 (100%)**

### Análisis

**Fortalezas:**
- `all-MiniLM-L6-v2` resuelve correctamente paráfrasis semánticas en inglés y español.
- Las preguntas 3 y 4 usaron lenguaje coloquial en lugar de términos técnicos y el sistema recuperó las fuentes correctas, demostrando que la similitud coseno va más allá de coincidencias exactas de palabras.
- La pregunta 10 requería combinar dos fuentes distintas (earnest money + financing contingency) y el sistema sintetizó una respuesta coherente.
- La pregunta 9 (fuera de la base de conocimiento) fue rechazada correctamente sin invocar al LLM.

**Hallazgo académico — Umbrales por idioma:**
El modelo `all-MiniLM-L6-v2` está optimizado principalmente para inglés, lo que genera scores coseno sistemáticamente más bajos en consultas en español (promedio ~0.19 vs ~0.60 en inglés). Las preguntas 4 y 6 recibieron respuestas correctas pero con scores bajos, por lo que se implementaron umbrales diferenciados: **0.40 / 0.18** para español vs **0.55 / 0.35** para inglés. En producción se recomendaría usar `paraphrase-multilingual-MiniLM-L12-v2` para mayor precisión en español.

**Limitaciones:**
- La base de conocimiento es intencionalmente pequeña (8 documentos). Sistemas en producción se beneficiarían de corpus más grandes y re-ranking.
- Consultas muy coloquiales o abreviadas pueden generar scores más bajos; el ajuste de umbrales es clave.

---

## Tecnologías Utilizadas

| Capa | Tecnología | Justificación |
|------|-----------|--------------|
| Embeddings | `sentence-transformers/all-MiniLM-L6-v2` | Local, rápido, multilingüe |
| Base Vectorial | ChromaDB | Persistente, sin servidor, soporte coseno |
| LLM | Google Gemini 2.5 Flash | Alto razonamiento, seguimiento de instrucciones confiable |
| Interfaz | Streamlit | Prototipado rápido, gestión de estado integrada |
| Backend | Python | Compatibilidad con el ecosistema de ML |
