"""
rag_system.py  —  RAG real con ChromaDB + sentence-transformers

Flujo:
  1. Cargar documentos .txt desde /knowledge
  2. Dividir en chunks (tamaño + overlap configurables)
  3. Generar embeddings locales con sentence-transformers
  4. Almacenar en ChromaDB (base de datos vectorial local)
  5. Buscar los chunks más relevantes por similitud semántica
"""

import os
import re
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer

# ─── Configuración ───────────────────────────────────────────────
KNOWLEDGE_DIR  = "knowledge"
CHROMA_DIR     = "chroma_db"
COLLECTION     = "florida_legal"
CHUNK_SIZE     = 400
CHUNK_OVERLAP  = 80
TOP_K          = 2

# Modelo local de embeddings (descarga automática la primera vez, ~90 MB)
# Elegido por: velocidad local, 384 dimensiones, buen soporte multilingüe
EMBED_MODEL    = "all-MiniLM-L6-v2"


# ─── Inicialización de herramientas ──────────────────────────────
print("[RAG] Cargando modelo de embeddings...")
_embedder = SentenceTransformer(EMBED_MODEL)

_chroma_client = chromadb.PersistentClient(
    path=CHROMA_DIR,
    settings=Settings(anonymized_telemetry=False),
)


# ─── 1. Chunking ─────────────────────────────────────────────────
def _chunk_text(text: str, size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    """
    Divide un texto en fragmentos de `size` caracteres con `overlap`
    caracteres de solapamiento entre chunks consecutivos.
    """
    text = re.sub(r"\n{3,}", "\n\n", text.strip())
    chunks = []
    start = 0
    while start < len(text):
        end = start + size
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        start += size - overlap
    return chunks


# ─── 2. Carga e indexación de documentos ─────────────────────────
def _ya_indexado(collection) -> bool:
    return collection.count() > 0


def cargar_e_indexar() -> chromadb.Collection:
    collection = _chroma_client.get_or_create_collection(
        name=COLLECTION,
        metadata={"hnsw:space": "cosine"},
    )

    if _ya_indexado(collection):
        print(f"[RAG] Colección '{COLLECTION}' ya indexada ({collection.count()} chunks). Reutilizando.")
        return collection

    print(f"[RAG] Indexando documentos de '{KNOWLEDGE_DIR}'...")
    all_chunks, all_ids, all_metas = [], [], []
    chunk_idx = 0

    for filename in sorted(os.listdir(KNOWLEDGE_DIR)):
        if not filename.endswith(".txt"):
            continue

        filepath = os.path.join(KNOWLEDGE_DIR, filename)
        with open(filepath, "r", encoding="utf-8") as f:
            texto = f.read()

        chunks = _chunk_text(texto)
        print(f"  · {filename} → {len(chunks)} chunks")

        for i, chunk in enumerate(chunks):
            all_chunks.append(chunk)
            all_ids.append(f"{filename}_chunk{i}_{chunk_idx}")
            all_metas.append({"source": filename, "chunk_index": i})
            chunk_idx += 1

    print(f"[RAG] Generando embeddings para {len(all_chunks)} chunks...")
    embeddings = _embedder.encode(all_chunks, show_progress_bar=False).tolist()

    collection.add(
        documents=all_chunks,
        embeddings=embeddings,
        ids=all_ids,
        metadatas=all_metas,
    )

    print(f"[RAG] ✅ Indexación completa: {collection.count()} chunks almacenados.")
    return collection


# ─── 3. Búsqueda semántica ────────────────────────────────────────
def buscar_contexto(pregunta: str, collection: chromadb.Collection, k: int = TOP_K) -> str:
    """
    Versión simple: devuelve solo el texto del contexto.
    """
    contexto, _ = buscar_contexto_con_fuentes(pregunta, collection, k)
    return contexto


def buscar_contexto_con_fuentes(
    pregunta: str, collection: chromadb.Collection, k: int = TOP_K
) -> tuple[str, list[dict]]:
    """
    Vectoriza la pregunta y recupera los `k` chunks más similares.
    Devuelve:
      - contexto: string listo para inyectar en el prompt
      - fuentes: lista de dicts con {source, score} para mostrar en la UI
    """
    query_embedding = _embedder.encode([pregunta]).tolist()

    resultados = collection.query(
        query_embeddings=query_embedding,
        n_results=k,
        include=["documents", "metadatas", "distances"],
    )

    docs      = resultados["documents"][0]
    metas     = resultados["metadatas"][0]
    distances = resultados["distances"][0]

    fragmentos = []
    fuentes = []

    for doc, meta, dist in zip(docs, metas, distances):
        score = round(1 - dist, 4)
        fragmentos.append(
            f"[Fuente: {meta['source']} | Relevancia: {score}]\n{doc}"
        )
        fuentes.append({
            "source": meta["source"].replace(".txt", "").replace("_", " ").title(),
            "score": score,
            "filename": meta["source"]
        })

    contexto = "\n\n---\n\n".join(fragmentos)
    return contexto, fuentes


# ─── API pública ──────────────────────────────────────────────────
def inicializar() -> chromadb.Collection:
    """Punto de entrada: devuelve la colección lista para consultas."""
    return cargar_e_indexar()