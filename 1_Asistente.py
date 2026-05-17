import os
import streamlit as st
from google import genai
from dotenv import load_dotenv
from rag_system import inicializar, buscar_contexto_con_fuentes

# ─── Configuración de página (debe ir PRIMERO) ────────────────────
st.set_page_config(
    page_title="Florida Legal Assistant",
    page_icon="⚖️",
    layout="wide",
)

# ─── CSS personalizado ────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&family=Spectral:wght@600&display=swap');

/* Base */
html, body, .stApp {
    background-color: #111111;
    color: #f3f4f6;
    font-family: 'Inter', sans-serif;
}

/* Hide Streamlit UI */
#MainMenu, footer, header {
    visibility: hidden;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background-color: #161616;
    border-right: 1px solid #262626;
}

[data-testid="stSidebar"] * {
    color: #f3f4f6;
}

/* Main title */
.main-title {
    font-family: 'Spectral', serif;
    font-size: 2rem;
    font-weight: 600;
    margin-bottom: 4px;
    color: #f8fafc;
}

.main-subtitle {
    color: #9ca3af;
    font-size: 0.9rem;
    margin-bottom: 28px;
}

/* Chat messages */
[data-testid="stChatMessage"] {
    background-color: #181818;
    border: 1px solid #262626;
    border-radius: 12px;
    padding: 10px;
    margin-bottom: 14px;
}

/* Chat text */
[data-testid="stChatMessageContent"] {
    font-size: 0.94rem;
    line-height: 1.75;
}

/* Chat input */
[data-testid="stChatInput"] textarea {
    background-color: #1c1c1c !important;
    border: 1px solid #2d2d2d !important;
    color: #f3f4f6 !important;
    border-radius: 10px !important;
}

/* Buttons */
.stButton button {
    background-color: #1d1d1d;
    color: #f3f4f6;
    border: 1px solid #303030;
    border-radius: 10px;
    transition: 0.2s;
}

.stButton button:hover {
    border-color: #d4b06a;
    color: #d4b06a;
}

/* Sources */
.source-pill {
    display: inline-block;
    background: #1f1f1f;
    border: 1px solid #333333;
    color: #d1d5db;
    border-radius: 999px;
    padding: 4px 10px;
    font-size: 0.72rem;
    margin-right: 6px;
    margin-top: 6px;
}

/* History cards */
.hist-card {
    background: #1b1b1b;
    border: 1px solid #2b2b2b;
    border-radius: 10px;
    padding: 10px;
    margin-bottom: 8px;
}

.hist-q {
    color: #f3f4f6;
    font-size: 0.8rem;
    font-weight: 500;
}

.hist-a {
    color: #9ca3af;
    font-size: 0.75rem;
    margin-top: 4px;
}

/* Divider */
hr {
    border-color: #262626;
}
</style>
""", unsafe_allow_html=True)

# ─── Carga de variables de entorno ───────────────────────────────
load_dotenv()
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

# ─── System Prompt con anti-alucinación ──────────────────────────
SYSTEM_PROMPT = """You are a Legal Assistant specialized in Florida Real Estate transactions.

Your role:
  Help users understand the process of buying houses in Florida.

LANGUAGE RULE:
  - Detect the language of the user's question.
  - Respond ONLY in that language (Spanish → Spanish only, English → English only).
  - Never mix languages in the same response.

ANTI-HALLUCINATION RULES (CRITICAL):
  - You MUST base your answer on the provided CONTEXT.
  - If the CONTEXT does not contain enough information to answer the question, respond EXACTLY with:
      English: "I cannot find that information in my knowledge base. Please consult a licensed Florida real estate attorney for this specific question."
      Spanish: "No encuentro esa información en mi base de conocimientos. Por favor consulta a un abogado inmobiliario certificado en Florida para esta pregunta específica."
  - NEVER invent legal facts, figures, deadlines, or procedures not present in the CONTEXT.
  - NEVER contradict the CONTEXT.
  - Do not speculate or guess — only use what is explicitly in the CONTEXT.

Rules:
  - Use the provided CONTEXT when it is relevant; prioritize it over general knowledge.
  - For greetings or off-topic questions, respond naturally and briefly.
  - Keep answers clear, structured, and professional.
"""

def construir_prompt(pregunta: str, contexto: str) -> str:
    return f"""CONTEXT retrieved from the knowledge base:
\"\"\"
{contexto}
\"\"\"

USER QUESTION:
{pregunta}

Provide a clear, well-structured answer based ONLY on the context above.
If the context does not contain the answer, say so explicitly as instructed."""


# ─── Inicialización de RAG (se ejecuta solo una vez) ─────────────
@st.cache_resource(show_spinner="⚙️ Cargando sistema RAG y modelo de embeddings...")
def cargar_sistema():
    return inicializar()

coleccion = cargar_sistema()

# ─── Estado de sesión ─────────────────────────────────────────────
if "mensajes" not in st.session_state:
    st.session_state.mensajes = []   # [{role, content, fuentes}]

# ─── SIDEBAR — Historial de conversación ─────────────────────────
with st.sidebar:
    st.markdown("""
<div class="main-title">
Florida Legal Assistant
</div>

<div class="main-subtitle">
Semantic search over Florida real estate documents.
</div>
""", unsafe_allow_html=True)
    st.markdown('<span class="rag-badge">RAG · ChromaDB · Gemini</span>', unsafe_allow_html=True)
    st.divider()

    col1, col2 = st.columns([2, 1])
    with col1:
        st.markdown("**📋 Historial**")
    with col2:
        if st.button("🗑️ Limpiar", use_container_width=True):
            st.session_state.mensajes = []
            st.rerun()

    st.markdown("")

    if not st.session_state.mensajes:
        st.markdown('<p style="color:#64748b; font-size:0.78rem; text-align:center;">Sin conversaciones aún.</p>', unsafe_allow_html=True)
    else:
        # Mostrar solo los turnos usuario-bot en orden inverso
        pares = []
        msgs = st.session_state.mensajes
        i = 0
        while i < len(msgs):
            if msgs[i]["role"] == "user" and i + 1 < len(msgs):
                pares.append((msgs[i], msgs[i+1]))
                i += 2
            else:
                i += 1

        for user_msg, bot_msg in reversed(pares):
            pregunta_corta = user_msg["content"][:60] + ("…" if len(user_msg["content"]) > 60 else "")
            resp_corta = bot_msg["content"][:80] + ("…" if len(bot_msg["content"]) > 80 else "")
            st.markdown(f"""
            <div class="hist-card">
                <div class="hist-q">💬 {pregunta_corta}</div>
                <div class="hist-a">{resp_corta}</div>
            </div>
            """, unsafe_allow_html=True)

    st.divider()
    


# ─── ÁREA PRINCIPAL ───────────────────────────────────────────────
st.markdown("# ⚖️ Florida Legal Assistant")


# Banner de bienvenida (solo si no hay mensajes)
if not st.session_state.mensajes:
    st.markdown("""
    <div class="welcome-banner">
        <h2>🏠 Florida Real Estate Assistant</h2>
        <p>Haz preguntas sobre el proceso de compra de propiedades en Florida.<br>
        El sistema usa búsqueda semántica sobre una base de conocimiento real.</p>
        
    </div>
    """, unsafe_allow_html=True)

    # Sugerencias de preguntas
    st.markdown("**💡 Preguntas sugeridas:**")
    cols = st.columns(2)
    sugerencias = [
        "What is earnest money?",
        "¿Qué derechos tengo como comprador?",
        "Inspection period explained",
        "¿Cómo funciona el cierre de la compra?",
    ]
    for i, sug in enumerate(sugerencias):
        with cols[i % 2]:
            if st.button(sug, use_container_width=True, key=f"sug_{i}"):
                st.session_state["sugerencia_clickeada"] = sug
                st.rerun()

# ─── Mostrar mensajes existentes ──────────────────────────────────
for msg in st.session_state.mensajes:
    with st.chat_message(msg["role"], avatar="👤" if msg["role"] == "user" else "⚖️"):
        st.markdown(msg["content"])

        # Mostrar fuentes si es un mensaje del asistente y tiene fuentes
        if msg["role"] == "assistant" and msg.get("fuentes"):
            fuentes_html = "".join([
                f'<span class="source-pill">📄 {f["source"]} &nbsp; <strong>{round(f["score"]*100)}%</strong></span>'
                for f in msg["fuentes"]
            ])
            st.markdown(
                f'<div style="margin-top:10px; padding-top:10px; border-top:1px solid #1e3a5f;">'
                f'<span style="color:#64748b; font-size:0.72rem;">Fuentes consultadas: </span>'
                f'{fuentes_html}</div>',
                unsafe_allow_html=True
            )

# ─── Manejar sugerencia clickeada ────────────────────────────────
pregunta_sugerida = st.session_state.pop("sugerencia_clickeada", None)

# ─── Input del usuario ────────────────────────────────────────────
pregunta_usuario = st.chat_input("Escribe tu pregunta en inglés o español…")

# Usar sugerencia si se clickeó un botón
pregunta_final = pregunta_sugerida or pregunta_usuario

if pregunta_final:
    # 1. Mostrar mensaje del usuario
    st.session_state.mensajes.append({"role": "user", "content": pregunta_final, "fuentes": []})
    with st.chat_message("user", avatar="👤"):
        st.markdown(pregunta_final)

    # 2. Recuperación semántica del contexto
    with st.chat_message("assistant", avatar="⚖️"):
        with st.spinner("🔍 Buscando en la base de conocimiento..."):
            contexto, fuentes = buscar_contexto_con_fuentes(pregunta_final, coleccion)

        # 3. Construcción del prompt aumentado + generación con Gemini
        prompt_completo = f"{SYSTEM_PROMPT}\n\n{construir_prompt(pregunta_final, contexto)}"

        with st.spinner("✍️ Generando respuesta..."):
            response = client.models.generate_content(
                model="gemini-2.5-flash",
                contents=prompt_completo
            )
            respuesta = response.text

        # 4. Mostrar respuesta
        st.markdown(respuesta)

        # 5. Mostrar fuentes consultadas
        if fuentes:
            fuentes_html = "".join([
                f'<span class="source-pill">📄 {f["source"]} &nbsp; <strong>{round(f["score"]*100)}%</strong></span>'
                for f in fuentes
            ])
            st.markdown(
                f'<div style="margin-top:10px; padding-top:10px; border-top:1px solid #1e3a5f;">'
                f'<span style="color:#64748b; font-size:0.72rem;">Fuentes consultadas: </span>'
                f'{fuentes_html}</div>',
                unsafe_allow_html=True
            )

    # Guardar en historial
    st.session_state.mensajes.append({
        "role": "assistant",
        "content": respuesta,
        "fuentes": fuentes
    })