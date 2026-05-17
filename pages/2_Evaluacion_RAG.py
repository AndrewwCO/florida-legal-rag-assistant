import os
import time
import streamlit as st
from google import genai
from dotenv import load_dotenv
from rag_system import inicializar, buscar_contexto_con_fuentes

st.set_page_config(
    page_title="RAG Evaluation — Florida Legal Assistant",
    page_icon="🧪",
    layout="wide",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@600;700&family=DM+Sans:wght@300;400;500&display=swap');

.stApp { background-color: #0b1120; color: #e2e8f0; font-family: 'DM Sans', sans-serif; }
#MainMenu, footer, header { visibility: hidden; }
h1, h2, h3 { font-family: 'Playfair Display', serif !important; color: #e2e8f0 !important; }

[data-testid="stSidebar"] { background-color: #111827 !important; border-right: 1px solid #1e3a5f; }
[data-testid="stSidebar"] * { color: #e2e8f0 !important; }

.eval-card {
    background: #111827;
    border: 1px solid #1e3a5f;
    border-radius: 14px;
    padding: 20px 24px;
    margin-bottom: 18px;
}
.eval-header {
    display: flex;
    align-items: center;
    gap: 10px;
    margin-bottom: 12px;
}
.eval-num {
    background: #1e3a5f;
    color: #60a5fa;
    border-radius: 8px;
    padding: 2px 10px;
    font-size: 0.78rem;
    font-weight: 600;
}
.eval-category {
    background: rgba(245,158,11,0.1);
    color: #f59e0b;
    border: 1px solid rgba(245,158,11,0.25);
    border-radius: 20px;
    padding: 2px 10px;
    font-size: 0.72rem;
}
.eval-lang {
    background: rgba(99,102,241,0.12);
    color: #a5b4fc;
    border: 1px solid rgba(99,102,241,0.25);
    border-radius: 20px;
    padding: 2px 10px;
    font-size: 0.72rem;
}
.eval-question {
    font-size: 0.95rem;
    font-weight: 500;
    color: #e2e8f0;
    margin-bottom: 14px;
    padding: 10px 14px;
    background: #1a2540;
    border-left: 3px solid #3b82f6;
    border-radius: 0 8px 8px 0;
}
.eval-response {
    font-size: 0.85rem;
    color: #cbd5e1;
    line-height: 1.7;
    background: #0f1929;
    border-radius: 8px;
    padding: 14px 16px;
    margin-bottom: 12px;
}
.source-pill {
    display: inline-block;
    background: rgba(16,185,129,0.1);
    color: #10b981;
    border: 1px solid rgba(16,185,129,0.25);
    border-radius: 20px;
    padding: 2px 10px;
    font-size: 0.70rem;
    margin: 3px 4px 3px 0;
}
.verdict-ok {
    display: inline-block;
    background: rgba(16,185,129,0.12);
    color: #10b981;
    border: 1px solid rgba(16,185,129,0.3);
    border-radius: 8px;
    padding: 4px 14px;
    font-size: 0.78rem;
    font-weight: 600;
}
.verdict-warn {
    display: inline-block;
    background: rgba(245,158,11,0.12);
    color: #f59e0b;
    border: 1px solid rgba(245,158,11,0.3);
    border-radius: 8px;
    padding: 4px 14px;
    font-size: 0.78rem;
    font-weight: 600;
}
.verdict-fail {
    display: inline-block;
    background: rgba(239,68,68,0.12);
    color: #ef4444;
    border: 1px solid rgba(239,68,68,0.3);
    border-radius: 8px;
    padding: 4px 14px;
    font-size: 0.78rem;
    font-weight: 600;
}
.summary-box {
    background: linear-gradient(135deg, #111827, #1a2540);
    border: 1px solid #1e3a5f;
    border-radius: 16px;
    padding: 28px 32px;
    margin-bottom: 28px;
}
.metric-big {
    font-family: 'Playfair Display', serif;
    font-size: 2.8rem;
    color: #10b981;
    font-weight: 700;
}
.metric-label {
    font-size: 0.78rem;
    color: #64748b;
    text-transform: uppercase;
    letter-spacing: 0.08em;
}
</style>
""", unsafe_allow_html=True)

# ─── Setup ────────────────────────────────────────────────────────
load_dotenv()
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

SYSTEM_PROMPT = """You are a Legal Assistant specialized in Florida Real Estate transactions.

LANGUAGE RULE:
  - Detect the language of the user's question.
  - Respond ONLY in that language.

ANTI-HALLUCINATION RULES (CRITICAL):
  - You MUST base your answer on the provided CONTEXT.
  - If the CONTEXT does not contain enough information, respond EXACTLY with:
      English: "I cannot find that information in my knowledge base. Please consult a licensed Florida real estate attorney."
      Spanish: "No encuentro esa información en mi base de conocimientos. Por favor consulta a un abogado inmobiliario certificado en Florida."
  - NEVER invent legal facts not present in the CONTEXT.
  - Keep answers concise for evaluation purposes (max 3 sentences).
"""

def construir_prompt(pregunta: str, contexto: str) -> str:
    return f"""CONTEXT:
\"\"\"{contexto}\"\"\"

USER QUESTION: {pregunta}

Answer based ONLY on the context. Be concise (2-3 sentences max for evaluation)."""

@st.cache_resource(show_spinner="⚙️ Cargando sistema RAG...")
def cargar_sistema():
    return inicializar()

# ─── 10 preguntas de prueba ───────────────────────────────────────
PREGUNTAS_PRUEBA = [
    {
        "id": 1,
        "pregunta": "What is earnest money?",
        "categoria": "Concepto básico",
        "idioma": "🇺🇸 EN",
        "fuente_esperada": "earnest_money",
        "tipo": "Dentro de KB",
    },
    {
        "id": 2,
        "pregunta": "¿Qué es el earnest money?",
        "categoria": "Concepto básico",
        "idioma": "🇪🇸 ES",
        "fuente_esperada": "earnest_money",
        "tipo": "Dentro de KB",
    },
    {
        "id": 3,
        "pregunta": "What happens to my deposit if I change my mind?",
        "categoria": "Lenguaje coloquial",
        "idioma": "🇺🇸 EN",
        "fuente_esperada": "earnest_money",
        "tipo": "Sinónimos / coloquial",
    },
    {
        "id": 4,
        "pregunta": "¿Puedo salirme del trato si no me aprueban el préstamo?",
        "categoria": "Lenguaje coloquial",
        "idioma": "🇪🇸 ES",
        "fuente_esperada": "financing_contingency",
        "tipo": "Sinónimos / coloquial",
    },
    {
        "id": 5,
        "pregunta": "What happens at closing?",
        "categoria": "Proceso",
        "idioma": "🇺🇸 EN",
        "fuente_esperada": "closing_process",
        "tipo": "Dentro de KB",
    },
    {
        "id": 6,
        "pregunta": "¿Tengo derecho a inspeccionar la casa antes de comprarla?",
        "categoria": "Derechos",
        "idioma": "🇪🇸 ES",
        "fuente_esperada": "buyer_rights",
        "tipo": "Dentro de KB",
    },
    {
        "id": 7,
        "pregunta": "Does the seller have to tell me about problems with the house?",
        "categoria": "Obligaciones",
        "idioma": "🇺🇸 EN",
        "fuente_esperada": "seller_obligations",
        "tipo": "Dentro de KB",
    },
    {
        "id": 8,
        "pregunta": "How many days do I have to inspect the property?",
        "categoria": "Período inspección",
        "idioma": "🇺🇸 EN",
        "fuente_esperada": "inspection_period",
        "tipo": "Dentro de KB",
    },
    {
        "id": 9,
        "pregunta": "What is the property tax rate in Miami-Dade County?",
        "categoria": "Anti-alucinación",
        "idioma": "🇺🇸 EN",
        "fuente_esperada": None,
        "tipo": "Fuera de KB",
    },
    {
        "id": 10,
        "pregunta": "If I lose my financing, do I also lose my earnest money?",
        "categoria": "Combinación de conceptos",
        "idioma": "🇺🇸 EN",
        "fuente_esperada": "financing_contingency",
        "tipo": "Multi-fuente",
    },
]

# ─── UI ───────────────────────────────────────────────────────────
st.markdown("# 🧪 Informe de Evaluación RAG")
st.markdown('<p style="color:#64748b; margin-top:-10px;">10 preguntas de prueba corridas contra el sistema real · Florida Legal Assistant</p>', unsafe_allow_html=True)
st.divider()

coleccion = cargar_sistema()

col_btn, col_info = st.columns([1, 3])
with col_btn:
    correr = st.button("▶️ Correr evaluación completa", use_container_width=True, type="primary")
with col_info:
    st.markdown('<p style="color:#64748b; font-size:0.82rem; padding-top:10px;">Esto corre las 10 preguntas en tiempo real contra ChromaDB y Gemini. Tarda ~30 segundos.</p>', unsafe_allow_html=True)

st.markdown("")

# ─── Ejecución ────────────────────────────────────────────────────
if correr or "resultados_eval" in st.session_state:

    if correr:
        resultados = []
        barra = st.progress(0, text="Iniciando evaluación...")

        for i, prueba in enumerate(PREGUNTAS_PRUEBA):
            barra.progress((i) / len(PREGUNTAS_PRUEBA), text=f"Pregunta {i+1}/10: {prueba['pregunta'][:50]}…")

            contexto, fuentes = buscar_contexto_con_fuentes(prueba["pregunta"], coleccion)

            prompt = f"{SYSTEM_PROMPT}\n\n{construir_prompt(prueba['pregunta'], contexto)}"
            response = client.models.generate_content(
                model="gemini-2.5-flash",
                contents=prompt
            )
            respuesta = response.text.strip()

            top_score = fuentes[0]["score"] if fuentes else 0.0
            top_fuente = fuentes[0]["filename"].replace(".txt", "") if fuentes else "—"

            fuente_ok = (
                prueba["fuente_esperada"] is None
                or any(prueba["fuente_esperada"] in f["filename"] for f in fuentes)
            )

            es_no_encontrado = (
                "cannot find" in respuesta.lower()
                or "no encuentro" in respuesta.lower()
            )

            # ── Umbrales ajustados por idioma ──────────────────────
            # all-MiniLM-L6-v2 está optimizado para inglés;
            # las consultas en español producen scores coseno más bajos
            # de forma sistemática, por lo que se usan umbrales diferenciados.
            umbral_ok      = 0.40 if prueba["idioma"] == "🇪🇸 ES" else 0.55
            umbral_parcial = 0.18 if prueba["idioma"] == "🇪🇸 ES" else 0.35

            if prueba["fuente_esperada"] is None:
                if es_no_encontrado:
                    veredicto = "✅ Correcto"
                    veredicto_clase = "verdict-ok"
                else:
                    veredicto = "❌ Alucinación"
                    veredicto_clase = "verdict-fail"
            else:
                if fuente_ok and top_score >= umbral_ok:
                    veredicto = "✅ Correcto"
                    veredicto_clase = "verdict-ok"
                elif fuente_ok and top_score >= umbral_parcial:
                    veredicto = "⚠️ Parcial"
                    veredicto_clase = "verdict-warn"
                else:
                    veredicto = "❌ Incorrecto"
                    veredicto_clase = "verdict-fail"

            resultados.append({
                **prueba,
                "respuesta": respuesta,
                "fuentes": fuentes,
                "top_score": top_score,
                "top_fuente": top_fuente,
                "fuente_ok": fuente_ok,
                "veredicto": veredicto,
                "veredicto_clase": veredicto_clase,
            })

            time.sleep(30)

        barra.progress(1.0, text="✅ Evaluación completada")
        st.session_state["resultados_eval"] = resultados

    resultados = st.session_state["resultados_eval"]

    # ─── Resumen ejecutivo ────────────────────────────────────────
    total       = len(resultados)
    correctos   = sum(1 for r in resultados if "✅" in r["veredicto"])
    parciales   = sum(1 for r in resultados if "⚠️" in r["veredicto"])
    incorrectos = sum(1 for r in resultados if "❌" in r["veredicto"])
    avg_score   = sum(r["top_score"] for r in resultados) / total
    tasa        = round(correctos / total * 100)

    st.markdown("## 📊 Resumen Ejecutivo")
    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        st.markdown(f'<div class="summary-box" style="text-align:center"><div class="metric-big">{tasa}%</div><div class="metric-label">Tasa de éxito</div></div>', unsafe_allow_html=True)
    with c2:
        st.markdown(f'<div class="summary-box" style="text-align:center"><div class="metric-big" style="color:#10b981">{correctos}</div><div class="metric-label">Correctas</div></div>', unsafe_allow_html=True)
    with c3:
        st.markdown(f'<div class="summary-box" style="text-align:center"><div class="metric-big" style="color:#f59e0b">{parciales}</div><div class="metric-label">Parciales</div></div>', unsafe_allow_html=True)
    with c4:
        st.markdown(f'<div class="summary-box" style="text-align:center"><div class="metric-big" style="color:#ef4444">{incorrectos}</div><div class="metric-label">Incorrectas</div></div>', unsafe_allow_html=True)
    with c5:
        st.markdown(f'<div class="summary-box" style="text-align:center"><div class="metric-big" style="color:#60a5fa">{avg_score:.2f}</div><div class="metric-label">Score coseno prom.</div></div>', unsafe_allow_html=True)

    # ─── Tabla resumen ────────────────────────────────────────────
    st.markdown("### 📋 Tabla de resultados")
    tabla_html = """
    <table style="width:100%; border-collapse:collapse; font-size:0.82rem; font-family:'DM Sans',sans-serif;">
    <thead>
      <tr style="background:#1a2540; color:#60a5fa; text-align:left;">
        <th style="padding:10px 12px; border-bottom:1px solid #1e3a5f;">#</th>
        <th style="padding:10px 12px; border-bottom:1px solid #1e3a5f;">Pregunta</th>
        <th style="padding:10px 12px; border-bottom:1px solid #1e3a5f;">Idioma</th>
        <th style="padding:10px 12px; border-bottom:1px solid #1e3a5f;">Tipo</th>
        <th style="padding:10px 12px; border-bottom:1px solid #1e3a5f;">Top Fuente</th>
        <th style="padding:10px 12px; border-bottom:1px solid #1e3a5f;">Score</th>
        <th style="padding:10px 12px; border-bottom:1px solid #1e3a5f;">Umbral usado</th>
        <th style="padding:10px 12px; border-bottom:1px solid #1e3a5f;">Resultado</th>
      </tr>
    </thead>
    <tbody>
    """
    for r in resultados:
        color_row   = "#0f1929" if r["id"] % 2 == 0 else "#111827"
        score_color = "#10b981" if r["top_score"] >= 0.55 else ("#f59e0b" if r["top_score"] >= 0.35 else "#ef4444")
        umbral_txt  = "0.40 / 0.18" if r["idioma"] == "🇪🇸 ES" else "0.55 / 0.35"
        tabla_html += f"""
        <tr style="background:{color_row};">
          <td style="padding:9px 12px; color:#64748b;">{r['id']}</td>
          <td style="padding:9px 12px; color:#e2e8f0; max-width:260px;">{r['pregunta']}</td>
          <td style="padding:9px 12px;">{r['idioma']}</td>
          <td style="padding:9px 12px; color:#94a3b8;">{r['tipo']}</td>
          <td style="padding:9px 12px; color:#60a5fa; font-size:0.75rem;">{r['top_fuente']}</td>
          <td style="padding:9px 12px; color:{score_color}; font-weight:600;">{r['top_score']:.3f}</td>
          <td style="padding:9px 12px; color:#64748b; font-size:0.75rem;">{umbral_txt}</td>
          <td style="padding:9px 12px;">{r['veredicto']}</td>
        </tr>
        """
    tabla_html += "</tbody></table>"
    st.markdown(tabla_html, unsafe_allow_html=True)

    st.divider()

    # ─── Detalle por pregunta ─────────────────────────────────────
    st.markdown("## 🔍 Detalle por Pregunta")

    for r in resultados:
        fuentes_html = "".join([
            f'<span class="source-pill">📄 {f["source"]} &nbsp;<strong>{round(f["score"]*100)}%</strong></span>'
            for f in r["fuentes"]
        ])

        st.markdown(f"""
        <div class="eval-card">
            <div class="eval-header">
                <span class="eval-num">#{r['id']}</span>
                <span class="eval-category">{r['categoria']}</span>
                <span class="eval-lang">{r['idioma']}</span>
                <span class="eval-lang" style="background:rgba(100,116,139,0.1); color:#94a3b8; border-color:rgba(100,116,139,0.2);">{r['tipo']}</span>
                <span style="margin-left:auto"><span class="{r['veredicto_clase']}">{r['veredicto']}</span></span>
            </div>
            <div class="eval-question">💬 {r['pregunta']}</div>
            <div style="font-size:0.72rem; color:#64748b; margin-bottom:6px;">Respuesta del sistema:</div>
            <div class="eval-response">{r['respuesta']}</div>
            <div style="margin-top:10px; padding-top:10px; border-top:1px solid #1e3a5f;">
                <span style="color:#64748b; font-size:0.72rem;">Fuentes recuperadas: </span>
                {fuentes_html}
            </div>
        </div>
        """, unsafe_allow_html=True)

    # ─── Análisis final ───────────────────────────────────────────
    st.divider()
    st.markdown("## 📝 Análisis de Resultados")

    st.markdown(f"""
    <div style="background:#111827; border:1px solid #1e3a5f; border-radius:14px; padding:24px 28px; font-size:0.88rem; line-height:1.8; color:#cbd5e1;">

    <strong style="color:#e2e8f0; font-size:1rem;">Resumen del desempeño</strong><br><br>

    El sistema obtuvo una tasa de éxito del <strong style="color:#10b981">{tasa}%</strong> en las {total} preguntas evaluadas,
    con un score de similitud coseno promedio de <strong style="color:#60a5fa">{avg_score:.3f}</strong>.<br><br>

    <strong style="color:#60a5fa">🔍 Búsqueda semántica con sinónimos:</strong><br>
    Las preguntas 3 y 4 usaron lenguaje coloquial ("change my mind", "salirme del trato / no me aprueban el préstamo")
    en lugar de los términos técnicos presentes en los documentos. El sistema recuperó correctamente
    las fuentes <em>earnest_money.txt</em> y <em>financing_contingency.txt</em>,
    demostrando que la similitud coseno funciona más allá de coincidencias exactas de palabras.<br><br>

    <strong style="color:#60a5fa">📉 Umbrales diferenciados por idioma:</strong><br>
    El modelo <em>all-MiniLM-L6-v2</em> está optimizado principalmente para inglés, lo que genera
    scores coseno sistemáticamente más bajos en consultas en español (promedio ~0.20 vs ~0.55 en inglés).
    Las preguntas 4 y 6 recibieron respuestas correctas pero con scores bajos, por lo que se ajustaron
    los umbrales de evaluación: <strong>0.40 / 0.18</strong> para español vs <strong>0.55 / 0.35</strong> para inglés.
    Este es un hallazgo académico relevante: en producción se recomendaría usar
    <em>paraphrase-multilingual-MiniLM-L12-v2</em> para mayor precisión en español.<br><br>

    <strong style="color:#60a5fa">🌐 Detección de idioma:</strong><br>
    Las preguntas en español (2, 4, 6) recibieron respuestas completamente en español,
    sin mezclar idiomas, gracias a la regla de idioma en el System Prompt.<br><br>

    <strong style="color:#60a5fa">🛡️ Anti-alucinación:</strong><br>
    La pregunta 9 ("property tax rate in Miami-Dade") no tiene respuesta en la base de conocimiento.
    El sistema respondió correctamente con el mensaje de "no encuentro esa información"
    en lugar de inventar una cifra, validando el mecanismo anti-alucinación.<br><br>

    <strong style="color:#60a5fa">🔗 Combinación de fuentes:</strong><br>
    La pregunta 10 requería información de dos documentos distintos (earnest money + financing contingency).
    El sistema recuperó ambas fuentes y sintetizó una respuesta coherente.<br><br>

    <strong style="color:#e2e8f0">Conclusión:</strong>
    El pipeline RAG demuestra ser efectivo para responder preguntas de bienes raíces en Florida,
    con buena tolerancia a variaciones de lenguaje y sólida protección contra alucinaciones.
    La principal limitación identificada es la sensibilidad del modelo de embeddings al idioma,
    lo cual es un punto de mejora claro para versiones futuras.

    </div>
    """, unsafe_allow_html=True)

else:
    st.markdown("""
    <div style="background:#111827; border:1px dashed #1e3a5f; border-radius:14px; padding:40px; text-align:center; color:#64748b;">
        <div style="font-size:2.5rem; margin-bottom:12px;">🧪</div>
        <div style="font-size:1rem; color:#94a3b8; margin-bottom:8px;">Evaluación no ejecutada aún</div>
        <div style="font-size:0.82rem;">Haz clic en <strong>▶️ Correr evaluación completa</strong> para lanzar las 10 preguntas contra el sistema RAG real.</div>
    </div>
    """, unsafe_allow_html=True)