import os
from flask import Flask, request, jsonify, render_template
from google import genai
from dotenv import load_dotenv
from rag_system import cargar_documentos, buscar_contexto

load_dotenv()

client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

app = Flask(__name__)

documentos = cargar_documentos()

chat_history = []


def construir_prompt(pregunta, contexto):

    prompt = f"""
You are a Legal Assistant specialized in Florida Real Estate transactions.

Your role:
Help users understand the process of buying houses in Florida.

LANGUAGE RULE:
- Detect the language of the user's question.
- If the question is in Spanish, respond ONLY in Spanish.
- If the question is in English, respond ONLY in English.

Rules:
- Use the provided context when it is relevant.
- If the context contains useful information, prioritize it.
- If the question is general (like greetings), respond normally.
- You may use general knowledge about real estate if needed.
- Do not invent legal facts that contradict the context.
- Always answer in Spanish first and then in English.

FORMAT:

Respuesta en Español:
...

Answer in English:
...

CONTEXT:
\"\"\"
{contexto}
\"\"\"

Examples:

Question: What happens during the inspection period?

Answer:
During the inspection period the buyer has the opportunity to evaluate the
condition of the property. The buyer may hire professional inspectors and
request repairs from the seller. If major issues are discovered, the buyer
may cancel the contract depending on the terms of the agreement.


Question: What is the closing process in real estate?

Answer:
The closing process is the final stage of a real estate transaction where
ownership of the property is officially transferred from the seller to the
buyer. During closing, legal documents are signed, payments are completed,
and the property title is transferred.

Question: What is an earnest money deposit?

Answer:
An earnest money deposit is a payment made by the buyer to show serious intent
to purchase a property. It is usually held in escrow until the closing process
and may be applied toward the purchase price or closing costs.

QUESTION:
{pregunta}

ANSWER:
"""

    return prompt


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/chat", methods=["POST"])
def chat():

    pregunta = request.json["mensaje"]

    contexto = buscar_contexto(pregunta, documentos)

    prompt = construir_prompt(pregunta, contexto)

    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt
    )

    respuesta = response.text

    chat_history.append({
        "pregunta": pregunta,
        "respuesta": respuesta
    })

    return jsonify({
        "respuesta": respuesta,
        "historial": chat_history
    })


if __name__ == "__main__":
    app.run(debug=True)