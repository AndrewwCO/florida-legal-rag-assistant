import os

def cargar_documentos():

    documentos = []

    carpeta = "knowledge"

    for archivo in os.listdir(carpeta):

        ruta = os.path.join(carpeta, archivo)

        with open(ruta, "r", encoding="utf-8") as f:
            texto = f.read()

            documentos.append(texto)

    return documentos


def buscar_contexto(pregunta, documentos):

    pregunta = pregunta.lower()

    resultados = []

    for doc in documentos:

        doc_lower = doc.lower()

        coincidencias = 0

        for palabra in pregunta.split():

            if palabra in doc_lower:
                coincidencias += 1

        if coincidencias >= 2:  # mínimo coincidencias
            resultados.append(doc)

    if resultados:
        return "\n\n".join(resultados[:2])

    # fallback → enviar algo aunque no haya coincidencia fuerte
    return documentos[0]