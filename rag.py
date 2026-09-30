# ==============================
# IMPORTS
# ==============================

import os

import numpy as np

import faiss

from sentence_transformers import SentenceTransformer

from google import genai

from langsmith import traceable


# Importer notre configuration
from config import (
    GOOGLE_API_KEY,
    DOCUMENTS_PATH,
    GEMINI_MODEL,
    EMBEDDING_MODEL,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    TOP_K
)


# ==============================
# CLIENT GEMINI
# ==============================

client = genai.Client(
    api_key=GOOGLE_API_KEY
)


# ==============================
# CHARGEMENT DES DOCUMENTS
# ==============================

documents = []


for filename in os.listdir(DOCUMENTS_PATH):

    if filename.endswith(".txt"):

        filepath = os.path.join(
            DOCUMENTS_PATH,
            filename
        )

        with open(
            filepath,
            "r",
            encoding="utf-8"
        ) as file:

            text = file.read()

        documents.append({
            "text": text,
            "source": filename
        })


print(
    f"{len(documents)} documents chargés."
)


# ==============================
# CHUNKING
# ==============================

def split_text(
    text,
    chunk_size=CHUNK_SIZE,
    overlap=CHUNK_OVERLAP
):

    chunks = []

    start = 0

    while start < len(text):

        end = start + chunk_size

        chunk = text[start:end]

        chunks.append(chunk)

        start += chunk_size - overlap

    return chunks


# ==============================
# CREATION DES CHUNKS
# ==============================

all_chunks = []


for document in documents:

    chunks = split_text(
        document["text"]
    )

    for i, chunk in enumerate(chunks):

        all_chunks.append({
            "text": chunk,
            "source": document["source"],
            "chunk_id": i
        })


print(
    f"{len(all_chunks)} chunks créés."
)


# ==============================
# MODELE D'EMBEDDINGS
# ==============================

embedding_model = SentenceTransformer(
    EMBEDDING_MODEL
)


# ==============================
# CREATION DES EMBEDDINGS
# ==============================

texts = [
    chunk["text"]
    for chunk in all_chunks
]


embeddings = embedding_model.encode(
    texts,
    normalize_embeddings=True
)


embeddings = np.array(
    embeddings,
    dtype="float32"
)


# ==============================
# CREATION DE L'INDEX FAISS
# ==============================

dimension = embeddings.shape[1]


index = faiss.IndexFlatIP(
    dimension
)


index.add(
    embeddings
)


print(
    f"{index.ntotal} vecteurs ajoutés à FAISS."
)


# ==============================
# APPEL GEMINI
# ==============================

@traceable(
    name="Appel Gemini"
)
def call_gemini(prompt):

    try:

        response = client.models.generate_content(
            model=GEMINI_MODEL,
            contents=prompt
        )

        return response.text

    except Exception as e:

        error_message = str(e)

        if "429" in error_message or "RESOURCE_EXHAUSTED" in error_message:

            return (
                "Le quota Gemini disponible pour ce projet "
                "a été atteint. Veuillez réessayer plus tard."
            )

        return (
            "Une erreur est survenue lors de l'appel à Gemini."
        )


# ==============================
# RECHERCHE FAISS
# ==============================

@traceable(
    name="Recherche FAISS"
)
def search_documents(
    question,
    k=TOP_K
):

    # Transformer la question
    # en embedding

    query_embedding = embedding_model.encode(
        [question],
        normalize_embeddings=True
    )


    query_embedding = np.array(
        query_embedding,
        dtype="float32"
    )


    # Recherche dans FAISS

    scores, indices = index.search(
        query_embedding,
        k
    )


    results = []


    for score, idx in zip(
        scores[0],
        indices[0]
    ):

        results.append({

            "text": all_chunks[idx]["text"],

            "source": all_chunks[idx]["source"],

            "chunk_id": all_chunks[idx]["chunk_id"],

            "score": float(score)

        })


    return results


# ==============================
# CONSTRUCTION DU CONTEXTE
# ==============================

@traceable(
    name="Construction du contexte"
)
def build_context(results):

    context = ""


    for result in results:

        context += (

            f"Source : {result['source']}\n"

            f"{result['text']}\n\n"

        )


    return context


# ==============================
# CREATION DU PROMPT
# ==============================

@traceable(
    name="Création du prompt"
)
def create_prompt(
    question,
    context
):

    prompt = f"""

Tu es l'assistant de NovaTech Solutions.

Tu dois répondre à la question uniquement à partir
des informations présentes dans le contexte fourni.

Règles :

1. N'invente aucune information.

2. Ne complète pas les informations manquantes.

3. Si la réponse n'est pas présente dans le contexte,
réponds exactement :

"Je ne trouve pas cette information dans les documents disponibles."

4. Réponds en français.

5. Sois clair et concis.

6. Si possible, indique la source utilisée.


CONTEXTE :

{context}


QUESTION :

{question}


RÉPONSE :

"""

    return prompt


# ==============================
# GARDE-FOU DE SECURITE
# ==============================

MOTS_INTERDITS = [

    "mot de passe",

    "password",

    "secret",

    "confidentiel",

    "voler un mot de passe",

    "identifiants",

    "credential"

]


def security_check(question):

    question_lower = question.lower()


    for word in MOTS_INTERDITS:

        if word in question_lower:

            return False


    return True


# ==============================
# PIPELINE RAG COMPLET
# ==============================

@traceable(
    name="NovaTech RAG"
)
def generate_answer(question):

    # --------------------------
    # 1. Vérification sécurité
    # --------------------------

    if not security_check(question):

        return {

            "answer":
            "Je ne peux pas fournir ou rechercher des informations sensibles.",

            "results": [],

            "context": "",

            "blocked": True

        }


    # --------------------------
    # 2. Recherche FAISS
    # --------------------------

    results = search_documents(
        question,
        k=TOP_K
    )


    # --------------------------
    # 3. Construction contexte
    # --------------------------

    context = build_context(
        results
    )


    # --------------------------
    # 4. Création prompt
    # --------------------------

    prompt = create_prompt(
        question,
        context
    )


    # --------------------------
    # 5. Appel Gemini
    # --------------------------

    answer = call_gemini(
        prompt
    )


    # --------------------------
    # 6. Résultat final
    # --------------------------

    return {

        "answer": answer,

        "results": results,

        "context": context,

        "blocked": False

    }