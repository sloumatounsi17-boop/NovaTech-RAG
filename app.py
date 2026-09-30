# ==============================
# IMPORTS
# ==============================

import gradio as gr

from rag import generate_answer


# ==============================
# FONCTION CHATBOT
# ==============================

def chatbot(question):

    # Appel du pipeline RAG
    result = generate_answer(question)


    # Récupérer la réponse
    answer = result["answer"]


    # --------------------------
    # Si la requête est bloquée
    # --------------------------

    if result["blocked"]:

        sources = (
            "Recherche bloquée par "
            "le garde-fou de sécurité."
        )


    # --------------------------
    # Sinon afficher les sources
    # --------------------------

    else:

        sources = ""


        for r in result["results"]:

            sources += (

                f"Source : {r['source']} | "

                f"Chunk : {r['chunk_id']} | "

                f"Score : {r['score']:.4f}\n"

            )


    return answer, sources


# ==============================
# CREATION DE L'INTERFACE
# ==============================

with gr.Blocks(
    title="NovaTech RAG"
) as demo:


    # --------------------------
    # Titre
    # --------------------------

    gr.Markdown(
        """
        # 🤖 NovaTech RAG Assistant

        Posez une question concernant
        NovaTech Solutions.
        """
    )


    # --------------------------
    # Question
    # --------------------------

    question = gr.Textbox(

        label="Votre question",

        placeholder=(
            "Exemple : "
            "Combien de jours de congé "
            "sont disponibles ?"
        )

    )


    # --------------------------
    # Bouton
    # --------------------------

    button = gr.Button(
        "🔍 Rechercher"
    )


    # --------------------------
    # Réponse
    # --------------------------

    answer = gr.Textbox(

        label="Réponse",

        lines=6

    )


    # --------------------------
    # Sources
    # --------------------------

    sources = gr.Textbox(

        label="Sources récupérées",

        lines=5

    )


    # --------------------------
    # Action du bouton
    # --------------------------

    button.click(

        fn=chatbot,

        inputs=question,

        outputs=[
            answer,
            sources
        ]

    )


# ==============================
# LANCEMENT DE GRADIO
# ==============================

if __name__ == "__main__":

    demo.launch(

        server_name="0.0.0.0",

        server_port=7860

    )