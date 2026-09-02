import streamlit as st

from main import (
    create_db,
    find_relevant,
    open_and_load_documents,
    load_text_document,
    save_to_db,
)

from foundry_local_sdk import Configuration, FoundryLocalManager


# ---------------------------------------------------------
# PAGE
# ---------------------------------------------------------

st.set_page_config(
    page_title="Istanbul Metro Assistant",
    layout="centered"
)


# ---------------------------------------------------------
# UI
# ---------------------------------------------------------

st.markdown(
    """
    <style>

    html, body, [class*="css"] {
        background-color: black !important;
        color: white !important;
        font-size: 16pt !important;
    }

    .stApp {
        background-color: black;
        color: white;
    }

    p, div, span, label {
        font-size: 16pt !important;
        color: white !important;
    }

    .stTextInput input {
        background-color: black !important;
        color: white !important;
        border: 1px solid white !important;
        font-size: 16pt !important;
    }

    .stButton button {
        background-color: black !important;
        color: white !important;
        border: 1px solid white !important;
        font-size: 16pt !important;
    }

    .stButton button:hover {
        border: 1px solid white !important;
        color: white !important;
    }

    header {
        visibility: hidden;
    }

    footer {
        visibility: hidden;
    }

    #MainMenu {
        visibility: hidden;
    }

    </style>
    """,
    unsafe_allow_html=True
)

# ---------------------------------------------------------
# INITIALIZE
# ---------------------------------------------------------

@st.cache_resource
def initialize_system():

    config = Configuration(
        app_name="foundry_local_rag"
    )

    FoundryLocalManager.initialize(config)

    manager = FoundryLocalManager.instance


    # -------------------------
    # EMBEDDING MODEL
    # -------------------------

    embedding_model = manager.catalog.get_model(
        "qwen3-embedding-0.6b"
    )

    embedding_model.download()
    embedding_model.load()

    embedding_client = (
        embedding_model.get_embedding_client()
    )


    # -------------------------
    # CHAT MODEL
    # -------------------------

    chat_model = manager.catalog.get_model(
        "qwen2.5-1.5b"
    )

    chat_model.download()
    chat_model.load()

    chat_client = (
        chat_model.get_chat_client()
    )


    return embedding_client, chat_client

try:

    embedding_client, chat_client = initialize_system()

except Exception as e:

    st.error(str(e))
    st.stop()

# ---------------------------------------------------------
# DATABASE CONNECTION
# ---------------------------------------------------------

cursor, connection = create_db()


# ---------------------------------------------------------
# INTRO TEXT
# ---------------------------------------------------------

st.markdown(
    """
This model answers questions about the Istanbul metro system. 
It uses a local database of text chunks to provide context for its answers. 
Please ask questions related to the Istanbul metro system.
        
EXAMPLE QUESTIONS:

How many stations are there in the M1A line?

What is the line identifier for the M5 line?

What is the M6 line's total length in km?

What are the operating hours of the M2 line?

Typos and misspelling may cause the model to not find the correct answer. Please check your spelling and try again.

"""
)


# ---------------------------------------------------------
# QUESTION
# ---------------------------------------------------------

st.markdown("### Question")

question = st.text_input(
    "Question",
    label_visibility="collapsed"
)


ask = st.button(
    "Ask"
)


# ---------------------------------------------------------
# ANSWER
# ---------------------------------------------------------

st.markdown("### Answer")

answer_area = st.empty()


if ask and question.strip():

    query = question.strip()


    # -----------------------------------------------------
    # EMBED QUERY
    # -----------------------------------------------------

    query_response = (
        embedding_client
        .generate_embedding(query)
    )

    query_embedding = (
        query_response.data[0].embedding
    )


    # -----------------------------------------------------
    # SEARCH DATABASE
    # -----------------------------------------------------

    results = find_relevant(
        query_embedding,
        cursor,
        top_k=3
    )

    context = "\n".join(
    f"- {text}"
    for text, score in results
)

    # -----------------------------------------------------
    # SAME PROMPT LOGIC AS MAIN.PY
    # -----------------------------------------------------

    messages = [
        {
            "role": "system",
            "content": (
                    "You are a helpful assistant for the Istanbul metro system. "
                    "Answer using only the provided context. "
                    "Do not make up answers. If you do not have the full context, say so. "
                    "Strict rule: Do not answer questions that are not related to the Istanbul metro system. Redirect back to the context if the question is off-topic. "
                    "Strict rule: never invent or change any answers. If the answer is fully in the context, answer exactly it. If the answer is not fully in the context, say so. "
                    "Strict rule: if the questions ask for a specific thing that doesn't exist in the context, say 'I don't have that information'. "
                    "Strict rule: if the questions asks for a station list, ask for a specific station number. If the number given does not match the context, say 'I don't know'. "
                    "After you retrieve the best match, evaulate if it truly answers the question. If it does not, say so. "
                    "Do not make up answers. If the context is insufficient, say so. "
                    "For 'who' 'what' 'when' 'where' 'why' 'how' questions, if the context does not provide an absolutely exact answer, say 'I don't have that information'. "
                    "Never convert related facts into an answer."
                    "You may format the vector daatabas context to make it easier to read, but do not change the meaning of the context. "
                    "Do not answer questions that need matching context with other context. If the context is insufficient, say 'I don't have that information'. "
                    "If unsure, say 'I don't have that information'.\n\n"
                f"Context:\n{context}"
            ),
        },

        {
            "role": "user",
            "content": query
        },
    ]


    # -----------------------------------------------------
    # STREAM ANSWER
    # -----------------------------------------------------

    full_answer = ""


    for chunk in (
        chat_client
        .complete_streaming_chat(messages)
    ):

        if (
            chunk.choices
            and len(chunk.choices) > 0
        ):

            content = (
                chunk
                .choices[0]
                .delta
                .content
            )

            if content:

                full_answer += content

                answer_area.markdown(
                    full_answer + "▌"
                )


    answer_area.markdown(
        full_answer
    )