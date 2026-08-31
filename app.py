import streamlit as st

# Import only functions that already exist in main.py
from main import create_db, find_relevant
from foundry_local_sdk import Configuration, FoundryLocalManager


st.set_page_config(
    page_title="Istanbul Metro Assistant",
    page_icon="🚇",
    layout="centered"
)


@st.cache_resource(show_spinner="Loading AI models...")
def init_system():
    # Initialize Foundry Local exactly like main.py
    config = Configuration(app_name="foundry_local_rag")
    FoundryLocalManager.initialize(config)

    manager = FoundryLocalManager.instance

    # Database
    cursor, connection = create_db()

    # Embedding model
    embedding_model = manager.catalog.get_model("qwen3-embedding-0.6b")
    embedding_model.download()
    embedding_model.load()
    embedding_client = embedding_model.get_embedding_client()

    # Chat model
    chat_model = manager.catalog.get_model("qwen2.5-1.5b")
    chat_model.download()
    chat_model.load()
    chat_client = chat_model.get_chat_client()

    return (
        cursor,
        connection,
        embedding_model,
        embedding_client,
        chat_model,
        chat_client
    )


try:
    (
        cursor,
        connection,
        embedding_model,
        embedding_client,
        chat_model,
        chat_client
    ) = init_system()

    st.sidebar.success("Metro System Ready")

except Exception as e:
    st.error("Error initializing the Metro RAG system.")
    st.exception(e)
    st.stop()


st.title("🚇 Istanbul Metro RAG Assistant")

with st.form(key="metro_form"):
    user_query = st.text_input(
        "Enter your question:",
        placeholder="e.g., Name all stations in M2"
    )

    submit_button = st.form_submit_button("Ask Assistant")


if submit_button and user_query.strip():

    with st.spinner("Searching metro knowledge base..."):

        # Generate embedding exactly like main.py
        query_response = embedding_client.generate_embedding(user_query)
        query_embedding = query_response.data[0].embedding

        # Dynamic top_k
        list_keywords = [
            "all",
            "list",
            "stations",
            "name",
            "route",
            "every"
        ]

        is_list_question = any(
            keyword in user_query.lower()
            for keyword in list_keywords
        )

        top_k = 6 if is_list_question else 3

        # Search database exactly using main.py function
        results = find_relevant(
            query_embedding,
            cursor,
            top_k=top_k
        )

        if not results:

            st.warning("No relevant information found in the database.")

        else:

            # main.py returns (text, score) tuples
            context = "\n".join(
                f"- {text}"
                for text, score in results
            )

            # Same prompt structure as main.py
            messages = [
                {
                    "role": "system",
                    "content": (
                        "You are a helpful assistant for the Istanbul metro system. "
                        "Answer using only the provided context. "
                        "Do not make up answers. If you do not have the full context, say so. "
                        "Strict rule: Do not answer questions that are not related to the Istanbul metro system. "
                        "Redirect back to the context if the question is off-topic. "
                        "After you retrieve the best match, evaluate if it truly answers the question. "
                        "If it does not, say 'I don't know'. "
                        "Avoid making up answers. "
                        "If the context is insufficient, say 'I don't know'. "
                        "Do not answer questions that need matching context with other context. "
                        "If the context is insufficient, say 'I don't know'. "
                        "If unsure, say 'I don't know'.\n\n"
                        f"Context:\n{context}"
                    )
                },
                {
                    "role": "user",
                    "content": user_query
                }
            ]

            st.markdown("### Answer")

            answer_placeholder = st.empty()
            full_answer = ""

            # Stream response exactly like main.py
            for chunk in chat_client.complete_streaming_chat(messages):

                if chunk.choices and len(chunk.choices) > 0:

                    content = chunk.choices[0].delta.content

                    if content:
                        full_answer += content
                        answer_placeholder.markdown(full_answer + "▌")

            # Final answer without cursor
            answer_placeholder.markdown(full_answer)

            # Debug section
            with st.expander("Retrieved Chunks (Debug)"):

                for index, (text, score) in enumerate(results, start=1):

                    st.markdown(
                        f"**[{index}] Score: {score:.4f}**"
                    )

                    st.text(text)