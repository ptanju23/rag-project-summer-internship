import math
import sqlite3
import json
from pathlib import Path
from foundry_local_sdk import Configuration, FoundryLocalManager

BASE_DIR = Path(__file__).resolve().parent

# Database setup
def create_db():
    connection = sqlite3.connect("metro.db")
    cursor = connection.cursor()
    cmd1 = "CREATE TABLE IF NOT EXISTS metro (id INTEGER PRIMARY KEY, document_name TEXT, chunk_text TEXT, embedding TEXT)"
    cursor.execute(cmd1)
    connection.commit()
    return cursor, connection

def save_to_db(doc_name, text, embedding, cursor, connection):
    cmd2 = "INSERT INTO metro (document_name, chunk_text, embedding) VALUES (?, ?, ?)"
    # Serialize embedding list to JSON string
    cursor.execute(cmd2, (doc_name, text, json.dumps(embedding)))
    connection.commit()

def close_db(cursor, connection): 
    cursor.close()
    connection.close()

def cosine_similarity(a, b):
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    return dot / (norm_a * norm_b) if norm_a and norm_b else 0.0

def find_relevant(query_embedding, cursor, top_k=3):
    """Retrieve top-k relevant text chunks from SQLite database."""
    cursor.execute("SELECT chunk_text, embedding FROM metro")
    rows = cursor.fetchall()
    
    scores = []
    for text, emb_str in rows:
        emb = json.loads(emb_str)
        score = cosine_similarity(query_embedding, emb)
        scores.append((text, score))
        
    scores.sort(key=lambda x: x[1], reverse=True)
    return scores[:top_k]

def open_and_load_documents():
    return [
        "metros.txt", "metro_m1a.txt", "metro_m1b.txt", "metro_m2.txt", 
        "metro_m3.txt", "metro_m4.txt", "metro_m5.txt", "metro_m6.txt", 
        "metro_m7.txt", "metro_m8.txt", "metro_m9.txt"
    ]

def load_text_document(file_path):
    path = Path(file_path)
    if not path.is_absolute():
        path = BASE_DIR / path

    if not path.exists():
        raise FileNotFoundError(f"Text document not found: {path}")

    return path.read_text(encoding="utf-8").strip()

def main():
    config = Configuration(app_name="foundry_local_rag")
    FoundryLocalManager.initialize(config)
    manager = FoundryLocalManager.instance

    cursor, connection = create_db()

    # Check if database already has indexed data
    cursor.execute("SELECT COUNT(*) FROM metro")
    count = cursor.fetchone()[0]

    # Load Embedding Model
    embedding_model = manager.catalog.get_model("qwen3-embedding-0.6b")
    embedding_model.download(
        lambda p: print(f"\rDownloading embedding model: {p:.1f}%", end="", flush=True)
    )
    print()
    embedding_model.load()
    embedding_client = embedding_model.get_embedding_client()

    # Index files if database is empty
    if count == 0:
        documents = open_and_load_documents()
        total_chunks = 0
        for doc_name in documents:
            try:
                text = load_text_document(doc_name)
                for line in text.splitlines():
                    chunk = line.strip()
                    if not chunk:
                        continue

                    response = embedding_client.generate_embedding(chunk)
                    embedding = response.data[0].embedding

                    save_to_db(doc_name, chunk, embedding, cursor, connection)
                    total_chunks += 1
            except FileNotFoundError as e:
                print(f"Skipping {doc_name}: {e}")
        print(f"Indexed {total_chunks} text chunks into SQLite.")
    else:
        print(f"Loaded existing database with {count} chunks.")

    # Load Chat Model
    chat_model = manager.catalog.get_model("qwen2.5-1.5b")
    chat_model.download(
        lambda p: print(f"\rDownloading chat model: {p:.1f}%", end="", flush=True)
    )
    print()
    chat_model.load()
    chat_client = chat_model.get_chat_client()

    print("\nModels loaded. Ready for questions.")
    print('Type "q" or "quit" to exit.\n')

    # Interaction Loop
    while True:
        query = input("Question: ").strip()
        if not query or query.lower() in ["quit", "q"]:
            break

        # Embed query & search SQLite
        query_response = embedding_client.generate_embedding(query)
        query_embedding = query_response.data[0].embedding

        results = find_relevant(query_embedding, cursor, top_k=3)
        context = "\n".join(f"- {text}" for text, score in results)

        messages = [
            {
                "role": "system",
                "content": (
                    "You are a helpful assistant for the Istanbul metro system. "
                    "Answer using only the provided context. "
                    "Do not make up answers. If you do not have the full context, say so."
                    "Strict rule: Do not answer questions that are not related to the Istanbul metro system. Redirect back to the context if the question is off-topic. "
                    "After you retrieve the best match, evaulate if it truly answers the question. If it does not, say 'I don't know'. "
                    "Avoid making up answers. If the context is insufficient, say 'I don't know'. "
                    "Do not answer questions that need matching context with other context. If the context is insufficient, say 'I don't know'. "
                    "If unsure, say 'I don't know'.\n\n"
                    f"Context:\n{context}"
                ),
            },
            {"role": "user", "content": query},
        ]

        print("Answer: ", end="", flush=True)
        for chunk in chat_client.complete_streaming_chat(messages):
        # Guard against empty choices array on terminating chunk
            if chunk.choices and len(chunk.choices) > 0:
                content = chunk.choices[0].delta.content
                if content:
                    print(content, end="", flush=True)
        print("\n")

    close_db(cursor, connection)
    embedding_model.unload()
    chat_model.unload()
    print("Models unloaded. Done!")

if __name__ == "__main__":
    main()