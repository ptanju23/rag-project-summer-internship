# Istanbul Metro RAG System

This repository implements a Retrieval-Augmented Generation (RAG) system for the Istanbul Metro system dataset. It uses Foundry Local for embedding and chat models, SQLite for a simple offline database, and StreamLit for a simple app ui.

## Project Structure

* **`requirements.txt`**: Lists the required resources for this project code to work.
* **`metro txt files`**: Contains the structured, RAG-optimized metro line text documentation to be ingested into the database.
* **`main.py`**: Implements the core technical pipeline, including generating embeddings, storing vectors in SQL, retrieving relevant context via vector search, and generating the LLM response.
* **`app.py`**: Provides a simple user interface with a dark/black theme while reusing the backend execution logic from `main.py`.
