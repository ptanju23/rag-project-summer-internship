# Istanbul Metro RAG System

This repository implements a Retrieval-Augmented Generation (RAG) system for the Istanbul Metro system dataset. It uses Foundry Local for embedding and chat models, SQLite for a simple offline database, and StreamLit for a simple app UI.

## Project Structure

* **`requirements.txt`**: Lists the required resources for this project code to work.
* **`metro txt files`**: Contains the structured, RAG-optimized metro line text documentation to be ingested into the database.
* **`main.py`**: Implements the core technical pipeline, including generating embeddings, storing vectors in SQL, retrieving relevant context via vector search, and generating the LLM response.
* **`app.py`**: Provides a simple user interface with a dark/black theme while reusing the backend execution logic from `main.py`.

## How to Run the Project

1. Run the core script in your terminal:
   ```bash
   python main.py
   
Executing main.py automatically creates metro.db, the SQLite database for the project.
Once initialized, the model will prompt you in the terminal to enter a query.
You can continue interacting with the model directly in the terminal, or type q or quit to exit.

2. Run the app in your terminal:
   ```bash
   streamlit run app.py
This will cause a tab to open it your browser that contains the simple UI for the project. You may interact with the model here, and close the tab or quit using your terminal when you are done.

Note that for this project to run, your python must have streamlit and foundry local downloaded. If that is not the case, you may download using the following command:
   ```bash
   pip install streamlit
   pip install foundry-local-sdk
