from pathlib import Path
from langchain_chroma import Chroma

DB_PATH = Path("db/chroma")


def create_vector_db(documents, embedding_model):
    """
    Create and persist Chroma vector database.
    """
    db = Chroma.from_documents(
        documents=documents,
        embedding=embedding_model,
        persist_directory=str(DB_PATH)
    )

    print("✅ Vector DB created and persisted")
    return db


def load_vector_db(embedding_model):
    """
    Load an existing Chroma vector database.
    """
    return Chroma(
        persist_directory=str(DB_PATH),
        embedding_function=embedding_model
    )
