from load_data import load_event_data
from src.feature_engineering import create_event_documents
from src.embedding import get_embedding_model
from src.vectordb import create_vector_db


def main():
    df = load_event_data()
    documents = create_event_documents(df)

    embedding_model = get_embedding_model()
    create_vector_db(documents, embedding_model)


if __name__ == "__main__":
    main()
