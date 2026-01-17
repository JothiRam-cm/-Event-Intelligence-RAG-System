import re
from src.embedding import get_embedding_model
from src.vectordb import load_vector_db


def normalize_id(value):
    if value is None:
        return None
    try:
        return str(int(float(value)))
    except:
        return None


def extract_component_id(query: str):
    match = re.search(r'component\s+(\d+)', query.lower())
    return match.group(1) if match else None


def get_relevant_events(query: str, k: int = 5):
    embedding_model = get_embedding_model()
    db = load_vector_db(embedding_model)

    component_id = normalize_id(extract_component_id(query))

    # ✅ METADATA PRE-FILTER
    if component_id:
        docs = db.similarity_search(
            query,
            k=k,
            filter={"component_id": {"$in": [component_id, f"{component_id}.0"]}}
        )
    else:
        docs = db.similarity_search(query, k=k)

    return docs
