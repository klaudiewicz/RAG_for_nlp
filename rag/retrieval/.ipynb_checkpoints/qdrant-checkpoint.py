import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from config import QDRANT_COLLECTION_NAME, client, model_emb

def get_embedding(text, is_query=True):
    prefix = "query: " if is_query else "passage: "
    return model_emb.encode(prefix + text, normalize_embeddings=True).tolist()

def search_qdrant(query_text, collection_name=QDRANT_COLLECTION_NAME, limit=15):
    query_vector = get_embedding(query_text, is_query=True)
    
    res = client.query_points(
        collection_name=collection_name,
        query=query_vector,
        limit=limit,
        with_payload=True
    )
    return res.points if hasattr(res, "points") else res

