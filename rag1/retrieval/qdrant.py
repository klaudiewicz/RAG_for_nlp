from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer

# Inicjalizacja klienta i modelu 
client = QdrantClient("localhost", port=6333)
model = SentenceTransformer("intfloat/multilingual-e5-small") 

def get_embedding(text, is_query=False):
    prefix = "query: " if is_query else "passage: "
    return model.encode(prefix + text, normalize_embeddings=True)

def search_qdrant(query_text, collection_name="culturax_semantic", limit=15):
    query_vector = get_embedding(query_text, is_query=True).tolist()
    
    res = client.query_points(
        collection_name=collection_name,
        query=query_vector,
        limit=limit,
        with_payload=True
    )
    return res.points if hasattr(res, "points") else res