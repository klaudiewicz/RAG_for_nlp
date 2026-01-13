import uuid
import sys
import os
import requests
import pandas as pd
from elasticsearch import helpers
from qdrant_client.models import Distance, VectorParams, PointStruct

# Ustawienie ścieżki
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Próba importu (z obsługą braku modułu)
try:
    from date_extraction_hybrid import hybrid_date_extraction
except ImportError:
    def hybrid_date_extraction(text): return {"final_years": []}

from config import (
    ES_HOST, 
    ES_INDEX_NAME, 
    QDRANT_COLLECTION_NAME, 
    DATA_PATH, 
    BATCH_SIZE,
    VECTOR_SIZE,
    ner_worker,
    model_emb,
    es,
    client
)

# --- MAPOWANIE ELASTICSEARCH ---
index_body = {
    "settings": {
        "analysis": {
            "analyzer": {
                "pl_lemma": {
                    "tokenizer": "standard",
                    "filter": ["lowercase"]
                }
            }
        }
    },
    "mappings": {
        "properties": {
            "id": { "type": "keyword" },
            "author": { "type": "keyword" },
            "topic": { "type": "keyword" },
            "date": { "type": "date" },
            "text": { 
                "type": "text", 
                "analyzer": "pl_lemma" 
            },
            "vector": {
                "type": "dense_vector",
                "dims": VECTOR_SIZE,            
                "index": True,            
                "similarity": "cosine" 
            },
            "named_entities": { "type": "keyword" }, 
            "places": { "type": "keyword" },            
            "years": { "type": "integer" }
        }
    }
}

def get_embedding(text, is_query=False, normalize_embeddings=True):
    prefix = "query: " if is_query else "passage: "
    text_with_prefix = prefix + text
    return model_emb.encode(text_with_prefix, normalize_embeddings=normalize_embeddings, device='cuda')

def create_es_index(index_name=ES_INDEX_NAME):
    url = f"{ES_HOST}/{index_name}"
    try:
        requests.delete(url)
    except:
        pass
    response = requests.put(url, json=index_body)
    if response.status_code != 200:
        print(f"Błąd tworzenia indeksu ES: {response.text}")

def run_es_indexing(es_client, docs_list, index_name=ES_INDEX_NAME):
    actions = [
        {
            "_index": index_name,
            "_id": doc["id"],
            "_source": doc
        } 
        for doc in docs_list
    ]
    helpers.bulk(es_client, actions)

def ensure_qdrant_collection(qdrant_client, collection_name=QDRANT_COLLECTION_NAME):
    try:
        qdrant_client.get_collection(collection_name)
    except:
        print(f"[Qdrant] Tworzenie kolekcji '{collection_name}'...")
        qdrant_client.recreate_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE)
        )

def prepare_metadata(text):
    try:
        entities = ner_worker(text)
        named_entities = []
        places = []
        for ent in entities:
            word = ent['word'].replace("##", "") 
            label = ent['entity_group']
            if label == 'LOC':
                places.append(word)
            else:
                named_entities.append(word)
        
        date_results = hybrid_date_extraction(text)
        years = [int(y) for y in date_results.get('final_years', []) if str(y).isdigit()]

        return {
            "named_entities": list(set(named_entities)),
            "places": list(set(places)),
            "years": list(set(years))
        }
    except Exception:
        return {"named_entities": [], "places": [], "years": []}

# --- NOWA FUNKCJA DO NAPRAWIANIA ID ---
def sanitize_id_for_qdrant(id_val):
    """
    Qdrant akceptuje tylko unsigned int lub poprawne UUID.
    Jeśli ID jest stringiem z błędami (np. '8c1g...'), zamieniamy go na hash UUID.
    """
    id_str = str(id_val)
    
    # 1. Jeśli to liczba (np. "1", "100"), jest OK
    if id_str.isdigit():
        return int(id_str)
    
    # 2. Sprawdź czy to poprawny UUID
    try:
        uuid_obj = uuid.UUID(id_str)
        return str(uuid_obj)
    except ValueError:
        # 3. Jeśli to błędny string (np. "8c1g..." lub "pytanie_1"), generujemy deterministyczny UUID
        return str(uuid.uuid5(uuid.NAMESPACE_DNS, id_str))

def index_batch(docs_list):
    """Wysyła paczkę do ES i Qdrant."""
    if not docs_list:
        return

    # 1. Elasticsearch (przyjmuje dowolne stringi jako ID)
    run_es_indexing(es, docs_list, ES_INDEX_NAME)

    # 2. Qdrant (wymaga sanityzacji ID)
    points = []
    for d in docs_list:
        safe_id = sanitize_id_for_qdrant(d["id"])
        
        points.append(PointStruct(
            id=safe_id,
            vector=d["vector"],
            payload={k: v for k, v in d.items() if k != "vector"}
        ))
    
    client.upsert(collection_name=QDRANT_COLLECTION_NAME, points=points)


# --- GŁÓWNA FUNKCJA ---
def create_indices(file_path=DATA_PATH):
    if not os.path.exists(file_path):
        print(f"[ERROR] Plik {file_path} nie istnieje.")
        return

    print(f"Wczytywanie danych z: {file_path}")
    df = pd.read_json(file_path, lines=True)

    create_es_index(ES_INDEX_NAME)
    ensure_qdrant_collection(client, QDRANT_COLLECTION_NAME)

    current_batch = []
    total_count = len(df)
    processed_count = 0
    
    print(f"Rozpoczynam przetwarzanie {total_count} rekordów...")
    
    for _, row in df.iterrows():
        try:
            text = row['text']
            vector = get_embedding(text)
            metadata = prepare_metadata(text)
            
            # Pobierz ID z pliku lub wygeneruj nowe
            raw_id = str(row['id']) if 'id' in row and pd.notna(row['id']) else str(uuid.uuid4())

            doc = {
                "id": raw_id,  # Tutaj trzymamy oryginał (może być błędny, naprawimy przy wysyłce do Qdrant)
                "text": text,
                "author": row.get('author'),
                "topic": row.get('topic'),
                "date": row.get('date'),
                "label": row.get('label'),
                "vector": vector.tolist(),
                "named_entities": metadata["named_entities"],
                "places": metadata["places"],
                "years": metadata["years"]
            }
            
            current_batch.append(doc)
            
            if len(current_batch) >= BATCH_SIZE:
                index_batch(current_batch)
                processed_count += len(current_batch)
                print(f"[PROGRESS] Przetworzono {processed_count}/{total_count}...")
                current_batch = [] 

        except Exception as e:
            print(f"[BŁĄD] Wiersz pominięty: {e}")
            continue

    if current_batch:
        index_batch(current_batch)
        print(f"[PROGRESS] Zapisano ostatnie {len(current_batch)} dokumentów.")
    
    print("Indeksowanie zakończone pomyślnie!")

if __name__ == "__main__":
    create_indices()