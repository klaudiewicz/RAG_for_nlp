import sys
import os
import json

# Dodanie ścieżki, aby zaimportować config
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from config import (
    ES_INDEX_NAME, 
    QDRANT_COLLECTION_NAME, 
    es, 
    client
)

def check_elasticsearch():
    print(f"\n=== KONTROLA ELASTICSEARCH (Index: {ES_INDEX_NAME}) ===")
    try:
        # 1. Sprawdź liczbę dokumentów
        count = es.count(index=ES_INDEX_NAME)['count']
        print(f"Liczba dokumentów w indeksie: {count}")
        
        if count == 0:
            print("[WARN] Indeks jest pusty!")
            return

        # 2. Pobierz próbkę
        response = es.search(index=ES_INDEX_NAME, size=1)
        hit = response['hits']['hits'][0]
        source = hit['_source']
        
        print("\n--- Przykładowy dokument (ES) ---")
        print(f"ID: {hit['_id']}")
        print(f"Autor: {source.get('author')}")
        print(f"Temat: {source.get('topic')}")
        print(f"Data (z pliku): {source.get('date')}")
        print(f"Lata (wykryte): {source.get('years')}")
        print(f"Miejsca (wykryte): {source.get('places')}")
        print(f"Fragment tekstu: {source.get('text', '')[:150]}...")
        
    except Exception as e:
        print(f"[BŁĄD ES] {e}")

def check_qdrant():
    print(f"\n=== KONTROLA QDRANT (Collection: {QDRANT_COLLECTION_NAME}) ===")
    try:
        # 1. Sprawdź liczbę punktów
        info = client.get_collection(collection_name=QDRANT_COLLECTION_NAME)
        print(f"Liczba wektorów w kolekcji: {info.points_count}")
        print(f"Status kolekcji: {info.status}")

        if info.points_count == 0:
            print("[WARN] Kolekcja jest pusta!")
            return

        # 2. Pobierz próbkę (scroll)
        points, _ = client.scroll(
            collection_name=QDRANT_COLLECTION_NAME,
            limit=1,
            with_vectors=True,
            with_payload=True
        )
        
        if points:
            point = points[0]
            print("\n--- Przykładowy punkt (Qdrant) ---")
            print(f"ID: {point.id}")
            # Sprawdzamy czy wektor istnieje i jaką ma długość (powinno być np. 384 lub 768)
            vector_len = len(point.vector) if point.vector else 0
            print(f"Długość wektora: {vector_len} (OK)" if vector_len > 0 else "Długość wektora: 0 (BŁĄD)")
            print(f"Payload (fragment): {str(point.payload)[:150]}...")
        else:
            print("Nie udało się pobrać punktów.")

    except Exception as e:
        print(f"[BŁĄD QDRANT] {e}")

if __name__ == "__main__":
    check_elasticsearch()
    check_qdrant()
    print("\nTest zakończony.")