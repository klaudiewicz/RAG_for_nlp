import sys
import os
import re
import json
import csv
from datetime import datetime

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from retrieval.elastic import search_es
from retrieval.qdrant import search_qdrant
from retrieval.fusion import retrieve_adaptive
from config import es, ES_INDEX_NAME

# Funkcja pomocnicza do RegEx
def search_regex_in_es(pattern, index_name=ES_INDEX_NAME, limit=20):
    body = {
        "query": {
            "match_all": {}
        },
        "size": 100 
    }
    res = es.search(index=index_name, body=body)
    hits = res['hits']['hits']
    
    matches = []
    for hit in hits:
        text = hit['_source']['text']
        if re.search(pattern, text, re.IGNORECASE):
            matches.append(hit['_source'])
    return matches[:limit]

# Funkcja do filtrowania po metadanych 
def filter_results(docs, min_year=2018, required_entity=None):
    filtered = []
    
    for doc in docs:
        meta = doc.get('metadata', doc)
        
        # 1. Zbieramy wszystkie możliwe lata z dokumentu
        candidate_years = []
        
        # A) Z pola 'years'
        extracted_years = meta.get('years', [])
        if isinstance(extracted_years, list):
            candidate_years.extend(extracted_years)
            
        # B) Z pola 'date'
        date_str = meta.get('date')
        if date_str:
            found_year = re.search(r'\d{4}', str(date_str))
            if found_year:
                candidate_years.append(int(found_year.group()))

        # 2. Czyścimy listę lat
        valid_years = []
        for y in candidate_years:
            try:
                valid_years.append(int(y))
            except (ValueError, TypeError):
                continue
        
        # 3. Sprawdzamy warunek roku
        has_valid_year = False
        if valid_years:
            has_valid_year = any(y >= min_year for y in valid_years)
        
        # 4. Sprawdzamy warunek encji
        has_valid_entity = True
        if required_entity:
            entities = meta.get('named_entities', [])
            has_valid_entity = any(required_entity.lower() in str(e).lower() for e in entities)
            
        if has_valid_year and has_valid_entity:
            doc['debug_year'] = valid_years 
            filtered.append(doc)
            
    return filtered

def run_experiment():
    # Lista do zbierania danych do CSV
    csv_rows = []
    csv_headers = ["Temat", "Metoda", "Odpowiedz"]

    topics = [
        {
            "query": "Pamięć, kontekst i mechanizm uwagi",
            "regex": r"(?:pamię[ćc]|kontekst|atencj|attention|LSTM|RNN|sekwencj)", 
            "filter_entity": "Aleksander Smywiński-Pohl",
        },
        {
            "query": "Symbol, znaczenie i interpretacja",
            "regex": r"(?:symbol|znaczeni|semanty|token|embedding|reprezentacj)", 
            "filter_entity": None
        },
        {
            "query": "Struktura języka i reguły formalne",
            "regex": r"(?:składni|gramaty|fleksj|regex|reguł|walencj)",
            "filter_entity": None
        }
    ]

    for t in topics:
        q = t["query"]
        pattern = t["regex"]
        print(f"\n{'='*60}")
        print(f"TEMAT: {q}")
        print(f"{'='*60}")

        # --- 1. RegExp ---
        print(f"\n--- [1] RegExp (wzorzec: '{pattern}') ---")
        regex_docs = search_regex_in_es(pattern)
        if regex_docs:
            for i, d in enumerate(regex_docs[:3]):
                print(f"[{i+1}] {d['text'][:100]}...")
                csv_rows.append([q, "RegExp", d['text']])
        else:
            print("Brak dopasowań regex w próbce.")
            csv_rows.append([q, "RegExp", "BRAK WYNIKÓW"])

        # --- 2. BM25 - Elasticsearch ---
        print(f"\n--- [2] BM25 (Elasticsearch) ---")
        es_docs = search_es(q, limit=3)
        if es_docs:
            for i, hit in enumerate(es_docs):
                src = hit['_source']
                print(f"[{i+1}] (Score: {hit['_score']:.2f}) {src['text'][:100]}...")
                csv_rows.append([q, "BM25 (ES)", src['text']])
        else:
            csv_rows.append([q, "BM25 (ES)", "BRAK WYNIKÓW"])

        # --- 3. Vector Search (Qdrant) ---
        print(f"\n--- [3] Vector Search (Qdrant) ---")
        qdrant_docs = search_qdrant(q, limit=3)
        if qdrant_docs:
            for i, point in enumerate(qdrant_docs):
                payload = point.payload
                print(f"[{i+1}] (Score: {point.score:.2f}) {payload['text'][:100]}...")
                csv_rows.append([q, "Vector (Qdrant)", payload['text']])
        else:
            csv_rows.append([q, "Vector (Qdrant)", "BRAK WYNIKÓW"])

        # --- 4. Hybryda (RRF) ---
        print(f"\n--- [4] Hybryda (RRF) ---")
        hybrid_docs, _, _, _ = retrieve_adaptive(q)
        top_hybrid = hybrid_docs[:3]
        if top_hybrid:
            for i, doc in enumerate(top_hybrid):
                print(f"[{i+1}] (Score: {doc['score']:.4f}) {doc['text'][:100]}...")
                csv_rows.append([q, "Hybryda (RRF)", doc['text']])
        else:
            csv_rows.append([q, "Hybryda (RRF)", "BRAK WYNIKÓW"])

        # --- 5. Filtrowanie (Nowe pola) ---
        print(f"\n--- [5] Filtrowanie Hybrydy (Lata >= 2025) ---")
        raw_hybrid_for_filter, _, _, _ = retrieve_adaptive(q) 
        filtered = filter_results(raw_hybrid_for_filter, min_year=2025, required_entity=t["filter_entity"])
        
        if filtered:
            for i, doc in enumerate(filtered[:3]):
                meta = doc.get('metadata', {})
                print(f"[{i+1}] [Lata: {doc.get('debug_year')}] {doc['text'][:100]}...")
                csv_rows.append([q, "Filtrowanie (Metadane)", doc['text']])
        else:
            print("Brak wyników spełniających kryteria filtracji.")
            csv_rows.append([q, "Filtrowanie (Metadane)", "BRAK WYNIKÓW spełniających kryteria"])

    filename = "wyniki_eksperymentu.csv"
    try:
        with open(filename, mode='w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(csv_headers) # Nagłówek
            writer.writerows(csv_rows)   # Dane
        print(f"\n[INFO] Zapisano pełne wyniki do pliku: {os.path.abspath(filename)}")
    except Exception as e:
        print(f"\n[BŁĄD] Nie udało się zapisać CSV: {e}")

if __name__ == "__main__":
    run_experiment()