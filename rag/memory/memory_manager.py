import json
import os
import sys
from datetime import datetime
from config import MEMORY_FILE

# Dodajemy ścieżkę do importów, aby widzieć plik rag_query_with_metadata.py
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

# Próba importu właściwej funkcji ekstrakcji
try:
    from rag_query_with_metadata import extract_metadata_hybrid
except ImportError:
    print("[WARNING] Nie udało się zaimportować extract_metadata_hybrid. Używam funkcji pustej.")
    def extract_metadata_hybrid(text): return {"entities": [], "dates": []}

def load_memory():
    if not os.path.exists(MEMORY_FILE):
        return {"pending_queries": []}
    with open(MEMORY_FILE, "r", encoding="utf-8") as f:
        return json.load(f)

def save_memory(data):
    os.makedirs(os.path.dirname(MEMORY_FILE), exist_ok=True)
    with open(MEMORY_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def add_to_pending_enhanced(query, reason):
    """
    Dodaje zapytanie do pamięci wraz z podpowiedziami (hints) o encjach i latach.
    """
    data = load_memory()
    
    # Ekstrakcja metadanych - teraz powinna wykryć 'LSTM'
    metadata = extract_metadata_hybrid(query)
    
    # Konwersja lat na inty
    years_ints = []
    for d in metadata.get('dates', []):
        if d.isdigit():
            years_ints.append(int(d))
    
    entry = {
        "id": len(data["pending_queries"]) + 1,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "query": query,
        "status": "retry_later",
        "attempts": 0,
        "reason": str(reason),
        "entities_hint": metadata.get('entities', []),
        "years_hint": years_ints
    }
    
    data["pending_queries"].append(entry)
    save_memory(data)
    print(f"[MEMORY] Dodano do oczekujących: '{query}' (Czekam na: {metadata})")

def check_new_documents_match(new_docs_metadata_list):
    """
    Sprawdza, czy nowe dokumenty pasują do oczekujących zapytań.
    """
    data = load_memory()
    updated = False
    
    pending_list = [p for p in data["pending_queries"] if p["status"] == "retry_later"]
    
    if not pending_list:
        print("[MEMORY] Brak oczekujących zapytań do sprawdzenia.")
        return "NO_PENDING"

    print(f"\n[MEMORY SCAN] Sprawdzam {len(pending_list)} oczekujących zapytań pod kątem nowych danych...")

    for pending in pending_list:
        match_found = False
        
        # Iterujemy przez metadane nowych dokumentów
        for doc_meta in new_docs_metadata_list:
            # 1. Sprawdzenie DATY
            date_match = False
            if not pending["years_hint"]:
                date_match = True 
            else:
                # Sprawdź czy jakakolwiek data z dokumentu jest w zakresie zainteresowania
                # Heurystyka: Czy dokument ma rok >= min(years_hint)?
                doc_years = doc_meta.get("years", [])
                req_years = pending["years_hint"]
                
                # Proste przecięcie zbiorów lub sprawdzenie zakresu
                if set(doc_years).intersection(set(req_years)):
                    date_match = True
                # Opcjonalnie: logika "po 2024" (jeśli dokument ma 2025, a user chciał 2024, to ok)
                elif doc_years and req_years and max(doc_years) >= min(req_years):
                    date_match = True

            # 2. Sprawdzenie ENCJI
            entity_match = False
            if not pending["entities_hint"]:
                entity_match = True
            else:
                # Normalizacja do małych liter
                doc_ents = [e.lower() for e in doc_meta.get("named_entities", []) + doc_meta.get("places", [])]
                pend_ents = [e.lower() for e in pending["entities_hint"]]
                
                # Czy jest jakakolwiek wspólna encja?
                if set(doc_ents).intersection(set(pend_ents)):
                    entity_match = True
            
            if date_match and entity_match:
                match_found = True
                break 
        
        if match_found:
            pending["status"] = "ready_to_notify"
            print(f"  [SUCCESS] -> ZNALEZIONO NOWE DANE DLA: '{pending['query']}'")
            updated = True
        else:
            print(f"  [INFO] Brak dopasowania dla: '{pending['query']}'")

    if updated:
        save_memory(data)
        return "FOUND_UPDATES"
    
    return "NO_MATCHES"

# --- TEST ---
if __name__ == "__main__":
    # Usuwamy stary plik pamięci dla czystego testu
    if os.path.exists(MEMORY_FILE):
        try:
            os.remove(MEMORY_FILE)
        except: pass

    print("--- 1. Symulacja dodania pytania o przyszłość ---")
    q = "Co takiego wprowadził  Hochreiter w 1997?"
    add_to_pending_enhanced(q, "Brak dokumentów z tego okresu")
    
    print("\n--- 2. Symulacja przyjścia nowego dokumentu ---")
    # Symulujemy metadane, jakie wygenerowałby create_indices.py
    new_doc_meta = [{
        "named_entities": ["Hochreiter"],
        "places": [],
        "years": [1997]
    }]
    
    print(f"Nowy dokument metadata: {new_doc_meta}")

    print("\n--- 3. Skanowanie pamięci ---")
    check_new_documents_match(new_doc_meta)
    
    # Weryfikacja
    mem = load_memory()
    status = mem["pending_queries"][0]["status"]
    print(f"\n[WYNIK KOŃCOWY] Status zapytania: {status}") 
