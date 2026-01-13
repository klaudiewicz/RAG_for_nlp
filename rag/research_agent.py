import sys
import os
import json
import time
from datetime import datetime

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from config import client_ollama, OLLAMA_MODEL, model_emb, MEMORY_FILE
from retrieval.fusion import retrieve_adaptive
from reasoning.prompt import generate_answer_variant
from reasoning.validation import validate_answer_hybrid

try:
    from rag_query_with_metadata import extract_metadata_hybrid, evaluate_response_quality
    from rag_temporal_logic import temporal_reranker
    from rag_query import decompose_query
    from memory.memory_manager import add_to_pending_enhanced, check_new_documents_match
    from data.create_indices import create_indices
except ImportError as e:
    print(f"[CRITICAL ERROR] Brakuje modułów z poprzednich kroków: {e}")
    sys.exit(1)

# --- KLASA AGENTA BADAWCZEGO ---

class ResearchAgent:
    def __init__(self):
        self.model = OLLAMA_MODEL
        print(f"[AGENT] Zainicjalizowano agenta badawczego ({self.model})")

    def run_pipeline(self, user_query):
        print(f"\n{'='*60}")
        print(f" AGENT START: {user_query}")
        print(f"{'='*60}")

        # 1. Analiza metadanych (NER + Daty)
        print(" [1/6] Analiza intencji i metadanych...")
        metadata = extract_metadata_hybrid(user_query)
        print(f"   -> Wykryto: {metadata}")

        # 2. Dekompozycja
        sub_queries = decompose_query(user_query)
        if len(sub_queries) > 1:
            print(f" [2/6] Dekompozycja na: {sub_queries}")
        else:
            print(" [2/6] Pytanie proste (brak dekompozycji).")

        # 3. Retrieval (Hybrydowy + Agregacja)
        print(" [3/6] Wyszukiwanie hybrydowe (Qdrant + BM25)...")
        all_docs = {}
        for sub_q in sub_queries:
            found, _, _, _ = retrieve_adaptive(sub_q)
            for d in found:
                key = d.get('id', d['text'][:50])
                if key not in all_docs:
                    if 'metadata' not in d:
                        d['metadata'] = {'source': 'unknown'}
                    all_docs[key] = d
        
        unique_docs = list(all_docs.values())
        print(f"   -> Znaleziono łącznie {len(unique_docs)} unikalnych kandydatów.")

        if not unique_docs:
            print("   [!] Brak dokumentów. Zapisuję do pamięci.")
            add_to_pending_enhanced(user_query, "Brak dokumentów w retrievalu")
            return "Przepraszam, nie znalazłem informacji w mojej bazie wiedzy. Zapisałem Twoje pytanie i wrócę do niego, gdy pojawią się nowe dane."

        # 4. Reranking Temporalny (Logika Agenta)
        if metadata.get('dates'):
            print(" [4/6] Uruchamiam filtr temporalny (Agent Czasu)...")
            # Limitujemy do top 5 dla wydajności LLM
            reranked_docs = temporal_reranker(unique_docs[:5], user_query)
            final_context = reranked_docs[:3]
        else:
            print(" [4/6] Brak wymogów czasowych. Standardowy ranking.")
            final_context = unique_docs[:3]

        # 5. Generowanie Odpowiedzi (RAG)
        print(" [5/6] Generowanie odpowiedzi...")
        constraints = ""
        if metadata['entities'] or metadata['dates']:
            constraints = f"[WYMAGANE ENCJE: {metadata['entities']}, ZAKRES CZASU: {metadata['dates']}] "
        
        answer, used_ids = generate_answer_variant(
            f"{constraints}{user_query}", 
            final_context, 
            client_ollama, 
            variant="B"
        )

        # 6. Weryfikacja i Quality Loop
        print(" [6/6] Weryfikacja jakości (Sędzia LLM)...")
        
        # A. Walidacja twarda (czy są cytaty?)
        is_valid, validation_errors = validate_answer_hybrid(answer, final_context, model_emb)
        
        # B. Ocena jakości merytorycznej
        quality_report = evaluate_response_quality(user_query, answer, final_context, metadata)
        
        score = quality_report.get('score', 0)
        label = quality_report.get('label', 'Unknown')

        print(f"\n--- RAPORT JAKOŚCI ---")
        print(f" Wynik: {score}/2 | Label: {label}")
        print(f" Walidacja techniczna: {'OK' if is_valid else 'BŁĄD'}")
        
        if not is_valid:
            print(f" Uwagi walidatora: {validation_errors}")

        # Decyzja końcowa
        if score == 0 or "BRAK INFORMACJI" in answer or label == "Noise":
            print("   [!] Odpowiedź niesatysfakcjonująca. Zapisuję do pamięci.")
            add_to_pending_enhanced(user_query, f"Niska jakość odpowiedzi (Score: {score}, Label: {label})")
            return f"Na podstawie dostępnych dokumentów nie mogę udzielić pewnej odpowiedzi. ({answer})"
        
        return answer

    def ingestion_hook(self, new_docs_metadata):
        """Symulacja zdarzenia: Pojawiły się nowe dokumenty."""
        print("\n[SYSTEM] Wykryto nowe dokumenty. Sprawdzam pamięć...")
        status = check_new_documents_match(new_docs_metadata)
        if status == "FOUND_UPDATES":
            print("[POWIADOMIENIE] Mam nowe informacje do Twoich starych pytań!")

# URUCHOMIENIE TESTOWE

if __name__ == "__main__":
    create_indices()
    agent = ResearchAgent()

    # Scenariusz 1: Pytanie, na które znamy odpowiedź (z historii)
    print("\n\n>>> SCENARIUSZ 1: Pytanie historyczne (sukces)")
    odp1 = agent.run_pipeline("Kto i kiedy wprowadził LSTM?")
    print(f"\n[ODPOWIEDŹ AGENTA]:\n{odp1}")

    # Scenariusz 2: Pytanie o przyszłość (porażka -> pamięć)
    print("\n\n>>> SCENARIUSZ 2: Pytanie o przyszłość (zapis do pamięci)")
    odp2 = agent.run_pipeline("Jakie modele NLP powstały po roku 2026?")
    print(f"\n[ODPOWIEDŹ AGENTA]:\n{odp2}")

    # Scenariusz 3: Symulacja "Next Day" - nowe dane
    print("\n\n>>> SCENARIUSZ 3: Ingestion Hook (Nowe dane)")
    new_data_meta = [{
        "text_snippet": "W 2027 roku Google wydało model GPT-7.",
        "named_entities": ["GPT-7", "Google"],
        "years": [2027]
    }]
    agent.ingestion_hook(new_data_meta)