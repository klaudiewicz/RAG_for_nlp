import sys
import os
import json
import re
from datetime import datetime

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, '..'))
sys.path.insert(0, parent_dir)

from data.create_indices import create_indices
from data.date_extraction_hybrid import hybrid_date_extraction

from reasoning.validation import validate_answer_hybrid
from reasoning.prompt import generate_answer_variant
from retrieval.fusion import retrieve_adaptive
from config import MEMORY_FILE, OLLAMA_MODEL, client_ollama, model_emb, ner_worker
from rag_query import decompose_query


def extract_metadata_hybrid(text):
    metadata = {"entities": [], "dates": []}
    try:
        # NER
        raw_entities = ner_worker(text)
        entities_list = []
        for ent in raw_entities:
            clean_word = ent['word'].replace("##", "")
            if len(clean_word) > 2 and ent['score'] > 0.85: 
                entities_list.append(clean_word)
        metadata["entities"] = list(set(entities_list))
        
        # Dates
        date_data = hybrid_date_extraction(text)
        years = date_data.get('final_years', [])
        metadata["dates"] = [str(y) for y in years]
    except Exception as e:
        pass 
    return metadata

# --- ZMODYFIKOWANY RAG ---

def rag_safe_mode_enhanced(user_input, verbose=False):
    # 1. Metadane
    metadata = extract_metadata_hybrid(user_input)
    
    # 2. Retrieval
    found_docs, _, _, _ = retrieve_adaptive(user_input) 
    
    if not found_docs:
        return "BRAK WYNIKÓW W BAZIE.", [], metadata

    filtered_docs = []
    if metadata['dates']:
        for d in found_docs:
            if any(date in d['text'] for date in metadata['dates']):
                filtered_docs.append(d)
        if not filtered_docs:
            filtered_docs = found_docs
    else:
        filtered_docs = found_docs

    final_docs = filtered_docs[:3] # Bierzemy top 3 najbardziej pasujące

    # 3. Generowanie
    answer_v1, _ = generate_answer_variant(user_input, final_docs, client_ollama, variant="B")

    return answer_v1, final_docs, metadata

# --- LLM JUDGE (EWALUACJA) ---

def evaluate_response_quality(query, answer, docs, metadata):
    # Jeśli odpowiedź jest pusta/błędna od razu zwróć słaby wynik
    if "BRAK WYNIKÓW" in answer or "Nie wiem" in answer:
         return {"label": "Noise", "score": 0, "hallucinations": "Brak wiedzy", "entities_in_answer": [], "dates_in_answer": []}

    context_text = "\n".join([d['text'][:150] for d in docs])
    
    system_prompt = (
        "Oceń odpowiedź RAG. Zwróć JSON:\n"
        "{\n"
        "  \"label\": \"Relevant\" lub \"Noise\" (czy odpowiedź ma sens),\n"
        "  \"entities_in_answer\": [lista znalezionych w odpowiedzi nazw własnych],\n"
        "  \"dates_in_answer\": [lista znalezionych w odpowiedzi dat],\n"
        "  \"hallucinations\": \"Brak\" lub krótki opis błędu,\n"
        "  \"score\": 0 (zła), 1 (średnia), 2 (idealna)\n"
        "}"
    )
    
    user_msg = f"PYTANIE: {query}\nKONTEKST: {context_text}\nODPOWIEDŹ: {answer}"
    
    try:
        response = client_ollama.chat.completions.create(
            model=OLLAMA_MODEL,
            messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": user_msg}],
            temperature=0.0,
            response_format={"type": "json_object"}
        )
        return json.loads(response.choices[0].message.content)
    except Exception:
        return {"label": "Error", "score": 0, "hallucinations": "JSON Error", "entities_in_answer": [], "dates_in_answer": []}

# --- URUCHOMIENIE ---

def run_quality_loop_benchmark():
    # Ścieżka do danych
    data_path = os.path.join(parent_dir, "rag/data", "benchmark.jsonl")
    
    print(f"[INIT] Sprawdzanie pliku: {data_path}")
    
    print("[INIT] Uruchamiam indeksowanie...")
    create_indices(data_path)

    queries = [
        "Dlaczego funkcje aktywacji Sigmoid i Tanh są rzadziej używane?",
        "Kto i w którym roku wprowadził sieci LSTM?",
        "Wymień bramki występujące w architekturze LSTM.",
        "Jaki problem rozwiązał mechanizm atencji Bahdanau?",
        "Na czym polega mechanizm Self-Attention?",
        "Jak obliczana jest atencja (Query, Key, Value)?",
        "Czym charakteryzuje się model BERT?",
        "Czym jest Cell State w LSTM?",
        "Jak zdefiniowana jest funkcja ReLU?",
        "Na czym polega idea sieci rekurencyjnych (RNN)?"
    ]

    print(f"\n{'='*100}")
    print(f"{'ZAPYTANIE':<40} | {'LABEL':<10} | {'ENCJE':<15} | {'DATY':<10} | {'OCENA':<5}")
    print(f"{'-'*100}")

    for q in queries:
        answer, docs, _ = rag_safe_mode_enhanced(q, verbose=False)
        eval_res = evaluate_response_quality(q, answer, docs, {})
        
        enc = ", ".join(eval_res.get('entities_in_answer', []))[:15]
        dat = ", ".join(eval_res.get('dates_in_answer', []))
        score = eval_res.get('score', 0)
        label = eval_res.get('label', 'Err')
        
        # Jeśli wynik jest słaby, sprawdźmy czy w ogóle coś znalazł
        if score == 0 and not docs:
            label = "NO DOCS"

        print(f"{q[:37]+'...':<40} | {label:<10} | {enc:<15} | {dat:<10} | {score:<5}")

if __name__ == "__main__":
    run_quality_loop_benchmark()