import sys
import os
import json
import re
from datetime import datetime

# Manipulacja ścieżką
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Importy własnych modułów
from reasoning.validation import validate_answer_hybrid
from reasoning.prompt import generate_answer_variant
from retrieval.fusion import retrieve_adaptive
from config import MEMORY_FILE, OLLAMA_MODEL, client_ollama, model_emb

def decompose_query(user_input):
    system_prompt = (
        "Jesteś asystentem od wyszukiwania informacji. Twoim zadaniem jest przeanalizowanie pytania użytkownika"
        "i rozbicie go na prostsze elementy, aby ułatwić wyszukiwanie w bazie wiedzy.\n"
        "Zwróć wynik WYŁĄCZNIE w formacie JSON o strukturze:\n"
        "{\n"
        '  "main_question": "Zreparafrazowane, jasne pytanie główne",\n'
        '  "sub_questions": ["Pytanie pomocnicze 1", "Pytanie pomocnicze 2", "Definicja kluczowego terminu"]\n'
        "}\n"
        "Nie dodawaj żadnego tekstu przed ani po JSON."
    )
    try:
        response = client_ollama.chat.completions.create(
            model=OLLAMA_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_input}
            ],
            temperature=0.0,
            response_format={"type": "json_object"} 
        )
        
        json_str = response.choices[0].message.content
        parsed_result = json.loads(json_str)
        return parsed_result

    except Exception as e:
        print(f"[BŁĄD DEKOMPOZYCJI] {e}")
        return {"main_question": user_input, "sub_questions": []}

def generate_clarification_question(user_input):
    system_prompt = (
        "Jesteś precyzyjnym filtrem semantycznym. "
        "Zdecyduj czy pytanie jest niejednoznaczne.\n"
        "Zwróć JSON: { 'is_ambiguous': boolean, 'reason': '...', 'clarifications': [] }"
    )
    try:
        response = client_ollama.chat.completions.create(
            model=OLLAMA_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_input}
            ],
            temperature=0.0,
            response_format={"type": "json_object"} 
        )
        return json.loads(response.choices[0].message.content)
    except Exception:
        return {"is_ambiguous": False, "reason": "Error", "clarifications": []}

def load_memory():
    if not os.path.exists(MEMORY_FILE):
        return {"pending_queries": []}
    with open(MEMORY_FILE, "r", encoding="utf-8") as f:
        return json.load(f)

def save_memory(data):
    # Upewnij się, że katalog istnieje
    os.makedirs(os.path.dirname(MEMORY_FILE), exist_ok=True)
    with open(MEMORY_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def add_to_pending(query, reason, retrieved_count=0):
    data = load_memory()
    new_id = len(data["pending_queries"]) + 1
    entry = {
        "id": new_id,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "query": query,
        "status": "pending",
        "reason": str(reason), 
        "docs_found": retrieved_count
    }
    data["pending_queries"].append(entry)
    save_memory(data)
    print(f"  [PAMIĘĆ] Zapisano do kolejki: '{query}'")

def rag_safe_mode(user_input):
    print(f"\n[SYSTEM] Start analizy: {user_input}")

    # 1. Dekompozycja i wyszukiwanie
    decomp = decompose_query(user_input)
    sub_queries = decomp.get("sub_questions", [user_input])
    
    all_docs_map = {}
    for q in sub_queries:
        found_docs, _, _, _ = retrieve_adaptive(q)
        for d in found_docs:
            key = d.get('id', d['text'][:100])
            all_docs_map[key] = d
    
    docs = list(all_docs_map.values())[:5]

    # 2. Obsługa braku dokumentów
    if not docs:
        print("  -> Brak dokumentów. Sprawdzam niejednoznaczność...")
        clarification = generate_clarification_question(user_input)
        
        if clarification.get("is_ambiguous", False):
            add_to_pending(user_input, reason="AMBIGUOUS (No docs)", retrieved_count=0)
            clar_list = "\n".join([f"- {q}" for q in clarification.get("clarifications", [])])
            return f"Pytanie jest niejednoznaczne. Czy chodziło Ci o:\n{clar_list}"
        else:
            add_to_pending(user_input, reason="NO KNOWLEDGE", retrieved_count=0)
            return "Niestety, baza wiedzy nie zawiera informacji na ten temat."

    # 3. Generowanie Wariantu B (Ścisły)
    print("  -> Próba 1: Generowanie odpowiedzi (wariant B)...")
    answer_v1, _ = generate_answer_variant(user_input, docs, client_ollama, variant="B")

    # POPRAWKA: Używamy model_emb jako validation_model
    is_valid, validation_errors = validate_answer_hybrid(answer_v1, docs, model_emb)
    
    if is_valid:
        print("  -> Walidacja: OK ✓")
        
        if "BRAK INFORMACJI" in answer_v1.upper():
            add_to_pending(user_input, reason="MODEL REFUSED (Docs irrelevant)", retrieved_count=len(docs))
            return answer_v1
        
        # Heurystyka "nadmiernego wnioskowania"
        inference_count = len(re.findall(r'\(wnioskuję że\.\.\.?\)', answer_v1))
        quotes_count = len(re.findall(r'[„"«»]', answer_v1)) / 2
        
        if inference_count > quotes_count + 2:
            print("  -> [INFO] Dużo wnioskowania, mało cytatów.")
            add_to_pending(user_input, reason="HIGH INFERENCE", retrieved_count=len(docs))
        
        return answer_v1

    # 4. Obsługa błędów walidacji
    print(f"  -> [ALARM] Błędy walidacji: {validation_errors}")
    
    if "BRAK INFORMACJI" in answer_v1.upper():
        add_to_pending(user_input, reason="VALIDATION: NO INFO", retrieved_count=len(docs))
        return answer_v1

    # 5. Fallback do Wariantu C (Elastyczny)
    print("  -> Próba 2: Wariant C (korekta)...")
    answer_v2, _ = generate_answer_variant(user_input, docs, client_ollama, variant="C")
    
    # Walidacja wariantu C
    is_valid_v2, validation_errors_v2 = validate_answer_hybrid(answer_v2, docs, model_emb)
    
    if is_valid_v2:
        print("  -> Walidacja V2: OK ✓")
        return answer_v2

    print("  -> [FAIL] Obie próby odrzucone.")
    
    final_answer = answer_v2 if len(answer_v2) > len(answer_v1) else answer_v1
    
    add_to_pending(
        user_input, 
        reason=f"VALIDATION FAILED. Errors: {validation_errors_v2}", 
        retrieved_count=len(docs)
    )
    
    return f"{final_answer}\n\n[System: Odpowiedź może zawierać nieścisłości - weryfikacja negatywna]"

if __name__ == "__main__":
    # Test
    print(rag_safe_mode("Co robią sieci neuronowe?"))