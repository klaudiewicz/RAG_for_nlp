import torch
from elasticsearch import Elasticsearch, helpers
import requests
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams
from qdrant_client.http.models import PointStruct
from sentence_transformers import SentenceTransformer
import uuid
import pandas as pd 
import openai
import pandas as pd
import time
from datetime import datetime

import sys
from reasoning.validation import validate_answer
from reasoning.prompt import generate_answer_variant
from retrieval.fusion import retrieve_adaptive
import openai
import json 
import re
import os


MODEL_OLLAMA = "llama3.1:8b" 
client_ollama = openai.OpenAI(
    base_url="http://localhost:11434/v1",  
    api_key="ollama"                      
)
MEMORY_FILE = "memory/pending.json"


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
            model="llama3.1:8b",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_input}
            ],
            temperature=0.0,
            response_format={"type": "json"} 
        )
        
        json_str = response.choices[0].message.content
        parsed_result = json.loads(json_str)
        return parsed_result

    except json.JSONDecodeError:
        print(f"[BŁĄD] Nie udało się sparsować JSON dla: {user_input}")
        return {
            "main_question": user_input,
            "sub_questions": [user_input]
        }
    except Exception as e:
        print(f"[BŁĄD API] {e}")
        return {"main_question": user_input, "sub_questions": []}
def generate_clarification_question(user_input):
    system_prompt = (
        "Jesteś asystentem dbającym o jasność komunikacji. Tematyka dotyczy przetwarzania języka naturalnego, transformatorów i sieci neuronowych."
        "Oceń, czy pytanie użytkownika jest jednoznaczne. Jeśli nie - zaproponuj 2-3 warianty interpretacji w formie pytań do użytkownika."
        "Zwróć wynik WYŁĄCZNIE w surowym formacie JSON, bez znaczników Markdown (```)."
        "Schemat:\n"
        "{\n"
        '  "is_ambiguous": true,\n'
        '  "reason": "Wyjaśnienie",\n'
        '  "clarifications": ["Pytanie 1", "Pytanie 2"]\n'
        "}\n"
        "Nie dodawaj żadnego komentarza."
    )
    try:
        response = client_ollama.chat.completions.create(
            model="llama3.1:8b",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_input}
            ],
            temperature=0.0,
            response_format={"type": "json"} 
        )
        
        raw_content = response.choices[0].message.content
        parsed_json = clean_json_response(raw_content)
        return parsed_json
    except json.JSONDecodeError:
        print(f"[BŁĄD PARSOWANIA] Model zwrócił błędny format dla: '{user_input}'")
        return {
            "is_ambiguous": False, 
            "reason": "Błąd techniczny modelu (zły JSON)", 
            "clarifications": []
        }
    except Exception as e:
        print(f"[BŁĄD API] {e}")
        return {"is_ambiguous": False, "reason": "Błąd API", "clarifications": []}

def clean_json_response(response_text):
    clean_text = re.sub(r'```json\s*', '', response_text, flags=re.IGNORECASE)
    clean_text = re.sub(r'```', '', clean_text)
    
    start_idx = clean_text.find('{')
    end_idx = clean_text.rfind('}')
    
    if start_idx != -1 and end_idx != -1:
        clean_text = clean_text[start_idx : end_idx + 1]
    else:
        return response_text.strip()

    try:
        return json.loads(clean_text)
    except json.JSONDecodeError:
        try:
            fixed_text = clean_text.replace('\\', '\\\\')
            return json.loads(fixed_text)
        except:
            raise 

def load_memory():
    if not os.path.exists(MEMORY_FILE):
        return {"pending_queries": []}
    with open(MEMORY_FILE, "r", encoding="utf-8") as f:
        return json.load(f)

def save_memory(data):
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
        "reason": reason,
        "docs_found": retrieved_count
    }
    
    data["pending_queries"].append(entry)
    save_memory(data)
    print(f"  [PAMIĘĆ] Zapisano pytanie do kolejki: '{query}' (Powód: {reason})")

def view_pending_queries():
    data = load_memory()
    pending = data.get("pending_queries", [])
    
    print(f"\n{'='*50}")
    print(f"STAN PAMIĘCI AGENTA: {len(pending)} oczekujących zadań")
    print(f"{'='*50}")
    
    if not pending:
        print("Brak nierozwiązanych pytań")
    else:
        df_mem = pd.DataFrame(pending)
        print(df_mem[["id", "status", "reason", "query"]])
    print("-" * 50 + "\n")

def rag_safe_mode(user_input):
    print(f"\n[SYSTEM] Start analizy: {user_input}")

    decomp = decompose_query(user_input)
    sub_queries = decomp.get("sub_questions", [user_input])
    
    all_docs_map = {}
    for q in sub_queries:
        found_docs, _, _, _ = retrieve_adaptive(q)
        for d in found_docs:
            key = d.get('id', d['text'][:100])
            all_docs_map[key] = d
    
    docs = list(all_docs_map.values())[:5]

    if not docs:
        print("  -> Brak dokumentów. Sprawdzam czy pytanie było jasne...")
        clarification = generate_clarification_question(user_input)
        
        if clarification.get("is_ambiguous", False):
            add_to_pending(user_input, reason="AMBIGUOUS QUERY (No docs)", retrieved_count=0)
            clar_list = "\n".join([f"- {q}" for q in clarification.get("clarifications", [])])
            return f"Nie znalazłem informacji. Twoje pytanie wydaje się niejednoznaczne. Czy chodziło Ci o:\n{clar_list}"
        else:
            add_to_pending(user_input, reason="OUT OF CORPUS (Clear query, no docs)", retrieved_count=0)
            return "Niestety, baza wiedzy nie zawiera informacji na ten temat."

    print("  -> Próba 1: Generowanie odpowiedzi z cytatami (wariant B)...")
    answer_v1, used_ids = generate_answer_variant(user_input, docs,client_ollama, variant="B")
    
    is_valid, bad_quotes = validate_answer(answer_v1, docs)
    
    if is_valid:
        print("  -> Walidacja: OK ✓")
        
        # Sprawdź czy BRAK INFORMACJI
        if "BRAK INFORMACJI" in answer_v1.upper():
            print("  -> [INFO] Model zwrócił BRAK INFORMACJI mimo dostępnych dokumentów")
            add_to_pending(
                user_input, 
                reason="OUT OF CORPUS (Docs found but irrelevant)", 
                retrieved_count=len(docs)
            )
            return answer_v1
        
        # Sprawdź czy niejasne - heurystyka: dużo wnioskowań, mało cytatów
        quotes_count = len(re.findall(r'[„"«»](.*?)[„"«»]|"(.*?)"', answer_v1))
        inference_count = len(re.findall(r'\(wnioskuję że\.\.\.?\)', answer_v1))
        personal_keywords = ["ulubion", "lubisz", "uważasz", "myślisz", "czujesz", "wolimy"]
        
        if (inference_count > quotes_count + 2) or any(kw in user_input.lower() for kw in personal_keywords):
            print("  -> [INFO] Pytanie może być niejasne (dużo wnioskowań, mało cytatów)")
            add_to_pending(
                user_input, 
                reason="AMBIGUOUS QUERY (Answer inferred, few quotes)", 
                retrieved_count=len(docs)
            )
        
        return answer_v1

    print(f"  -> [ALARM] Wykryto potencjalne problemy w odpowiedzi")
    
    if "BRAK INFORMACJI" in answer_v1.upper():
        print("  -> Model sam zaproponował: BRAK INFORMACJI")
        add_to_pending(
            user_input, 
            reason="OUT OF CORPUS (Validation: no answer)", 
            retrieved_count=len(docs)
        )
        return answer_v1
    
    if bad_quotes:
        print(f"  -> Problematyczne cytaty: {len(bad_quotes)} szt.")
    
    print("  -> Próba 2: Wariant C (elastyczny, z wnioskowaniem)...")
    answer_v2, _ = generate_answer_variant(user_input, docs, variant="C")
    
    is_valid_v2, bad_quotes_v2 = validate_answer(answer_v2, docs)
    
    if is_valid_v2:
        print("  -> Walidacja V2: OK ✓ (Wariant C zaakceptowany)")
        
        # Loguj jeśli V2 to BRAK INFORMACJI
        if "BRAK INFORMACJI" in answer_v2.upper():
            add_to_pending(
                user_input, 
                reason="OUT OF CORPUS (V2: no answer)", 
                retrieved_count=len(docs)
            )
        
        return answer_v2

    print("  -> [INFO] Ani B ani C nie przeszły walidacji ścisłej")
    print("  -> Zwracam najlepszą dostępną odpowiedź...")
    
    if len(answer_v2.strip()) > len(answer_v1.strip()) and "BRAK INFORMACJI" not in answer_v2.upper():
        add_to_pending(
            user_input, 
            f"VALIDATION FAILED - Returned V2 (elastic)", 
            retrieved_count=len(docs)
        )
        return answer_v2
    
    add_to_pending(
        user_input, 
        f"QA PIPELINE FAILED - Best effort returned",
        retrieved_count=len(docs)
    )
    
    if bad_quotes_v2:
        return f"{answer_v1}\n\n[Uwaga: Odpowiedź może wymagać weryfikacji]"
    
    return answer_v1

	
# if __name__ == "__main__":
#     print(rag_safe_mode("Co robią sieci?"))
