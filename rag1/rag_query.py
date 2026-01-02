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
import sys
from reasoning.validation import decompose_query, generate_clarification_question, add_to_pending, validate_answer
from reasoning.prompt import generate_answer_variant
from retrieval.fusion import retrieve_adaptive
import openai
MODEL_OLLAMA = "llama3.1:8b" 
client_ollama = openai.OpenAI(
    base_url="http://localhost:11434/v1",  
    api_key="ollama"                      
)

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
    answer_v1, used_ids = generate_answer_variant(user_input, docs, variant="B")
    
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

	
if __name__ == "__main__":
    print(rag_safe_mode("Co robią sieci?"))
