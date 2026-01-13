import re
import json
import openai
import random
import sys
import os
import sys
from transformers import pipeline, AutoTokenizer, AutoModelForTokenClassification
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from config import client_ollama, ner_worker, OLLAMA_MODEL

# RegExp
def extract_dates_regexp(text):
    patterns = [
        # Format YYYY-MM-DD
        r'\b\d{4}-\d{2}-\d{2}\b',             
        # Format DD.MM.YYYY
        r'\b\d{2}\.\d{2}\.\d{4}\b',           
        # Format DD-MM-YYYY 
        r'\b\d{2}-\d{2}-\d{4}\b',             
        # Format: w 2020 roku / w 2020 r.
        r'\bw\s\d{4}\s(?:roku|r\.)\b',        
        # Format: 2022 r.
        r'\b\d{4}\sr\.\b',                    
        # Same lata (np. 1998, 2023) z grupą non-capturing (?:...)
        r'\b(?:19|20)\d{2}\b',
        # Daty słowne (np. 15 czerwca 2026)
		r'\b\d{1,2}\s(?:stycznia|lutego|marca|kwietnia|maja|czerwca|lipca|sierpnia|września|października|listopada|grudnia)\s\d{4}\b'
    ]
    found = []
    for p in patterns:
        # flaga IGNORECASE dla miesięcy słownych
        matches = re.findall(p, text, re.IGNORECASE)
        found.extend(matches)
    
    return list(set(found))

# NER
def extract_dates_ner(text):
    try:
        entities = ner_worker(text)
        dates = [e['word'] for e in entities if e['entity_group'] in ['DATE', 'TIME'] and len(e['word']) >= 4]
        return list(set(dates))
    except Exception as e:
        print(f"[BŁĄD NER] {e}")
        return []

# LLM
def extract_dates_llm(text):
    prompt = f"""Wyodrębnij z poniższego tekstu wszystkie daty i zakresy czasowe.
Zwróć wynik WYŁĄCZNIE w formacie JSON:
{{
  "dates": ["YYYY-MM-DD", "..."],
  "years": ["YYYY", "..."],
  "ranges": ["od YYYY do YYYY", "..."]
}}

Tekst:
{text}"""

    try:
        response = client_ollama.chat.completions.create(
            model=OLLAMA_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            response_format={"type": "json_object"}
        )
        return json.loads(response.choices[0].message.content)
    except Exception as e:
        print(f"[BŁĄD LLM] {e}")
        return {"dates": [], "years": [], "ranges": []}

# HYBRYDA
def hybrid_date_extraction(text):
    # RegExp + NER
    re_dates = extract_dates_regexp(text)
    ner_dates = extract_dates_ner(text)
    
    combined_baseline = list(set(re_dates + ner_dates))
    
    triggers = ["między", "od", "do", "przełomie", "wiek"]
    needs_llm = len(combined_baseline) == 0 or any(t in text.lower() for t in triggers)
    
    llm_data = None
    if needs_llm:
        #print("  [DEBUG] Uruchamiam LLM...")
        llm_data = extract_dates_llm(text)
    
    all_source_text = str(combined_baseline) + str(llm_data)
    final_years = list(set(re.findall(r'\b(?:19|20)\d{2}\b', all_source_text)))
    
    return {
        "baseline": combined_baseline,
        "llm_refined": llm_data,
        "final_years": sorted(final_years)
    }

# TESTOWANIE
if __name__ == "__main__":
    with open("dane_ner.jsonl", 'r', encoding='utf-8') as f:
            lines = f.readlines()
            samples = random.sample(lines, min(25, len(lines)))
            
            for i, line in enumerate(samples):
                data = json.loads(line)
                text = data.get('text', '')[:500] 
                
                print(f"Tekst: {text}")
                result = hybrid_date_extraction(text)
                print(f"Wynik: {json.dumps(result, indent=2, ensure_ascii=False)}")

