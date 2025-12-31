import json
import os
import re
import time
from datetime import datetime

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


def validate_answer(answer, retrieved_docs):
    # Jeśli odpowiedź to BRAK INFORMACJI - OK
    if "BRAK INFORMACJI" in answer.upper():
        return True, []
    
    # Szukamy cytatów
    quotes = re.findall(r'[„"«»](.*?)[„"«»]|"(.*?)"', answer)
    quotes = [q[0] or q[1] for q in quotes if q[0] or q[1]]
    
    # Jeśli brak cytatów, ale odpowiedź nie mówi "BRAK INFORMACJI"
    # → to może być wnioskowanie logiczne (OK dla wariantu C)
    if not quotes:
        # Sprawdzamy czy odpowiedź wyglądać na rzeczywistą odpowiedź
        if len(answer.strip()) > 20 and answer.strip() != "":
            return True, []  # Akceptujemy wnioskowanie
        return False, ["Brak treści w odpowiedzi"]
    
    source_texts = [normalize_text(doc['text']) for doc in retrieved_docs]
    combined_source = " ".join(source_texts)
    
    invalid_quotes = []
    for quote in quotes:
        if len(quote.strip()) < 5: 
            continue
        
        norm_quote = normalize_text(quote)
        
        # Exact match
        if norm_quote in combined_source:
            continue
        
        # Fuzzy match 
        matcher = difflib.SequenceMatcher(None, norm_quote, combined_source)
        match = matcher.find_longest_match(0, len(norm_quote), 0, len(combined_source))
        
        if match.size > len(norm_quote) * 0.75:  
            continue
        
        invalid_quotes.append(quote[:80] + "...") 
    
    return (len(invalid_quotes) == 0, invalid_quotes)
