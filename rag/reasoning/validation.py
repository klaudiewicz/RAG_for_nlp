import json
import os
import re
import time
from datetime import datetime

MEMORY_FILE = "memory/pending.json"
def normalize_text(text):
    return " ".join(text.lower().split())
	
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
