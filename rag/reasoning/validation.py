import re
import difflib
import nltk
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from nltk.tokenize import sent_tokenize

try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

def normalize_text(text):
    return " ".join(text.lower().split())

def check_quotes(answer, combined_source_text):
    """Sprawdza, czy tekst w cudzysłowach faktycznie istnieje w źródle."""
    quotes = re.findall(r'[„"«»](.*?)[„"«»]|"(.*?)"', answer)
    quotes = [q[0] or q[1] for q in quotes if q[0] or q[1]]
    
    invalid_quotes = []
    
    for quote in quotes:
        if len(quote.strip()) < 10:  # Ignorujemy bardzo krótkie cytaty
            continue
            
        norm_quote = normalize_text(quote)
        
        # Exact match
        if norm_quote in combined_source_text:
            continue
            
        # Fuzzy match (dla literówek/interpunkcji)
        # Sprawdzamy fragmenty, nie cały tekst na raz, jeśli tekst jest ogromny
        matcher = difflib.SequenceMatcher(None, norm_quote, combined_source_text)
        match = matcher.find_longest_match(0, len(norm_quote), 0, len(combined_source_text))
        
        # Jeśli dopasowano mniej niż 85% cytatu -> uznajemy za fałsz
        if match.size < len(norm_quote) * 0.85:
            invalid_quotes.append(quote)
            
    return invalid_quotes

def check_semantics(answer, retrieved_docs, embedding_model, threshold=0.75):
    """
    Sprawdza, czy każde zdanie odpowiedzi ma pokrycie semantyczne w dokumentach.
    Zwraca listę zdań podejrzanych o halucynacje.
    """
    # 1. Podziel odpowiedź na zdania
    answer_sentences = sent_tokenize(answer)
    if not answer_sentences:
        return []

    # 2. Przygotuj tekst źródłowy (dzielimy na mniejsze fragmenty dla precyzji)
    source_texts = [doc['text'] for doc in retrieved_docs]
    
    # 3. Generuj embeddingi
    answer_embs = embedding_model.encode(answer_sentences, convert_to_numpy=True)
    source_embs = embedding_model.encode(source_texts, convert_to_numpy=True)
    
    suspicious_sentences = []

    # 4. Sprawdź każde zdanie odpowiedzi
    # Obliczamy podobieństwo każdego zdania do WSZYSTKICH dokumentów
    similarity_matrix = cosine_similarity(answer_embs, source_embs)
    
    for i, sentence in enumerate(answer_sentences):
        # Ignorujemy krótkie zdania łącznikowe/formatowanie
        if len(sentence) < 15:
            continue
            
        # Bierzemy maksymalne podobieństwo danego zdania do jakiegokolwiek dokumentu
        max_sim = np.max(similarity_matrix[i])
        
        if max_sim < threshold:
            suspicious_sentences.append(f"{sentence} (sim: {max_sim:.2f})")
            
    return suspicious_sentences

def validate_answer_hybrid(answer, retrieved_docs, embedding_model):
    if "BRAK INFORMACJI" in answer.upper():
        return True, "OK"
        
    if not answer.strip():
        return False, "Pusta odpowiedź"

    norm_source_texts = [normalize_text(doc['text']) for doc in retrieved_docs]
    combined_source_norm = " ".join(norm_source_texts)

    errors = []

    # 1. Sprawdź cytaty (Syntax Check)
    invalid_quotes = check_quotes(answer, combined_source_norm)
    if invalid_quotes:
        errors.append(f"Zmyślone cytaty: {invalid_quotes}")

    # 2. Sprawdź sens (Semantic Check)
    # Jeśli cytaty są OK, ale nie ma cytatów - musimy sprawdzić, czy model nie zmyśla faktów
    # Uruchamiamy to zawsze, chyba że odpowiedź jest bardzo krótka
    if len(answer) > 30:
        suspicious_sentences = check_semantics(answer, retrieved_docs, embedding_model)
        if suspicious_sentences:
            errors.append(f"Brak potwierdzenia w źródłach dla zdań: {suspicious_sentences}")

    is_valid = len(errors) == 0
    return is_valid, errors