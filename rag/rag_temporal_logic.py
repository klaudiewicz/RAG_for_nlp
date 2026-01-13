import json
import os
import sys

# Dodajemy ścieżkę do projektu
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from config import client_ollama, OLLAMA_MODEL, model_emb
from retrieval.fusion import retrieve_adaptive
from reasoning.prompt import generate_answer_variant

# --- 1. PROMPT LLM DO ANALIZY DAT (Bez zmian) ---

def analyze_temporal_relevance(text_chunk, user_question):
    system_prompt = (
        "Jesteś ekspertem od logiki temporalnej w NLP. "
        "Twoim zadaniem jest analiza dat w tekście względem pytania."
    )
    
    user_prompt = f"""
    Twoim zadaniem jest:
    1. Wyodrębnić wszystkie daty z poniższego tekstu (lata, daty dzienne, okresy).
    2. Zinterpretować je względem zapytania użytkownika: "{user_question}".
    3. Określić, czy daty zawarte w tekście są ISTOTNE dla odpowiedzi na to konkretne pytanie.
       - Jeśli pytanie dotyczy roku 1997, a tekst opisuje 2023 -> relevant_for_question: false.
       - Jeśli pytanie nie ma ram czasowych (jest ogólne) -> relevant_for_question: true.
       - Jeśli tekst nie zawiera dat, ale temat pasuje -> relevant_for_question: true.

    Tekst do analizy:
    "{text_chunk[:500]}..."

    Zwróć wynik WYŁĄCZNIE w formacie JSON:
    {{
      "dates": ["YYYY", "YYYY-MM-DD", ...],
      "relevant_for_question": true,
      "explanation": "Krótkie uzasadnienie decyzji"
    }}
    """

    try:
        response = client_ollama.chat.completions.create(
            model=OLLAMA_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.0,
            response_format={"type": "json_object"}
        )
        return json.loads(response.choices[0].message.content)
    except Exception as e:
        # print(f"[BŁĄD LLM TIME] {e}")
        return {"dates": [], "relevant_for_question": True, "explanation": "Error"}

# WARSTWA FILTRACJI I RERANKINGU

def temporal_reranker(docs, user_question):
    """
    Obniża priorytet dokumentów, które LLM uznał za nieistotne czasowo.
    Dodatkowo upewnia się, że dokumenty mają strukturę zgodną z prompt.py.
    """
    print(f"\n[TEMPORAL FILTER] Analiza {len(docs)} dokumentów pod kątem czasu...")
    
    reranked_docs = []
    
    for doc in docs:
        if 'metadata' not in doc:
            # Jeśli dokument jest płaski, tworzymy sztuczne metadane z dostępnych pól
            doc['metadata'] = {
                'author': doc.get('author', 'Nieznany'),
                'topic': doc.get('topic', 'Ogólny'),
                'date': doc.get('date', 'Brak daty')
            }

        analysis = analyze_temporal_relevance(doc['text'], user_question)
        
        is_relevant = analysis.get("relevant_for_question", True)
        dates_found = analysis.get("dates", [])
        explanation = analysis.get("explanation", "")
        
        original_score = doc.get('score', 1.0)
        
        if is_relevant:
            final_score = original_score
            status = "KEEP"
        else:
            final_score = original_score * 0.1
            status = "DEMOTE"

        doc['_temporal_analysis'] = {
            "status": status,
            "dates": dates_found,
            "reason": explanation
        }
        doc['final_score'] = final_score
        
        reranked_docs.append(doc)
        print(f"  -> Doc ID: {doc.get('id', '?')[:8]} | Dates: {dates_found} | {status} ({explanation})")

    reranked_docs.sort(key=lambda x: x['final_score'], reverse=True)
    return reranked_docs

# --- 3. PIPELINE RAG Z OBSŁUGĄ CZASU ---

def rag_with_temporal_logic(user_input):
    print(f"\n--- RAG TEMPORAL START: {user_input} ---")
    
    # 1. Retrieval
    raw_docs, _, _, _ = retrieve_adaptive(user_input)
    
    if not raw_docs:
        return "Brak dokumentów w bazie.", []

    # 2. Temporal Reranking
    candidates = raw_docs[:5]
    sorted_docs = temporal_reranker(candidates, user_input)
    
    # 3. Wybór Top-K
    final_context = sorted_docs[:3]
    
    # 4. Generowanie odpowiedzi
    answer, _ = generate_answer_variant(user_input, final_context, client_ollama, variant="B")
    
    return answer, final_context

# --- TESTY ---

if __name__ == "__main__":
    queries = [
        "Co działo się w NLP w 1997 roku?",
        "Jakie modele powstały po roku 2015?",
        "Opisz działanie neuronu."
    ]
    
    mock_docs = [
        {
            "id": "doc_lstm", 
            "text": "W 1997 roku Hochreiter i Schmidhuber wprowadzili sieci LSTM, rozwiązując problem gradientu.",
            "score": 0.9,
            "metadata": {"author": "Test Author", "topic": "LSTM History"} # To było kluczowe!
        },
        {
            "id": "doc_transformer", 
            "text": "W 2017 roku opublikowano pracę 'Attention Is All You Need', wprowadzając Transformery.",
            "score": 0.88,
            "metadata": {"author": "Test Author", "topic": "Transformer History"} # To było kluczowe!
        }
    ]
    
    def retrieve_adaptive(q): return mock_docs, [], [], []

    for q in queries:
        try:
            answer, ctx = rag_with_temporal_logic(q)
            
            print(f"\nODPOWIEDŹ: {answer}")
            print("-" * 50)
            print("KOLEJNOŚĆ DOKUMENTÓW W KONTEKŚCIE:")
            for i, d in enumerate(ctx):
                meta = d.get('_temporal_analysis', {})
                print(f"{i+1}. {d['text'][:60]}... [{meta.get('status')}]")
        except Exception as e:
            print(f"\n[CRITICAL ERROR]: {e}")