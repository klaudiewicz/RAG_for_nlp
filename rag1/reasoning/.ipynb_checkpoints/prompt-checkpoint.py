
def generate_answer_variant(user_input, docs, variant="A"):
    if not docs:
        return "Brak kontekstu.", []

    context_text = ""
    used_ids = []
    
    for i, doc in enumerate(docs):
        meta = doc['metadata']
        snippet = chunk_document(doc['text'], token_limit=500)[0] 
        source_desc = f"[{i+1}] (Autor: {meta.get('author', 'Nieznany')})"
        context_text += f"{source_desc}\nTreść: {snippet}\n\n"
        used_ids.append(doc.get('id', f"doc_{i}"))

    if variant == "A":
        system_msg = (
            "Jesteś pomocnym asystentem AI. Odpowiedz na pytanie na podstawie fragmentów. "
            "Możesz łączyć informacje i wyciągać wnioski."
        )
        user_msg = f"Fragmenty:\n{context_text}\n\nPytanie:\n{user_input}"
        
    elif variant == "B":
        system_msg = (
            "Jesteś asystentem, który dokładnie współpracuje z dokumentami.\n"
            "INSTRUKCJE:\n"
            "1. ZAWSZE czytaj dokumenty uważnie przed odpowiedzią.\n"
            "2. Gdy przytaczasz fakt z dokumentu - dodaj cytat: \"dosłowny fragment\".\n"
            "3. Gdy wyciągasz wniosek logiczny - napisz: (wnioskuję że...) bez cudzysłowu.\n"
            "4. Możesz łączyć informacje z różnych dokumentów.\n"
            "5. Jeśli tematu nie ma w dokumentach - napisz: BRAK INFORMACJI."
        )
        user_msg = f"DOKUMENTY:\n{context_text}\n\nPYTANIE:\n{user_input}\n\nODPOWIEDŹ:"
        
    elif variant == "C":
        system_msg = (
            "Jesteś asystentem AI. Odpowiedz na pytanie, bazując WYŁĄCZNIE na poniższych fragmentach.\n"
            "Możesz łączyć informacje z różnych fragmentów i wyciągać logiczne wnioski.\n"
            "Jeśli w tekście znajduje się odpowiedni cytat, wypisz go po odpowiedzi w postaci: CYTAT: <cytat>.\n"
            "Jeśli pytanie zawiera słowa wieloznaczne, interpretuj je w kontekście dostarczonych dokumentów.\n"
            "Jeśli tekst nie dotyczy, napisz TYLKO \"BRAK INFORMACJI\"."
        )
        user_msg = f"Fragmenty:\n{context_text}\n\nPytanie:\n{user_input}"

    try:
        response = client_ollama.chat.completions.create(
            model="llama3.1:8b",
            messages=[
                {"role": "system", "content": system_msg}, 
                {"role": "user", "content": user_msg}
            ],
            temperature=0.1  # Nieznacznie wyższa dla wnioskowania
        )
        return response.choices[0].message.content, used_ids
    except Exception as e:
        return f"Błąd API: {e}", []