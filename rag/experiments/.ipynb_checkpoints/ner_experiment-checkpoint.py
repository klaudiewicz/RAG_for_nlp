import json
import sys
import os

# Dodaj ścieżkę do projektu
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from config import client_ollama, OLLAMA_MODEL, ner_worker

SAMPLE_TEXT = """
Architektura Transformera została opracowana w Google i opisana w 2017 roku w artykule Attention Is All You Need.
W pracy z 2017 r. autorzy z Google przedstawili Transformer, oparty na mechanizmie attention.
Mechanizm uwagi, wykorzystany w Transformerach, wywodzi się z badań Bahdanau z 2014 roku.
Dzięki Transformerom możliwe stało się trenowanie dużych modeli językowych, takich jak GPT oraz BERT.
Artykuł Attention Is All You Need został napisany przez Ashisha Vaswaniego, Noama Shazeera i współautorów w 2017.
Autorami pracy byli m.in. Jakob Uszkoreit, Łukasz Kaiser oraz Illia Polosukhin, związani wówczas z Google.
W porównaniu z RNN i LSTM, architekturze Transformera nie są potrzebne mechanizmy rekurencyjne.
Transformery znalazły zastosowanie w NLP, wizji komputerowej oraz systemach multimodalnych.
26 lutego 2025 roku Ministerstwo Cyfryzacji zaprezentowało PLLuM, czyli Polish Large Language Model.
Projekt PLLuM-u został zrealizowany przez konsorcjum, którego liderem była Politechnika Wrocławska.
W pracach nad PLLuM uczestniczył Uniwersytet Łódzki, reprezentowany m.in. przez prof. Piotra Pęzika.
Model PLLuM trenowano na danych w języku polskim, pozyskiwanych w sposób etyczny.
Na dalszy rozwój PLLuM-u Ministerstwo Cyfryzacji przyznało 19 mln zł.
Zastosowanie PLLuM planowane jest m.in. w aplikacji mObywatel oraz w administracji publicznej.
Modele PLLuM udostępniono na platformie Hugging Face w 2025 roku.
"""

# --- A. PROMPT PROSTY ---
def ner_prompt_simple(text):
    prompt = f"""Wyodrębnij z tekstu wszystkie nazwane encje:
- osoby,
- organizacje,
- miejsca.

Zwróć wynik WYŁĄCZNIE w formacie JSON:
{{
  "persons": ["..."],
  "organizations": ["..."],
  "locations": ["..."]
}}

Tekst:
{text}"""
    
    response = client_ollama.chat.completions.create(
        model=OLLAMA_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
        response_format={"type": "json_object"}
    )
    return json.loads(response.choices[0].message.content)

# --- B. PROMPT KONTEKSTOWY ---
def ner_prompt_contextual(text):
    prompt = f"""Z poniższego tekstu wyodrębnij encje w kontekście NAUKI:
- scientists: osoby, które pełnią rolę naukowców / badaczy (ignoruj inne osoby),
- organizations: organizacje, w których działają ci naukowcy, badacze,
- locations: miejsca kluczowe dla kontekstu ich decyzji naukowych, badań.

Zwróć wynik WYŁĄCZNIE w formacie JSON:
{{
  "scientists": ["..."],
  "organizations": ["..."],
  "locations": ["..."]
}}

Tekst:
{text}"""
    
    response = client_ollama.chat.completions.create(
        model=OLLAMA_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
        response_format={"type": "json_object"}
    )
    return json.loads(response.choices[0].message.content)

# --- C. TRADYCYJNY NER (BERT) ---
def run_traditional_ner(text):
    results = ner_worker(text)
    # Grupowanie wyników
    output = {"PER": [], "ORG": [], "LOC": []}
    for ent in results:
        label = ent['entity_group']
        word = ent['word'].replace("##", "")
        if label in output:
            output[label].append(word)
    # Deduplikacja
    for k in output:
        output[k] = list(set(output[k]))
    return output

# --- PORÓWNANIE ---
if __name__ == "__main__":
    print(f"--- TEKST WEJŚCIOWY ---\n{SAMPLE_TEXT.strip()}\n")
    
    print("\n>>> 1. TRADYCYJNY NER (BERT)")
    print(json.dumps(run_traditional_ner(SAMPLE_TEXT), indent=2, ensure_ascii=False))
    
    print("\n>>> 2. LLM PROMPT PROSTY")
    print(json.dumps(ner_prompt_simple(SAMPLE_TEXT), indent=2, ensure_ascii=False))
    
    print("\n>>> 3. LLM PROMPT KONTEKSTOWY (SCIENCE)")
    print(json.dumps(ner_prompt_contextual(SAMPLE_TEXT), indent=2, ensure_ascii=False))