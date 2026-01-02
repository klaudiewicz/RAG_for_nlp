# Modular RAG System with Semantic Memory & Validation Layer

Projekt systemu **Retrieval-Augmented Generation (RAG)**, realizowany w ramach laboratorium Przetwarzania Języka Naturalnego (PJN). System charakteryzuje się modularną architekturą, hybrydowym wyszukiwaniem oraz rygorystyczną warstwą walidacji odpowiedzi w celu eliminacji halucynacji.

## Główne Funkcjonalności

* **Hybrydowy silnik retrieval:** Łączenie wyszukiwania semantycznego (**Qdrant**) z leksykalnym (**Elasticsearch**) przy użyciu algorytmu **RRF (Reciprocal Rank Fusion)** .


* **Adaptacyjny dobór wag:** System dynamicznie dobiera wagi dla silników wyszukiwania na podstawie analizy zapytania (np. wyższe wagi dla Elasticsearch przy zapytaniach o akronimy i liczby) .


* **Warstwa walidacji (Safe Mode):** Rygorystyczne sprawdzanie odpowiedzi pod kątem obecności i poprawności cytatów z dokumentów źródłowych (fuzzy matching cytatów) .


* **Inteligentna pamięć agenta:** Automatyczne katalogowanie zapytań, na które system nie znalazł odpowiedzi (OUT OF CORPUS), w celu późniejszej rozbudowy bazy wiedzy przez administratora .


* **Dekompozycja zapytań:** Rozbijanie złożonych pytań na pod-zapytania w celu zwiększenia precyzji wyszukiwania .



## Architektura techniczna

* **LLM:** Ollama (Llama 3.1: 8B).


* **Bazy wektorowe:** Qdrant (Kolekcja `culturax_semantic`) .


* **Wyszukiwanie pełnotekstowe:** Elasticsearch (Indeks `culturax_vectors`).


* **Framework API:** FastAPI.



## Struktura projektu

```text
├── main.py                # Serwer FastAPI i definicje endpointów
├── rag_query.py           # Główny silnik logiczny RAG
├── retrieval/             # Moduły wyszukiwania
│   ├── fusion.py          # Implementacja RRF i retrieve_adaptive
│   ├── elastic.py         # Integracja z Elasticsearch
│   └── qdrant.py          # Integracja z Qdrant
├── reasoning/             # Logika analizy i weryfikacji
│   ├── validation.py      # Walidacja cytatów i obsługa pamięci
│   └── prompt.py          # Szablony promptów (Warianty A, B, C)
│   └── chunking.py        # Podział dokumentów na chunki
└── data/                  # Pliki bazy wiedzy i pamięć (JSON)

```

## Prompty

1. **Wariant A (Luźny):** Skupiony na wysokim Recall, dopuszcza wiedzę własną modelu .


2. **Wariant B (Restrykcyjny):** Wymaga dosłownych cytatów; jeśli brak danych w tekście, zwraca "BRAK INFORMACJI" .


3. **Wariant C (Zbalansowany):** Dopuszcza wnioskowanie logiczne ("wnioskuję, że..."), ale nadal wymaga zakotwiczenia w źródłach .

## Uruchomienie

### Wymagania

* Uruchomiona usługa **Ollama** z modelem `llama3.1:8b`.

* Dostęp do instancji **Elasticsearch** i **Qdrant**.

### Instalacja i start

```bash
# Instalacja zależności
pip install fastapi uvicorn pydantic openai qdrant-client elasticsearch

# Uruchomienie serwera API
uvicorn main:app --reload

```

## Przykładowe zapytania (API)

**Zapytanie o skrót techniczny (HMM):**
System przypisuje wagi 0.8/0.2 (ES/Qdrant) i zwraca precyzyjną definicję.

**Zapytanie spoza korpusu (Przepis na pizzę):**
System poprawnie odmawia odpowiedzi i zapisuje zapytanie do `pamiec_nierozwiazane.json` .

---

Autor projektu: Klaudia Stodółkiewicz 

Materiały źródłowe: Wykłady dr. Aleksandra Smywińskiego-Pohla, opracowania Jakuba 'morgula' Adamczyka.
