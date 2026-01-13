import json
import random
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from config import ner_worker


def run_test_on_corpus(file_path, num_samples=25):
    results = []
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            samples = random.sample(lines, min(num_samples, len(lines)))
            
            for i, line in enumerate(samples):
                data = json.loads(line)
                text = data.get('text', '')[:500] 
                
                # Ekstrakcja NER
                entities = ner_worker(text)
                
                print(f"\n--- Próbka {i+1} ---")
                print(f"Tekst: {text[:100]}...")
                print(f"Encje: {[(e['word'], e['entity_group']) for e in entities]}")
                
    except FileNotFoundError:
        print(f"[BŁĄD] Nie znaleziono pliku {file_path}")

if __name__ == "__main__":
    run_test_on_corpus("dane_ner.jsonl")