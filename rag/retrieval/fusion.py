from typing import List, Dict, Any
from retrieval.qdrant import search_qdrant
from retrieval.elastic import search_es
import re

def choose_weights(query):

    has_number = bool(re.search(r'\d', query))
    has_caps_acronym = bool(re.search(r'\b[A-Z]{2,}\b', query))
    has_mixed_acronym = bool(re.search(r'\b[A-Z][a-z]+[A-Z]\w*\b', query))
    potential_names = re.findall(r'\b[A-Z][a-z]{3,}\b', query)
    
    has_proper_name = False
    if potential_names:
        first_word = query.split()[0]
        if len(query.split()) == 1 or not query.startswith(potential_names[0]):
            has_proper_name = True
            
    if has_number or has_caps_acronym or has_mixed_acronym or has_proper_name:
        return {"es": 0.8, "qdrant": 0.2}, "Faktograficzne (ES)"
    else:
        return {"es": 0.4, "qdrant": 0.6}, "Semantyczne (Qdrant)"


def rrf_fusion_weighted(sem_results, key_results, weights, k=60):
    scores = {}
    doc_map = {} 
    
    w_qdrant = weights.get('qdrant', 1.0)
    w_es = weights.get('es', 1.0)

    # Qdrant 
    for rank, point in enumerate(sem_results):
        payload = point.payload if hasattr(point, 'payload') else point.get('payload', {})
        doc_id = payload.get('id', getattr(point, 'id', None))
        
        if doc_id:
            score_val = w_qdrant * (1 / (k + rank + 1))
            scores[doc_id] = scores.get(doc_id, 0) + score_val
            doc_map[doc_id] = payload

    # Elasticsearch 
    for rank, hit in enumerate(key_results):
        doc_id = hit['_source']['id']
        if doc_id:
            score_val = w_es * (1 / (k + rank + 1))
            scores[doc_id] = scores.get(doc_id, 0) + score_val
            
            if doc_id not in doc_map:
                doc_map[doc_id] = hit['_source']

    # Sortowanie
    sorted_ids = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)
    
    return [
        {
            "id": pid, 
            "text": doc_map[pid].get("text", ""), 
            "metadata": doc_map[pid], 
            "score": scores[pid]
        } 
        for pid in sorted_ids
    ]

def filter_retrieved(docs, min_tokens=30, max_docs=5):
    valid_docs = []
    dropped_count = 0   
    for doc in docs:
        word_count = len(doc['text'].split())
        if word_count >= min_tokens:
            valid_docs.append(doc)
        else:
            dropped_count += 1            
    final_docs = valid_docs[:max_docs]    
    return final_docs, dropped_count

def retrieve_adaptive(user_input):
    qdrant_hits = search_qdrant(user_input)
    es_hits = search_es(user_input)
    weights, query_type = choose_weights(user_input)
    merged_docs = rrf_fusion_weighted(qdrant_hits, es_hits, weights, k=60)    
    final_docs, dropped_count = filter_retrieved(merged_docs, min_tokens=30, max_docs=5)
    
    return final_docs, query_type, dropped_count, weights

# stare

def rrf_fusion(sem_results, key_results, k=60):
    scores = {}
    doc_map = {}
    
    # Qdrant 
    for rank, point in enumerate(sem_results):
        payload = point.payload if hasattr(point, 'payload') else point.get('payload', {})
        doc_id = payload.get('id', getattr(point, 'id', None))
        
        if doc_id:
            scores[doc_id] = scores.get(doc_id, 0) + (1 / (k + rank + 1))
            doc_map[doc_id] = payload

    # Elasticsearch 
    for rank, hit in enumerate(key_results):
        doc_id = hit['_source'].get('id')
        if doc_id:
            scores[doc_id] = scores.get(doc_id, 0) + (1 / (k + rank + 1))
            if doc_id not in doc_map:
                doc_map[doc_id] = hit['_source']

    # Posortowanie i formatowanie
    sorted_ids = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)
    
    return [
        {
            "id": pid,
            "text": doc_map[pid].get("text", ""),
            "metadata": doc_map[pid],
            "score": scores[pid]
        }
        for pid in sorted_ids
    ]
