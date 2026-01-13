import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from config import ES_INDEX_NAME, es

def search_es(query_text, index_name=ES_INDEX_NAME, limit=15):
    es_query = {
        "query": {
            "multi_match": {
                "query": query_text,
                "fields": ["text", "topic", "author^2"],
                "type": "best_fields"
            }
        },
        "size": limit
    }
    res = es.search(index=index_name, body=es_query)
    return res["hits"]["hits"]