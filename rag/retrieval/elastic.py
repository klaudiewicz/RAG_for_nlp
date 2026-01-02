from elasticsearch import Elasticsearch

# Inicjalizacja ES
es = Elasticsearch("http://localhost:9200")

def search_es(query_text, index_name="culturax_vectors", limit=15):
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