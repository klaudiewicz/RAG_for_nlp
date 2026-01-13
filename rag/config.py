# rag/config.py
import os
import openai
from qdrant_client import QdrantClient
from elasticsearch import Elasticsearch
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline

QDRANT_COLLECTION_NAME = "culturax_enriched"
QDRANT_HOST = "localhost"
QDRANT_PORT = 6333

ES_INDEX_NAME = "culturax_enriched"
ES_HOST = "http://localhost:9200"

NER_MODEL = "Babelscape/wikineural-multilingual-ner"
EMBEDDING_MODEL = 'intfloat/multilingual-e5-small'

OLLAMA_MODEL = "llama3.1:8b"
OLLAMA_URL="http://localhost:11434/v1"


DATA_PATH = "data/dane_new.jsonl"
BATCH_SIZE = 50 
VECTOR_SIZE = 384

MEMORY_FILE = "memory/pending.json"


# Initialization
client = QdrantClient(QDRANT_HOST, port=QDRANT_PORT)
es = Elasticsearch(ES_HOST)

tokenizer = AutoTokenizer.from_pretrained(NER_MODEL)
model_ner = AutoModelForTokenClassification.from_pretrained(NER_MODEL)
ner_worker = pipeline("ner", model=model_ner, tokenizer=tokenizer, aggregation_strategy="simple")

client_ollama = openai.OpenAI(
    base_url=OLLAMA_URL,  
    api_key="ollama"                      
)
model_emb = SentenceTransformer(EMBEDDING_MODEL)