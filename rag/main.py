from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel, Field
from typing import List, Optional, Literal
import time
from rag_query import rag_safe_mode
from retrieval.fusion import retrieve_adaptive 

class RagRequest(BaseModel):
    query: str = Field(..., description="Pytanie użytkownika", min_length=3)
    mode: Literal["safe", "creative", "debug"] = Field(
        default="safe", 
        description="Tryb działania: 'safe' (restrykcyjny z walidacją), 'creative' (wariant A/C), 'debug' (z logami)"
    )
    conversation_id: Optional[str] = None

class SourceDoc(BaseModel):
    id: str
    author: str
    snippet: str
    score: float

class RagResponse(BaseModel):
    status: Literal["success", "ambiguous", "no_info", "error"]
    answer: str
    sources: List[SourceDoc] = []
    clarifications: List[str] = [] # Dla pytań niejednoznacznych [cite: 356]
    memory_action: Optional[str] = None # Czy zapisano do pamięci? [cite: 404]
    processing_time: float

app = FastAPI(title="RAG Safe Mode API", version="1.0")

# ENDPOINT PODSTAWOWY
@app.get("/ask")
async def ask_minimal(query: str):
    start_time = time.time()
    result = rag_safe_mode(query) 
    if isinstance(result, str):
        return {
            "query": query,
            "answer": result,
            "duration": round(time.time() - start_time, 2)
        }
    return {
        "query": query,
        "answer": result.get("answer", "Wystąpił błąd przetwarzania."),
        "duration": round(time.time() - start_time, 2)
    }

# ENDPOINT ROZSZERZONY 
@app.post("/rag", response_model=RagResponse)
async def rag_full(request: RagRequest, background_tasks: BackgroundTasks):
    start_time = time.time()

    try:
        response_data = rag_safe_mode(request.query)

        if "niejednoznaczne" in response_data.lower() and "czy chodziło ci o" in response_data.lower():
            clarifications = extract_bullet_points(response_data) 
            return RagResponse(
                status="ambiguous",
                answer="Pytanie wymaga doprecyzowania.",
                clarifications=clarifications,
                processing_time=time.time() - start_time
            )

        if "BRAK INFORMACJI" in response_data.upper():
            return RagResponse(
                status="no_info",
                answer="Brak informacji w bazie wiedzy.",
                memory_action="Saved to pending_queries (OUT OF CORPUS)",
                processing_time=time.time() - start_time
            )

        return RagResponse(
            status="success",
            answer=response_data,
            sources=[], 
            processing_time=time.time() - start_time
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

def extract_bullet_points(text):
    return [line.strip("- ").strip() for line in text.split("\n") if line.strip().startswith("-")]