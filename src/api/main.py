from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager

from src.models.agent import create_agent, ChatAgent
from src.models.schemas import QueryRequest, AddDocumentRequest, HealthResponse, RAGResponse
from src.data.pipeline import IngestionPipeline
from src import config

import uvicorn


agent_instance: Optional[ChatAgent] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global agent_instance
    agent_instance = create_agent()
    yield


app = FastAPI(
    title="RAG API",
    description="Production RAG Service with multi-format ingestion and reranking",
    version=config.API_VERSION,
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health", response_model=HealthResponse)
async def health_check():
    return HealthResponse(
        status="healthy",
        version=config.API_VERSION,
        components={
            "agent": "ready" if agent_instance else "not_initialized",
            "vector_store": "ready",
            "llm_model": config.OLLAMA_CONFIG['model_name']
        }
    )


@app.post("/query", response_model=RAGResponse)
async def query(request: QueryRequest):
    if not agent_instance:
        raise HTTPException(status_code=503, detail="Agent not initialized")
    
    try:
        response = agent_instance.query(request.question)
        return RAGResponse(**response)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/add")
async def add_documents(request: AddDocumentRequest):
    try:
        pipeline = IngestionPipeline()
        count = pipeline.ingest_files(request.file_paths)
        return {"status": "success", "chunks_added": count}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/ingest")
async def ingest_folder(folder_path: Optional[str] = None, incremental: bool = False):
    try:
        pipeline = IngestionPipeline()
        if incremental:
            count = pipeline.ingest_incremental(folder_path)
        else:
            count = pipeline.ingest_folder(folder_path)
        return {"status": "success", "chunks_ingested": count}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


def run_server():
    uvicorn.run(
        "src.api.main:app",
        host=config.API_HOST,
        port=config.API_PORT,
        reload=True
    )


if __name__ == "__main__":
    run_server()
