from typing import List, TypedDict, Optional
import json

from langgraph.graph import END, StateGraph, START
from langchain_ollama.chat_models import ChatOllama
from langchain_core.output_parsers import StrOutputParser

from langchain_core.globals import set_llm_cache
from langchain_community.cache import SQLiteCache

from src.models.rag import create_retriever
from src.models.reranker import rerank_documents
from src.models.schemas import RAGResponse, Citation
from src.helper.utils import format_docs
from src.models.prompt import create_ragqa_prompt
from src import config


set_llm_cache(SQLiteCache(database_path=config.LLM_CACHE_PATH))


def get_llm_client() -> ChatOllama:
    """
    Create and return an Ollama LLM client.
    
    Returns:
        ChatOllama instance with configured settings.
    """
    client = ChatOllama(**config.OLLAMA_CONFIG)
    return client


class GraphState(TypedDict):
    question: str
    generation: str
    documents: List
    citations: List
    confidence: float


class ChatAgent:
    def __init__(self) -> None:
        self.llm = get_llm_client()
        self.retriever = create_retriever()
        self.workflow = None
        self.agent = None

    def build(self):
        self._create_workflow()

    def _create_workflow(self):
        self.workflow = StateGraph(GraphState)
        self.workflow.add_node("retrieve", self._retrieve_docs)
        self.workflow.add_node("rerank", self._rerank_docs)
        self.workflow.add_node("generate", self._rag_qa)

        self.workflow.add_edge(START, "retrieve")
        self.workflow.add_edge("retrieve", "rerank")
        self.workflow.add_edge("rerank", "generate")
        self.workflow.add_edge("generate", END)
        self.agent = self.workflow.compile()

    def _retrieve_docs(self, state):
        question = state["question"]
        documents = self.retriever.invoke(question)
        return {"documents": documents, "question": question}

    def _rerank_docs(self, state):
        question = state["question"]
        documents = state["documents"]
        if not documents:
            return {"documents": [], "question": question}
        
        reranked = rerank_documents(question, documents, top_k=config.RERANK_TOP_K)
        return {"documents": reranked, "question": question}

    def _extract_citations(self, documents: List) -> List[Citation]:
        citations = []
        for doc in documents:
            meta = doc.metadata
            citations.append(Citation(
                source=meta.get("source", "unknown"),
                page=meta.get("page"),
                slide=meta.get("slide"),
                relevant_text=meta.get("relevant_text", doc.page_content[:200])
            ))
        return citations

    def _rag_qa(self, state):
        prompt = create_ragqa_prompt()
        question = state["question"]
        documents = state["documents"]
        
        if not documents:
            return {
                "documents": [],
                "question": question,
                "generation": "I don't have enough context to answer this question.",
                "citations": [],
                "confidence": 0.0
            }
        
        rag_chain = prompt | self.llm | StrOutputParser()
        generation = rag_chain.invoke({
            "context": format_docs(documents),
            "question": question
        })
        
        citations = self._extract_citations(documents)
        confidence = min(1.0, len(documents) * 0.2)
        
        return {
            "documents": documents,
            "question": question,
            "generation": generation,
            "citations": citations,
            "confidence": confidence
        }

    def chat(self, question: str) -> RAGResponse:
        if self.agent is None:
            raise RuntimeError("Agent not built. Call build() first.")
        result = self.agent.invoke({"question": question})
        return RAGResponse(
            answer=result["generation"],
            citations=[c.model_dump() for c in result.get("citations", [])],
            confidence=result.get("confidence", 0.0)
        )

    def query(self, question: str) -> dict:
        response = self.chat(question)
        return response.to_dict()


def create_agent() -> ChatAgent:
    agent = ChatAgent()
    agent.build()
    return agent


if __name__ == "__main__":
    agent = create_agent()
    response = agent.query("What is in the documents?")
    print(json.dumps(response, indent=2))
