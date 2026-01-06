import os

from langchain_ollama import OllamaEmbeddings

from src.data.data_loader import chunk_documents
from src.data.data_loader import DataLoader
from src.data.embeddings import generate_embeddings, get_vector_store
from src import config


def create_document_embedding():
    data_loader = DataLoader()
    docs = data_loader.load(config.RAW_DATA_PATH)
    docs = chunk_documents(docs)
    generate_embeddings(docs=docs)


def create_retriever(top_k: int = None):
    top_k = top_k or config.RETRIEVER_TOP_K

    index_file = os.path.join(config.EMBEDDING_PATH, "index.faiss")
    pkl_file = os.path.join(config.EMBEDDING_PATH, "index.pkl")
    if not (os.path.exists(index_file) and os.path.exists(pkl_file)):
        create_document_embedding()

    db = get_vector_store(persist_path=config.EMBEDDING_PATH)
    retriever = db.as_retriever(
        search_type="similarity",
        search_kwargs={"k": top_k},
    )
    return retriever


if __name__ == "__main__":
    # create_document_embedding()
    retriever = create_retriever()
    docs = retriever.invoke("what are few drills techiques?")
    for doc in docs:
        print(doc.page_content)
        print(doc.metadata)
