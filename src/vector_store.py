import shutil
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.vectorstores import VectorStoreRetriever

from src.config import (
    EMBEDDING_MODEL, EMBEDDING_DEVICE, EMBEDDING_NORMALIZE,
    CHROMA_DIR, TOP_K, SEARCH_TYPE,
)


def create_embedding_model() -> HuggingFaceEmbeddings:
    print(f"Embedding modell: {EMBEDDING_MODEL}...")
    model = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": EMBEDDING_DEVICE},
        encode_kwargs={"normalize_embeddings": EMBEDDING_NORMALIZE},
    )
    dim = len(model.embed_query("test"))
    print(f"Betöltve (dimenzió: {dim})")
    return model


def create_vectorstore(
    chunks: list[Document],
    embedding_model: HuggingFaceEmbeddings,
    *,
    force_rebuild: bool = True,
) -> Chroma:
    if force_rebuild and CHROMA_DIR.exists():
        shutil.rmtree(CHROMA_DIR)

    print(f"Vector store építés ({len(chunks)} chunk)...")
    store = Chroma.from_documents(
        documents=chunks,
        embedding=embedding_model,
        persist_directory=str(CHROMA_DIR),
        collection_name="arxiv_papers",
    )
    print(f"{store._collection.count()} dokumentum indexelve")
    return store


def get_retriever(vectorstore: Chroma) -> VectorStoreRetriever:
    return vectorstore.as_retriever(
        search_type=SEARCH_TYPE,
        search_kwargs={"k": TOP_K},
    )


def test_retriever(retriever: VectorStoreRetriever, queries: list[str]) -> None:
    print("Retrieval teszt:\n")
    for q in queries:
        results = retriever.invoke(q)
        print(f"  Q: {q}")
        for i, doc in enumerate(results):
            src = doc.metadata.get("title", "?")[:60]
            pg = doc.metadata.get("page", "?")
            preview = doc.page_content[:100].replace("\n", " ")
            print(f"    [{i+1}] {src} (p.{pg}): {preview}...")
        print()