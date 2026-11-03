import chromadb
from pathlib import Path
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent  # src/ → simple-rag/
CHROMA_PATH = PROJECT_ROOT / "chroma_db"
COLLECTION_NAME = "documents"

def get_collection(persist_dir: str, collection_name: str) -> chromadb.Collection:
    client = chromadb.PersistentClient(path=persist_dir)
    collection = client.get_or_create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"}
    )
    return collection

def add_chunks(collection, embedded_chunks: list[dict]) -> None:
    ids = [f'{embedding["source"]}:{embedding["page"]}:{embedding["chunk_index"]}' for embedding in embedded_chunks]
    embeddings = [embedding["embedding"].tolist() for embedding in embedded_chunks]
    documents = [embedding["text"] for embedding in embedded_chunks]
    metadatas = [{"source": embedding["source"], "page": embedding["page"], "chunk_index": embedding["chunk_index"]} for embedding in embedded_chunks]

    assert len(ids) == len(embeddings) == len(documents) == len(metadatas), "not equal data fields len in embedded_chunks"

    collection.add(
        ids=ids,
        embeddings=embeddings,
        documents=documents,
        metadatas=metadatas
    )

def query_collection(collection, query_embedding: np.ndarray, k: int) -> list[dict]:
    results = collection.query(
        query_embeddings=query_embedding.tolist(),
        n_results=k
    )
    new_results = []
    for i in range(len(results["ids"][0])):
        new_results.append({
            "id": results["ids"][0][i],
            "text": results["documents"][0][i],
            "source": results["metadatas"][0][i]["source"],
            "page": results["metadatas"][0][i]["page"],
            "chunk_index": results["metadatas"][0][i]["chunk_index"],
            "distance": results["distances"][0][i],
        })
    return new_results


if __name__ == "__main__":
    from chunker import chunk_documents_fixed
    from loader import load_documents
    from embedder import embed_documents, embed_text

    # 1. index
    docs = load_documents("data")
    chunks = chunk_documents_fixed(docs, 500, 100, 50)
    embedded = embed_documents(chunks)
    collection = get_collection(str(CHROMA_PATH), COLLECTION_NAME)
    add_chunks(collection, embedded)

    # 2. query
    query = "What is multi-head attention?"
    query_embedding = embed_text(query)
    results = query_collection(collection, query_embedding, k=3)

    print(f"Total in collection: {collection.count()}")
    for r in results:
        print(f"[{r['distance']:.3f}] {r['source']} p.{r['page']}")
        print(r['text'][:200], "\n---")