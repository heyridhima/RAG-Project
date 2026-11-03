from sentence_transformers import SentenceTransformer
import numpy as np


MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
_model = None  # lazy load

def get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        _model = SentenceTransformer(MODEL_NAME)
    return _model

def embed_text(text: str) -> np.ndarray:
    model = get_model()
    embedded_text = model.encode(text, normalize_embeddings=True)
    return embedded_text

def embed_documents(chunks: list[dict]) -> list[dict]:
    batch_size = 32
    model = get_model()
    batched_chunks = [chunks[i:i + batch_size] for i in range(0, len(chunks), batch_size)]
    for batch in batched_chunks:
        batch_text = [chunk["text"] for chunk in batch]
        embeddings = model.encode(batch_text, normalize_embeddings=True)
        for chunk, embedding in zip(batch, embeddings):
            chunk["embedding"] = embedding
    return chunks


if __name__ == "__main__":
    from chunker import chunk_documents_fixed
    from loader import load_documents

    docs = load_documents("data")
    chunks = chunk_documents_fixed(docs, 500, 100, 50)
    embedded = embed_documents(chunks)

    print(f"Embeddings count: {len(embedded)}")
    print(f"Vector dim: {embedded[0]['embedding'].shape}")
    print(f"First 5 values: {embedded[0]['embedding'][:5]}")
    print(f"Vector norm: {np.linalg.norm(embedded[0]['embedding'])}")  # ~1.0, bcs of normalize

    sentences = [
        "Cat is on the mat.",
        "A feline rests on the rug.",
        "Today's weather is sunny.",
    ]
    embeddings = [embed_text(z) for z in sentences]

    from numpy import dot  # dot = cos similarity after normalize
    print(f"cat vs feline: {dot(embeddings[0], embeddings[1]):.3f}")
    print(f"cat vs weather: {dot(embeddings[0], embeddings[2]):.3f}")