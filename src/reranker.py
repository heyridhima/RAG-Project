from sentence_transformers import CrossEncoder

MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"
_model = None

def get_model() -> "CrossEncoder":
    """Load cross-encoder on first use, return cached instance."""
    global _model
    if _model is None:
        _model = CrossEncoder(MODEL_NAME)
    return _model

def rerank(query: str, chunks: list[dict], top_k: int = 5) -> list[dict]:
    """Score (query, chunk_text) pairs with cross-encoder, sort by score,
    return top_k chunks. Adds 'rerank_score' field to each returned chunk.
    """
    # 1. Edge case: empty chunks -> return []
    if len(chunks) == 0:
        return []
    
    # 2. Get model
    model = get_model()
    # 3. Build pairs
    pairs = [(query, c['text']) for c in chunks]
    # 4. scores
    scores = model.predict(pairs)

    # 5. Attach score to each chunk
    for i, c in enumerate(chunks):
        c['rerank_score'] = float(scores[i])
    # 6. Sort chunks by rerank_score desc
    chunks = sorted(chunks, key=lambda c: c['rerank_score'], reverse=True)

    # 7. Return top_k slice
    return chunks[:top_k]

if __name__ == "__main__":
    from retriever import retrieve

    query = "What is multi-head attention?"
    initial = retrieve(query, k=20, max_distance=2.0)

    print("=== BEFORE rerank (top 10 by distance) ===")
    for i, c in enumerate(initial[:10], 1):
        print(f"  [{i}] dist={c['distance']:.3f}  {c['source']} p.{c['page']}: {c['text'][:80]}...")

    reranked = rerank(query, initial, top_k=5)

    print("\n=== AFTER rerank (top 5 by cross-encoder) ===")
    for i, c in enumerate(reranked, 1):
        print(f"  [{i}] rerank={c['rerank_score']:.3f} dist={c['distance']:.3f}  {c['source']} p.{c['page']}: {c['text'][:80]}...")