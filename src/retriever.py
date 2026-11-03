from embedder import embed_text
from vector_store import get_collection, query_collection, CHROMA_PATH, COLLECTION_NAME
from typing import Optional

_collection = None

def get_or_load_collection():
    global _collection
    if _collection is None:
        _collection = get_collection(str(CHROMA_PATH), COLLECTION_NAME)
    return _collection

def retrieve(query: str, source: Optional[str] = None, k: int = 5, max_distance: float = 0.7) -> list[dict]:
    collection = get_or_load_collection()
    embedded_query = embed_text(query)
    results = query_collection(collection, embedded_query, k=k)
    thresholded_results = [r for r in results if r["distance"] <= max_distance]
    if source is not None:
        thresholded_results = [r for r in thresholded_results if r["source"] == source]
    return thresholded_results


if __name__ == "__main__":
    print("=" * 70)
    print("TEST 1: On-topic queries (relevant)")
    print("=" * 70)
    queries = [
        "What is multi-head attention?",
        "How does scaled dot-product attention work?",
        "What is retrieval-augmented generation?",
    ]
    for q in queries:
        print(f"\n--- Query: {q}")
        results = retrieve(q, k=3)
        if not results:
            print("  No results above threshold.")
        for r in results:
            print(f"  [{r['distance']:.3f}] {r['source']} p.{r['page']}: {r['text'][:100]}...")

    print("\n" + "=" * 70)
    print("TEST 2: Trap query (not in the documents)")
    print("=" * 70)
    trap_query = "Who won the 2022 FIFA World Cup?"

    print(f"\n--- With threshold max_distance=0.7:")
    results_filtered = retrieve(trap_query, k=3, max_distance=0.7)
    print(f"  Result count: {len(results_filtered)}")
    for r in results_filtered:
        print(f"  [{r['distance']:.3f}] {r['source']} p.{r['page']}: {r['text'][:100]}...")

    print(f"\n--- Without threshold (max_distance=2.0):")
    results_unfiltered = retrieve(trap_query, k=3, max_distance=2.0)
    print(f"  Result count: {len(results_unfiltered)}")
    for r in results_unfiltered:
        print(f"  [{r['distance']:.3f}] {r['source']} p.{r['page']}: {r['text'][:100]}...")

    print("\n" + "=" * 70)
    print("TEST 3: Source filtering (only attention.pdf)")
    print("=" * 70)
    q = "What is attention?"
    print(f"\n--- Query: {q} (only attention.pdf)")
    results = retrieve(q, source="attention.pdf", k=3)
    for r in results:
        print(f"  [{r['distance']:.3f}] {r['source']} p.{r['page']}: {r['text'][:100]}...")

    print(f"\n--- Query: {q} (only rag.pdf)")
    results = retrieve(q, source="rag.pdf", k=3)
    for r in results:
        print(f"  [{r['distance']:.3f}] {r['source']} p.{r['page']}: {r['text'][:100]}...")

    print("\n" + "=" * 70)
    print("TEST 4: Effect of k on result count and quality")
    print("=" * 70)
    q = "transformer architecture"
    for k in [1, 3, 10]:
        results = retrieve(q, k=k, max_distance=2.0)  # no threshold, see all
        print(f"\n--- k={k}, returned {len(results)} results")
        distances = [r["distance"] for r in results]
        if distances:
            print(f"    distance range: {min(distances):.3f} - {max(distances):.3f}")