import argparse
from rich.console import Console
from rich.markdown import Markdown
from retriever import retrieve
from reranker import rerank
from generator import generate_answer

def run_query_loop(k: int = 5, max_distance: float = 0.7, use_reranker: bool = False, retrieve_n: int = 20) -> None:
    """Interactive loop — ask, retrieve, generate, print."""

    print("Welcome to Simple RAG. Type 'quit' to exit.")
    console = Console()

    while True:
        try:
            query = input("\nYour question: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nBye!")
            break
        
        if not query:
            continue   # empty - ignore
        
        if query.lower() in {"quit", "exit", "q"}:
            print("Bye!")
            break
        
        # Retrieve. With reranking on, cast a wider net (retrieve_n candidates)
        # and let the cross-encoder narrow them down to k.
        initial_k = retrieve_n if use_reranker else k
        chunks = retrieve(query, k=initial_k, max_distance=max_distance)
        if not chunks:
            print("\nI don't have relevant information in the indexed documents.")
            continue

        if use_reranker:
            chunks = rerank(query, chunks, top_k=k)

        sources = [f"- [{i}] {chunk['source']}, p.{chunk['page']} (relevance: {1 - chunk['distance']:.3f})" for i, chunk in enumerate(chunks, start=1)]
        sources_text = "\n".join(sources)

        answer = generate_answer(query, chunks)

        answer_text = f"""# Q: {query}

{answer}

## Sources used
{sources_text}
"""
        console.print(Markdown(answer_text))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--k', 
                        help="Top-K for retrieval metrics.",
                        type=int, 
                        default=5)
    parser.add_argument('--max-distance', 
                        help="Max disctance between embeddings - threshold.",
                        type=float, 
                        default=0.7)
    parser.add_argument("--reranker",
                        action="store_true",
                        help="Enable cross-encoder reranking (retrieve --retrieve-n, then rerank to top-K).")
    parser.add_argument("--retrieve-n",
                        type=int,
                        default=20,
                        help="How many candidates to retrieve before reranking. Ignored without --reranker.")
    args = parser.parse_args()
    
    print("Loading models...")
    from embedder import get_model
    get_model()  # lazy load now, not after the first query
    print("Ready!")

    run_query_loop(args.k, args.max_distance, args.reranker, args.retrieve_n)