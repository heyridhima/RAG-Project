import os
import textwrap
from dotenv import load_dotenv
from anthropic import Anthropic

load_dotenv(override=True)
api_key = os.getenv("ANTHROPIC_API_KEY")

def build_context(chunks: list[dict]) -> str:
    """Format retrieved chunks into a single context string."""

    parts = []
    for i, chunk in enumerate(chunks, 1):
        parts.append(f"[Source {i}: {chunk['source']}, page: {chunk['page']}]\n{chunk['text']}")
    return "\n\n".join(parts)

def build_prompt(query: str, context: str) -> tuple[str, str]:
    """Return (system_prompt, user_prompt)."""

    system_prompt = textwrap.dedent("""\
        You are a precise question-answering assistant. Your task:

        1. Answer the user's question using ONLY the information provided in the context below.
        2. Do not use your own knowledge or assumptions.
        3. If the context does not contain enough information to answer, say "I don't have enough information in the provided context to answer this question."
        4. Cite sources using the [Source N] notation.
        5. Be concise and factual.""").strip()

    user_prompt = f"Context:\n{context}\n\nQuestion: {query}"

    return (system_prompt, user_prompt)

def generate_answer(query: str, chunks: list[dict]) -> str:
    """Build prompt, call LLM, return answer text."""

    context = build_context(chunks)
    system_prompt, user_prompt = build_prompt(query, context)

    if not api_key:
        raise RuntimeError("ANTHROPIC_API_KEY not set in .env")
    client = Anthropic(api_key=api_key)
    response = client.messages.create(
        model="claude-haiku-4-5",
        temperature=0,
        max_tokens=1024,
        system=system_prompt,
        messages=[{"role": "user", "content": user_prompt}],
    )
    return response.content[0].text


if __name__ == "__main__":
    from retriever import retrieve
    
    queries = [
        "What is multi-head attention?",
        "Who won the 2022 FIFA World Cup?",  # trap question
    ]
    
    for q in queries:
        print(f"\n=== Query: {q}")
        chunks = retrieve(q, k=5, max_distance=0.7)
        if not chunks:
            print("No relevant chunks found.")
            continue
        answer = generate_answer(q, chunks)
        print(f"\nAnswer: {answer}")
        print(f"\nSources used: {len(chunks)}")
        for i, c in enumerate(chunks, 1):
            print(f"  [Source {i}] {c['source']} p.{c['page']} (dist={c['distance']:.3f})")