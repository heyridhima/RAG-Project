def chunk_text_fixed(text: str, chunk_size: int, overlap: int, min_chunk_size: int) -> list[str]:
    assert overlap < chunk_size, "overlap can not be grater than chunk_size"
    start = 0
    chunks = []
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        if len(chunk) < min_chunk_size:
            if not chunks:
                chunks.append(chunk)
            else:
                chunks[-1] = chunks[-1][:-overlap] + chunk
        else:
            chunks.append(chunk)
        start = end - overlap
    return chunks

def chunk_documents_fixed(docs: list[dict], chunk_size: int, overlap: int, min_chunk_size: int) -> list[dict]:
    chunks = []
    for doc in docs:
        chunked_text = chunk_text_fixed(doc["text"], chunk_size, overlap, min_chunk_size)
        for chunk_index, chunk in enumerate(chunked_text):
            chunks.append({
                "text": chunk,
                "source": doc["source"],
                "page": doc["page"],
                "chunk_index": chunk_index
            })
    return chunks

def _merge_pieces(
    pieces: list[str], chunk_size: int, overlap: int, separator: str
) -> list[str]:
    """Greedy merge: glue pieces left-to-right up to chunk_size, carrying
    `overlap` characters from the previous chunk to preserve context.
    """
    chunks = []
    current_chunk = ""
    for piece in pieces:
        if (
            current_chunk
            and len(current_chunk) + len(separator) + len(piece) > chunk_size
        ):
            chunks.append(current_chunk)
            current_chunk = current_chunk[-overlap:] if overlap > 0 else ""

        if current_chunk:
            current_chunk += separator + piece
        else:
            current_chunk = piece

    if current_chunk:
        chunks.append(current_chunk)
    return chunks


def chunk_text_recursive(
    text: str,
    chunk_size: int,
    overlap: int,
    min_chunk_size: int = 50,
    separators: list[str] = None,
) -> list[str]:
    """Recursively split text on a hierarchy of separators (paragraph -> line ->
    sentence -> word -> char) and greedily merge the resulting pieces into
    chunks of at most chunk_size, with `overlap` characters carried from the
    previous chunk to preserve context.
    """
    assert overlap < chunk_size, "overlap must be less than chunk_size"
    if separators is None:
        separators = ["\n\n", "\n", ". ", " ", ""]

    # Base case: whole text already fits in a single chunk
    if len(text) <= chunk_size:
        return [text]

    # Pick the first separator that actually appears in the text. The empty
    # string is the per-character fallback and always "matches", so this loop
    # is guaranteed to terminate.
    chosen_sep = ""
    remaining_separators = []
    for sep in separators:
        if sep == "" or sep in text:
            chosen_sep = sep
            remaining_separators = separators[separators.index(sep) + 1:]
            break

    if chosen_sep == "":
        pieces = list(text)
    else:
        pieces = text.split(chosen_sep)

    # Drop empties that come from consecutive separators (e.g. "\n\n\n\n")
    pieces = [p for p in pieces if p]

    # Walk through pieces. Small ones go into a buffer that we merge at this
    # level using `chosen_sep`. When we hit a piece bigger than chunk_size,
    # we first flush the buffer, then RECURSE — the recursive call returns
    # already-finalized chunks, so we extend `final_chunks` directly with
    # them (we must not re-merge them, otherwise we duplicate the overlap).
    final_chunks = []
    buffer = []
    for piece in pieces:
        if len(piece) <= chunk_size:
            buffer.append(piece)
        else:
            if buffer:
                final_chunks.extend(_merge_pieces(buffer, chunk_size, overlap, chosen_sep))
                buffer = []
            final_chunks.extend(
                chunk_text_recursive(
                    piece, chunk_size, overlap, min_chunk_size, remaining_separators
                )
            )
    if buffer:
        final_chunks.extend(_merge_pieces(buffer, chunk_size, overlap, chosen_sep))

    # Post-processing: if the very last chunk is shorter than min_chunk_size,
    # merge it into the previous one so we never emit tiny tails.
    if len(final_chunks) >= 2 and len(final_chunks[-1]) < min_chunk_size:
        tail = final_chunks.pop()
        if overlap > 0 and len(tail) > overlap:
            # Skip the overlap region so we don't duplicate text
            final_chunks[-1] = final_chunks[-1] + tail[overlap:]
        else:
            final_chunks[-1] = final_chunks[-1] + tail

    return final_chunks

def chunk_documents_recursive(
    docs: list[dict],
    chunk_size: int,
    overlap: int,
    min_chunk_size: int = 50,
    separators: list[str] = None,
) -> list[dict]:
    """Identical interface to chunk_documents_fixed but uses recursive splitter."""
    chunks = []
    for doc in docs:
        chunked_text = chunk_text_recursive(
            doc["text"], chunk_size, overlap, min_chunk_size, separators
        )
        for chunk_index, chunk in enumerate(chunked_text):
            chunks.append({
                "text": chunk,
                "source": doc["source"],
                "page": doc["page"],
                "chunk_index": chunk_index,
            })
    return chunks

if __name__ == "__main__":
    from loader import load_documents
    
    processed_files = load_documents("data")
    chunked_text = chunk_documents_fixed(processed_files, 500, 100, 50)
    print(f"Number of pages: {len(processed_files)}")
    print(f"Number of chunks: {len(chunked_text)}")
    print(f"Min chunk len: {min(len(c['text']) for c in chunked_text)}")
    print(f"Max chunk len: {max(len(c['text']) for c in chunked_text)}")
    print(f"Example: {chunked_text[5]}")