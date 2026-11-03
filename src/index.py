import argparse
import chromadb
import time
from pathlib import Path
from loader import load_documents
from chunker import chunk_documents_fixed, chunk_documents_recursive
from embedder import embed_documents
from vector_store import get_collection, add_chunks

PROJECT_ROOT = Path(__file__).resolve().parent.parent  # src/ → simple-rag/
CHROMA_PATH = PROJECT_ROOT / "chroma_db"
COLLECTION_NAME = "documents"

def reindex(
    data_dir: str,
    chroma_path: str,
    collection_name: str,
    chunk_size: int,
    overlap: int,
    min_chunk_size: int,
    chunker: str = "fixed",
) -> None:
    t_start = time.time()

    # load docs
    print(f"[1/6] Loading documents from {data_dir}/...")
    docs = load_documents(data_dir)

    # chunking docs
    print(f"[2/6] Chunking {len(docs)} pages with '{chunker}' chunker...")
    if chunker == "fixed":
        chunked_docs = chunk_documents_fixed(docs, chunk_size, overlap, min_chunk_size)
    elif chunker == "recursive":
        # Default separators: paragraph -> line -> sentence -> word -> char
        chunked_docs = chunk_documents_recursive(docs, chunk_size, overlap, min_chunk_size)
    elif chunker == "recursive-clean":
        # Skip \n\n and \n — pypdf emits them as soft-wraps (line breaks
        # inside columns), not as semantic paragraph boundaries.
        chunked_docs = chunk_documents_recursive(
            docs, chunk_size, overlap, min_chunk_size,
            separators=[". ", " ", ""],
        )
    else:
        raise ValueError(
            f"Unknown chunker: {chunker!r} "
            "(expected 'fixed', 'recursive' or 'recursive-clean')"
        )

    # embed docs
    print(f"[3/6] Embedding {len(chunked_docs)} chunks...")
    embedded_docs = embed_documents(chunked_docs)

    # delete old collection
    print(f"[4/6] Deleting old collection {collection_name}...")
    client = chromadb.PersistentClient(path=str(chroma_path))
    
    try:
        client.delete_collection(name=collection_name)
        print(f"      Deleted.")
    except Exception:
        print(f"      No existing collection to delete.")

    # create new collection
    print(f"[5/6] Creating new collection in {str(chroma_path)}...")
    collection = get_collection(str(chroma_path), collection_name)

    # add chunks
    print(f"[6/6] Indexing to ChromaDB...")
    add_chunks(collection, embedded_docs)

    # print stats
    docs_names = set([i["source"] for i in docs])
    elapsed = time.time() - t_start
    print(f"Done! Indexed {len(chunked_docs)} chunks from {len(docs_names)} documents in {elapsed:.1f}s")


if __name__ == "__main__":
    # arg parser
    parser = argparse.ArgumentParser()
    parser.add_argument('--chunk-size', 
                        help="chunk size",
                        type=int, 
                        default=500)
    parser.add_argument('--overlap', 
                        help="overlap of chunks",
                        type=int, 
                        default=100)
    parser.add_argument('--min-chunk-size', 
                        help="min size of chunk. If size of chunk < min_chunk_size, the chunk is added to chunk before", 
                        type=int,
                        default=50)
    parser.add_argument('--data-dir',
                        help="directory of the data folder",
                        type=str,
                        default="data")
    parser.add_argument('--chunker',
                        help="which chunker to use",
                        type=str,
                        choices=["fixed", "recursive", "recursive-clean"],
                        default="fixed")
    args = parser.parse_args()

    reindex(
        data_dir=args.data_dir,
        chroma_path=str(CHROMA_PATH),
        collection_name=COLLECTION_NAME,
        chunk_size=args.chunk_size,
        overlap=args.overlap,
        min_chunk_size=args.min_chunk_size,
        chunker=args.chunker,
    )