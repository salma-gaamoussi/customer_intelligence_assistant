import argparse
import os
import time
from dataclasses import dataclass
from pathlib import Path

import psycopg2
from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pgvector.psycopg2 import register_vector
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, PreTrainedTokenizerBase

load_dotenv()

DEFAULT_DOCUMENTS_DIR = Path("data/documents")

MODEL_MAX_SEQ_LENGTH = 256
TOKENIZER_ID = "sentence-transformers/all-MiniLM-L6-v2"

FIXED_CHUNK_SIZE_TOKENS = 200
FIXED_CHUNK_OVERLAP_TOKENS = 30

# RecursiveCharacterTextSplitter sizes in characters, not tokens. ~3.75
# chars/token is a typical ratio for English, so 750 chars lands near the
# fixed strategy's 200-token target; the post-chunk token check catches
# any chunk where that estimate is off
RECURSIVE_CHUNK_SIZE_CHARS = 750
RECURSIVE_CHUNK_OVERLAP_CHARS = 110


@dataclass
class IngestSummary:
    source_file: str
    chunks_created: int
    avg_chunk_length: float
    max_chunk_tokens: int


def get_database_url() -> str:
    database_url = os.getenv("DATABASE_URL")
    if not database_url:
        raise RuntimeError("DATABASE_URL is not set.")
    return database_url


def get_embedding_model_name() -> str:
    return os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")


def extract_text(pdf_path: Path) -> str:
    reader = PdfReader(str(pdf_path))
    pages = [page.extract_text() or "" for page in reader.pages]
    return "\n".join(pages)


def get_tokenizer() -> PreTrainedTokenizerBase:
    return AutoTokenizer.from_pretrained(TOKENIZER_ID)


def count_tokens(text: str, tokenizer: PreTrainedTokenizerBase) -> int:
    return len(tokenizer.encode(text, add_special_tokens=False))


def chunk_fixed(
    text: str,
    tokenizer: PreTrainedTokenizerBase,
    chunk_size: int = FIXED_CHUNK_SIZE_TOKENS,
    overlap: int = FIXED_CHUNK_OVERLAP_TOKENS,
) -> list[str]:

    token_ids = tokenizer.encode(text, add_special_tokens=False)
    if not token_ids:
        return []
    step = chunk_size - overlap
    chunks: list[str] = []
    start = 0
    while start < len(token_ids):
        window = token_ids[start : start + chunk_size]
        chunks.append(tokenizer.decode(window, skip_special_tokens=True))
        if start + chunk_size >= len(token_ids):
            break
        start += step
    return chunks


def chunk_recursive(
    text: str,
    tokenizer: PreTrainedTokenizerBase,
    chunk_size: int = RECURSIVE_CHUNK_SIZE_CHARS,
    overlap: int = RECURSIVE_CHUNK_OVERLAP_CHARS,
) -> list[str]:

    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=overlap)
    return splitter.split_text(text)


CHUNKERS = {
    "fixed": chunk_fixed,
    "recursive": chunk_recursive,
}


def check_token_limits(
    chunks: list[str], tokenizer: PreTrainedTokenizerBase, source_file: str, strategy: str
) -> int:

    max_tokens = 0
    for i, chunk in enumerate(chunks):
        n_tokens = count_tokens(chunk, tokenizer)
        max_tokens = max(max_tokens, n_tokens)
        if n_tokens > MODEL_MAX_SEQ_LENGTH:
            print(
                f"WARNING: {source_file} ({strategy}) chunk {i} has {n_tokens} tokens, "
                f"exceeds model max of {MODEL_MAX_SEQ_LENGTH} and will be truncated by the embedding model"
            )
    return max_tokens


def embed_chunks(chunks: list[str], model: SentenceTransformer) -> list[list[float]]:
    embeddings = model.encode(chunks, show_progress_bar=False, normalize_embeddings=True)
    return embeddings.tolist()


def delete_existing_chunks(conn: psycopg2.extensions.connection, source_file: str, strategy: str) -> None:
    with conn.cursor() as cur:
        cur.execute(
            "DELETE FROM rag.chunks WHERE source_file = %s AND chunking_strategy = %s",
            (source_file, strategy),
        )


def insert_chunks(
    conn: psycopg2.extensions.connection,
    chunks: list[str],
    embeddings: list[list[float]],
    source_file: str,
    strategy: str,
) -> None:
    with conn.cursor() as cur:
        for chunk_text, embedding in zip(chunks, embeddings):
            cur.execute(
                """
                INSERT INTO rag.chunks (chunk_text, embedding, source_file, chunking_strategy)
                VALUES (%s, %s, %s, %s)
                """,
                (chunk_text, embedding, source_file, strategy),
            )


def ingest_file(
    pdf_path: Path,
    strategy: str,
    model: SentenceTransformer,
    conn: psycopg2.extensions.connection,
) -> IngestSummary:
    text = extract_text(pdf_path)
    chunks = CHUNKERS[strategy](text)
    source_file = pdf_path.name

    # Delete-then-insert makes re-ingesting a file idempotent per (source_file, strategy).
    delete_existing_chunks(conn, source_file, strategy)
    if chunks:
        embeddings = embed_chunks(chunks, model)
        insert_chunks(conn, chunks, embeddings, source_file, strategy)
    conn.commit()

    avg_length = sum(len(c) for c in chunks) / len(chunks) if chunks else 0.0
    return IngestSummary(source_file=source_file, chunks_created=len(chunks), avg_chunk_length=avg_length)


def find_pdfs(documents_dir: Path) -> list[Path]:
    return sorted(documents_dir.glob("*.pdf"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Ingest PDFs into pgvector.")
    parser.add_argument("--strategy", choices=sorted(CHUNKERS), required=True, help="Chunking strategy to use.")
    parser.add_argument(
        "--documents-dir", type=Path, default=DEFAULT_DOCUMENTS_DIR, help="Directory of PDFs to ingest."
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    pdf_paths = find_pdfs(args.documents_dir)
    if not pdf_paths:
        print(f"No PDFs found in {args.documents_dir}")
        return

    print(f"Loading embedding model '{get_embedding_model_name()}'...")
    model = SentenceTransformer(get_embedding_model_name())

    conn = psycopg2.connect(get_database_url())
    register_vector(conn)

    try:
        for pdf_path in pdf_paths:
            start = time.perf_counter()
            summary = ingest_file(pdf_path, args.strategy, model, conn)
            elapsed = time.perf_counter() - start
            print(
                f"{summary.source_file}: {summary.chunks_created} chunks created, "
                f"avg length {summary.avg_chunk_length:.0f} chars "
                f"(strategy={args.strategy}, {elapsed:.1f}s)"
            )
    finally:
        conn.close()


if __name__ == "__main__":
    main()
