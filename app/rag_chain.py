import os
import psycopg2
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from pgvector.psycopg2 import register_vector
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer

load_dotenv()

TOP_K = 4
CHUNKING_STRATEGIES = {"fixed", "recursive"}
NO_INFO_MESSAGE = "I don't have that information in the policy documents."

PROMPT_TEMPLATE = """You are a policy assistant for a telecom retention team.
Answer the question using ONLY the information in the numbered source chunks below.
For every claim you make, cite the source file in square brackets, e.g. [retention_discount_policy.pdf].
Do not use any outside knowledge, and do not guess.
If the chunks do not contain enough information to answer the question, respond with exactly this
sentence and nothing else: "{no_info_message}"

Question: {question}

Source chunks:
{chunks_block}
"""


class RetrievedChunk(BaseModel):
    chunk_text: str
    source_file: str
    similarity: float


class RAGResult(BaseModel):
    answer: str
    chunks: list[RetrievedChunk]


def get_database_url() -> str:
    database_url = os.getenv("DATABASE_URL")
    if not database_url:
        raise RuntimeError("DATABASE_URL is not set.")
    return database_url


def get_chunking_strategy() -> str:
    strategy = os.getenv("RAG_CHUNKING_STRATEGY", "recursive")
    if strategy not in CHUNKING_STRATEGIES:
        raise RuntimeError(
            f"RAG_CHUNKING_STRATEGY={strategy!r} is not one of {sorted(CHUNKING_STRATEGIES)}. "
            "Check .env and make sure ingestion/ingest.py has been run with that strategy."
        )
    return strategy


def get_embedding_model() -> SentenceTransformer:
    model_name = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
    return SentenceTransformer(model_name)


def embed_question(question: str, model: SentenceTransformer) -> list[float]:
    embedding = model.encode([question], normalize_embeddings=True)[0]
    return embedding.tolist()


def retrieve_chunks(
    question_embedding: list[float],
    strategy: str,
    conn: psycopg2.extensions.connection,
    top_k: int = TOP_K,
) -> list[RetrievedChunk]:

    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT chunk_text, source_file, embedding <=> %s::vector AS distance
            FROM rag.chunks
            WHERE chunking_strategy = %s
            ORDER BY embedding <=> %s::vector
            LIMIT %s
            """,
            (question_embedding, strategy, question_embedding, top_k),
        )
        rows = cur.fetchall()

    return [
        RetrievedChunk(chunk_text=chunk_text, source_file=source_file, similarity=1.0 - distance)
        for chunk_text, source_file, distance in rows
    ]


def build_prompt(question: str, chunks: list[RetrievedChunk]) -> str:
    chunks_block = "\n\n".join(
        f'Source file: "{chunk.source_file}"\nContent: {chunk.chunk_text}' for chunk in chunks
    )
    return PROMPT_TEMPLATE.format(question=question, chunks_block=chunks_block, no_info_message=NO_INFO_MESSAGE)


def generate_answer(question: str, chunks: list[RetrievedChunk]) -> str:
    if not chunks:
        return NO_INFO_MESSAGE

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    prompt = build_prompt(question, chunks)
    response = llm.invoke(prompt)
    return response.content


def ask_rag_chain(question: str) -> RAGResult:
    model = get_embedding_model()
    question_embedding = embed_question(question, model)
    strategy = get_chunking_strategy()

    conn = psycopg2.connect(get_database_url())
    register_vector(conn)
    try:
        chunks = retrieve_chunks(question_embedding, strategy, conn)
    finally:
        conn.close()

    if not chunks:
        print(
            f"WARNING: no chunks retrieved for chunking_strategy={strategy!r} — "
            "check rag.chunks has been ingested with this strategy."
        )

    answer = generate_answer(question, chunks)
    return RAGResult(answer=answer, chunks=chunks)
