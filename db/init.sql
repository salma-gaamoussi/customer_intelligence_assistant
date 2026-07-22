CREATE EXTENSION IF NOT EXISTS vector;

CREATE SCHEMA IF NOT EXISTS rag;
CREATE SCHEMA IF NOT EXISTS telco;

CREATE TABLE IF NOT EXISTS rag.chunks (
    id SERIAL PRIMARY KEY,
    chunk_text TEXT NOT NULL,
    embedding vector(384) NOT NULL,
    source_file TEXT NOT NULL,
    chunking_strategy TEXT NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_rag_chunks_source_strategy
    ON rag.chunks (source_file, chunking_strategy);

-- Read-only role for the LangChain SQL agent: SELECT-only on telco, no
-- grants on rag at all (agent cannot see or query rag.chunks).
DO $$
BEGIN
    IF NOT EXISTS (SELECT FROM pg_roles WHERE rolname = 'telco_readonly') THEN
        CREATE ROLE telco_readonly WITH LOGIN PASSWORD 'telco_readonly_pw';
    END IF;
END
$$;

GRANT USAGE ON SCHEMA telco TO telco_readonly;
GRANT SELECT ON ALL TABLES IN SCHEMA telco TO telco_readonly;
ALTER DEFAULT PRIVILEGES IN SCHEMA telco GRANT SELECT ON TABLES TO telco_readonly;
