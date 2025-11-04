CREATE EXTENSION IF NOT EXISTS vector;

-- tiny smoke tables (we'll replace later)
CREATE TABLE IF NOT EXISTS datasets (
  dataset_id SERIAL PRIMARY KEY,
  name TEXT NOT NULL,
  source_org TEXT,
  source_url TEXT,
  description TEXT,
  first_ingested_at TIMESTAMP DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS meta_docs (
  doc_id SERIAL PRIMARY KEY,
  title TEXT,
  body TEXT,
  embedding vector(1536)
);

-- ---------- RAG storage ----------
CREATE TABLE IF NOT EXISTS chunks (
  id BIGSERIAL PRIMARY KEY,
  dataset_uid TEXT NOT NULL,
  org TEXT NOT NULL,
  chunk_id TEXT NOT NULL,
  content TEXT NOT NULL,
  embedding vector(1536) -- matches OpenAI text-embedding-3-small
);
CREATE INDEX IF NOT EXISTS idx_chunks_dataset ON chunks (org, dataset_uid);
CREATE INDEX IF NOT EXISTS idx_chunks_ivfflat ON chunks USING ivfflat (embedding vector_l2_ops) WITH (lists = 100);
