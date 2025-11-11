# Baltimore Mental Health Platform
## Capstone Project Presentation

---

## Project Overview

A comprehensive mental health data platform that enables discovery, ingestion, and intelligent querying of mental health datasets from CDC for Baltimore City, Maryland.

### Key Features
- **AI-Powered Search**: Semantic search using OpenAI embeddings
- **Multi-Level Data**: County and census tract granularity
- **RESTful API**: Easy integration with other systems
- **Scalable Storage**: PostgreSQL + MinIO object storage
- **Interactive UI**: API documentation and data exploration

---

## Technical Architecture

### Technology Stack
- **Backend**: FastAPI (Python)
- **Database**: PostgreSQL 16 with pgvector extension
- **Object Storage**: MinIO (S3-compatible)
- **AI/ML**: OpenAI GPT-4 and text-embedding-3-small
- **Deployment**: Docker Compose

### Architecture Diagram
```
┌─────────────┐
│   User      │
└──────┬──────┘
       │
       ▼
┌─────────────────┐
│  FastAPI        │ ← RESTful API
│  Backend        │   Port 8000
└────┬────────┬───┘
     │        │
     ▼        ▼
┌─────────┐ ┌────────────┐
│PostgreSQL│ │   MinIO    │
│+ pgvector│ │  Storage   │
│Port 5432 │ │ Port 9001  │
└──────────┘ └────────────┘
     │             │
     ▼             ▼
  Metadata    Parquet Files
  Embeddings  (Raw Data)
```

---

## Data Sources & Coverage

### Baltimore Mental Health Datasets (4 Total)

| Dataset | Rows | Description | Granularity |
|---------|------|-------------|-------------|
| PLACES County 2024 | 3,144 | Mental health indicators | County |
| PLACES Census Tract 2024 | 50,000 | Depression, anxiety measures | Census Tract |
| Injury/Overdose/Violence | 50,000 | Suicide, overdose data | County |
| Anxiety/Depression Indicators | 16,794 | COVID-19 era symptoms | State/National |

**Total Data Points**: 119,938 rows
**AI Chunks Created**: 270 with embeddings
**Coverage**: Baltimore City, Maryland

---

## User Stories Implemented

### 1. Public Health Researcher
**Story**: "As a public health researcher, I want to search for mental health datasets for Baltimore City"

**Implementation**:
- Catalog search API: `/catalog/search?org=CDC&q=baltimore`
- Dataset listing: `/datasets`
- Status checking: `/rag_status?org=CDC`

### 2. Data Analyst
**Story**: "As a data analyst, I want to connect to the database using SQL tools"

**Implementation**:
- PostgreSQL accessible on port 5432
- DBeaver/pgAdmin compatible
- 4 tables: datasets, data_chunks, chunks, meta_docs

### 3. Policy Maker
**Story**: "As a policy maker, I want to see quick statistics about mental health data"

**Implementation**:
- Statistics endpoint: `/stats`
- Real-time health check: `/health`
- Dataset summaries with counts

### 4. System Administrator
**Story**: "As a system admin, I want to view raw data files"

**Implementation**:
- MinIO console at http://localhost:9001
- Parquet files viewable and downloadable
- S3-compatible API for automation

### 5. Developer
**Story**: "As a developer, I want API documentation"

**Implementation**:
- Interactive Swagger UI: http://localhost:8000/docs
- OpenAPI specification
- Try endpoints directly in browser

### 6. Data Engineer
**Story**: "As a data engineer, I want to verify data quality"

**Implementation**:
- Embedding verification queries
- Data quality checks in PostgreSQL
- 100% embedding coverage (270/270)

---

## Demonstration Flow

### Part 1: System Setup (2 minutes)
1. Start platform: `docker compose up -d`
2. Check health: `curl http://localhost:8000/health`
3. Verify services: `docker compose ps`

### Part 2: Data Exploration (3 minutes)
1. View datasets: API call to `/datasets`
2. Check AI status: API call to `/rag_status?org=CDC`
3. Show statistics: API call to `/stats`

### Part 3: Database Access (3 minutes)
1. Connect DBeaver to PostgreSQL
2. Run query to show datasets and chunks
3. Demonstrate data quality check

### Part 4: Storage & API (2 minutes)
1. Open MinIO console
2. Show Parquet files in `mh-raw` bucket
3. Open Swagger API docs
4. Try live API endpoint

---

## Key Achievements

✅ **Data Integration**
- Successfully ingested 119,938 rows from CDC
- 4 different dataset types for Baltimore
- County and census tract granularity

✅ **AI Implementation**
- 270 AI chunks with OpenAI embeddings
- Vector similarity search ready
- 1536-dimensional embedding vectors

✅ **Scalable Architecture**
- Fully containerized with Docker
- PostgreSQL with specialized vector extension
- S3-compatible object storage
- RESTful API design

✅ **Production Ready**
- Health monitoring endpoints
- Error handling and logging
- Database constraints and indexing
- API documentation

---

## Technical Highlights

### Database Schema
```sql
-- Datasets table (metadata)
CREATE TABLE datasets (
  dataset_id SERIAL PRIMARY KEY,
  name TEXT NOT NULL,
  source_org TEXT,
  source_url TEXT,
  description TEXT,
  first_ingested_at TIMESTAMP DEFAULT NOW(),
  UNIQUE (source_org, source_url)
);

-- Data chunks with vector embeddings
CREATE TABLE data_chunks (
  id BIGSERIAL PRIMARY KEY,
  org TEXT NOT NULL,
  dataset_uid TEXT NOT NULL,
  chunk_id TEXT NOT NULL UNIQUE,
  content TEXT NOT NULL,
  summary TEXT,
  metadata JSONB,
  embedding vector(1536),
  created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);
```

### Vector Search Index
- **Type**: IVFFlat (Inverted File Flat)
- **Operation**: Cosine similarity
- **Dimensions**: 1536 (OpenAI text-embedding-3-small)
- **Performance**: Optimized for semantic search

---

## API Endpoints

### Core Endpoints
- `GET /health` - System health check
- `GET /datasets` - List all datasets
- `GET /stats` - Platform statistics
- `GET /rag_status` - AI indexing status

### Data Operations
- `POST /ingest` - Ingest new dataset
- `POST /index_dataset` - Create AI embeddings
- `POST /batch_index` - Batch index datasets
- `DELETE /datasets/{uid}/index` - Remove index

### Search & Query
- `POST /semantic/search` - Semantic metadata search
- `POST /rag/query` - Natural language queries
- `POST /answer/query` - AI-powered Q&A

---

## Demo Commands

### Check System Status
```bash
# Health check
curl http://localhost:8000/health

# View all services
docker compose ps
```

### View Data
```bash
# List datasets
curl http://localhost:8000/datasets | python -m json.tool

# Check AI status
curl http://localhost:8000/rag_status?org=CDC | python -m json.tool

# Get statistics
curl http://localhost:8000/stats | python -m json.tool
```

### Database Queries
```sql
-- View all datasets with AI chunks
SELECT
    d.name,
    COUNT(dc.id) as ai_chunks,
    d.first_ingested_at
FROM datasets d
LEFT JOIN data_chunks dc ON d.source_org = dc.org
GROUP BY d.dataset_id, d.name, d.first_ingested_at;

-- Verify embedding quality
SELECT
    org,
    dataset_uid,
    COUNT(*) as total,
    COUNT(embedding) as with_embeddings
FROM data_chunks
GROUP BY org, dataset_uid;
```

---

## Future Enhancements

### Phase 2 Features
- [ ] Streamlit UI for non-technical users
- [ ] More data sources (SAMHSA, local health dept)
- [ ] Automated daily data updates
- [ ] Email alerts for new datasets
- [ ] Data visualization dashboard

### Phase 3 Features
- [ ] Multi-user authentication
- [ ] Role-based access control
- [ ] Data export to Excel/CSV
- [ ] Custom report generation
- [ ] Mobile responsive UI

---

## Resources & Links

### Live URLs
- **API Docs**: http://localhost:8000/docs
- **MinIO Console**: http://localhost:9001
- **Database**: localhost:5432 (DBeaver/pgAdmin)

### Documentation
- Main README: `README.md`
- Demo Guide: `DEMO.md`
- Demo Script: `demo_script.bat`

### Credentials
- **PostgreSQL**: app_user / changeme
- **MinIO**: minioadmin / minioadmin123

---

## Questions & Discussion

Thank you for your attention!

**Contact**: [Your Name]
**Repository**: [GitHub URL if applicable]
**Date**: November 2025
