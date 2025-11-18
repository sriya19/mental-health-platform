# Baltimore Mental Health Platform - Demo Guide

## Overview
This platform provides AI-powered access to mental health datasets for Baltimore City, Maryland, enabling semantic search and natural language querying.

## Quick Start

### 1. Start the Platform
```bash
docker compose up -d
```

### 2. Verify All Services are Running
```bash
docker compose ps
```

Expected output: All services should be "Up" (postgres, minio, backend)

### 3. Check System Health
```bash
curl http://localhost:8000/health
```

Expected: `{"ok":true,"db":true,"bucket":"mh-raw"}`

---

## User Stories & Demonstrations

### User Story 1: View Available Datasets

**Who:** Public Health Researcher
**Goal:** Discover what Baltimore mental health data is available

**Demo:**
```bash
curl http://localhost:8000/datasets
```

**Expected Result:** List of 4 CDC datasets for Baltimore

---

### User Story 2: Check AI Indexing Status

**Who:** Data Scientist
**Goal:** Verify datasets are indexed for AI queries

**Demo:**
```bash
curl http://localhost:8000/rag_status?org=CDC
```

**Expected Result:**
- 4 indexed datasets
- 270 total AI chunks
- Status: "ready"

---

### User Story 3: Database Access with DBeaver

**Who:** Data Analyst
**Goal:** Run SQL queries on mental health data

**Connection Details:**
- Host: localhost
- Port: 5432
- Database: mh_catalog
- Username: app_user
- Password: changeme

**Sample Query:**
```sql
-- View all datasets with chunk counts
SELECT
    d.name,
    COUNT(dc.id) as ai_chunks,
    d.first_ingested_at
FROM datasets d
LEFT JOIN data_chunks dc ON d.source_org = dc.org
GROUP BY d.dataset_id, d.name, d.first_ingested_at
ORDER BY d.first_ingested_at DESC;
```

---

### User Story 4: View Raw Data Files

**Who:** System Administrator
**Goal:** Access raw Parquet data files

**Demo:**
1. Open: http://localhost:9001
2. Login: minioadmin / minioadmin123
3. Navigate to bucket: `mh-raw`
4. View files in: `raw/CDC/`

**Files:**
- `i46a-9kgh.parquet` - PLACES County Data (3,144 rows)
- `yjkw-uj5s.parquet` - PLACES Census Tract Data (50,000 rows)
- `psx4-wq38.parquet` - Injury/Overdose/Violence (50,000 rows)
- `8pt5-q6wp.parquet` - Anxiety/Depression Indicators (16,794 rows)

---

### User Story 5: API Documentation

**Who:** Developer
**Goal:** Explore available API endpoints

**Demo:**
1. Open: http://localhost:8000/docs
2. Interactive Swagger UI with all endpoints
3. Try any endpoint directly from the browser

**Key Endpoints:**
- `GET /health` - Health check
- `GET /datasets` - List datasets
- `GET /rag_status` - Check AI indexing
- `GET /stats` - System statistics

---

### User Story 6: Verify Data Quality

**Who:** Data Engineer
**Goal:** Ensure AI embeddings are properly created

**Demo:**
```bash
docker exec pg psql -U app_user -d mh_catalog -c "
SELECT
    org,
    dataset_uid,
    COUNT(*) as total_chunks,
    COUNT(embedding) as chunks_with_embeddings
FROM data_chunks
GROUP BY org, dataset_uid;
"
```

**Expected Result:** All chunks should have embeddings (100% coverage)

---

## Key Features Demonstrated

1. **Multi-Source Data Integration**
   - CDC PLACES data (county and census tract level)
   - Injury/overdose/violence statistics
   - Mental health indicators

2. **AI-Powered Indexing**
   - 270 semantic chunks with OpenAI embeddings
   - Vector similarity search ready
   - 1536-dimensional vectors

3. **Scalable Architecture**
   - Docker containerized services
   - PostgreSQL with pgvector extension
   - MinIO S3-compatible object storage
   - FastAPI RESTful backend

4. **Data Coverage**
   - 119,938 total rows ingested
   - Baltimore City and Maryland coverage
   - County and census tract granularity

---

## Technical Architecture

### Components
- **PostgreSQL (pgvector)**: Stores metadata and vector embeddings
- **MinIO**: S3-compatible storage for Parquet files
- **FastAPI**: RESTful API backend
- **OpenAI**: Embedding generation for semantic search

### Data Flow
1. Datasets ingested from CDC Socrata API
2. Data stored as Parquet files in MinIO
3. Content chunked and embedded using OpenAI
4. Vectors stored in PostgreSQL for similarity search
5. API provides access to data and AI queries

---

## Stopping the Platform

```bash
docker compose down
```

To stop and remove all data:
```bash
docker compose down -v
```

---

## Support & Documentation

- API Docs: http://localhost:8000/docs
- MinIO Console: http://localhost:9001
- Database: localhost:5432 (via DBeaver/pgAdmin)
- Main README: See README.md for full documentation
