# Mental Health Platform - Team Setup Guide

This guide helps team members get the complete platform running with ALL datasets and indexed data.

## What's Included

✅ **Full Application Code**
- Backend FastAPI service
- Streamlit UI
- Docker setup

✅ **Complete Data Backup** (40MB in `backups/`)
- 66+ ingested datasets (CDC, SAMHSA, Baltimore)
- 1,077 indexed chunks for RAG Q&A
- All PostgreSQL database tables

✅ **All Utility Scripts** (`scripts/`)
- Export, download, index, ingest, test tools

## Quick Start (10 minutes)

### Prerequisites
- Docker Desktop installed and running
- Python 3.10+
- Git

### Step 1: Clone Repository
```bash
git clone <your-repo-url>
cd mental-health-platform
```

### Step 2: Set Up Environment
```bash
# Copy environment template
cp .env.example .env

# Edit .env and add your OpenAI API key
# OPENAI_API_KEY=sk-proj-your-key-here
```

### Step 3: Start Docker Containers
```bash
docker-compose up -d
```

Wait ~30 seconds for all services to start.

### Step 4: Restore All Data
```bash
python scripts/utils/restore_data.py
```

This will:
- Restore PostgreSQL database (66 datasets metadata)
- Restore MinIO S3 files (all dataset parquet files)
- Restore indexed chunks (1,077 searchable data chunks)

**Takes ~2-3 minutes**

### Step 5: Start UI
```bash
streamlit run ui_app/streamlit_app.py
```

Open browser: **http://localhost:8501**

## Verification

### Check Services
```bash
docker-compose ps
```

All 4 containers should be "healthy" or "running":
- ✅ postgres (pg)
- ✅ minio
- ✅ backend
- ✅ create-bucket

### Check Data
```bash
# Check indexed datasets
curl http://localhost:8000/rag_status?org=All

# Should show:
# - indexed_datasets: 30+
# - total_chunks: 1000+
```

### Test Search
1. Open UI: http://localhost:8501
2. Go to "Discover Data" tab
3. Enter user story: "I want to analyze mental health trends in Baltimore"
4. Should see ~10+ relevant datasets

## What You Get

### Datasets (66+)
- **CDC datasets**: Mental health surveillance, BRFSS, YRBSS
- **SAMHSA datasets**: Substance abuse, treatment data
- **Baltimore datasets**: Local mental health indicators

### Indexed Data (1,077 chunks)
- Enables AI-powered Q&A over actual data
- Ask questions like "What is the suicide rate in Maryland?"
- System searches through indexed data chunks

### Features
- ✅ AI-powered dataset discovery (searches CDC/SAMHSA live)
- ✅ AI-generated visualizations
- ✅ RAG-based question answering
- ✅ Full dataset preview and export
- ✅ Multi-organization search

## Troubleshooting

### "Port already in use"
```bash
# Stop existing containers
docker-compose down

# Change ports in docker-compose.yml if needed
```

### "Database restore failed"
```bash
# Reset everything
docker-compose down -v
docker-compose up -d
sleep 30
python scripts/utils/restore_data.py
```

### "OpenAI API error"
- Make sure you added `OPENAI_API_KEY` to `.env`
- Verify key is valid at https://platform.openai.com/api-keys

### "No datasets found"
```bash
# Check database connection
docker-compose exec postgres psql -U app_user -d mh_catalog -c "SELECT COUNT(*) FROM datasets;"

# Should return 60+
```

## Development

### Run Backend Directly
```bash
cd backend
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000
```

### Run Tests
```bash
# Test search functionality
python scripts/test/test_search.py

# Test Baltimore indexing
python scripts/test/test_baltimore_search.py
```

### Ingest New Datasets
```bash
# Search and ingest from CDC
curl -X POST "http://localhost:8000/ingest" \
  -H "Content-Type: application/json" \
  -d '{"org": "CDC", "query": "mental health", "auto_index": true}'
```

## Project Structure

```
mental-health-platform/
├── backend/           # FastAPI backend
├── ui_app/            # Streamlit UI
├── infra/             # Database schema
├── backups/           # Data backups (INCLUDED IN REPO)
├── scripts/           # Utility scripts
│   ├── export/        # Data export tools
│   ├── download/      # Dataset downloaders
│   ├── index/         # Indexing tools
│   ├── ingest/        # Data ingesters
│   ├── test/          # Test scripts
│   └── utils/         # General utilities
├── docs/              # Documentation
├── docker-compose.yml # Docker setup
└── .env               # Your config (NOT in repo)
```

## API Endpoints

**Backend:** http://localhost:8000

### Data Discovery
- `GET /catalog/search?org=CDC&q=mental health` - Search catalog
- `POST /semantic/search` - AI-powered semantic search

### Data Access
- `GET /datasets?org=All` - List ingested datasets
- `GET /datasets/preview?org=CDC&uid=abc-123&rows=1000` - Preview data

### RAG/Q&A
- `POST /answer/ask` - Ask questions about data
- `GET /rag_status?org=All` - Check indexing status

### Data Management
- `POST /ingest` - Ingest new dataset
- `POST /index_dataset` - Index dataset for RAG
- `POST /upload_csv` - Upload local CSV

## Support

If you encounter issues:
1. Check this guide's Troubleshooting section
2. Verify all prerequisites are installed
3. Check Docker Desktop is running
4. Contact the team lead

## Credits

**Built by:** Mental Health Data Platform Team
**Course:** Data Science Capstone
**Date:** November 2024

---

**Happy coding!** 🚀
