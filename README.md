# Mental Health Data Platform

An AI-powered data discovery and analysis platform for mental health datasets from government sources (CDC, SAMHSA, Baltimore Open Data).

![Platform Version](https://img.shields.io/badge/version-1.0.0-blue)
![Python](https://img.shields.io/badge/python-3.10+-green)
![Docker](https://img.shields.io/badge/docker-required-blue)
![License](https://img.shields.io/badge/license-MIT-green)

## Overview

This platform enables data scientists, researchers, and public health professionals to discover, analyze, and visualize mental health datasets using AI-powered tools. It combines real-time government data API integration with intelligent semantic search and question-answering capabilities.

### Key Features

- **AI-Powered Dataset Discovery** - Natural language search across CDC, SAMHSA, and Baltimore Open Data
- **Semantic Search** - OpenAI embeddings for intelligent dataset matching across 66+ pre-ingested datasets
- **RAG-Based Q&A** - Ask questions about actual data with context-aware answers
- **Auto-Generated Visualizations** - AI analyzes data structure and creates appropriate charts automatically
- **Full Data Pipeline** - Ingest, store, index, and query datasets through a unified interface
- **Real-Time Search** - Live API calls to government data portals (no scraping, no caching)

## Architecture

```
┌─────────────────┐
│  Streamlit UI   │  (Port 8501)
└────────┬────────┘
         │
┌────────▼────────┐
│  FastAPI Backend│  (Port 8000)
└────────┬────────┘
         │
    ┌────┴────┬──────────┬──────────┐
    │         │          │          │
┌───▼───┐ ┌──▼──┐  ┌───▼────┐ ┌───▼────┐
│ PG DB │ │MinIO│  │ OpenAI │ │Socrata │
│+vector│ │ S3  │  │  API   │ │  API   │
└───────┘ └─────┘  └────────┘ └────────┘
```

### Tech Stack

- **Backend**: FastAPI, Python 3.10+, SQLAlchemy
- **Database**: PostgreSQL 16 with pgvector extension
- **Storage**: MinIO (S3-compatible object storage)
- **Frontend**: Streamlit
- **AI/ML**: OpenAI GPT-4o-mini, OpenAI Embeddings
- **Data APIs**: Socrata (CDC, SAMHSA), Baltimore Open Data
- **Containers**: Docker & Docker Compose
- **Data Format**: Apache Parquet

## Quick Start

### Prerequisites

- Docker Desktop (running)
- Python 3.10+
- Git
- OpenAI API key ([Get one here](https://platform.openai.com/api-keys))

### 1. Clone Repository

```bash
git clone https://github.com/sriya19/mental-health-platform.git
cd mental-health-platform
git checkout sankarsh-updates
```

### 2. Set Up Environment

```bash
# Copy environment template
cp .env.example .env

# Edit .env and add your OpenAI API key
# OPENAI_API_KEY=sk-proj-your-key-here
```

### 3. Start Docker Containers

```bash
docker-compose up -d
```

Wait ~30 seconds for all services to initialize.

### 4. Restore Data (Recommended)

```bash
# Restore 66 datasets + 1,077 indexed chunks
python scripts/utils/restore_data.py
```

This gives you a fully populated system with all datasets and indexed data ready to use.

### 5. Start UI

```bash
streamlit run ui_app/streamlit_app.py
```

Open browser: **http://localhost:8501**

## Features in Detail

### 1. Dataset Discovery

**How it works:**
- Enter a user story: "I want to analyze suicide rates across demographics"
- System makes real-time API calls to CDC, SAMHSA, and Baltimore Open Data portals
- Returns ranked, relevant datasets with metadata

**No web scraping, no caching** - all data comes directly from government APIs.

### 2. Semantic Search

- Uses OpenAI embeddings (1536 dimensions) for intelligent matching
- Searches across 66+ pre-ingested datasets
- Filters by organization, category, and relevance score

### 3. RAG-Based Question Answering

- Ask natural language questions: "What is the suicide rate in Maryland?"
- System retrieves relevant chunks from 1,077 indexed records
- Provides answers with source citations and data context

### 4. AI-Generated Visualizations

- AI analyzes dataset structure automatically
- Generates appropriate charts (bar, line, scatter, histogram, heatmap)
- Provides insights and statistics with each visualization

### 5. Data Management

- **Ingest**: Download datasets from CDC/SAMHSA/Baltimore
- **Store**: PostgreSQL metadata + MinIO S3 for parquet files
- **Index**: Vector embeddings for semantic search
- **Export**: Download data in various formats

## Usage

### Discover Datasets

1. Navigate to **"Discover Data"** tab
2. Enter a user story:
   - "I want to analyze youth mental health trends"
   - "Need data on substance abuse treatment facilities"
3. Select organization(s): CDC, SAMHSA, Baltimore, or All
4. Click **"Search for Datasets"**
5. Browse results, preview data, and download

### Ask Questions (RAG)

1. Navigate to **"Ask Questions"** tab
2. Enter a question:
   - "What is the mental health prevalence in Maryland?"
   - "Show me suicide statistics by county"
3. System searches indexed data chunks
4. Returns answer with data sources

### Generate Visualizations

1. Go to **"Visualize Data"** tab
2. Select an ingested dataset
3. Click **"Generate AI Visualizations"**
4. View auto-generated charts and insights

### Ingest New Datasets

Use the backend API:

```bash
curl -X POST "http://localhost:8000/ingest" \
  -H "Content-Type: application/json" \
  -d '{
    "org": "CDC",
    "query": "mental health",
    "auto_index": true,
    "limit": 5
  }'
```

## Project Structure

```
mental-health-platform/
├── backend/              # FastAPI application
│   ├── app/
│   │   ├── main.py       # API endpoints
│   │   ├── catalog.py    # Dataset search (Socrata API)
│   │   ├── semantic.py   # Semantic search with embeddings
│   │   ├── rag.py        # RAG indexing and retrieval
│   │   ├── answer.py     # Q&A system
│   │   ├── viz.py        # AI visualizations
│   │   ├── baltimore_indexer.py  # Baltimore data integration
│   │   └── config.py     # Configuration management
│   └── requirements.txt
├── ui_app/               # Streamlit frontend
│   └── streamlit_app.py
├── infra/                # Infrastructure
│   ├── init.sql          # Database schema
│   └── seed_data.sql     # Sample data (optional)
├── scripts/              # Utility scripts
│   ├── export/          # Backup and export tools
│   ├── download/        # Dataset downloaders
│   ├── index/           # Indexing utilities
│   ├── ingest/          # Data ingesters
│   ├── test/            # Test scripts
│   └── utils/           # General utilities (restore, check status)
├── backups/              # Data backups (40MB, included in repo)
│   ├── database_backup.sql  # PostgreSQL dump
│   ├── minio-data/          # MinIO data files
│   └── README.md            # Restoration instructions
├── docs/                 # Documentation
│   ├── TEAM_SETUP.md        # Team setup guide
│   ├── BALTIMORE_DATA_SOURCES.md
│   ├── DEMO.md
│   └── ...
├── docker-compose.yml    # Container orchestration
├── .env.example          # Environment template
├── .gitignore
└── README.md             # This file
```

## Datasets Included

The platform comes with **66+ pre-ingested datasets** and **1,077 indexed chunks** for RAG:

### CDC (Centers for Disease Control)
- Mental Health Surveillance System
- Behavioral Risk Factor Surveillance System (BRFSS)
- Youth Risk Behavior Survey (YRBSS)
- Suicide statistics and mortality data
- Substance abuse indicators
- Depression and anxiety prevalence

### SAMHSA (Substance Abuse & Mental Health Services)
- National Survey on Drug Use and Health (NSDUH)
- Treatment Episode Data Set (TEDS)
- Mental health service utilization
- State-level mental health indicators
- Treatment facility locator data

### Baltimore Open Data
- Mental health service locations
- Community health indicators
- Overdose statistics
- Social determinants of health
- Local treatment resources

## API Endpoints

### Base URL: `http://localhost:8000`

#### Data Discovery
- `GET /catalog/search?org=CDC&q=mental health` - Search live catalogs
- `POST /semantic/search` - AI-powered semantic search
- `GET /datasets?org=All` - List ingested datasets
- `GET /datasets/preview?org=CDC&uid=abc-123` - Preview dataset

#### RAG/Q&A
- `POST /answer/ask` - Ask questions about data
- `GET /rag_status?org=All` - Check indexing status
- `POST /index_dataset` - Index dataset for RAG
- `POST /batch_index?org=CDC` - Batch index all datasets

#### Data Management
- `POST /ingest` - Ingest new dataset
- `POST /upload_csv` - Upload local CSV file
- `POST /visualizations/generate` - Generate visualizations

#### Health & Stats
- `GET /health` - Service health check
- `GET /stats` - System statistics

**Full API documentation**: http://localhost:8000/docs

## Configuration

### Environment Variables

Edit `.env` file with your configuration:

```bash
# Required: OpenAI API key
OPENAI_API_KEY=sk-proj-your-key-here
OPENAI_MODEL=gpt-4o-mini

# Database (defaults work with Docker)
POSTGRES_HOST=localhost
POSTGRES_PORT=5432
POSTGRES_DB=mh_catalog
POSTGRES_USER=app_user
POSTGRES_PASSWORD=changeme

# MinIO S3 (defaults work with Docker)
MINIO_ENDPOINT=localhost:9000
MINIO_ACCESS_KEY=minioadmin
MINIO_SECRET_KEY=minioadmin

# Optional: Socrata API token (increases rate limits)
SOCRATA_APP_TOKEN=your-token-here
SOCRATA_TIMEOUT=20

# Redis (optional, for caching)
REDIS_URL=redis://localhost:6379/0
```

### Service Ports

- **Backend API**: http://localhost:8000
- **Backend Docs**: http://localhost:8000/docs
- **Streamlit UI**: http://localhost:8501
- **MinIO Console**: http://localhost:9001
- **PostgreSQL**: localhost:5432

## Team Setup

For team members cloning this repository, see [TEAM_SETUP.md](docs/TEAM_SETUP.md) for detailed setup instructions.

**Quick version:**
```bash
git clone https://github.com/sriya19/mental-health-platform.git
cd mental-health-platform
git checkout sankarsh-updates
cp .env.example .env
# Edit .env and add OPENAI_API_KEY
docker-compose up -d
python scripts/utils/restore_data.py
streamlit run ui_app/streamlit_app.py
```

## Development

### Run Backend Locally (Without Docker)

```bash
cd backend
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000
```

### Run Tests

```bash
# Test search functionality
python scripts/test/test_search.py

# Test Baltimore integration
python scripts/test/test_baltimore_search.py

# Test unified search
python scripts/test/test_unified_search.py

# Test AI keyword generation
python scripts/test/test_ai_keywords.py
```

### Create Data Backup

```bash
python scripts/export/backup_data.py
```

Creates backups in `backups/` directory:
- `database_backup.sql` - PostgreSQL dump
- `minio-data/` - MinIO object files

### Restore Data

```bash
python scripts/utils/restore_data.py
```

Restores all datasets and indexed chunks from `backups/` directory.

## Troubleshooting

### "Port already in use"

```bash
docker-compose down
# Change ports in docker-compose.yml if needed
docker-compose up -d
```

### "Database connection failed"

```bash
# Verify containers are running
docker-compose ps

# Check database
docker-compose exec postgres psql -U app_user -d mh_catalog -c "SELECT COUNT(*) FROM datasets;"

# Reset database
docker-compose down -v
docker-compose up -d
sleep 30
python scripts/utils/restore_data.py
```

### "OpenAI API error"

- Verify `OPENAI_API_KEY` is set correctly in `.env`
- Check key validity at https://platform.openai.com/api-keys
- Ensure you have API credits available
- Check rate limits if getting 429 errors

### "No datasets found"

```bash
# Restore data backups
python scripts/utils/restore_data.py

# Or ingest new data
curl -X POST "http://localhost:8000/ingest" \
  -H "Content-Type: application/json" \
  -d '{"org": "CDC", "query": "mental health", "limit": 10}'
```

### "Streamlit won't start"

```bash
# Install dependencies
pip install streamlit requests pandas plotly

# Check if port 8501 is available
# On Windows:
netstat -ano | findstr :8501

# On macOS/Linux:
lsof -i :8501
```

## Performance Notes

- **Indexed chunks**: 1,077 data chunks for fast RAG queries
- **Search latency**: ~2-3 seconds for semantic search across 60+ datasets
- **API rate limits**:
  - Socrata: 1,000 requests/hour (10,000 with app token)
  - OpenAI: Varies by tier (check your limits)
- **Costs**:
  - OpenAI: ~$0.01-0.05 per search/visualization request
  - Storage: Minimal (parquet files are compressed)

## Data Privacy & Ethics

- All datasets are publicly available government data
- No personally identifiable information (PII) is stored
- Data used for research and public health analysis only
- Complies with CDC and SAMHSA data use agreements
- OpenAI API calls follow their data usage policies

## Contributing

### For Team Members

1. Clone the repository
2. Create a feature branch: `git checkout -b feature/your-feature`
3. Make your changes
4. Run tests: `python scripts/test/test_*.py`
5. Commit: `git commit -m "Description"`
6. Push: `git push origin feature/your-feature`
7. Create a Pull Request

### Adding New Data Sources

1. Add catalog search logic in `backend/app/catalog.py`
2. Add ingest logic in `backend/app/main.py`
3. Update tests in `scripts/test/`
4. Document in `docs/`

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Credits

**Built by**: Mental Health Data Platform Team
**Course**: Data Science Capstone
**Date**: November 2024

## Acknowledgments

- **CDC** and **SAMHSA** for providing open data APIs
- **Baltimore Open Data** for local datasets
- **OpenAI** for GPT-4o-mini and embeddings
- **Socrata** for data catalog infrastructure
- **pgvector** for efficient vector similarity search
- **MinIO** for S3-compatible object storage

## Contact & Support

For questions, issues, or contributions:

1. **GitHub Issues**: https://github.com/sriya19/mental-health-platform/issues
2. **Team Lead**: [Your Contact]
3. **Documentation**: Check the `docs/` folder

## Citation

If you use this platform in your research, please cite:

```bibtex
@software{mental_health_platform_2024,
  title = {Mental Health Data Platform: AI-Powered Data Discovery and Analysis},
  author = {Mental Health Data Platform Team},
  year = {2024},
  url = {https://github.com/sriya19/mental-health-platform}
}
```

---

**Happy analyzing!** If you find this platform useful, please star the repository ⭐ and share with colleagues working in public health and mental health research.

For more information, see:
- [Team Setup Guide](docs/TEAM_SETUP.md)
- [Baltimore Data Sources](docs/BALTIMORE_DATA_SOURCES.md)
- [Demo Guide](docs/DEMO.md)
- [API Documentation](http://localhost:8000/docs)
