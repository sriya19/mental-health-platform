# Data Restore Instructions

This directory contains backup data for the Mental Health Platform.

## What's Included:
- `database_backup.sql` - PostgreSQL database with all ingested datasets metadata and indexed chunks
- `minio-data/` - All Parquet files (actual dataset content)

## How to Restore:

### Step 1: Start the containers
```bash
docker-compose up -d
```

### Step 2: Wait for services to be healthy (30 seconds)
```bash
sleep 30
```

### Step 3: Restore the database
```bash
docker exec -i pg psql -U app_user -d mh_catalog < backups/database_backup.sql
```

### Step 4: Restore MinIO data
```bash
docker cp backups/minio-data/mh-raw mental-health-platform-minio-1:/data/
```

### Step 5: Restart backend to pick up changes
```bash
docker-compose restart backend
```

### Step 6: Open the UI
Navigate to http://localhost:8501

You should now see all 30+ datasets already ingested and indexed!

## For Team Members Adding New Datasets:
After restoring, you can:
1. Search for datasets
2. Ingest new datasets (won't affect existing ones)
3. Index new datasets
4. Everyone works on the same data

## File Sizes:
- Database: ~24MB (all metadata + indexed chunks)
- MinIO data: Varies based on datasets (Parquet files)

## Note:
Make sure your `.env` file has your own OpenAI API key before running.
