"""
Backup all ingested datasets and database to share with team/professor
"""
import os
import subprocess
import shutil
from pathlib import Path

def backup_database():
    """Export PostgreSQL database"""
    print("[BACKUP] Backing up PostgreSQL database...")

    backup_dir = Path("backups")
    backup_dir.mkdir(exist_ok=True)

    # Database dump
    cmd = 'docker exec pg pg_dump -U app_user mh_catalog > backups/database_backup.sql'
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)

    if result.returncode == 0:
        size = os.path.getsize("backups/database_backup.sql") / (1024 * 1024)
        print(f"[SUCCESS] Database backed up: {size:.2f} MB")
        return True
    else:
        print(f"[ERROR] Database backup failed: {result.stderr}")
        return False

def backup_minio():
    """Export MinIO/S3 data (Parquet files)"""
    print("\n[BACKUP] Backing up MinIO data...")

    backup_dir = Path("backups/minio-data")
    backup_dir.mkdir(parents=True, exist_ok=True)

    # Copy MinIO data using docker cp
    cmd = 'docker cp mental-health-platform-minio-1:/data/mh-raw backups/minio-data/'
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)

    if result.returncode == 0:
        # Count files
        files = list(backup_dir.rglob("*.parquet"))
        total_size = sum(f.stat().st_size for f in files) / (1024 * 1024)
        print(f"[SUCCESS] MinIO data backed up: {len(files)} files, {total_size:.2f} MB")
        return True
    else:
        print(f"[ERROR] MinIO backup failed: {result.stderr}")
        return False

def create_restore_instructions():
    """Create README for restoring data"""
    readme = """# Data Restore Instructions

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
"""

    with open("backups/README.md", "w") as f:
        f.write(readme)

    print("\n[INFO] Created restore instructions: backups/README.md")

def main():
    print("="*60)
    print("BACKING UP MENTAL HEALTH PLATFORM DATA")
    print("="*60)

    db_success = backup_database()
    minio_success = backup_minio()

    if db_success or minio_success:
        create_restore_instructions()

        print("\n" + "="*60)
        print("[SUCCESS] BACKUP COMPLETE!")
        print("="*60)
        print("\nBackup location: ./backups/")
        print("\nTo share with team/professor:")
        print("1. Commit the backups/ directory to Git")
        print("2. Push to GitHub")
        print("3. They clone and run: python restore_data.py")
        print("\nOR")
        print("1. Compress backups/ folder")
        print("2. Share via Google Drive / email")
        print("="*60)
    else:
        print("\n[ERROR] Backup failed. Check errors above.")

if __name__ == "__main__":
    main()
