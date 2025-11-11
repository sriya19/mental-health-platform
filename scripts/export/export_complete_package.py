"""
Complete export package for professor - includes Docker containers AND data
Creates a single package with everything needed to run the platform
"""
import subprocess
import os
import shutil
from pathlib import Path
import time

def check_docker():
    """Check if Docker is running"""
    result = subprocess.run("docker info", shell=True, capture_output=True)
    return result.returncode == 0

def backup_database(export_dir):
    """Export PostgreSQL database"""
    print("\n[STEP 1/5] Backing up database...")

    backup_dir = export_dir / "backups"
    backup_dir.mkdir(parents=True, exist_ok=True)

    cmd = f'docker exec pg pg_dump -U app_user mh_catalog > "{backup_dir}/database_backup.sql"'
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)

    if result.returncode == 0:
        size = os.path.getsize(backup_dir / "database_backup.sql") / (1024 * 1024)
        print(f"[SUCCESS] Database backed up: {size:.2f} MB")
        return True
    else:
        print(f"[ERROR] Database backup failed: {result.stderr}")
        return False

def backup_minio(export_dir):
    """Export MinIO/S3 data (Parquet files)"""
    print("\n[STEP 2/5] Backing up MinIO data...")

    backup_dir = export_dir / "backups" / "minio-data"
    backup_dir.mkdir(parents=True, exist_ok=True)

    cmd = f'docker cp mental-health-platform-minio-1:/data/mh-raw "{backup_dir}/"'
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)

    if result.returncode == 0:
        files = list(backup_dir.rglob("*.parquet"))
        total_size = sum(f.stat().st_size for f in files) / (1024 * 1024)
        print(f"[SUCCESS] MinIO data backed up: {len(files)} files, {total_size:.2f} MB")
        return True
    else:
        print(f"[ERROR] MinIO backup failed: {result.stderr}")
        return False

def export_docker_images(export_dir):
    """Export all Docker images as tar files"""
    print("\n[STEP 3/5] Exporting Docker images...")
    print("This may take 5-10 minutes...\n")

    images_dir = export_dir / "docker-images"
    images_dir.mkdir(exist_ok=True)

    images = [
        ("mental-health-platform-backend", "backend-image.tar"),
        ("pgvector/pgvector:pg16", "postgres-image.tar"),
        ("minio/minio:latest", "minio-image.tar"),
    ]

    for image_name, filename in images:
        print(f"[EXPORT] Exporting {image_name}...")
        output_path = images_dir / filename

        cmd = f'docker save {image_name} -o "{output_path}"'
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)

        if result.returncode == 0:
            size = os.path.getsize(output_path) / (1024 * 1024)
            print(f"[SUCCESS] Exported: {size:.1f} MB")
        else:
            print(f"[ERROR] Failed: {result.stderr}")
            return False

    return True

def create_setup_scripts(export_dir):
    """Create all setup scripts for professor"""
    print("\n[STEP 4/5] Creating setup scripts...")

    # Script 1: Load Docker images
    load_script = """@echo off
echo ============================================================
echo STEP 1: LOADING DOCKER IMAGES
echo ============================================================

echo.
echo [INFO] Loading Docker images...
echo This will take a few minutes...
echo.

cd docker-images

docker load -i backend-image.tar
if %ERRORLEVEL% NEQ 0 (
    echo [ERROR] Failed to load backend image
    pause
    exit /b 1
)

docker load -i postgres-image.tar
if %ERRORLEVEL% NEQ 0 (
    echo [ERROR] Failed to load postgres image
    pause
    exit /b 1
)

docker load -i minio-image.tar
if %ERRORLEVEL% NEQ 0 (
    echo [ERROR] Failed to load minio image
    pause
    exit /b 1
)

cd ..

echo.
echo ============================================================
echo [SUCCESS] ALL IMAGES LOADED!
echo ============================================================
echo.
echo Next: Run 2_start_containers.bat
echo.
pause
"""

    # Script 2: Start containers
    start_script = """@echo off
echo ============================================================
echo STEP 2: STARTING CONTAINERS
echo ============================================================

echo.
echo [INFO] Starting Docker containers...
echo.

docker-compose up -d

if %ERRORLEVEL% NEQ 0 (
    echo [ERROR] Failed to start containers
    pause
    exit /b 1
)

echo.
echo [INFO] Waiting 30 seconds for services to be ready...
timeout /t 30 /nobreak

echo.
echo ============================================================
echo [SUCCESS] CONTAINERS RUNNING!
echo ============================================================
echo.
echo Next: Run 3_restore_data.bat
echo.
pause
"""

    # Script 3: Restore data
    restore_script = """@echo off
echo ============================================================
echo STEP 3: RESTORING DATA
echo ============================================================

echo.
echo [INFO] Restoring database and datasets...
echo.

python restore_data.py

if %ERRORLEVEL% NEQ 0 (
    echo [ERROR] Failed to restore data
    pause
    exit /b 1
)

echo.
echo ============================================================
echo [SUCCESS] SETUP COMPLETE!
echo ============================================================
echo.
echo Open your browser: http://localhost:8501
echo.
echo You will see:
echo   - 30+ datasets already loaded
echo   - 1,077 indexed chunks for Q&A
echo   - Full working platform
echo.
pause
"""

    # Write scripts
    with open(export_dir / "1_load_images.bat", "w") as f:
        f.write(load_script)

    with open(export_dir / "2_start_containers.bat", "w") as f:
        f.write(start_script)

    with open(export_dir / "3_restore_data.bat", "w") as f:
        f.write(restore_script)

    # Copy necessary files
    print("[INFO] Copying project files...")

    # Copy docker-compose.yml
    shutil.copy("docker-compose.yml", export_dir / "docker-compose.yml")

    # Copy .env.example
    shutil.copy(".env.example", export_dir / ".env.example")

    # Copy restore_data.py
    shutil.copy("restore_data.py", export_dir / "restore_data.py")

    print("[SUCCESS] Setup scripts created")
    return True

def create_professor_readme(export_dir):
    """Create comprehensive README for professor"""
    print("\n[STEP 5/5] Creating documentation...")

    readme = """# Mental Health Platform - Complete Package

This package contains EVERYTHING needed to run the Mental Health Platform with all datasets.

## What's Included:

### 1. Docker Images (in docker-images/)
- `backend-image.tar` (~500 MB) - FastAPI backend with all code
- `postgres-image.tar` (~150 MB) - PostgreSQL with pgvector
- `minio-image.tar` (~200 MB) - MinIO S3 storage

### 2. Student's Work (in backups/)
- `database_backup.sql` (24 MB) - 30+ ingested datasets metadata
- `minio-data/` - All Parquet files (35 datasets)
- 1,077 indexed chunks for question-answering

### 3. Setup Scripts
- `1_load_images.bat` - Load Docker containers
- `2_start_containers.bat` - Start the platform
- `3_restore_data.bat` - Load student's datasets

---

## Quick Start (10 minutes):

### Prerequisites:
- **Docker Desktop** - Must be installed and running
- **Python 3.10+** - Check with: `python --version`
- **OpenAI API key** - Contact student for test key

### Step 1: Extract Package
Extract this entire folder to your desired location.

### Step 2: Setup Environment
```bash
# Copy the example environment file
copy .env.example .env

# Edit .env and add your OpenAI API key:
OPENAI_API_KEY=sk-your-key-here
```

### Step 3: Run Setup Scripts
```bash
# Double-click each script in order:
1_load_images.bat       # Loads Docker containers (5 min)
2_start_containers.bat  # Starts platform (30 sec)
3_restore_data.bat      # Loads student's datasets (1 min)
```

### Step 4: Open Application
Open browser: http://localhost:8501

---

## What You'll See:

✅ **30+ Health Datasets**
   - CDC surveillance data
   - SAMHSA behavioral health surveys
   - Baltimore City health data

✅ **AI-Powered Search**
   - Semantic search with OpenAI embeddings
   - Intent extraction and query enhancement
   - Relevance ranking

✅ **Question Answering (RAG)**
   - 1,077 indexed text chunks
   - Ask natural language questions
   - Example: "What are drug overdose trends in Baltimore?"

✅ **Data Visualizations**
   - Interactive charts
   - Statistical summaries
   - Dataset analytics

✅ **Full Working Platform**
   - All student's work pre-loaded
   - No manual data ingestion needed
   - Ready for demonstration

---

## System Architecture:

```
Frontend (Streamlit)  →  Backend (FastAPI)  →  PostgreSQL (pgvector)
                                            →  MinIO (S3 Storage)
                                            →  OpenAI API
```

**Technologies Demonstrated:**
- Docker containerization
- Vector databases (pgvector)
- AI/ML integration (OpenAI embeddings)
- REST API design
- Data pipeline engineering
- Cloud storage (S3-compatible)

---

## Troubleshooting:

**Docker not running:**
1. Start Docker Desktop
2. Wait until "Docker Desktop is running" shows
3. Run scripts again

**Port already in use:**
```bash
docker-compose down
# Then start again
```

**Restore failed:**
```bash
# Check containers are running:
docker-compose ps

# Should see: backend, pg, minio all "Up"
```

**API key error:**
- Make sure .env file has valid OPENAI_API_KEY
- Get test key from student if needed

**Can't access http://localhost:8501:**
```bash
# Check if Streamlit is running:
docker-compose logs backend

# Restart if needed:
docker-compose restart backend
```

---

## File Sizes:

- Docker images: ~850 MB
- Database backup: ~24 MB
- Parquet files: ~1-50 MB
- **Total package: ~1-1.5 GB**

---

## Evaluation Criteria:

This project demonstrates:

1. **Docker & DevOps**
   - Multi-container architecture
   - Container orchestration
   - Volume management
   - Environment configuration

2. **Database Design**
   - PostgreSQL with vector extension
   - Schema design for health data
   - Indexing strategy
   - Data persistence

3. **AI/ML Integration**
   - OpenAI embeddings (1536-dim vectors)
   - Semantic search implementation
   - RAG (Retrieval-Augmented Generation)
   - Intent extraction with LLMs

4. **Data Engineering**
   - Multi-source data ingestion (CDC, SAMHSA)
   - Parquet file handling
   - ETL pipelines
   - Data validation

5. **API Development**
   - RESTful API design
   - FastAPI framework
   - Async operations
   - Error handling

6. **Frontend Development**
   - Streamlit web interface
   - Interactive visualizations
   - State management
   - User experience

7. **Real-World Application**
   - 30+ actual datasets ingested
   - Production-ready deployment
   - Reproducible environment
   - Complete documentation

---

## Contact:

If you encounter any issues or need assistance:
- Contact the student
- Request OpenAI API test key
- Check system requirements

---

**Built by:** [Student Name]
**Course:** [Course Name]
**Date:** November 2024
**Technologies:** Python, Docker, PostgreSQL, FastAPI, Streamlit, OpenAI, MinIO

---

## Commands Reference:

```bash
# View running containers
docker-compose ps

# View logs
docker-compose logs backend
docker-compose logs pg

# Stop platform
docker-compose down

# Start platform (after initial setup)
docker-compose up -d

# Restart a service
docker-compose restart backend
```

Enjoy exploring the Mental Health Platform!
"""

    with open(export_dir / "README.md", "w", encoding="utf-8") as f:
        f.write(readme)

    print("[SUCCESS] Documentation created")
    return True

def create_student_instructions():
    """Create instructions for student on how to share"""

    instructions = """# How to Send Package to Professor

You have created a complete package in `professor-complete-package/` folder.

## What's in the Package:

✅ Docker containers (3 tar files, ~850 MB)
✅ Your database backup (24 MB with 30+ datasets)
✅ Your Parquet files (35 datasets)
✅ Setup scripts (3 automated batch files)
✅ Complete documentation

Total size: ~1-1.5 GB

---

## Sharing Options:

### Option 1: Google Drive (Recommended)

1. **Compress the package:**
   - Right-click `professor-complete-package` folder
   - Send to → Compressed (zipped) folder
   - Result: ~500-700 MB zip file

2. **Upload to Google Drive:**
   - Upload the zip file
   - Right-click → Share → Get link
   - Set to "Anyone with the link can view"

3. **Email template:**

```
Subject: Mental Health Platform - Complete Docker Package

Dear Professor [Name],

I've prepared a complete package of my Mental Health Platform capstone project.

Download link: [Your Google Drive Link]

Quick start:
1. Extract the zip file
2. Make sure Docker Desktop is running
3. Double-click: 1_load_images.bat
4. Double-click: 2_start_containers.bat
5. Double-click: 3_restore_data.bat
6. Open browser: http://localhost:8501

The package includes:
- All pre-built Docker containers (~850 MB)
- 30+ datasets I've already ingested and indexed
- 1,077 searchable text chunks for Q&A
- Complete working application

Setup time: ~10 minutes
System requirements: Docker Desktop, Python 3.10+

Note: You'll need an OpenAI API key. I can provide a test key for evaluation.

Please let me know if you encounter any issues!

Best regards,
[Your Name]
```

### Option 2: OneDrive

Same process as Google Drive - upload zip and share link.

### Option 3: USB Drive (In-Person)

If meeting professor in person:
1. Copy `professor-complete-package` folder to USB drive
2. Include printed copy of README.md
3. Provide OpenAI API key on paper

### Option 4: Cloud Storage Alternative

If file is too large for your cloud storage:
- WeTransfer (free up to 2GB)
- Dropbox
- MEGA (free 20GB)

---

## What Professor Gets:

### Your Code:
- FastAPI backend implementation
- Streamlit UI
- Semantic search with AI
- RAG system for Q&A
- Data ingestion pipeline

### Your Work:
- 30+ datasets already loaded
- 1,077 indexed chunks
- All Parquet files
- Working demonstration

### Easy Setup:
- 3 simple batch scripts
- Automated restoration
- Complete documentation
- ~10 minutes to run

---

## Important Notes:

### Before Sharing:

✅ Make sure package is complete:
   - Check docker-images/ has 3 tar files
   - Check backups/ has database_backup.sql
   - Check backups/minio-data/ has Parquet files

✅ Test the package locally (optional):
   - Extract to different folder
   - Run setup scripts
   - Verify everything works

### Security:

🔒 The package does NOT include:
   - Your .env file (API keys)
   - Passwords or secrets

⚠️ Professor needs their own OpenAI API key:
   - You can provide a test key
   - Or they can use their own

### File Sizes to Expect:

- Uncompressed: ~1-1.5 GB
- Compressed (zip): ~500-700 MB
- Upload time: 5-30 minutes depending on connection

---

## Presentation Tips:

When demonstrating to professor:

1. **Show the easy setup:**
   - "Just 3 clicks to run everything"
   - Highlight automation

2. **Demo the features:**
   - Search: "What are drug overdose trends in Baltimore?"
   - Q&A: Ask questions about the data
   - Show visualizations

3. **Explain the architecture:**
   - Docker containerization
   - Vector database with pgvector
   - AI-powered search
   - RAG for question-answering

4. **Highlight the data:**
   - "30+ datasets from CDC, SAMHSA, Baltimore"
   - "1,077 indexed chunks"
   - "Real production data, not test data"

5. **Technical achievements:**
   - Multi-container Docker setup
   - OpenAI integration
   - Semantic search implementation
   - Data pipeline engineering

---

## Backup Plan:

If professor has issues:
1. Be available for quick support
2. Have a video demo ready
3. Consider live demo session

---

Good luck with your presentation! 🎓
"""

    with open("SEND_TO_PROFESSOR.md", "w", encoding="utf-8") as f:
        f.write(instructions)

    print(f"\n[SUCCESS] Created: SEND_TO_PROFESSOR.md")

def main():
    print("="*60)
    print("CREATING COMPLETE PACKAGE FOR PROFESSOR")
    print("="*60)
    print("\nThis will create a package with:")
    print("  - Docker containers (~850 MB)")
    print("  - Your database backup (30+ datasets)")
    print("  - Your Parquet files")
    print("  - Setup scripts and documentation")
    print("\nExpected time: 10-15 minutes")
    print("Required disk space: ~2-3 GB")
    print("\n[INFO] Starting export process...")
    print("="*60)

    # Check Docker
    if not check_docker():
        print("\n[ERROR] Docker is not running! Start Docker Desktop first.")
        return

    # Create export directory
    export_dir = Path("professor-complete-package")
    if export_dir.exists():
        print(f"\n[WARNING] {export_dir} already exists. Removing...")
        shutil.rmtree(export_dir)

    export_dir.mkdir(exist_ok=True)

    print(f"\n[INFO] Creating package in: {export_dir}/")

    # Execute all steps
    success = True

    success = success and backup_database(export_dir)
    success = success and backup_minio(export_dir)
    success = success and export_docker_images(export_dir)
    success = success and create_setup_scripts(export_dir)
    success = success and create_professor_readme(export_dir)

    if success:
        create_student_instructions()

        # Calculate total size
        total_size = sum(f.stat().st_size for f in export_dir.rglob("*") if f.is_file()) / (1024 * 1024)

        print("\n" + "="*60)
        print("[SUCCESS] COMPLETE PACKAGE CREATED!")
        print("="*60)
        print(f"\nPackage location: {export_dir}/")
        print(f"Total size: {total_size:.1f} MB (~{total_size/1024:.2f} GB)")
        print("\nContents:")
        print("  [+] docker-images/        - 3 Docker containers")
        print("  [+] backups/              - Database + Parquet files")
        print("  [+] 1_load_images.bat     - Setup script 1")
        print("  [+] 2_start_containers.bat - Setup script 2")
        print("  [+] 3_restore_data.bat    - Setup script 3")
        print("  [+] README.md             - Professor instructions")
        print("  [+] docker-compose.yml    - Container config")
        print("  [+] .env.example          - Environment template")
        print("\nNext steps:")
        print("1. Read: SEND_TO_PROFESSOR.md")
        print("2. Compress professor-complete-package/ folder")
        print("3. Upload to Google Drive")
        print("4. Send link to professor")
        print("="*60)
    else:
        print("\n[ERROR] Package creation failed. Check errors above.")

if __name__ == "__main__":
    main()
