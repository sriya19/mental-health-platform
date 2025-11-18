"""
Export complete Docker setup for professor
Creates a package with all containers and data
"""
import subprocess
import os
from pathlib import Path

def export_docker_images():
    """Export all Docker images as tar files"""
    print("="*60)
    print("EXPORTING DOCKER IMAGES FOR PROFESSOR")
    print("="*60)

    export_dir = Path("professor-package")
    export_dir.mkdir(exist_ok=True)

    images = [
        ("mental-health-platform-backend", "backend-image.tar"),
        ("pgvector/pgvector:pg16", "postgres-image.tar"),
        ("minio/minio:latest", "minio-image.tar"),
    ]

    print("\n[INFO] Exporting Docker images...")
    print("This may take 5-10 minutes...\n")

    for image_name, filename in images:
        print(f"[EXPORT] Exporting {image_name}...")
        output_path = export_dir / filename

        cmd = f'docker save {image_name} -o "{output_path}"'
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)

        if result.returncode == 0:
            size = os.path.getsize(output_path) / (1024 * 1024)
            print(f"[SUCCESS] Exported: {size:.1f} MB")
        else:
            print(f"[ERROR] Failed: {result.stderr}")

    return export_dir

def create_load_script(export_dir):
    """Create script for professor to load images"""

    load_script = """@echo off
echo ============================================================
echo LOADING MENTAL HEALTH PLATFORM DOCKER IMAGES
echo ============================================================

echo.
echo [INFO] Loading Docker images...
echo This will take a few minutes...
echo.

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

echo.
echo ============================================================
echo [SUCCESS] ALL IMAGES LOADED!
echo ============================================================
echo.
echo Next steps:
echo 1. Go back to project folder
echo 2. Run: docker-compose up -d
echo 3. Run: python restore_data.py
echo 4. Open: http://localhost:8501
echo.
pause
"""

    with open(export_dir / "1_load_images.bat", "w") as f:
        f.write(load_script)

    print(f"\n[INFO] Created: {export_dir}/1_load_images.bat")

def create_readme(export_dir):
    """Create instructions for professor"""

    readme = """# Mental Health Platform - Professor Package

This package contains everything needed to run the Mental Health Platform.

## What's Included:

1. **Docker Images** (Pre-built containers)
   - `backend-image.tar` (~500 MB) - FastAPI backend
   - `postgres-image.tar` (~150 MB) - PostgreSQL with pgvector
   - `minio-image.tar` (~200 MB) - MinIO S3 storage

2. **Database Backup** (in ../backups/)
   - 30+ ingested datasets
   - 1,077 indexed chunks

3. **Scripts**
   - `1_load_images.bat` - Load Docker images
   - `restore_data.py` - Restore database

## Quick Start (5 minutes):

### Prerequisites:
- Docker Desktop installed and running
- Python 3.10+
- OpenAI API key (contact student for test key)

### Step 1: Load Docker Images
```
cd professor-package
1_load_images.bat
```

### Step 2: Setup Environment
```
cd ..  (back to project folder)
copy .env.example .env
# Edit .env and add OPENAI_API_KEY
```

### Step 3: Start Containers
```
docker-compose up -d
```

### Step 4: Restore Data
```
python restore_data.py
```

### Step 5: Open Application
```
Browser: http://localhost:8501
```

## What You'll See:

✅ 30+ health datasets (CDC, SAMHSA, Baltimore data)
✅ AI-powered search and discovery
✅ Question answering with 1,077 indexed chunks
✅ Data visualizations and analytics
✅ Full working platform with all student's work

## Troubleshooting:

**Docker not running:**
- Start Docker Desktop
- Wait until fully started
- Run steps again

**Port already in use:**
- Stop existing containers: `docker-compose down`
- Or change ports in docker-compose.yml

**Restore failed:**
- Make sure backups/ folder exists
- Verify containers running: `docker-compose ps`

## File Sizes:

- Total package: ~850 MB - 1.5 GB
- Includes all pre-built containers and data
- No internet download needed during setup

## Support:

Contact student if you encounter issues.

---

**Built by:** [Your Name]
**Course:** [Course Name]
**Date:** November 2024
"""

    with open(export_dir / "README.md", "w") as f:
        f.write(readme)

    print(f"[INFO] Created: {export_dir}/README.md")

def create_deployment_guide():
    """Create guide for student on what to send professor"""

    guide = """# How to Send to Professor

You have created a complete package in `professor-package/` folder.

## What's in the Package:

1. Docker images (3 tar files, ~850 MB total)
2. Load script (1_load_images.bat)
3. Instructions (README.md)

## How to Share:

### Option 1: Google Drive (Recommended)

1. Compress the entire project folder:
   - Right-click `mental-health-platform` folder
   - Send to → Compressed (zipped) folder

2. Upload to Google Drive

3. Share link with professor

4. Email template:

```
Subject: Mental Health Platform - Docker Package

Dear Professor,

I've prepared a complete Docker package of my Mental Health Platform project.

Download link: [Google Drive Link]

Quick start:
1. Extract the zip file
2. cd professor-package
3. Run: 1_load_images.bat
4. Follow README.md instructions

The package includes:
- All pre-built Docker containers
- 30+ datasets already loaded
- Complete working application

Estimated setup time: 5 minutes

Let me know if you need the OpenAI API key for testing.

Best regards,
[Your Name]
```

### Option 2: USB Drive

If file is too large for email:
1. Copy entire `mental-health-platform` folder to USB
2. Give to professor in person
3. Include printed copy of professor-package/README.md

### Option 3: OneDrive/Dropbox

Same as Google Drive option.

## File Size:

- Just Docker images: ~850 MB
- Complete project: ~1 GB
- Compressed (zip): ~500-700 MB

## What Professor Needs:

✅ Docker Desktop
✅ Python 3.10+
✅ OpenAI API key (you can provide a test key)
✅ 5 minutes to set up

They do NOT need:
❌ Install dependencies manually
❌ Build Docker images
❌ Ingest datasets
❌ Index data

Everything is pre-built and ready to run!

---

**Note:** If professor requests DockerHub instead, let me know and
I can help you push the images there.
"""

    with open("PROFESSOR_DEPLOYMENT.md", "w") as f:
        f.write(guide)

    print(f"\n[SUCCESS] Created: PROFESSOR_DEPLOYMENT.md")

def main():
    print("\nThis will export Docker images for your professor.")
    print("Expected time: 5-10 minutes")
    print("Required disk space: ~1-2 GB")

    response = input("\nContinue? (yes/no): ").strip().lower()

    if response != "yes":
        print("Cancelled.")
        return

    # Export images
    export_dir = export_docker_images()

    # Create helper scripts
    create_load_script(export_dir)
    create_readme(export_dir)
    create_deployment_guide()

    print("\n" + "="*60)
    print("[SUCCESS] EXPORT COMPLETE!")
    print("="*60)
    print(f"\nPackage created in: {export_dir}/")
    print("\nNext steps:")
    print("1. Read: PROFESSOR_DEPLOYMENT.md")
    print("2. Compress the project folder")
    print("3. Upload to Google Drive")
    print("4. Share with professor")
    print("="*60)

if __name__ == "__main__":
    main()
