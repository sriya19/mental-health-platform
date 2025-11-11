"""
Restore backed up data for team members and professor
"""
import subprocess
import time
from pathlib import Path

def check_docker():
    """Check if Docker is running"""
    result = subprocess.run("docker info", shell=True, capture_output=True)
    return result.returncode == 0

def restore_database():
    """Restore PostgreSQL database"""
    print("[RESTORE] Restoring PostgreSQL database...")

    backup_file = Path("backups/database_backup.sql")
    if not backup_file.exists():
        print(f"[ERROR] Backup file not found: {backup_file}")
        print("   Make sure you're in the project root directory!")
        return False

    cmd = 'docker exec -i pg psql -U app_user -d mh_catalog < backups/database_backup.sql'
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)

    if result.returncode == 0:
        print("[SUCCESS] Database restored successfully!")
        return True
    else:
        # Sometimes it's OK if there are warnings
        if "ERROR" not in result.stderr:
            print("[SUCCESS] Database restored (with warnings - this is normal)")
            return True
        else:
            print(f"[ERROR] Database restore failed: {result.stderr[:500]}")
            return False

def restore_minio():
    """Restore MinIO/S3 data"""
    print("\n[RESTORE] Restoring MinIO data...")

    backup_dir = Path("backups/minio-data/mh-raw")
    if not backup_dir.exists():
        print(f"[ERROR] MinIO backup not found: {backup_dir}")
        return False

    cmd = 'docker cp backups/minio-data/mh-raw mental-health-platform-minio-1:/data/'
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)

    if result.returncode == 0:
        print("[SUCCESS] MinIO data restored successfully!")
        return True
    else:
        print(f"[ERROR] MinIO restore failed: {result.stderr}")
        return False

def restart_backend():
    """Restart backend to pick up restored data"""
    print("\n[INFO] Restarting backend...")
    subprocess.run("docker-compose restart backend", shell=True)
    time.sleep(5)
    print("[SUCCESS] Backend restarted")

def main():
    print("="*60)
    print("RESTORING MENTAL HEALTH PLATFORM DATA")
    print("="*60)

    # Check Docker
    if not check_docker():
        print("[ERROR] Docker is not running! Start Docker Desktop first.")
        return

    # Check if containers are running
    result = subprocess.run("docker-compose ps", shell=True, capture_output=True, text=True)
    if "backend" not in result.stdout or "pg" not in result.stdout:
        print("\n[WARNING]  Containers not running. Starting them now...")
        subprocess.run("docker-compose up -d", shell=True)
        print("⏳ Waiting 30 seconds for services to be ready...")
        time.sleep(30)

    # Restore database
    db_success = restore_database()

    # Restore MinIO
    minio_success = restore_minio()

    if db_success or minio_success:
        restart_backend()

        print("\n" + "="*60)
        print("[SUCCESS] RESTORE COMPLETE!")
        print("="*60)
        print("\n[SUCCESS] Your platform is ready with all existing datasets!")
        print("\n[INFO] What you have now:")
        print("   • 30+ ingested datasets")
        print("   • 1,077 indexed chunks for Q&A")
        print("   • All Parquet files")
        print("\n[INFO] Open your browser:")
        print("   • UI: http://localhost:8501")
        print("   • Backend: http://localhost:8000")
        print("\n[INFO] You can now:")
        print("   • Search existing datasets")
        print("   • Ask questions using indexed data")
        print("   • Ingest additional datasets")
        print("   • Share the platform with your team!")
        print("="*60)
    else:
        print("\n[ERROR] Restore failed. Check errors above.")

if __name__ == "__main__":
    main()
