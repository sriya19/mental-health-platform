# Sharing Guide: Mental Health Platform with Data

This guide explains how to share your project with your team and professor, including all datasets.

## What You Have Now:

✅ **Database Backup** (`backups/database_backup.sql`)
   - 24 MB SQL dump
   - Contains:
     - All 30+ ingested datasets metadata
     - 1,077 indexed chunks for Q&A
     - Dataset configurations
     - Search history

✅ **MinIO Data Backup** (`backups/minio-data/`)
   - 35 Parquet files
   - All actual dataset content

✅ **Restore Scripts**
   - `restore_data.py` - Automatic restoration for team/professor
   - `backups/README.md` - Instructions

---

## Option 1: GitHub (Recommended for Team Collaboration)

### Step 1: Prepare for GitHub

**Note:** GitHub has a 100MB file size limit. Your database is 24MB, so it fits!

```bash
# Check current git status
git status

# Add the backup files
git add backups/
git add backup_data.py
git add restore_data.py
git add SHARING_GUIDE.md

# Create a commit
git commit -m "Add database and dataset backups for team sharing"

# Push to GitHub
git push origin sankarsh-updates
```

### Step 2: Share with Team/Professor

Send them:
1. **GitHub repo URL**
2. **Instructions:**

```
Hi! To run the Mental Health Platform with all my datasets:

1. Clone the repo:
   git clone [your-repo-url]
   cd mental-health-platform

2. Create .env file with your OpenAI API key:
   cp .env.example .env
   # Edit .env and add your OPENAI_API_KEY

3. Restore the data:
   python restore_data.py

4. Open the UI:
   http://localhost:8501

That's it! You'll see all 30+ datasets already loaded.
```

---

## Option 2: Google Drive / OneDrive (For Professor Review)

### Step 1: Compress the Backup

```bash
# Create a zip file with everything needed
cd "C:\Users\sanka\OneDrive\Desktop\Final Capstone Project"
# Compress mental-health-platform folder

(Or use 7-Zip/WinRAR to compress)
```

### Step 2: Upload to Cloud Storage

1. Upload compressed folder to Google Drive / OneDrive
2. Share link with professor
3. Include instructions (see below)

### Instructions for Professor:

```
QUICK START INSTRUCTIONS
========================

Prerequisites:
- Docker Desktop installed
- Python 3.10+ installed
- OpenAI API key (I can provide one for testing)

Steps:
1. Extract the zip file
2. Open terminal in the project folder
3. Create .env file:
   cp .env.example .env
   (Add OpenAI API key)
4. Run: python restore_data.py
5. Open browser: http://localhost:8501

What you'll see:
- 30+ health datasets already loaded
- 1,077 searchable chunks
- AI-powered search and Q&A
- Visualizations and analytics
```

---

## Option 3: For Large Files (If Backups Exceed 100MB)

If your backups are too large for GitHub:

### Use Git LFS (Large File Storage)

```bash
# Install Git LFS
git lfs install

# Track large files
git lfs track "backups/*.sql"
git lfs track "backups/minio-data/**/*.parquet"

# Add .gitattributes
git add .gitattributes

# Now commit and push as normal
git add backups/
git commit -m "Add data backups with Git LFS"
git push
```

---

## What Team Members Can Do:

After restoring your data, they can:

✅ **View existing datasets** - See all 30+ datasets you ingested
✅ **Search datasets** - Use the enhanced AI search
✅ **Ask questions** - Query the 1,077 indexed chunks
✅ **Add new datasets** - Ingest additional data (won't affect yours)
✅ **Export data** - Download any dataset as CSV/Excel
✅ **Visualize data** - Create charts and statistics

---

## Security Notes:

🔒 **What's NOT shared:**
- Your `.env` file (contains API keys) - excluded by .gitignore
- Passwords/secrets - not in backups

✅ **What IS shared:**
- Dataset metadata and content (public CDC/SAMHSA data)
- Indexed chunks
- Project code

⚠️ **Important:**
- Don't commit `.env` file
- Each team member needs their own OpenAI API key
- Database passwords in `.env.example` should be changed for production

---

## Updating Backups:

If you add more datasets later:

```bash
# Re-run backup script
python backup_data.py

# Commit new backups
git add backups/
git commit -m "Update datasets: added [description]"
git push
```

---

## Troubleshooting:

**"Docker not running":**
```
- Start Docker Desktop
- Wait until it's fully started
- Run restore_data.py again
```

**"Restore failed":**
```
- Make sure you're in the project root directory
- Check if backups/ folder exists
- Verify Docker containers are running: docker-compose ps
```

**"API key error":**
```
- Each person needs their own OpenAI API key in .env
- Get one at: https://platform.openai.com/api-keys
```

---

## File Size Summary:

- Database backup: ~24 MB (metadata + indexed chunks)
- MinIO backup: ~1-50 MB depending on datasets
- Total project: < 100 MB (fits on GitHub)

---

## Demo for Professor:

When presenting:

1. Show the restored data (30+ datasets)
2. Demo AI search: "What are drug overdose trends in Baltimore?"
3. Show Q&A: Ask questions about the indexed data
4. Demonstrate new dataset ingestion
5. Show visualizations and analytics

The professor will see your work with real data, not an empty system!

---

## Questions?

If team members need help:
1. Check backups/README.md
2. Run: python restore_data.py
3. Contact you if issues persist

Good luck with your demo! 🎓
