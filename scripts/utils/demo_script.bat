@echo off
REM ========================================
REM Baltimore Mental Health Platform - Demo Script
REM ========================================

echo.
echo ========================================
echo USER STORY 1: System Health Check
echo ========================================
echo.
echo Checking if the platform is running...
curl -s http://localhost:8000/health
echo.
echo.
pause

echo.
echo ========================================
echo USER STORY 2: View Available Datasets
echo ========================================
echo.
echo Showing all Baltimore mental health datasets...
curl -s http://localhost:8000/datasets | python -m json.tool
echo.
echo.
pause

echo.
echo ========================================
echo USER STORY 3: Check AI Indexing Status
echo ========================================
echo.
echo Verifying AI embeddings are ready...
curl -s http://localhost:8000/rag_status?org=CDC | python -m json.tool
echo.
echo.
pause

echo.
echo ========================================
echo USER STORY 4: View System Statistics
echo ========================================
echo.
echo Getting platform statistics...
curl -s http://localhost:8000/stats | python -m json.tool
echo.
echo.
pause

echo.
echo ========================================
echo USER STORY 5: Database Quality Check
echo ========================================
echo.
echo Checking data quality in PostgreSQL...
docker exec pg psql -U app_user -d mh_catalog -c "SELECT org, dataset_uid, COUNT(*) as total_chunks, COUNT(embedding) as chunks_with_embeddings FROM data_chunks GROUP BY org, dataset_uid;"
echo.
echo.
pause

echo.
echo ========================================
echo USER STORY 6: API Documentation
echo ========================================
echo.
echo Opening API documentation in browser...
start http://localhost:8000/docs
echo API docs opened in your browser!
echo.
pause

echo.
echo ========================================
echo USER STORY 7: MinIO Object Storage
echo ========================================
echo.
echo Opening MinIO console in browser...
echo Login: minioadmin / minioadmin123
start http://localhost:9001
echo MinIO console opened in your browser!
echo.
pause

echo.
echo ========================================
echo Demo Complete!
echo ========================================
echo.
echo Summary:
echo - Platform Status: Running
echo - Datasets Loaded: 4 Baltimore mental health datasets
echo - Total Rows: 119,938
echo - AI Chunks: 270 with embeddings
echo - Services: PostgreSQL, MinIO, FastAPI all operational
echo.
echo Next Steps:
echo 1. Connect to DBeaver (localhost:5432, database: mh_catalog)
echo 2. Explore API endpoints at http://localhost:8000/docs
echo 3. View raw data files at http://localhost:9001
echo.
pause
