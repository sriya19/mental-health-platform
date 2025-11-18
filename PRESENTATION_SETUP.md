# Mental Health Platform - Presentation Setup Guide

## Your App is Now Live!

**Public URL**: https://mindcube-mh-platform.loca.lt

**QR Code**: `presentation_qr_code.png` (in this folder)

---

## For Your Presentation

### Quick Steps:
1. **Display the QR Code** (`presentation_qr_code.png`) on your screen or slides
2. **Tell your audience** to scan the QR code with their phone cameras
3. **They'll be directed** to your live Mental Health Platform
4. **No WiFi restrictions** - works on cellular data too!

### Alternative Access:
If someone can't scan the QR code, they can type this URL directly:
```
https://mindcube-mh-platform.loca.lt
```

---

## Technical Details

### Active Services:
- **Frontend (Streamlit)**: https://mindcube-mh-platform.loca.lt
- **Backend API**: https://mh-backend-api.loca.lt (internal use)
- **Database**: PostgreSQL (local, via Docker)
- **Storage**: MinIO (local, via Docker)

### Background Processes Running:
1. Docker Compose (backend services)
2. Streamlit app (with public backend connection)
3. LocalTunnel for backend API (port 8000)
4. LocalTunnel for Streamlit (port 8501)

---

## IMPORTANT - Keep Running During Presentation

**DO NOT CLOSE:**
- This terminal/command window
- Docker Desktop
- Your laptop

**Keep your laptop:**
- Plugged in (to prevent sleep)
- Connected to internet (WiFi or ethernet)

---

## If Something Goes Wrong

### If the URL stops working:
1. Check if all Docker containers are running: `docker-compose ps`
2. Check if tunnels are still active (look for terminal windows)
3. Restart tunnels if needed:
   ```bash
   # Backend tunnel
   lt --port 8000 --subdomain mh-backend-api

   # Streamlit tunnel
   lt --port 8501 --subdomain mindcube-mh-platform
   ```

### To completely restart everything:
```bash
# Stop all services
docker-compose down

# Start Docker services
docker-compose up -d

# Restart Streamlit with public backend
set BACKEND_URL=https://mh-backend-api.loca.lt && streamlit run ui_app/streamlit_app.py

# Start backend tunnel
lt --port 8000 --subdomain mh-backend-api

# Start frontend tunnel
lt --port 8501 --subdomain mindcube-mh-platform
```

---

## After the Presentation

### To stop everything:
```bash
# Stop Docker services
docker-compose down

# Kill Streamlit (close the terminal or Ctrl+C)

# Kill tunnels (close the terminals or Ctrl+C)
```

### To restart for another presentation:
Just run the commands in "If Something Goes Wrong" section above.

---

## Features Your Audience Can Explore

1. **Search Datasets**: Semantic search across mental health datasets
2. **Ask Questions**: RAG-powered Q&A about mental health data
3. **Visualize Data**: AI-powered visualizations and insights
4. **Explore Catalog**: Browse indexed mental health datasets

---

## Presentation Tips

1. **Test the URL** yourself on your phone before the presentation starts
2. **Have a backup** - take screenshots of key features in case of network issues
3. **Prepare a demo** - walk through a sample search query to show the platform
4. **Highlight AI features** - the RAG system and AI-powered visualizations are impressive
5. **Mention the stack** - Docker, FastAPI, Streamlit, PostgreSQL, OpenAI

---

## Demo Search Queries to Try:

- "As a public health researcher, I need data on adolescent depression prevalence"
- "Show me datasets about mental health treatment outcomes"
- "What data is available on substance abuse and mental health?"
- "Find datasets related to suicide prevention"

---

Good luck with your presentation! 🎉
