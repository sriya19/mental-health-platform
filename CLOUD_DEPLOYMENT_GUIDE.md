# Mental Health Platform - Cloud Deployment Guide

## Deploy Your App to the Cloud (FREE & Permanent)

This guide will help you deploy your Mental Health Platform to the cloud with a **permanent, public URL** that anyone can access.

---

## Option 1: Render (Recommended - Simplest)

**Pros:**
- ✅ 100% Free tier available
- ✅ Automatic HTTPS
- ✅ Easy PostgreSQL setup
- ✅ Direct GitHub integration
- ✅ Permanent URL

**Cons:**
- ⚠️ Free tier services sleep after 15 minutes of inactivity (wake up in ~30 seconds)
- ⚠️ 750 hours/month limit (enough for demos/presentations)

### Step-by-Step Instructions:

#### 1. Prepare Your Repository

```bash
# Make sure your code is committed
git add .
git commit -m "Prepare for cloud deployment"
git push origin master
```

#### 2. Sign Up for Render

1. Go to https://render.com
2. Click "Get Started for Free"
3. Sign up with your GitHub account
4. Authorize Render to access your repositories

#### 3. Deploy the Backend API

1. From Render Dashboard, click **"New +"** → **"Web Service"**
2. Connect your GitHub repository: `mental-health-platform`
3. Configure:
   - **Name**: `mental-health-backend`
   - **Region**: Choose closest to you
   - **Branch**: `master`
   - **Root Directory**: `backend`
   - **Environment**: `Docker`
   - **Instance Type**: `Free`
4. Click **"Create Web Service"**
5. Wait for deployment (5-10 minutes)
6. **Copy the backend URL** (e.g., `https://mental-health-backend.onrender.com`)

#### 4. Deploy PostgreSQL Database

1. Click **"New +"** → **"PostgreSQL"**
2. Configure:
   - **Name**: `mental-health-db`
   - **Database**: `mh_catalog`
   - **User**: `app_user`
   - **Region**: Same as backend
   - **Instance Type**: `Free`
3. Click **"Create Database"**
4. **Copy the Internal Database URL** from the database page

#### 5. Update Backend Environment Variables

1. Go to your backend service → **"Environment"** tab
2. Add these variables:
   ```
   POSTGRES_HOST=<from database internal connection>
   POSTGRES_PORT=5432
   POSTGRES_DB=mh_catalog
   POSTGRES_USER=app_user
   POSTGRES_PASSWORD=<from database connection>
   OPENAI_API_KEY=<your OpenAI API key>
   MINIO_ENDPOINT=localhost:9000
   MINIO_ROOT_USER=minioadmin
   MINIO_ROOT_PASSWORD=minioadmin123
   S3_BUCKET=datasets
   ```
3. Click **"Save Changes"** (backend will redeploy automatically)

#### 6. Deploy the Streamlit Frontend

1. Click **"New +"** → **"Web Service"**
2. Connect the same repository
3. Configure:
   - **Name**: `mental-health-app`
   - **Region**: Same as backend
   - **Branch**: `master`
   - **Root Directory**: `.` (leave empty or use root)
   - **Environment**: `Python 3`
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `streamlit run ui_app/streamlit_app.py`
   - **Instance Type**: `Free`
4. Add environment variable:
   ```
   BACKEND_URL=<your backend URL from step 3>
   OPENAI_API_KEY=<your OpenAI API key>
   ```
5. Click **"Create Web Service"**
6. Wait for deployment (3-5 minutes)

#### 7. Get Your Public URL

Your app is now live! The frontend URL will be something like:
```
https://mental-health-app.onrender.com
```

This is your **permanent public URL** - share it with anyone!

---

## Option 2: Railway (Alternative)

**Pros:**
- ✅ $5 free credit per month
- ✅ Supports full Docker Compose
- ✅ No sleep mode
- ✅ Simple deployment

**Cons:**
- ⚠️ Limited free credits ($5/month = ~140 hours)

### Quick Railway Deployment:

1. **Sign up**: https://railway.app
2. **Install Railway CLI**:
   ```bash
   npm install -g @railway/cli
   ```
3. **Login**:
   ```bash
   railway login
   ```
4. **Initialize**:
   ```bash
   railway init
   ```
5. **Deploy**:
   ```bash
   railway up
   ```
6. **Get URL**:
   ```bash
   railway open
   ```

---

## Option 3: Streamlit Community Cloud (Frontend Only)

**Best for**: If you want to deploy JUST the Streamlit UI quickly

**Limitations**:
- Only deploys the frontend
- Backend must be deployed separately
- Cannot use Docker services (PostgreSQL, MinIO)

### Instructions:

1. Go to https://streamlit.io/cloud
2. Sign in with GitHub
3. Click "New app"
4. Select your repository and `ui_app/streamlit_app.py`
5. Add environment variables (BACKEND_URL, OPENAI_API_KEY)
6. Click "Deploy"

**Note**: You'll still need to deploy the backend separately using Render or Railway.

---

## After Deployment

### 1. Test Your Deployment

Visit your frontend URL and verify:
- ✅ App loads without errors
- ✅ Search works
- ✅ Backend connection is successful
- ✅ Database queries work

### 2. Update QR Code

Run this to generate a new QR code with your permanent URL:

```bash
# Edit generate_qr.py and change the URL
python generate_qr.py
```

### 3. Share Your App

Your app is now live! Share the URL:
- **Public URL**: `https://your-app.onrender.com`
- **QR Code**: `presentation_qr_code.png`
- **GitHub**: `https://github.com/your-username/mental-health-platform`

---

## Troubleshooting

### Backend not connecting:
- Check BACKEND_URL in frontend environment variables
- Ensure backend health check passes: `https://backend-url/health`
- Check backend logs in Render dashboard

### Database connection issues:
- Verify PostgreSQL is running in Render
- Check database credentials in backend environment
- Ensure backend and database are in same region

### App sleeping (Render free tier):
- Free services sleep after 15 min inactivity
- They wake up automatically when accessed (~30 sec)
- To prevent: upgrade to paid tier ($7/month) or use a service like UptimeRobot to ping your app

### Environment variables:
- Double-check all required variables are set
- Redeploy after changing environment variables
- Check logs for missing variable errors

---

## Cost Summary

### Free Tier Comparison:

| Platform | Cost | Limitations |
|----------|------|-------------|
| **Render** | Free | Services sleep after 15 min, 750 hrs/month |
| **Railway** | $5/month credit | ~140 hours with full stack |
| **Streamlit Cloud** | Free | Frontend only, 1GB RAM limit |

### Recommended Setup (FREE):

- **Frontend**: Streamlit Community Cloud (free)
- **Backend + DB**: Render (free)
- **Total Cost**: $0/month

---

## Production Tips

1. **Keep services awake**: Use https://uptimerobot.com (free) to ping your app every 5 minutes
2. **Monitor usage**: Check Render dashboard for usage statistics
3. **Optimize**: Implement caching in your app to reduce database queries
4. **Backup data**: Export important datasets regularly
5. **Security**: Never commit .env files, use environment variables

---

## Need Help?

- **Render Docs**: https://render.com/docs
- **Railway Docs**: https://docs.railway.app
- **Streamlit Docs**: https://docs.streamlit.io

---

Good luck with your deployment! 🚀
