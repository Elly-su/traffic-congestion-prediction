# Deployment Guide - Traffic Prediction Dashboard

## 🚀 Deployment Options

This guide covers **5 deployment methods** for your Streamlit dashboard, ranked from easiest to most complex.

---

## ⭐ Option 1: Streamlit Community Cloud (Recommended - FREE!)

**Best for:** Quick deployment, sharing with others, educational projects  
**Cost:** FREE  
**Time:** 5-10 minutes  
**Difficulty:** ⭐ Easy

### Prerequisites
- GitHub account
- Your project pushed to GitHub

### Step-by-Step Instructions

#### 1. Prepare Your Repository

First, make sure your project is on GitHub:

```bash
# If not already a git repo, initialize it
cd C:\Users\ellio\.gemini\antigravity\scratch\traffic_congestion_prediction
git init
git add .
git commit -m "Initial commit with Streamlit dashboard"

# Create a new repository on GitHub, then:
git remote add origin https://github.com/YOUR_USERNAME/traffic-prediction-dashboard.git
git branch -M main
git push -u origin main
```

#### 2. Create `.streamlit/config.toml` (Optional but Recommended)

Create a folder `.streamlit` and add `config.toml`:

```toml
[theme]
primaryColor = "#1f77b4"
backgroundColor = "#FFFFFF"
secondaryBackgroundColor = "#F0F2F6"
textColor = "#262730"
font = "sans serif"

[server]
headless = true
port = 8501
enableCORS = false
```

#### 3. Update `requirements.txt`

Make sure your `requirements.txt` includes all dependencies with versions:

```txt
pandas>=1.5.0
numpy>=1.23.0
matplotlib>=3.6.0
seaborn>=0.12.0
scikit-learn>=1.2.0
streamlit>=1.28.0
plotly>=5.17.0
```

#### 4. Deploy to Streamlit Cloud

1. **Go to**: https://share.streamlit.io/
2. **Sign in** with GitHub
3. **Click "New app"**
4. **Configure**:
   - Repository: `YOUR_USERNAME/traffic-prediction-dashboard`
   - Branch: `main`
   - Main file path: `app.py`
5. **Click "Deploy"**

**Your app will be live at**: `https://YOUR_USERNAME-traffic-prediction-dashboard.streamlit.app`

### ✅ Advantages
- ✅ Completely FREE
- ✅ Automatic HTTPS
- ✅ Auto-deploys on git push
- ✅ Easy to share (just send the URL)
- ✅ No server maintenance

### ❌ Limitations
- Limited to 1 GB RAM (may struggle with large models)
- Public apps sleep after inactivity
- 3 private apps max on free tier

---

## 🐳 Option 2: Docker Deployment

**Best for:** Consistent deployments, local network, or cloud platforms  
**Cost:** Free locally, varies on cloud  
**Time:** 15-30 minutes  
**Difficulty:** ⭐⭐ Moderate

### Create Dockerfile

Create `Dockerfile` in your project root:

```dockerfile
# Use official Python runtime
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Copy requirements first (for caching)
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application
COPY . .

# Expose Streamlit port
EXPOSE 8501

# Health check
HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health

# Run the app
ENTRYPOINT ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

### Create `.dockerignore`

```
__pycache__
*.pyc
*.pyo
*.pyd
.Python
env/
venv/
.git
.gitignore
*.md
.vscode/
```

### Build and Run

```bash
# Build the image
docker build -t traffic-dashboard .

# Run the container
docker run -p 8501:8501 traffic-dashboard

# Access at http://localhost:8501
```

### Deploy to Cloud with Docker

**Option A: AWS ECS**
```bash
# Tag and push to ECR
aws ecr create-repository --repository-name traffic-dashboard
docker tag traffic-dashboard:latest YOUR_AWS_ACCOUNT.dkr.ecr.us-east-1.amazonaws.com/traffic-dashboard:latest
docker push YOUR_AWS_ACCOUNT.dkr.ecr.us-east-1.amazonaws.com/traffic-dashboard:latest
```

**Option B: Google Cloud Run**
```bash
# Deploy directly
gcloud run deploy traffic-dashboard \
  --source . \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated
```

---

## 🌐 Option 3: Heroku Deployment

**Best for:** Small to medium apps, easy scaling  
**Cost:** $5-7/month (Eco dynos)  
**Time:** 10-20 minutes  
**Difficulty:** ⭐⭐ Moderate

### Prerequisites
- Heroku account: https://signup.heroku.com/
- Heroku CLI: https://devcenter.heroku.com/articles/heroku-cli

### Setup Files

#### 1. Create `Procfile`

```
web: sh setup.sh && streamlit run app.py
```

#### 2. Create `setup.sh`

```bash
mkdir -p ~/.streamlit/

echo "\
[server]\n\
headless = true\n\
port = $PORT\n\
enableCORS = false\n\
\n\
" > ~/.streamlit/config.toml
```

#### 3. Create `runtime.txt`

```
python-3.10.13
```

### Deploy

```bash
# Login to Heroku
heroku login

# Create new app
heroku create traffic-prediction-dashboard

# Deploy
git push heroku main

# Open the app
heroku open
```

### Scale if Needed

```bash
# Upgrade to better dyno
heroku ps:scale web=1:standard-1x

# View logs
heroku logs --tail
```

---

## ☁️ Option 4: AWS Deployment (Full Control)

**Best for:** Production apps, enterprise use  
**Cost:** ~$10-50/month (depends on usage)  
**Time:** 30-60 minutes  
**Difficulty:** ⭐⭐⭐ Advanced

### Option A: AWS EC2

```bash
# 1. Launch EC2 instance (t3.medium recommended)
# 2. SSH into instance
ssh -i your-key.pem ubuntu@your-instance-ip

# 3. Install dependencies
sudo apt update
sudo apt install python3-pip nginx -y

# 4. Clone your repo
git clone https://github.com/YOUR_USERNAME/traffic-prediction-dashboard.git
cd traffic-prediction-dashboard

# 5. Install Python packages
pip3 install -r requirements.txt

# 6. Run with nohup (background)
nohup streamlit run app.py --server.port 8501 &

# 7. Configure nginx reverse proxy
sudo nano /etc/nginx/sites-available/streamlit

# Add:
server {
    listen 80;
    server_name your-domain.com;

    location / {
        proxy_pass http://localhost:8501;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection 'upgrade';
        proxy_set_header Host $host;
        proxy_cache_bypass $http_upgrade;
    }
}

# Enable site
sudo ln -s /etc/nginx/sites-available/streamlit /etc/nginx/sites-enabled/
sudo nginx -t
sudo systemctl restart nginx
```

### Option B: AWS Elastic Beanstalk

```bash
# Install EB CLI
pip install awsebcli

# Initialize EB
eb init -p python-3.10 traffic-dashboard

# Create environment and deploy
eb create traffic-dashboard-env
eb deploy
```

---

## 🏠 Option 5: Local Network Deployment

**Best for:** Internal company use, demos  
**Cost:** FREE  
**Time:** 5 minutes  
**Difficulty:** ⭐ Easy

### Make Accessible on Local Network

```bash
# Run with network access
streamlit run app.py --server.address 0.0.0.0
```

**Access from other devices on your network:**
- Find your IP: `ipconfig` (Windows) or `ifconfig` (Mac/Linux)
- Share URL: `http://YOUR_LOCAL_IP:8501`

### Setup as Windows Service (Always Running)

Create `run-dashboard.bat`:

```batch
@echo off
cd C:\Users\ellio\.gemini\antigravity\scratch\traffic_congestion_prediction
streamlit run app.py --server.address 0.0.0.0
```

Use NSSM to create a Windows service:
```bash
# Download NSSM from nssm.cc
nssm install TrafficDashboard "C:\path\to\run-dashboard.bat"
nssm start TrafficDashboard
```

---

## 📋 Comparison Table

| Method | Cost | Difficulty | Maintenance | Best For |
|--------|------|------------|-------------|----------|
| **Streamlit Cloud** | FREE | ⭐ | None | Quick sharing, demos |
| **Docker** | Varies | ⭐⭐ | Low | Consistent deploys |
| **Heroku** | $5-7/mo | ⭐⭐ | Low | Simple production |
| **AWS EC2** | $10+/mo | ⭐⭐⭐ | Medium | Full control |
| **Local Network** | FREE | ⭐ | Low | Internal use |

---

## 🔒 Security Best Practices

### For All Deployments

1. **Environment Variables**: Never commit sensitive data
   ```python
   import os
   API_KEY = os.getenv('API_KEY')
   ```

2. **Authentication**: Add login if needed
   ```bash
   pip install streamlit-authenticator
   ```

3. **HTTPS**: Always use SSL in production
   - Streamlit Cloud: Automatic
   - Others: Use Let's Encrypt or cloud provider SSL

4. **Rate Limiting**: Protect against abuse
   ```python
   from streamlit_extras.no_default_selectbox import selectbox
   ```

---

## 🚀 Recommended Path for You

Based on your project (educational/portfolio):

### 1️⃣ **Start with Streamlit Cloud** (Easiest)
- Perfect for sharing with professors/employers
- Free and professional-looking URL
- No maintenance required

### 2️⃣ **Later: Docker + Cloud Run** (If needed)
- Better for large models
- More control
- Still relatively easy

### 3️⃣ **Advanced: AWS** (Only if required)
- For "real" production use
- When you need full control

---

## 📦 Pre-Deployment Checklist

Before deploying, ensure:

- [ ] `requirements.txt` is complete and has versions
- [ ] No hardcoded file paths (use relative paths)
- [ ] `.gitignore` excludes large model files if needed
- [ ] README has clear description
- [ ] Test locally one more time
- [ ] Remove any debug/print statements
- [ ] Add error handling for missing files

---

## 🎯 Quick Start: Streamlit Cloud (5 Minutes)

```bash
# 1. Create GitHub repo
git init
git add .
git commit -m "Traffic prediction dashboard"
git remote add origin https://github.com/YOUR_USERNAME/traffic-dashboard.git
git push -u origin main

# 2. Go to share.streamlit.io
# 3. Click "New app"
# 4. Connect your repo
# 5. Done! Share your URL
```

**That's it!** Your dashboard is now live and accessible worldwide. 🌍

---

## 💡 Tips for Production

1. **Add Loading States**: Use `st.spinner()` for better UX
2. **Cache Everything**: Use `@st.cache_data` and `@st.cache_resource`
3. **Monitor Usage**: Streamlit Cloud provides basic analytics
4. **Update Regularly**: Keep dependencies up to date
5. **Test Mobile**: Check how it looks on phones

---

## 🆘 Troubleshooting

**Issue**: "ModuleNotFoundError"
- **Fix**: Ensure all packages in `requirements.txt`

**Issue**: App is slow
- **Fix**: Add more caching, reduce data loaded

**Issue**: Models too large for Streamlit Cloud
- **Fix**: Use Docker + Cloud Run or compress models

**Issue**: Port already in use
- **Fix**: `streamlit run app.py --server.port 8502`

---

## 📚 Resources

- Streamlit Deployment Docs: https://docs.streamlit.io/streamlit-cloud
- Docker Tutorial: https://docs.docker.com/get-started/
- Heroku Python Guide: https://devcenter.heroku.com/articles/getting-started-with-python
- AWS Deployment: https://aws.amazon.com/getting-started/

---

**Need Help?** Let me know which deployment method you'd like to use, and I can provide more specific guidance!
