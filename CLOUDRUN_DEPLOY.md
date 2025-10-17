# Deploy DiaRemot to Google Cloud - Quick Start

## 🎯 What This Does

Deploys your diaremot audio processing pipeline as a **live web service** on Google Cloud that:
- Accepts audio file uploads via HTTP
- Processes them through the full pipeline
- Returns diarized transcripts with emotion analysis
- Auto-scales based on demand
- You only pay when processing files

---

## ✅ Files Created

All files are in: `D:\diaremot\diaremot2-on\`

```
📄 deploy-cloudrun.ps1          ← RUN THIS (Windows)
📄 deploy-cloudrun.sh           ← RUN THIS (Linux/Mac)
🐳 Dockerfile.cloudrun          ← Web service container
⚙️ cloudbuild-cloudrun.yaml     ← Build configuration
📁 src/diaremot/api.py          ← FastAPI REST API
📖 CLOUDRUN_DEPLOY.md           ← This file
```

---

## 🚀 Deploy in 3 Steps

### Step 1: Prerequisites

```powershell
# Install Google Cloud SDK (if not already installed)
# Download from: https://cloud.google.com/sdk/docs/install

# Set your project
gcloud config set project YOUR_PROJECT_ID

# Login
gcloud auth login
```

### Step 2: Deploy

```powershell
# Windows PowerShell
cd D:\diaremot\diaremot2-on
.\deploy-cloudrun.ps1
```

**OR**

```bash
# Linux/Mac
cd /path/to/diaremot2-on
./deploy-cloudrun.sh
```

The script will:
1. ✅ Enable required Google Cloud APIs
2. ✅ Build Docker image (8-10 minutes)
3. ✅ Deploy to Cloud Run
4. ✅ Give you a service URL

### Step 3: Use It

```bash
# Get your service URL (from deploy output)
SERVICE_URL="https://diaremot-service-xxxxx.run.app"

# Upload audio file
curl -X POST $SERVICE_URL/process \
  -F "audio=@my_audio.wav" \
  -F "output_format=csv" \
  -o transcript.csv

# Or get JSON
curl -X POST $SERVICE_URL/process \
  -F "audio=@my_audio.wav"
```

---

## 📡 API Endpoints

### GET /
Service info and available endpoints

### GET /health
Health check (returns model availability status)

### POST /process
Upload audio file for processing

**Parameters:**
- `audio` (file, required) - Audio file (WAV, MP3, M4A, FLAC)
- `output_format` (string, optional) - Response format:
  - `json` (default) - Returns JSON with results
  - `csv` - Returns CSV transcript file
  - `all` - Returns ZIP with all output files

**Example:**
```bash
curl -X POST $SERVICE_URL/process \
  -F "audio=@recording.wav" \
  -F "output_format=csv"
```

---

## 🐍 Python Client Example

```python
import requests

SERVICE_URL = "https://your-service-xxxxx.run.app"

# Upload and process audio
with open("audio.wav", "rb") as f:
    response = requests.post(
        f"{SERVICE_URL}/process",
        files={"audio": f},
        data={"output_format": "json"}
    )

result = response.json()
print(f"Status: {result['status']}")
print(f"Segments: {result['result']['num_segments']}")
print(f"Speakers: {result['result']['num_speakers']}")

# Download CSV
response = requests.post(
    f"{SERVICE_URL}/process",
    files={"audio": open("audio.wav", "rb")},
    data={"output_format": "csv"}
)

with open("transcript.csv", "wb") as f:
    f.write(response.content)
```

---

## 💰 Pricing

**Free Tier (monthly):**
- 2M requests
- 360K CPU-seconds
- 180K GB-seconds

**After Free Tier:**
- ~$0.03 per 10-minute audio file
- Only pay when processing

**Example costs:**
- 100 files/month: ~$3
- 1000 files/month: ~$30
- 10000 files/month: ~$300

---

## 🔧 Configuration Options

### Deploy with custom resources:

```powershell
# Windows
.\deploy-cloudrun.ps1 us-central1 latest 8Gi 8

# Linux/Mac
./deploy-cloudrun.sh us-central1 latest 8Gi 8
```

Parameters:
1. Region (default: `us-central1`)
2. Image tag (default: `latest`)
3. Memory (default: `4Gi`)
4. CPU count (default: `4`)

### Update existing service:

```bash
# Just re-run the deploy script
.\deploy-cloudrun.ps1
```

---

## 📊 Monitoring

### View Logs

```bash
gcloud run services logs tail diaremot-service --region us-central1
```

### Service Details

```bash
gcloud run services describe diaremot-service --region us-central1
```

### Access Console

Visit: https://console.cloud.google.com/run

---

## 🛠️ Troubleshooting

### Service won't start
```bash
# Check logs
gcloud run services logs read diaremot-service --region us-central1 --limit 100

# Common issues:
# - FFmpeg not installed → Check Dockerfile.cloudrun
# - Port not 8080 → Check api.py PORT env variable
```

### Timeouts on long files
```bash
# Already set to max (3600s)
# For longer processing, use Cloud Tasks + Cloud Run
```

### Models not found
```bash
# Models need to be either:
# 1. Mounted from GCS bucket (recommended)
# 2. Baked into image (slow, expensive)
# 3. Downloaded on startup (cold start delay)

# To mount GCS bucket:
gcloud run deploy diaremot-service \
  --add-volume name=models,type=cloud-storage,bucket=YOUR_BUCKET \
  --add-volume-mount volume=models,mount-path=/srv/models
```

---

## 🗑️ Clean Up

### Delete service:
```bash
gcloud run services delete diaremot-service --region us-central1
```

### Delete image:
```bash
gcloud container images delete gcr.io/PROJECT_ID/diaremot2-on:latest
```

---

## 📚 More Information

For complete documentation, advanced configuration, and production deployment:
- See `CLOUD_BUILD_GUIDE.md` for build details
- Visit: https://cloud.google.com/run/docs

---

## ❓ FAQ

**Q: Do I need to keep my computer on?**  
A: No. After deployment, the service runs on Google's servers.

**Q: How do I update the code?**  
A: Make changes, then re-run `.\deploy-cloudrun.ps1`

**Q: Can I use my own domain?**  
A: Yes. Map custom domain in Cloud Run console.

**Q: Is this production-ready?**  
A: Yes, but consider:
- Adding authentication (`--no-allow-unauthenticated`)
- Setting up monitoring/alerting
- Mounting models from GCS
- Configuring CDN for static assets

**Q: What about the original Cloud Build files?**  
A: They're still there:
- `cloudbuild.yaml` - Builds CLI Docker images
- `Dockerfile` - CLI version container
- Use those for building images for other platforms

---

## 🎉 Success!

After deployment, you have:
- ✅ Live HTTP API processing audio files
- ✅ Auto-scaling from 0 to 100+ instances
- ✅ Pay-per-use pricing
- ✅ No servers to manage
- ✅ Global availability

**Your audio processing pipeline is now in the cloud!**
