# ✅ FILES CREATED - Cloud Run Deployment

## Location
`D:\diaremot\diaremot2-on\`

## New Files for Running Code on Google Cloud

### 🚀 Deployment Scripts (Use These!)
- **`deploy-cloudrun.ps1`** - Windows PowerShell deployment script
- **`deploy-cloudrun.sh`** - Linux/Mac bash deployment script

### 🐳 Docker & Build
- **`Dockerfile.cloudrun`** - Web service container definition
- **`cloudbuild-cloudrun.yaml`** - Cloud Build configuration for web service

### 📡 API Service
- **`src/diaremot/api.py`** - FastAPI REST API wrapper
  - `POST /process` - Upload audio, get transcript
  - `GET /health` - Health check

### 📖 Documentation
- **`CLOUDRUN_DEPLOY.md`** - Complete deployment guide (READ THIS!)

---

## Original Files (Still There)
- **`cloudbuild.yaml`** - Build CLI images
- **`Dockerfile`** - CLI container
- **`CLOUD_BUILD_GUIDE.md`** - Build documentation
- **`CLOUD_BUILD_SUMMARY.md`** - Build summary

---

## 🎯 Quick Start

### Windows:
```powershell
cd D:\diaremot\diaremot2-on
.\deploy-cloudrun.ps1
```

### Linux/Mac:
```bash
cd /path/to/diaremot2-on
./deploy-cloudrun.sh
```

**That's it!** The script will:
1. Build your code into a Docker image on Google Cloud
2. Deploy it as a web service
3. Give you a URL to upload audio files

---

## 💡 What You Get

After running the deploy script:

```
Your diaremot pipeline is now a web API!

Upload audio → https://your-service-xxxxx.run.app/process
                        ↓
              Diarization + ASR + Emotion Analysis
                        ↓
         Returns: transcript with speakers & emotions
```

**Example usage:**
```bash
curl -X POST https://your-service-url.run.app/process \
  -F "audio=@my_recording.wav" \
  -o transcript.csv
```

---

## 📂 File Sizes
- deploy-cloudrun.ps1: ~3.5 KB
- deploy-cloudrun.sh: ~3.0 KB
- Dockerfile.cloudrun: ~1.7 KB
- cloudbuild-cloudrun.yaml: ~2.1 KB
- api.py: ~5.0 KB
- CLOUDRUN_DEPLOY.md: ~6.5 KB

---

## ❓ Quick FAQ

**Q: Where are the files?**  
A: In `D:\diaremot\diaremot2-on\` - open File Explorer and navigate there

**Q: Which file do I run?**  
A: `deploy-cloudrun.ps1` (Windows) or `deploy-cloudrun.sh` (Linux/Mac)

**Q: Do I need to install anything?**  
A: Just Google Cloud SDK (`gcloud` command)

**Q: How much does it cost?**  
A: ~$0.03 per 10-minute audio file processed (free tier: 2M requests/month)

**Q: What's the difference between the two Dockerfiles?**  
- `Dockerfile` = CLI version (for running locally or in containers)
- `Dockerfile.cloudrun` = Web service version (for Cloud Run)

**Q: Can I test locally first?**  
A: Yes! See `CLOUDRUN_DEPLOY.md` for local testing instructions

---

## 📞 Next Steps

1. **Read:** `CLOUDRUN_DEPLOY.md` (full documentation)
2. **Deploy:** Run `.\deploy-cloudrun.ps1`
3. **Test:** Upload an audio file to your service URL
4. **Monitor:** Check logs in Google Cloud Console

---

## 🎉 You're Ready!

Everything you need to run diaremot on Google Cloud is in this directory.

**Just run the deploy script and you're live!**
