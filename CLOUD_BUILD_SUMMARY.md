# Cloud Build Setup - Summary

## Files Created

### Core Configuration
1. **`cloudbuild.yaml`** - Cloud Build pipeline configuration
   - Tests package
   - Builds Docker image
   - Pushes to GCR
   - Multi-step automated build

2. **`Dockerfile`** - Multi-stage Docker image
   - Python 3.11 slim base
   - FFmpeg included
   - Optimized layers for caching
   - Models mounted at `/srv/models`
   - Output at `/app/outputs`

3. **`.dockerignore`** - Build optimization
   - Excludes cache, logs, temp files
   - Reduces image size
   - Faster builds

### Documentation
4. **`CLOUD_BUILD_GUIDE.md`** - Complete deployment guide
   - GCP setup instructions
   - Cloud Build usage
   - Cloud Run deployment
   - GKE deployment
   - Model management strategies
   - CI/CD integration examples
   - Troubleshooting guide

### Helper Scripts
5. **`scripts/cloud_build.sh`** - Linux/macOS build script
6. **`scripts/cloud_build.ps1`** - Windows PowerShell build script

---

## Quick Start

### 1. Local Docker Build
```bash
# Build locally
docker build -t diaremot2-on:latest .

# Run
docker run --rm \
    -v /path/to/models:/srv/models:ro \
    -v $(pwd)/outputs:/app/outputs \
    diaremot2-on:latest \
    run --input audio.wav --outdir /app/outputs
```

### 2. Cloud Build (Manual)
```bash
# Submit to Cloud Build
gcloud builds submit --config=cloudbuild.yaml .

# Or use helper script
./scripts/cloud_build.sh v2.2.0
```

### 3. Cloud Build (Automated - GitHub)
1. Connect repo to Cloud Build
2. Create trigger pointing to `cloudbuild.yaml`
3. Push to main branch → automatic build

---

## Architecture

### Build Pipeline (cloudbuild.yaml)
```
┌─────────────────────────────────────────────┐
│ Step 1: Test Package                        │
│ - Install deps                              │
│ - Run pytest                                │
│ - Validate installation                     │
└──────────────┬──────────────────────────────┘
               │
               ▼
┌─────────────────────────────────────────────┐
│ Step 2: Build Docker Image                  │
│ - Multi-stage build                         │
│ - Tag with :latest and :BUILD_ID            │
└──────────────┬──────────────────────────────┘
               │
               ▼
┌─────────────────────────────────────────────┐
│ Step 3: Push to GCR                         │
│ - gcr.io/PROJECT/diaremot2-on:latest        │
│ - gcr.io/PROJECT/diaremot2-on:BUILD_ID      │
└─────────────────────────────────────────────┘
```

### Docker Image Layers
```
┌──────────────────────────────────────┐
│ python:3.11-slim (Base)              │
├──────────────────────────────────────┤
│ System Dependencies (FFmpeg)         │
├──────────────────────────────────────┤
│ Python Packages (requirements.txt)   │
├──────────────────────────────────────┤
│ Source Code (src/)                   │
├──────────────────────────────────────┤
│ Package Installation (pip install -e)│
├──────────────────────────────────────┤
│ Environment Setup (/srv/models, etc) │
└──────────────────────────────────────┘
```

---

## Configuration Options

### Build Variables
```yaml
substitutions:
  _IMAGE_NAME: 'diaremot2-on'      # Change image name
  _IMAGE_TAG: 'latest'             # Change tag
  _GCR_HOSTNAME: 'gcr.io'          # Use Artifact Registry
  _PYTHON_VERSION: '3.11'          # Python version
```

### Runtime Environment
```dockerfile
ENV DIAREMOT_MODEL_DIR=/srv/models
ENV HF_HOME=/app/.cache
ENV OMP_NUM_THREADS=4
```

Override at runtime:
```bash
docker run -e DIAREMOT_MODEL_DIR=/custom/path ...
```

---

## Deployment Targets

### 1. Cloud Run (Serverless)
```bash
gcloud run deploy diaremot-service \
    --image gcr.io/PROJECT/diaremot2-on:latest \
    --memory 4Gi \
    --cpu 4
```

**Pros**: Auto-scaling, pay-per-use  
**Cons**: Cold starts, max 1 hour timeout

### 2. GKE (Kubernetes)
```bash
kubectl apply -f k8s/deployment.yaml
```

**Pros**: Full control, persistent workloads  
**Cons**: Cluster management overhead

### 3. Compute Engine VM
```bash
gcloud compute instances create-with-container diaremot-vm \
    --container-image=gcr.io/PROJECT/diaremot2-on:latest
```

**Pros**: Simple, persistent disk  
**Cons**: Always running, fixed resources

---

## Model Management Strategies

### Option A: Cloud Storage Mount
```bash
# Models in GCS bucket
gsutil -m cp -r models/* gs://PROJECT-models/

# Mount in container (gcsfuse or native)
docker run -v gs://PROJECT-models:/srv/models:ro ...
```

### Option B: Persistent Disk (GKE)
```yaml
volumes:
- name: models
  persistentVolumeClaim:
    claimName: models-pvc
```

### Option C: Download on Startup
```dockerfile
# Add to Dockerfile
RUN wget https://releases.com/models.zip && \
    unzip models.zip -d /srv/models
```

⚠️ **Recommended**: Option A or B for production

---

## CI/CD Integration

### GitHub Actions
```yaml
- name: Cloud Build
  run: gcloud builds submit --config=cloudbuild.yaml .
```

### GitLab CI
```yaml
cloud-build:
  script:
    - gcloud builds submit --config=cloudbuild.yaml .
```

### Jenkins
```groovy
stage('Cloud Build') {
    sh 'gcloud builds submit --config=cloudbuild.yaml .'
}
```

---

## Cost Estimates

### Typical Usage
- **10 builds/day** × 5 min/build = 50 min/day
- **1500 min/month** - 120 free = 1380 billable
- **Cost**: ~$4.14/month

### Storage
- **Image size**: ~2GB
- **5 tags** stored = 10GB
- **Cost**: ~$0.50/month

**Total**: ~$5/month for moderate usage

---

## Monitoring

### View Builds
```bash
# List recent
gcloud builds list --limit=10

# Watch live
gcloud builds log BUILD_ID --stream
```

### Metrics
- Build duration
- Success rate
- Resource usage

Available in Cloud Console → Cloud Build → History

---

## Next Steps

1. ✅ Set up GCP project
2. ✅ Enable APIs (Cloud Build, GCR)
3. ✅ Submit first build: `./scripts/cloud_build.sh`
4. ⬜ Set up automated trigger (GitHub/GitLab)
5. ⬜ Deploy to Cloud Run or GKE
6. ⬜ Configure model storage (GCS or PV)
7. ⬜ Set up monitoring/alerting

---

## Troubleshooting

| Issue | Solution |
|-------|----------|
| Permission denied | Grant Cloud Build SA correct IAM roles |
| Build timeout | Increase timeout in `cloudbuild.yaml` |
| OOM during build | Use larger machine type |
| Models not found | Check volume mount paths |
| Image too large | Review `.dockerignore`, multi-stage build |

---

## Support

- **Cloud Build Docs**: https://cloud.google.com/build/docs
- **Dockerfile Reference**: https://docs.docker.com/engine/reference/builder/
- **Project Issues**: https://github.com/your-org/diaremot2-on/issues

---

## License

MIT License - See LICENSE file
