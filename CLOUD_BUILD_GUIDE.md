# Cloud Build Setup for DiaRemot2-On

Complete guide for deploying **diaremot2-on** on Google Cloud Platform using Cloud Build.

---

## Overview

The Cloud Build configuration (`cloudbuild.yaml`) automates:
1. **Testing** - Installs deps, runs pytest
2. **Building** - Creates optimized Docker image
3. **Pushing** - Uploads to Google Container Registry (GCR)
4. **Tagging** - Tags with `:latest` and `:BUILD_ID`

---

## Prerequisites

### Local Setup

```bash
# Install Google Cloud SDK
curl https://sdk.cloud.google.com | bash
exec -l $SHELL

# Initialize gcloud
gcloud init

# Authenticate
gcloud auth login
gcloud auth configure-docker
```

### GCP Project Setup

```bash
# Set your project ID
export PROJECT_ID="your-gcp-project-id"
gcloud config set project $PROJECT_ID

# Enable required APIs
gcloud services enable cloudbuild.googleapis.com
gcloud services enable containerregistry.googleapis.com
gcloud services enable artifactregistry.googleapis.com

# Grant Cloud Build service account permissions
PROJECT_NUMBER=$(gcloud projects describe $PROJECT_ID --format="value(projectNumber)")
gcloud projects add-iam-policy-binding $PROJECT_ID \
    --member="serviceAccount:${PROJECT_NUMBER}@cloudbuild.gserviceaccount.com" \
    --role="roles/artifactregistry.writer"
```

---

## Quick Start

### 1. Manual Cloud Build Trigger

```bash
# From project root
cd D:\diaremot\diaremot2-on

# Submit build to Cloud Build
gcloud builds submit \
    --config=cloudbuild.yaml \
    --substitutions=_IMAGE_TAG=v2.2.0 \
    .

# Watch build logs
gcloud builds log <BUILD_ID> --stream
```

### 2. Create Automated Build Trigger

#### Via Console
1. Go to [Cloud Build Triggers](https://console.cloud.google.com/cloud-build/triggers)
2. Click **CREATE TRIGGER**
3. Configure:
   - **Name**: `diaremot-on-build`
   - **Event**: Push to branch
   - **Source**: Connect your repository
   - **Branch**: `^main$` (or your preferred branch)
   - **Configuration**: `cloudbuild.yaml`
   - **Substitution variables**:
     - `_IMAGE_TAG`: `${BRANCH_NAME}-${SHORT_SHA}`

#### Via gcloud CLI

```bash
gcloud builds triggers create github \
    --name="diaremot-on-build" \
    --repo-name="diaremot2-on" \
    --repo-owner="your-github-username" \
    --branch-pattern="^main$" \
    --build-config="cloudbuild.yaml" \
    --substitutions="_IMAGE_TAG=latest"
```

---

## Configuration

### Substitution Variables

Override at build time:

```bash
gcloud builds submit \
    --config=cloudbuild.yaml \
    --substitutions=\
_IMAGE_NAME=diaremot-production,\
_IMAGE_TAG=v2.2.0,\
_PYTHON_VERSION=3.11,\
_GCR_HOSTNAME=gcr.io \
    .
```

Available substitutions:
- `_IMAGE_NAME` - Docker image name (default: `diaremot2-on`)
- `_IMAGE_TAG` - Image tag (default: `latest`)
- `_PYTHON_VERSION` - Python version (default: `3.11`)
- `_GCR_HOSTNAME` - Registry hostname (default: `gcr.io`)

### Build Options

Edit `cloudbuild.yaml` to customize:

```yaml
options:
  machineType: 'E2_HIGHCPU_8'  # CPU type (E2_HIGHCPU_8, N1_HIGHCPU_8, etc.)
  diskSizeGb: 100               # Disk size
  logging: CLOUD_LOGGING_ONLY   # Logging destination
```

---

## Running the Container

### Pull from GCR

```bash
# Pull latest
docker pull gcr.io/${PROJECT_ID}/diaremot2-on:latest

# Or specific build
docker pull gcr.io/${PROJECT_ID}/diaremot2-on:BUILD_ID_HERE
```

### Run Locally

#### Basic usage (no models)
```bash
docker run --rm \
    gcr.io/${PROJECT_ID}/diaremot2-on:latest \
    diagnostics
```

#### With model volume mount
```bash
docker run --rm \
    -v /path/to/models:/srv/models:ro \
    -v $(pwd)/outputs:/app/outputs \
    gcr.io/${PROJECT_ID}/diaremot2-on:latest \
    run --input /app/test_audio.wav --outdir /app/outputs
```

#### Interactive shell
```bash
docker run -it --rm \
    -v /path/to/models:/srv/models:ro \
    gcr.io/${PROJECT_ID}/diaremot2-on:latest \
    /bin/bash
```

### On Cloud Run

Deploy to Cloud Run for serverless execution:

```bash
# Deploy
gcloud run deploy diaremot-service \
    --image gcr.io/${PROJECT_ID}/diaremot2-on:latest \
    --platform managed \
    --region us-central1 \
    --memory 4Gi \
    --cpu 4 \
    --timeout 3600 \
    --max-instances 10 \
    --allow-unauthenticated

# Update with new image
gcloud run services update diaremot-service \
    --image gcr.io/${PROJECT_ID}/diaremot2-on:v2.2.0 \
    --region us-central1
```

### On GKE (Kubernetes)

Create deployment:

```yaml
# deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: diaremot-deployment
spec:
  replicas: 2
  selector:
    matchLabels:
      app: diaremot
  template:
    metadata:
      labels:
        app: diaremot
    spec:
      containers:
      - name: diaremot
        image: gcr.io/YOUR_PROJECT_ID/diaremot2-on:latest
        resources:
          requests:
            memory: "4Gi"
            cpu: "2"
          limits:
            memory: "8Gi"
            cpu: "4"
        volumeMounts:
        - name: models
          mountPath: /srv/models
          readOnly: true
        - name: outputs
          mountPath: /app/outputs
        env:
        - name: DIAREMOT_MODEL_DIR
          value: "/srv/models"
      volumes:
      - name: models
        persistentVolumeClaim:
          claimName: models-pvc
      - name: outputs
        emptyDir: {}
```

Apply:
```bash
kubectl apply -f deployment.yaml
```

---

## Model Management

### Option 1: Cloud Storage Bucket

Store models in GCS and mount:

```bash
# Create bucket
gsutil mb gs://${PROJECT_ID}-diaremot-models

# Upload models
gsutil -m cp -r /path/to/models/* gs://${PROJECT_ID}-diaremot-models/

# Make models accessible
gsutil iam ch allUsers:objectViewer gs://${PROJECT_ID}-diaremot-models
```

Use gcsfuse or cloud-native mounting in GKE.

### Option 2: Persistent Disk (GKE)

Create PV/PVC in Kubernetes:

```yaml
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: models-pvc
spec:
  accessModes:
  - ReadOnlyMany
  resources:
    requests:
      storage: 10Gi
  storageClassName: standard-rwo
```

### Option 3: Bake into Image (Not Recommended)

Only for small models (<100MB):

```dockerfile
# Add to Dockerfile
COPY models/ /srv/models/
```

⚠️ **Warning**: Makes image huge and slow to build/push.

---

## CI/CD Pipeline

### GitHub Actions Integration

`.github/workflows/cloudbuild.yml`:

```yaml
name: Cloud Build

on:
  push:
    branches: [main, develop]
  pull_request:
    branches: [main]

jobs:
  build:
    runs-on: ubuntu-latest
    
    steps:
    - uses: actions/checkout@v3
    
    - id: auth
      uses: google-github-actions/auth@v1
      with:
        credentials_json: ${{ secrets.GCP_CREDENTIALS }}
    
    - name: Set up Cloud SDK
      uses: google-github-actions/setup-gcloud@v1
    
    - name: Submit to Cloud Build
      run: |
        gcloud builds submit \
          --config=cloudbuild.yaml \
          --substitutions=_IMAGE_TAG=${GITHUB_REF##*/}-${GITHUB_SHA::8} \
          .
```

### GitLab CI Integration

`.gitlab-ci.yml`:

```yaml
stages:
  - build

cloud-build:
  stage: build
  image: google/cloud-sdk:alpine
  script:
    - echo $GCP_SERVICE_KEY | base64 -d > ${HOME}/gcp-key.json
    - gcloud auth activate-service-account --key-file ${HOME}/gcp-key.json
    - gcloud config set project $GCP_PROJECT_ID
    - gcloud builds submit --config=cloudbuild.yaml .
  only:
    - main
    - develop
```

---

## Monitoring & Debugging

### View Build History

```bash
# List recent builds
gcloud builds list --limit=10

# Get specific build details
gcloud builds describe BUILD_ID

# Stream build logs
gcloud builds log BUILD_ID --stream
```

### Debug Failed Builds

```bash
# Get build logs
gcloud builds log BUILD_ID > build.log

# Check builder logs step-by-step
gcloud builds log BUILD_ID --stream | grep "Step #"
```

### Common Issues

**Issue**: `Error 403: Permission denied`
```bash
# Fix: Grant Cloud Build service account correct permissions
PROJECT_NUMBER=$(gcloud projects describe $PROJECT_ID --format="value(projectNumber)")
gcloud projects add-iam-policy-binding $PROJECT_ID \
    --member="serviceAccount:${PROJECT_NUMBER}@cloudbuild.gserviceaccount.com" \
    --role="roles/storage.admin"
```

**Issue**: `Build timeout`
```yaml
# Fix: Increase timeout in cloudbuild.yaml
timeout: 3600s  # 1 hour
```

**Issue**: `Out of memory`
```yaml
# Fix: Use larger machine
options:
  machineType: 'N1_HIGHCPU_32'
  diskSizeGb: 200
```

---

## Cost Optimization

### Build Time Optimization

1. **Use caching**:
```yaml
# Add to cloudbuild.yaml
options:
  pool:
    name: 'projects/PROJECT_ID/locations/REGION/workerPools/POOL_NAME'
```

2. **Parallel steps**:
```yaml
- name: 'python:3.11'
  id: 'test-1'
  waitFor: ['-']
  ...
  
- name: 'python:3.11'
  id: 'test-2'
  waitFor: ['-']
  ...
```

3. **Smaller base images**:
```dockerfile
FROM python:3.11-slim  # ✅ Good
# vs
FROM python:3.11       # ❌ Larger
```

### Pricing

- **Build time**: $0.003/build-minute (first 120 min/day free)
- **Storage**: GCR charges for image storage
- **Data transfer**: Egress charges apply

Example: 10 builds/day × 5 min/build × 30 days = 1500 minutes = ~$4.50/month

---

## Best Practices

1. **Pin versions**: Use specific image tags, not `:latest` in production
2. **Multi-stage builds**: Reduce final image size (see `Dockerfile`)
3. **Secrets**: Use Secret Manager, not env vars
4. **Health checks**: Always include in Dockerfile
5. **Resource limits**: Set CPU/memory limits for containers
6. **Logging**: Use structured logging (JSON) for better monitoring

---

## Advanced: Artifact Registry

Modern alternative to GCR:

```bash
# Create repository
gcloud artifacts repositories create diaremot \
    --repository-format=docker \
    --location=us-central1 \
    --description="DiaRemot speech pipeline images"

# Update cloudbuild.yaml
substitutions:
  _GCR_HOSTNAME: 'us-central1-docker.pkg.dev'
```

---

## Security

### Vulnerability Scanning

```bash
# Enable scanning
gcloud container images scan gcr.io/${PROJECT_ID}/diaremot2-on:latest

# View results
gcloud container images list-tags gcr.io/${PROJECT_ID}/diaremot2-on \
    --format='get(digest)' | head -n 1 | xargs -I {} \
    gcloud container images describe gcr.io/${PROJECT_ID}/diaremot2-on@{} \
    --show-package-vulnerability
```

### Binary Authorization

Enforce signed images:

```bash
# Create policy
gcloud container binauthz policy import policy.yaml

# Attest build
gcloud beta container binauthz attestations create \
    --artifact-url=gcr.io/${PROJECT_ID}/diaremot2-on@sha256:DIGEST \
    --attestor=ATTESTOR_NAME \
    --signature-file=SIGNATURE_FILE
```

---

## Support

- **Cloud Build Docs**: https://cloud.google.com/build/docs
- **Dockerfile Best Practices**: https://docs.docker.com/develop/dev-best-practices/
- **GKE Guide**: https://cloud.google.com/kubernetes-engine/docs

---

## License

MIT License - See LICENSE file for details
