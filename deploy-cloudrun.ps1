# Complete build and deploy script for Cloud Run (PowerShell)

$ErrorActionPreference = "Stop"

Write-Host "=== DiaRemot Cloud Run Deployment ===" -ForegroundColor Green

# Get project ID
$PROJECT_ID = gcloud config get-value project 2>$null
if ([string]::IsNullOrEmpty($PROJECT_ID)) {
    Write-Host "ERROR: No GCP project set" -ForegroundColor Red
    Write-Host "Run: gcloud config set project YOUR_PROJECT_ID"
    exit 1
}

# Configuration
$REGION = if ($args.Count -gt 0) { $args[0] } else { "us-central1" }
$SERVICE_NAME = "diaremot-service"
$IMAGE_NAME = "diaremot2-on"
$IMAGE_TAG = if ($args.Count -gt 1) { $args[1] } else { "latest" }
$MEMORY = if ($args.Count -gt 2) { $args[2] } else { "4Gi" }
$CPU = if ($args.Count -gt 3) { $args[3] } else { "4" }

Write-Host "Configuration:" -ForegroundColor Green
Write-Host "  Project: $PROJECT_ID"
Write-Host "  Region: $REGION"
Write-Host "  Service: $SERVICE_NAME"
Write-Host "  Image: gcr.io/$PROJECT_ID/${IMAGE_NAME}:${IMAGE_TAG}"
Write-Host "  Resources: $MEMORY RAM, $CPU CPUs"
Write-Host ""

$confirmation = Read-Host "Continue? (y/N)"
if ($confirmation -ne 'y' -and $confirmation -ne 'Y') {
    Write-Host "Aborted."
    exit 0
}

# Step 1: Enable required APIs
Write-Host "Step 1/4: Enabling APIs..." -ForegroundColor Yellow
gcloud services enable cloudbuild.googleapis.com
gcloud services enable run.googleapis.com
gcloud services enable containerregistry.googleapis.com
Write-Host "✓ APIs enabled" -ForegroundColor Green
Write-Host ""

# Step 2: Build with Cloud Build
Write-Host "Step 2/4: Building image with Cloud Build..." -ForegroundColor Yellow
gcloud builds submit `
    --config=cloudbuild-cloudrun.yaml `
    --substitutions="_IMAGE_NAME=$IMAGE_NAME,_IMAGE_TAG=$IMAGE_TAG" `
    .

if ($LASTEXITCODE -ne 0) {
    Write-Host "✗ Build failed" -ForegroundColor Red
    exit 1
}
Write-Host "✓ Image built successfully" -ForegroundColor Green
Write-Host ""

# Step 3: Deploy to Cloud Run
Write-Host "Step 3/4: Deploying to Cloud Run..." -ForegroundColor Yellow
gcloud run deploy $SERVICE_NAME `
    --image gcr.io/$PROJECT_ID/${IMAGE_NAME}:${IMAGE_TAG} `
    --region $REGION `
    --platform managed `
    --memory $MEMORY `
    --cpu $CPU `
    --timeout 3600 `
    --max-instances 10 `
    --allow-unauthenticated `
    --execution-environment gen2

if ($LASTEXITCODE -ne 0) {
    Write-Host "✗ Deployment failed" -ForegroundColor Red
    exit 1
}
Write-Host "✓ Deployed successfully" -ForegroundColor Green
Write-Host ""

# Step 4: Get service URL and test
Write-Host "Step 4/4: Testing service..." -ForegroundColor Yellow
$SERVICE_URL = gcloud run services describe $SERVICE_NAME `
    --region $REGION `
    --format 'value(status.url)'

Write-Host "Service URL: $SERVICE_URL"
Write-Host ""

# Test health endpoint
Write-Host "Testing health endpoint..."
try {
    $response = Invoke-WebRequest -Uri "$SERVICE_URL/health" -UseBasicParsing
    if ($response.StatusCode -eq 200) {
        Write-Host "✓ Service is healthy" -ForegroundColor Green
    }
} catch {
    Write-Host "✗ Service health check failed" -ForegroundColor Red
}

Write-Host ""
Write-Host "=== Deployment Complete ===" -ForegroundColor Green
Write-Host ""
Write-Host "Service URL: $SERVICE_URL"
Write-Host ""
Write-Host "Test the API:"
Write-Host "  curl $SERVICE_URL/health"
Write-Host "  curl -X POST $SERVICE_URL/process -F 'audio=@test.wav'"
Write-Host ""
Write-Host "View logs:"
Write-Host "  gcloud run services logs tail $SERVICE_NAME --region $REGION"
Write-Host ""
