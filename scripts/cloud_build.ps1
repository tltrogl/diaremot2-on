# Quick Cloud Build submission script for diaremot2-on (PowerShell)

$ErrorActionPreference = "Stop"

Write-Host "=== DiaRemot Cloud Build Submission ===" -ForegroundColor Green

# Check if gcloud is installed
if (!(Get-Command gcloud -ErrorAction SilentlyContinue)) {
    Write-Host "ERROR: gcloud CLI not found. Install from: https://cloud.google.com/sdk/docs/install" -ForegroundColor Red
    exit 1
}

# Get current project
$PROJECT_ID = gcloud config get-value project 2>$null
if ([string]::IsNullOrEmpty($PROJECT_ID)) {
    Write-Host "ERROR: No GCP project set. Run: gcloud config set project YOUR_PROJECT_ID" -ForegroundColor Red
    exit 1
}

Write-Host "Using GCP Project: $PROJECT_ID" -ForegroundColor Green

# Parse arguments
$IMAGE_TAG = if ($args.Count -gt 0) { $args[0] } else { "latest" }
$IMAGE_NAME = if ($args.Count -gt 1) { $args[1] } else { "diaremot2-on" }

Write-Host "Building image: gcr.io/$PROJECT_ID/${IMAGE_NAME}:${IMAGE_TAG}" -ForegroundColor Yellow

# Confirm
$confirmation = Read-Host "Continue? (y/N)"
if ($confirmation -ne 'y' -and $confirmation -ne 'Y') {
    Write-Host "Aborted."
    exit 0
}

# Submit build
Write-Host "Submitting build to Cloud Build..." -ForegroundColor Green
gcloud builds submit `
    --config=cloudbuild.yaml `
    --substitutions="_IMAGE_NAME=$IMAGE_NAME,_IMAGE_TAG=$IMAGE_TAG" `
    .

Write-Host "Build submitted successfully!" -ForegroundColor Green
Write-Host "View logs at: https://console.cloud.google.com/cloud-build/builds" -ForegroundColor Green
