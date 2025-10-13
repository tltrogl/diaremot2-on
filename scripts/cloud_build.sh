#!/bin/bash
# Quick Cloud Build submission script for diaremot2-on

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}=== DiaRemot Cloud Build Submission ===${NC}"

# Check if gcloud is installed
if ! command -v gcloud &> /dev/null; then
    echo -e "${RED}ERROR: gcloud CLI not found. Install from: https://cloud.google.com/sdk/docs/install${NC}"
    exit 1
fi

# Get current project
PROJECT_ID=$(gcloud config get-value project 2>/dev/null)
if [ -z "$PROJECT_ID" ]; then
    echo -e "${RED}ERROR: No GCP project set. Run: gcloud config set project YOUR_PROJECT_ID${NC}"
    exit 1
fi

echo -e "${GREEN}Using GCP Project: ${PROJECT_ID}${NC}"

# Parse arguments
IMAGE_TAG="${1:-latest}"
IMAGE_NAME="${2:-diaremot2-on}"

echo -e "${YELLOW}Building image: gcr.io/${PROJECT_ID}/${IMAGE_NAME}:${IMAGE_TAG}${NC}"

# Confirm
read -p "Continue? (y/N): " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Aborted."
    exit 0
fi

# Submit build
echo -e "${GREEN}Submitting build to Cloud Build...${NC}"
gcloud builds submit \
    --config=cloudbuild.yaml \
    --substitutions="_IMAGE_NAME=${IMAGE_NAME},_IMAGE_TAG=${IMAGE_TAG}" \
    .

echo -e "${GREEN}Build submitted successfully!${NC}"
echo -e "${GREEN}View logs at: https://console.cloud.google.com/cloud-build/builds${NC}"
