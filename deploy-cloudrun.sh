#!/bin/bash
# Complete build and deploy script for Cloud Run

set -e

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

echo -e "${GREEN}=== DiaRemot Cloud Run Deployment ===${NC}"

# Get project ID
PROJECT_ID=$(gcloud config get-value project 2>/dev/null)
if [ -z "$PROJECT_ID" ]; then
    echo -e "${RED}ERROR: No GCP project set${NC}"
    echo "Run: gcloud config set project YOUR_PROJECT_ID"
    exit 1
fi

# Configuration
REGION="${1:-us-central1}"
SERVICE_NAME="diaremot-service"
IMAGE_NAME="diaremot2-on"
IMAGE_TAG="${2:-latest}"
MEMORY="${3:-4Gi}"
CPU="${4:-4}"

echo -e "${GREEN}Configuration:${NC}"
echo "  Project: $PROJECT_ID"
echo "  Region: $REGION"
echo "  Service: $SERVICE_NAME"
echo "  Image: gcr.io/$PROJECT_ID/$IMAGE_NAME:$IMAGE_TAG"
echo "  Resources: $MEMORY RAM, $CPU CPUs"
echo ""

read -p "Continue? (y/N): " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Aborted."
    exit 0
fi

# Step 1: Enable required APIs
echo -e "${YELLOW}Step 1/4: Enabling APIs...${NC}"
gcloud services enable cloudbuild.googleapis.com
gcloud services enable run.googleapis.com
gcloud services enable containerregistry.googleapis.com
echo -e "${GREEN}✓ APIs enabled${NC}"
echo ""

# Step 2: Build with Cloud Build
echo -e "${YELLOW}Step 2/4: Building image with Cloud Build...${NC}"
gcloud builds submit \
    --config=cloudbuild-cloudrun.yaml \
    --substitutions="_IMAGE_NAME=$IMAGE_NAME,_IMAGE_TAG=$IMAGE_TAG" \
    .

if [ $? -ne 0 ]; then
    echo -e "${RED}✗ Build failed${NC}"
    exit 1
fi
echo -e "${GREEN}✓ Image built successfully${NC}"
echo ""

# Step 3: Deploy to Cloud Run
echo -e "${YELLOW}Step 3/4: Deploying to Cloud Run...${NC}"
gcloud run deploy $SERVICE_NAME \
    --image gcr.io/$PROJECT_ID/$IMAGE_NAME:$IMAGE_TAG \
    --region $REGION \
    --platform managed \
    --memory $MEMORY \
    --cpu $CPU \
    --timeout 3600 \
    --max-instances 10 \
    --allow-unauthenticated \
    --execution-environment gen2

if [ $? -ne 0 ]; then
    echo -e "${RED}✗ Deployment failed${NC}"
    exit 1
fi
echo -e "${GREEN}✓ Deployed successfully${NC}"
echo ""

# Step 4: Get service URL and test
echo -e "${YELLOW}Step 4/4: Testing service...${NC}"
SERVICE_URL=$(gcloud run services describe $SERVICE_NAME \
    --region $REGION \
    --format 'value(status.url)')

echo "Service URL: $SERVICE_URL"
echo ""

# Test health endpoint
echo "Testing health endpoint..."
HTTP_CODE=$(curl -s -o /dev/null -w "%{http_code}" $SERVICE_URL/health)

if [ "$HTTP_CODE" == "200" ]; then
    echo -e "${GREEN}✓ Service is healthy${NC}"
else
    echo -e "${RED}✗ Service health check failed (HTTP $HTTP_CODE)${NC}"
fi

echo ""
echo -e "${GREEN}=== Deployment Complete ===${NC}"
echo ""
echo "Service URL: $SERVICE_URL"
echo ""
echo "Test the API:"
echo "  curl $SERVICE_URL/health"
echo "  curl -X POST $SERVICE_URL/process -F 'audio=@test.wav'"
echo ""
echo "View logs:"
echo "  gcloud run services logs tail $SERVICE_NAME --region $REGION"
echo ""
