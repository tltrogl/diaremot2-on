#!/bin/bash
set -e

# DiaRemot Setup Script for Linux/macOS
# This script sets up the Python environment and installs all dependencies

PYTHON_VERSION="3.11"
VENV_DIR=".venv"
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "========================================"
echo "DiaRemot Setup Script"
echo "========================================"
echo ""

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check for Python
echo "Checking for Python ${PYTHON_VERSION}..."
if command -v python${PYTHON_VERSION} &> /dev/null; then
    PYTHON_CMD="python${PYTHON_VERSION}"
elif command -v python3.11 &> /dev/null; then
    PYTHON_CMD="python3.11"
elif command -v python3 &> /dev/null; then
    PYTHON_CMD="python3"
    CURRENT_VERSION=$($PYTHON_CMD --version 2>&1 | cut -d' ' -f2 | cut -d'.' -f1,2)
    if [[ "$CURRENT_VERSION" != "3.11" ]] && [[ "$CURRENT_VERSION" != "3.12" ]]; then
        echo -e "${RED}Error: Python 3.11 or 3.12 required, found $CURRENT_VERSION${NC}"
        exit 1
    fi
else
    echo -e "${RED}Error: Python ${PYTHON_VERSION} not found!${NC}"
    echo "Please install Python 3.11 or 3.12 and try again."
    exit 1
fi

PYTHON_VERSION_FULL=$($PYTHON_CMD --version 2>&1 | cut -d' ' -f2)
echo -e "${GREEN}Found: Python ${PYTHON_VERSION_FULL}${NC}"
echo ""

# Check for FFmpeg
echo "Checking for FFmpeg..."
if command -v ffmpeg &> /dev/null; then
    FFMPEG_VERSION=$(ffmpeg -version 2>&1 | head -n1 | cut -d' ' -f3)
    echo -e "${GREEN}Found: FFmpeg ${FFMPEG_VERSION}${NC}"
else
    echo -e "${YELLOW}Warning: FFmpeg not found!${NC}"
    echo "FFmpeg is required for audio processing. Install with:"
    echo "  Ubuntu/Debian: sudo apt-get install ffmpeg"
    echo "  macOS:         brew install ffmpeg"
    echo ""
    read -p "Continue anyway? (y/N) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi
echo ""

# Create virtual environment
if [ -d "$VENV_DIR" ]; then
    echo -e "${YELLOW}Virtual environment already exists at ${VENV_DIR}${NC}"
    read -p "Remove and recreate? (y/N) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo "Removing existing virtual environment..."
        rm -rf "$VENV_DIR"
    else
        echo "Using existing virtual environment..."
    fi
fi

if [ ! -d "$VENV_DIR" ]; then
    echo "Creating virtual environment..."
    $PYTHON_CMD -m venv "$VENV_DIR"
    echo -e "${GREEN}Virtual environment created${NC}"
fi
echo ""

# Activate virtual environment
echo "Activating virtual environment..."
source "$VENV_DIR/bin/activate"
echo -e "${GREEN}Virtual environment activated${NC}"
echo ""

# Upgrade pip, wheel, setuptools
echo "Upgrading pip, wheel, and setuptools..."
python -m pip install --upgrade pip wheel setuptools
echo -e "${GREEN}Package managers upgraded${NC}"
echo ""

# Install project dependencies
echo "Installing project dependencies..."
if [ -f "requirements.txt" ]; then
    pip install -r requirements.txt
    echo -e "${GREEN}Dependencies installed${NC}"
else
    echo -e "${RED}Error: requirements.txt not found!${NC}"
    exit 1
fi
echo ""

# Install PyTorch CPU (if not already installed)
echo "Checking PyTorch installation..."
if python -c "import torch" 2>/dev/null; then
    echo -e "${GREEN}PyTorch already installed${NC}"
else
    echo "Installing PyTorch CPU..."
    pip install --index-url https://download.pytorch.org/whl/cpu torch
    echo -e "${GREEN}PyTorch CPU installed${NC}"
fi
echo ""

# Install package in editable mode
echo "Installing diaremot package in editable mode..."
pip install -e .
echo -e "${GREEN}Package installed${NC}"
echo ""

# Create necessary directories
echo "Creating necessary directories..."
mkdir -p .cache/hf .cache/torch .cache/transformers
mkdir -p checkpoints outputs logs data
echo -e "${GREEN}Directories created${NC}"
echo ""

# Check model directory
echo "Checking model directory..."
if [ -z "$DIAREMOT_MODEL_DIR" ]; then
    if [ -d "/srv/models" ]; then
        MODEL_DIR="/srv/models"
    elif [ -d "$HOME/models" ]; then
        MODEL_DIR="$HOME/models"
    elif [ -d "models" ]; then
        MODEL_DIR="models"
    else
        echo -e "${YELLOW}Warning: No model directory found${NC}"
        echo "Set DIAREMOT_MODEL_DIR environment variable or create:"
        echo "  /srv/models (recommended)"
        echo "  ~/models"
        echo "  ./models"
        MODEL_DIR=""
    fi
else
    MODEL_DIR="$DIAREMOT_MODEL_DIR"
fi

if [ -n "$MODEL_DIR" ]; then
    echo -e "${GREEN}Model directory: ${MODEL_DIR}${NC}"
else
    echo -e "${YELLOW}No model directory configured${NC}"
fi
echo ""

# Run diagnostics
echo "Running diagnostics..."
python -m diaremot.cli diagnostics
echo ""

# Setup complete
echo "========================================"
echo -e "${GREEN}Setup Complete!${NC}"
echo "========================================"
echo ""
echo "To activate the environment in future sessions:"
echo "  source ${VENV_DIR}/bin/activate"
echo ""
echo "To run the pipeline:"
echo "  python -m diaremot.cli run --input audio.wav --outdir outputs/"
echo ""
echo "To run smoke test:"
echo "  python -m diaremot.cli smoke --outdir outputs/"
echo ""
echo "Environment variables to set (add to ~/.bashrc or ~/.zshrc):"
echo "  export DIAREMOT_MODEL_DIR=${MODEL_DIR:-"/srv/models"}"
echo "  export HF_HOME=./.cache"
echo "  export HUGGINGFACE_HUB_CACHE=./.cache/hub"
echo "  export TRANSFORMERS_CACHE=./.cache/transformers"
echo "  export TORCH_HOME=./.cache/torch"
echo "  export OMP_NUM_THREADS=4"
echo "  export MKL_NUM_THREADS=4"
echo "  export NUMEXPR_MAX_THREADS=4"
echo "  export TOKENIZERS_PARALLELISM=false"
echo ""
