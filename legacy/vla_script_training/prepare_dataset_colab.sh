#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════════
# prepare_dataset_colab.sh — Helper script to prepare local dataset for Google Colab
# ═══════════════════════════════════════════════════════════════

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
DATASET_HOME="${PROJECT_ROOT}/data/hf_lerobot_cache"
DATASET_PATH="${DATASET_HOME}/local/ur5_cube"
ZIP_OUTPUT="${SCRIPT_DIR}/ur5_cube_converted.zip"

echo "══════════════════════════════════════════════════"
echo "  Dataset Preparation for Google Colab"
echo "══════════════════════════════════════════════════"

# Check if local converted dataset exists
if [ ! -d "${DATASET_PATH}" ]; then
    echo "❌ Error: Converted dataset not found at: ${DATASET_PATH}"
    echo "Please run python convert_data_cube.py first to generate it."
    exit 1
fi

echo "✔ Found local converted dataset at: ${DATASET_PATH}"
echo "Select how you would like to prepare your dataset for Google Colab:"
echo ""
echo "Option 1: Push dataset to Hugging Face Hub (Recommended & Easiest)"
echo "------------------------------------------------------------"
echo "This uploads your 235MB dataset directly to your Hugging Face account."
echo "Google Colab will then stream it automatically during training."
echo ""
echo "To do this, run these commands in your terminal:"
echo "  1. huggingface-cli login"
echo "  2. python -c \"from lerobot.datasets.lerobot_dataset import LeRobotDataset; dataset = LeRobotDataset('local/ur5_cube'); dataset.push_to_hub('your_hf_username/ur5_cube')\""
echo ""
echo "Option 2: Create a Compressed ZIP Archive"
echo "-----------------------------------------"
echo "We will package your local dataset into: ${ZIP_OUTPUT}"
echo "You can upload this 235MB zip file to Google Drive or directly to Colab."
echo ""

read -rp "Do you want to create the ZIP archive now? (y/n): " choice
if [[ "${choice}" =~ ^[Yy]$ ]]; then
    echo "📦 Creating ZIP archive..."
    
    # We navigate to parent folder of the local dataset so we zip the folder structure starting from local/ur5_cube
    cd "${DATASET_HOME}"
    zip -r "${ZIP_OUTPUT}" local/ur5_cube
    
    echo ""
    echo "✅ ZIP archive created successfully!"
    echo "📁 File Location: ${ZIP_OUTPUT}"
    echo "📁 Size: $(du -sh "${ZIP_OUTPUT}" | cut -f1)"
    echo ""
    echo "👉 How to use in Google Colab:"
    echo "1. Upload ${ZIP_OUTPUT} to your Google Drive or direct to Colab runtime."
    echo "2. Follow the unzipping instructions in the Colab notebook to restore it."
else
    echo "Skipped ZIP creation."
fi
echo "══════════════════════════════════════════════════"
