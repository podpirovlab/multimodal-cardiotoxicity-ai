#!/usr/bin/env bash
# Download PTB-XL v1.0.3 (open access, CC BY 4.0) from PhysioNet into data/ptb-xl.
# ~1.7 GB zip, ~3 GB unpacked. Run once from the repository root:
#     bash scripts/download_ptbxl.sh
set -euo pipefail
URL="https://physionet.org/static/published-projects/ptb-xl/ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3.zip"
mkdir -p data
cd data
if [ ! -f ptb-xl.zip ]; then
  echo "Downloading PTB-XL (~1.7 GB)..."
  curl -L --fail --retry 5 -C - -o ptb-xl.zip "$URL"
fi
echo "Unpacking..."
unzip -q -o ptb-xl.zip
rm -rf ptb-xl
mv ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3 ptb-xl
echo "Done: data/ptb-xl/ptbxl_database.csv"
echo "Next: python train_ptbxl.py --data data/ptb-xl --epochs 30 --export-onnx"
