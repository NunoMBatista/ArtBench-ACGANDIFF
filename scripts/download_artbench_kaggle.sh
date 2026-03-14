#!/usr/bin/env bash
set -euo pipefail

# Must have the following:
# - kaggle CLI installed (pip install kaggle)
# - ~/.kaggle/acess_token configured (paste API token from Kaggle account)

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
DATA_DIR="${1:-$REPO_ROOT/data}"
DATASET_SLUG="alexanderliao/artbench10"

mkdir -p "$DATA_DIR"

kaggle datasets download -d "$DATASET_SLUG" -p "$DATA_DIR" --unzip

echo "Download complete. Expected structure:"
echo "  $DATA_DIR/ArtBench-10/ArtBench-10.csv"
echo "  $DATA_DIR/ArtBench-10/artbench-10-python/artbench-10-batches-py"
