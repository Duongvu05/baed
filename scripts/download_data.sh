#!/usr/bin/env bash
# Download the raw datasets into $BAED_ROOT/data/graphs (default: repo root).
#
#   bash scripts/download_data.sh                 # elliptic dgraph tfinance reddit
#   bash scripts/download_data.sh elliptic        # only some of them
#
# Sources (the authors' Baidu Pan link cannot be fetched non-interactively):
#   elliptic  PyG mirror of the Kaggle Elliptic data set (data.pyg.org)
#   dgraph    HF mirror of DGraphFin (huggingface.co/datasets/dingyiz/fraud_detection_collections)
#   tfinance  BWGNN authors' Google Drive (DGL graph file, as expected by the code)
#   reddit    GADBench repo (DGL graph) converted to reddit.mat (Network/Attributes/Label)
# photo and Amazon .mat files are not mirrored publicly; put them in data/graphs/ by hand.
set -euo pipefail
ROOT="${BAED_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}"
G="$ROOT/data/graphs"
mkdir -p "$G"
cd "$G"
DATASETS=("$@")
[ ${#DATASETS[@]} -eq 0 ] && DATASETS=(elliptic dgraph tfinance reddit)

fetch() { # url out
  [ -s "$2" ] && { echo "  have $2"; return; }
  curl -fL --retry 10 --retry-delay 3 -C - -o "$2.part" "$1" && mv "$2.part" "$2"
}

for d in "${DATASETS[@]}"; do
  echo "== $d"
  case "$d" in
    elliptic)
      mkdir -p elliptic_bitcoin_dataset
      for f in classes edgelist features; do
        fetch "https://data.pyg.org/datasets/elliptic/elliptic_txs_${f}.csv.zip" "elliptic_txs_${f}.csv.zip"
        unzip -oq "elliptic_txs_${f}.csv.zip" -d elliptic_bitcoin_dataset
      done
      ls -la elliptic_bitcoin_dataset ;;
    dgraph)
      fetch "https://huggingface.co/datasets/dingyiz/fraud_detection_collections/resolve/main/data/dgraph_fin/dgraphfin.npz" dgraphfin.npz ;;
    tfinance)
      if [ ! -s tfinance ]; then
        uvx gdown --folder "https://drive.google.com/drive/folders/1PpNwvZx_YRSCDiHaBUmRIS3x1rZR7fMr" -O bwgnn_gdrive
        f=$(find bwgnn_gdrive -iname 'tfinance*' | head -1)
        case "$f" in
          *.zip) unzip -oq "$f" -d . ;;
          *) cp "$f" tfinance ;;
        esac
      fi
      ls -la tfinance ;;
    reddit)
      fetch "https://raw.githubusercontent.com/squareRoot3/GADBench/master/datasets/reddit" reddit_dgl
      (cd "$ROOT" && uv run python scripts/dgl_to_mat.py "$G/reddit_dgl" "$G/reddit.mat") ;;
    *) echo "unknown dataset $d"; exit 1 ;;
  esac
done
