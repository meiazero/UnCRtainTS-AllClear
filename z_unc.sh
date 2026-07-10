#!/bin/bash
#SBATCH -t 80:00:00                      # Time limit (hh:mm:ss)
#SBATCH --partition=gpu                  # Partition
#SBATCH --gres=gpu:1                     # Number of GPUs
#SBATCH --mem-per-cpu=60G
#SBATCH --cpus-per-task=4                # Number of CPU cores per task
#SBATCH -N 1                             # Number of nodes
#SBATCH --output=watch_folder/data-%x-%j.log
#SBATCH --requeue                        # Requeue job if it fails

# Point these at your checkout and your AllClear copy.
REPO="${REPO:-$(cd "$(dirname "$0")" && pwd)}"
ALLCLEAR="${ALLCLEAR:?set ALLCLEAR to the AllClear repo root}"

cd "$REPO"

uv run python train_allclear.py \
    --experiment_name "allclear_v1" \
    --allclear_repo        "$ALLCLEAR" \
    --allclear_root        "$ALLCLEAR/data" \
    --allclear_train_split "$ALLCLEAR/metadata/datasets/train_tx3_s2-s1_10pct.json" \
    --allclear_val_split   "$ALLCLEAR/metadata/datasets/val_tx3_s2-s1-landsat_100pct.json" \
    --allclear_test_split  "$ALLCLEAR/metadata/datasets/test_tx3_s2-s1_100pct.json" \
    --num_workers 4
