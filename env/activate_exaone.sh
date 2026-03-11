#!/usr/bin/env bash

set -euo pipefail

source /home/ubuntu/anaconda3/etc/profile.d/conda.sh
conda activate exaone
source /home/ubuntu/Desktop/LG\ Aimers/env/exaone.env

echo "CONDA_DEFAULT_ENV=${CONDA_DEFAULT_ENV:-}"
echo "WORKSPACE=${WORKSPACE:-}"
echo "MODEL_ID=${MODEL_ID:-}"
echo "DATASET_ID=${DATASET_ID:-}"
