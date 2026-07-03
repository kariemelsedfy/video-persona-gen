#!/bin/bash
#SBATCH --job-name=avagen-preprocess
#SBATCH --partition=gpu
#SBATCH --gres=gpu:0
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=outputs/logs/%x-%j.out

set -euo pipefail

python scripts/preprocess_dataset.py "$@"
