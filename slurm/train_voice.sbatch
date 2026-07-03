#!/bin/bash
#SBATCH --job-name=avagen-train-voice
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=outputs/logs/%x-%j.out

set -euo pipefail

echo "Voice training script placeholder"
