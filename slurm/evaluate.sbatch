#!/bin/bash
#SBATCH --job-name=avagen-eval
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=outputs/logs/%x-%j.out

set -euo pipefail

echo "Evaluation script placeholder"
