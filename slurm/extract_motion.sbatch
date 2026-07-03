#!/bin/bash
#SBATCH --job-name=avagen-motion-extract
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=08:00:00
#SBATCH --output=outputs/logs/%x-%j.out

set -euo pipefail

echo "Motion extraction script placeholder"
