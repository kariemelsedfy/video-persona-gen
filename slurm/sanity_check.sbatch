#!/bin/bash
#SBATCH --job-name=avagen-sanity
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:15:00
#SBATCH --output=outputs/logs/%x-%j.out

set -euo pipefail

python - <<'PY'
import platform
print("python_ok", platform.python_version())
try:
    import torch
    print("cuda_available", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("device_count", torch.cuda.device_count())
        print("device_name", torch.cuda.get_device_name(0))
except Exception as exc:
    print("torch_check_failed", exc)
PY
