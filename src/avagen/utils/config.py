from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from avagen.utils.paths import ensure_dir


def load_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with config_path.open("r", encoding="utf-8") as handle:
        if config_path.suffix in {".yaml", ".yml"}:
            import yaml

            return yaml.safe_load(handle) or {}
        if config_path.suffix == ".json":
            return json.load(handle)

    raise ValueError(f"Unsupported config extension: {config_path.suffix}")


def dump_json(data: Any, path: str | Path) -> Path:
    output_path = Path(path)
    ensure_dir(output_path.parent)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, sort_keys=True)
    return output_path
