from __future__ import annotations

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[3]


def repo_path(*parts: str) -> Path:
    return PROJECT_ROOT.joinpath(*parts)


def ensure_dir(path: str | Path) -> Path:
    directory = Path(path)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def to_repo_relative(path: str | Path) -> str:
    candidate = Path(path).resolve()
    try:
        return candidate.relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return str(candidate)
