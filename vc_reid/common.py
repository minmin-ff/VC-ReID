from __future__ import annotations

import hashlib
import json
import os
import random
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

import numpy as np


@dataclass(frozen=True)
class Paths:
    suite_root: Path
    interactive_root: Path
    base_dir: Path
    train_features: Path
    test_features: Path
    train_meta: Path
    test_meta: Path
    train_videos: Path
    test_videos: Path
    train_annotations: Path
    test_annotations: Path
    train_bbox: Path
    test_bbox: Path
    train_static_json: Path
    test_static_json: Path
    qwen_path: Path
    clip_path: Path
    full_checkpoint: Path
    artifacts_root: Path
    results_root: Path


def load_paths(config_path: str | os.PathLike[str]) -> Paths:
    with open(config_path, "r", encoding="utf-8") as handle:
        raw = json.load(handle)
    missing = sorted(set(Paths.__annotations__) - set(raw))
    if missing:
        raise ValueError(f"Missing path keys: {', '.join(missing)}")
    return Paths(**{key: Path(raw[key]) for key in Paths.__annotations__})


def add_parent_to_path(paths: Paths) -> None:
    import sys

    value = str(paths.interactive_root)
    if value not in sys.path:
        sys.path.insert(0, value)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except ImportError:
        pass


def stable_uniform(key: str, seed: int) -> float:
    digest = hashlib.sha256(f"{seed}:{key}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") / float(2**64)


def utc_timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def make_run_dir(paths: Paths, experiment_id: str, tag: str) -> Path:
    safe_tag = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in tag)
    run_dir = paths.results_root / experiment_id / f"{utc_timestamp()}_{safe_tag}"
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def atomic_json_dump(data: Any, target: str | os.PathLike[str]) -> None:
    path = Path(target)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=str(path.parent), delete=False
    ) as handle:
        json.dump(data, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
        temporary = Path(handle.name)
    temporary.replace(path)


def require_paths(mapping: Dict[str, Path]) -> None:
    missing = [f"{name}: {path}" for name, path in mapping.items() if not path.exists()]
    if missing:
        raise FileNotFoundError("Required paths do not exist:\n" + "\n".join(missing))


def load_json(path: str | os.PathLike[str]) -> Any:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)
