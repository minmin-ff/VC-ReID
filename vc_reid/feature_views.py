from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np


VIEW_DIMS: Dict[str, int] = {
    "global": 2048,
    "action": 2048,
    "static": 512,
    "global_static": 2560,
    "global_action": 4096,
    "full": 4608,
}


def select_view(feature: np.ndarray, view: str) -> np.ndarray:
    flat = np.asarray(feature, dtype=np.float32).reshape(-1)
    if flat.size != 4608:
        raise ValueError(f"Expected 4608-D source feature, got {flat.size}")
    if view == "global":
        output = flat[:2048]
    elif view == "action":
        output = flat[2048:4096]
    elif view == "static":
        output = flat[4096:4608]
    elif view == "global_static":
        output = np.concatenate((flat[:2048], flat[4096:4608]))
    elif view == "global_action":
        output = flat[:4096]
    elif view == "full":
        output = flat
    else:
        raise ValueError(f"Unknown feature view: {view}")
    if output.size != VIEW_DIMS[view]:
        raise AssertionError((view, output.size, VIEW_DIMS[view]))
    return output.astype(np.float32, copy=False)


def discover_feature_files(root: str | Path) -> List[Tuple[Path, str, str]]:
    samples: List[Tuple[Path, str, str]] = []
    for path in sorted(Path(root).glob("*/*.npy")):
        pid = path.parent.name
        tracklet = path.stem
        camera = tracklet.split("_")[0]
        samples.append((path, pid, camera))
    return samples


def branch_norms(feature: np.ndarray) -> Dict[str, float]:
    flat = np.asarray(feature, dtype=np.float32).reshape(-1)
    return {
        "global": float(np.linalg.norm(flat[:2048])),
        "action": float(np.linalg.norm(flat[2048:4096])),
        "static": float(np.linalg.norm(flat[4096:4608])),
    }
