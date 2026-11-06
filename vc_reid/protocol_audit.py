from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List

import numpy as np

from .common import atomic_json_dump, load_paths, require_paths
from .feature_views import branch_norms, discover_feature_files


def audit_split(root: Path, expected_dim: int = 4608) -> Dict[str, object]:
    samples = discover_feature_files(root)
    dimensions: Counter[int] = Counter()
    cameras: Counter[str] = Counter()
    pids: Counter[str] = Counter()
    nonfinite = 0
    zero_branches = Counter()
    static_fingerprints: Dict[str, set[bytes]] = defaultdict(set)
    for path, pid, camera in samples:
        feature = np.load(path).astype(np.float32).reshape(-1)
        dimensions[int(feature.size)] += 1
        cameras[camera] += 1
        pids[pid] += 1
        if not np.isfinite(feature).all():
            nonfinite += 1
        if feature.size == expected_dim:
            for branch, norm in branch_norms(feature).items():
                if norm <= 1e-8:
                    zero_branches[branch] += 1
            static_fingerprints[pid].add(feature[4096:4608].tobytes())
    return {
        "root": str(root),
        "tracklets": len(samples),
        "identities": len(pids),
        "cameras": dict(sorted(cameras.items())),
        "dimensions": {str(key): value for key, value in sorted(dimensions.items())},
        "nonfinite_files": nonfinite,
        "zero_branch_files": dict(zero_branches),
        "identities_with_one_static_vector": sum(
            len(values) == 1 for values in static_fingerprints.values()
        ),
        "identity_tracklet_min": min(pids.values()) if pids else 0,
        "identity_tracklet_max": max(pids.values()) if pids else 0,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit the frozen Dialog-MARS protocol")
    parser.add_argument("--config", default="configs/paths.json")
    parser.add_argument("--output")
    args = parser.parse_args()
    paths = load_paths(args.config)
    require_paths(
        {
            "train_features": paths.train_features,
            "test_features": paths.test_features,
            "train_meta": paths.train_meta,
            "test_meta": paths.test_meta,
            "full_checkpoint": paths.full_checkpoint,
        }
    )
    report = {
        "protocol": {
            "query_gallery_source": "same test feature root",
            "junk_rule": "exclude same-pid same-camera tracklets",
            "warning": (
                "The current evaluator uses every test tracklet as both a query and a "
                "gallery candidate before applying the junk rule. Use this exact protocol "
                "for every reproduced baseline or replace it everywhere with a frozen "
                "query/gallery manifest."
            ),
        },
        "train": audit_split(paths.train_features),
        "test": audit_split(paths.test_features),
    }
    output = Path(args.output) if args.output else paths.artifacts_root / "protocol_audit.json"
    atomic_json_dump(report, output)
    print(f"Audit written to {output}")
    print(report)


if __name__ == "__main__":
    main()
