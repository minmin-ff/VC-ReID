from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def validate(root: Path, expected: int) -> None:
    files = sorted(root.glob("*/*.npy"))
    if len(files) != expected:
        raise RuntimeError(f"{root}: expected {expected} files, found {len(files)}")
    bad_shape = []
    nonfinite = []
    for path in files:
        feature = np.load(path, mmap_mode="r")
        if feature.size != 4608:
            bad_shape.append((str(path), tuple(feature.shape)))
        if not np.isfinite(feature).all():
            nonfinite.append(str(path))
    if bad_shape or nonfinite:
        raise RuntimeError(
            f"{root}: bad_shape={bad_shape[:5]} nonfinite={nonfinite[:5]}"
        )
    print(f"validated {root}: {len(files)} finite 4608-D features")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-root", required=True)
    parser.add_argument("--test-root", required=True)
    parser.add_argument("--expected-train", type=int, default=8297)
    parser.add_argument("--expected-test", type=int, default=8062)
    args = parser.parse_args()
    validate(Path(args.train_root), args.expected_train)
    validate(Path(args.test_root), args.expected_test)


if __name__ == "__main__":
    main()
