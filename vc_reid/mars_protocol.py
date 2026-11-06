from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import re
from typing import Dict, Iterable, List

import numpy as np
from scipy.io import loadmat

from .common import atomic_json_dump


def read_names(path: Path) -> List[str]:
    with open(path, "r", encoding="utf-8") as handle:
        return [line.strip() for line in handle if line.strip()]


def official_tracklet_number(row: np.ndarray, names: List[str]) -> int:
    start = int(row[0])
    first_frame = names[start - 1]
    match = re.search(r"C\dT(\d+)F\d+", first_frame)
    if match is None:
        raise ValueError(f"Cannot parse tracklet number from {first_frame}")
    return int(match.group(1))


def positive_feature_path(
    feature_root: Path,
    pid: int,
    camera: int,
    tracklet_number: int,
) -> Path:
    return (
        feature_root
        / f"{pid:04d}"
        / f"cam{camera}_track{tracklet_number:04d}.npy"
    )


def distractor_feature_path(
    feature_root: Path,
    official_index: int,
    camera: int,
) -> Path:
    return (
        feature_root
        / "0000"
        / f"cam{camera}_official{official_index:05d}.npy"
    )


def make_entry(
    row: np.ndarray,
    official_index: int,
    names: List[str],
    feature_path: Path,
    subset: str,
    custom_tracklet: str | None,
) -> Dict[str, object]:
    start, end, pid, camera = (int(value) for value in row)
    frame_names = names[start - 1 : end]
    if not frame_names:
        raise ValueError(f"Empty official tracklet at row {official_index}")
    expected_pid = "00-1" if pid == -1 else f"{pid:04d}"
    actual_pids = {name[:4] for name in frame_names}
    actual_cameras = {int(name[5]) for name in frame_names}
    if actual_pids != {expected_pid}:
        raise ValueError(
            f"PID mismatch at row {official_index}: {actual_pids} != {expected_pid}"
        )
    if actual_cameras != {camera}:
        raise ValueError(
            f"Camera mismatch at row {official_index}: {actual_cameras} != {camera}"
        )
    return {
        "official_index": official_index,
        "subset": subset,
        "pid": expected_pid,
        "pid_numeric": pid,
        "camera": f"cam{camera}",
        "camera_numeric": camera,
        "start_frame_index": start,
        "end_frame_index": end,
        "frame_count": len(frame_names),
        "first_frame": frame_names[0],
        "last_frame": frame_names[-1],
        "custom_tracklet": custom_tracklet,
        "feature_path": str(feature_path),
        "feature_exists": feature_path.exists(),
    }


def write_csv(entries: List[Dict[str, object]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(entries[0]))
        writer.writeheader()
        writer.writerows(entries)


def build_protocol(
    info_root: Path,
    train_feature_root: Path,
    test_feature_root: Path,
    distractor_feature_root: Path,
    output_root: Path,
) -> Dict[str, object]:
    track_train = loadmat(info_root / "tracks_train_info.mat")[
        "track_train_info"
    ]
    track_test = loadmat(info_root / "tracks_test_info.mat")["track_test_info"]
    query_indices = (
        loadmat(info_root / "query_IDX.mat")["query_IDX"].squeeze().astype(int) - 1
    )
    train_names = read_names(info_root / "train_name.txt")
    test_names = read_names(info_root / "test_name.txt")
    if track_train.shape != (8298, 4):
        raise ValueError(f"Unexpected official train shape: {track_train.shape}")
    if track_test.shape != (12180, 4):
        raise ValueError(f"Unexpected official test shape: {track_test.shape}")
    if len(query_indices) != 1980 or len(set(query_indices.tolist())) != 1980:
        raise ValueError("Unexpected official query index set")

    train_entries: List[Dict[str, object]] = []
    for index, row in enumerate(track_train):
        pid, camera = int(row[2]), int(row[3])
        tracklet_number = official_tracklet_number(row, train_names)
        tracklet = f"cam{camera}_track{tracklet_number:04d}"
        path = positive_feature_path(
            train_feature_root, pid, camera, tracklet_number
        )
        train_entries.append(
            make_entry(row, index, train_names, path, "train", tracklet)
        )

    query_set = set(query_indices.tolist())
    query_entries: List[Dict[str, object]] = []
    gallery_entries: List[Dict[str, object]] = []
    junk_entries: List[Dict[str, object]] = []
    positive_entries: List[Dict[str, object]] = []
    for index, row in enumerate(track_test):
        pid, camera = int(row[2]), int(row[3])
        custom_tracklet = None
        if pid > 0:
            tracklet_number = official_tracklet_number(row, test_names)
            custom_tracklet = f"cam{camera}_track{tracklet_number:04d}"
            path = positive_feature_path(
                test_feature_root, pid, camera, tracklet_number
            )
        elif pid == 0:
            path = distractor_feature_path(
                distractor_feature_root, index, camera
            )
        else:
            path = distractor_feature_root / "junk" / f"official{index:05d}.npy"

        if index in query_set:
            if pid <= 0:
                raise ValueError(f"Official query {index} has non-positive PID {pid}")
            entry = make_entry(
                row, index, test_names, path, "query", custom_tracklet
            )
            query_entries.append(entry)
            positive_entries.append(entry)
        elif pid != -1:
            entry = make_entry(
                row, index, test_names, path, "gallery", custom_tracklet
            )
            gallery_entries.append(entry)
            if pid > 0:
                positive_entries.append(entry)
        else:
            junk_entries.append(
                make_entry(row, index, test_names, path, "junk", None)
            )

    output_root.mkdir(parents=True, exist_ok=True)
    manifests = {
        "train": train_entries,
        "query": query_entries,
        "gallery": gallery_entries,
        "junk": junk_entries,
        "positive_test": positive_entries,
    }
    for name, entries in manifests.items():
        atomic_json_dump(entries, output_root / f"{name}.json")
        write_csv(entries, output_root / f"{name}.csv")

    report = {
        "source": "Official MARS metadata from liangzheng06/MARS-evaluation",
        "protocol": {
            "query": "the 1,980 rows listed by query_IDX.mat",
            "gallery": (
                "the complement of the query rows in tracks_test_info.mat, "
                "excluding PID -1 junk and retaining PID 0 distractors"
            ),
            "junk_filter": "same PID and same camera at metric time",
        },
        "counts": {
            "official_train": len(train_entries),
            "official_query": len(query_entries),
            "official_gallery": len(gallery_entries),
            "official_gallery_positive": sum(
                int(row["pid_numeric"]) > 0 for row in gallery_entries
            ),
            "official_gallery_distractor_pid0": sum(
                int(row["pid_numeric"]) == 0 for row in gallery_entries
            ),
            "excluded_junk_pid_minus1": len(junk_entries),
            "positive_test_total": len(positive_entries),
        },
        "feature_coverage": {
            name: {
                "available": sum(bool(row["feature_exists"]) for row in entries),
                "missing": sum(not bool(row["feature_exists"]) for row in entries),
            }
            for name, entries in manifests.items()
        },
        "missing_train_examples": [
            row for row in train_entries if not row["feature_exists"]
        ][:20],
        "missing_gallery_examples": [
            row for row in gallery_entries if not row["feature_exists"]
        ][:20],
    }
    atomic_json_dump(report, output_root / "audit.json")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the official MARS protocol")
    parser.add_argument(
        "--info-root", default="reference/mars_official_info"
    )
    parser.add_argument(
        "--train-feature-root",
        default="artifacts/features/train",
    )
    parser.add_argument(
        "--test-feature-root",
        default="artifacts/features/test",
    )
    parser.add_argument(
        "--distractor-feature-root",
        default="artifacts/standard_mars_features/test",
    )
    parser.add_argument("--output-root", default="artifacts/protocol/official_mars")
    args = parser.parse_args()
    report = build_protocol(
        Path(args.info_root),
        Path(args.train_feature_root),
        Path(args.test_feature_root),
        Path(args.distractor_feature_root),
        Path(args.output_root),
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
