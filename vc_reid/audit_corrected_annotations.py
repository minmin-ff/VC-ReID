from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List

from .common import atomic_json_dump


PROMPT_VERSION = "tracklet-independent-v1"
ACTIONS = (
    "walking", "running", "waving hand", "squatting", "turning around",
    "falling down", "talking", "looking back", "raising hand", "jumping",
    "riding bike", "pacing back and forth", "standing", "using phone",
    "smoking", "carrying object",
)


REQUIRED_FIELDS = (
    "description",
    "gender_presentation",
    "top_color",
    "top_type",
    "sleeve_length",
    "bottom_color",
    "bottom_type",
    "shoe_color",
    "shoe_type",
    "hair_color",
    "hair_length",
    "backpack",
    "hat",
    "glasses",
    "mask",
    "carrying_item",
    "actions",
)


def load_rows(path: Path) -> List[Dict[str, object]]:
    with open(path, "r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, list):
        raise ValueError(path)
    return value


def audit_manifest(
    manifest_path: Path, annotation_root: Path
) -> Dict[str, object]:
    rows = load_rows(manifest_path)
    errors = []
    descriptors_by_pid = defaultdict(list)
    actions = Counter()
    unknown_fields = Counter()
    for row in rows:
        index = int(row["official_index"])
        subset = str(row["subset"])
        path = annotation_root / subset / f"official_{index:05d}.json"
        if not path.exists():
            errors.append({"official_index": index, "error": "missing_annotation"})
            continue
        try:
            with open(path, "r", encoding="utf-8") as handle:
                value = json.load(handle)
            if value.get("prompt_version") != PROMPT_VERSION:
                raise ValueError("prompt_version mismatch")
            for key in ("official_index", "pid", "camera", "first_frame", "last_frame"):
                expected = row[key]
                if value.get(key) != expected:
                    raise ValueError(f"{key} mismatch: {value.get(key)} != {expected}")
            sampled = value.get("sampled_frames", [])
            if len(sampled) != 8 or sampled[0] != row["first_frame"] or sampled[-1] != row["last_frame"]:
                raise ValueError("sampled-frame provenance mismatch")
            annotation = value.get("annotation")
            if not isinstance(annotation, dict):
                raise ValueError("annotation is not an object")
            missing_fields = [field for field in REQUIRED_FIELDS if field not in annotation]
            if missing_fields:
                raise ValueError(f"missing fields: {missing_fields}")
            description = str(annotation["description"]).strip()
            if not description:
                raise ValueError("empty description")
            descriptors_by_pid[str(row["pid"])].append(description)
            for field in REQUIRED_FIELDS[:-1]:
                if str(annotation[field]).strip().lower() == "unknown":
                    unknown_fields[field] += 1
            if not isinstance(annotation["actions"], list):
                raise ValueError("actions is not a list")
            for action in annotation["actions"]:
                if action.get("name") not in ACTIONS:
                    raise ValueError(f"invalid action: {action}")
                start = float(action["start_fraction"])
                end = float(action["end_fraction"])
                if not 0 <= start <= end <= 1:
                    raise ValueError(f"invalid action boundary: {action}")
                actions[action["name"]] += 1
        except Exception as error:
            errors.append({"official_index": index, "error": repr(error)})
    multi_tracklet = {
        pid: values for pid, values in descriptors_by_pid.items() if len(values) > 1
    }
    one_description = sum(len(set(values)) == 1 for values in multi_tracklet.values())
    return {
        "manifest": str(manifest_path),
        "expected": len(rows),
        "valid": len(rows) - len(errors),
        "errors": errors[:100],
        "error_count": len(errors),
        "action_counts": dict(actions),
        "unknown_field_counts": dict(unknown_fields),
        "multi_tracklet_identities": len(multi_tracklet),
        "multi_tracklet_identities_with_one_exact_description": one_description,
        "shared_description_ratio": one_description / max(1, len(multi_tracklet)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit corrected per-tracklet annotations")
    parser.add_argument("--train-manifest", required=True)
    parser.add_argument("--query-manifest", required=True)
    parser.add_argument("--gallery-manifest", required=True)
    parser.add_argument("--annotation-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--strict", action="store_true")
    args = parser.parse_args()
    annotation_root = Path(args.annotation_root)
    reports = {
        name: audit_manifest(Path(path), annotation_root)
        for name, path in (
            ("train", args.train_manifest),
            ("query", args.query_manifest),
            ("gallery", args.gallery_manifest),
        )
    }
    report = {
        "prompt_version": PROMPT_VERSION,
        "splits": reports,
        "passed": all(item["error_count"] == 0 for item in reports.values()),
    }
    atomic_json_dump(report, args.output)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if args.strict and not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
