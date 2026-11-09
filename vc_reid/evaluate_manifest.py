from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
import torch.nn.functional as F

from .common import add_parent_to_path, atomic_json_dump, load_paths
from .feature_views import VIEW_DIMS, select_view
from .retrieval_metrics import evaluate_with_details


def load_manifest(path: str | Path) -> List[Dict[str, object]]:
    with open(path, "r", encoding="utf-8") as handle:
        rows = json.load(handle)
    if not isinstance(rows, list):
        raise ValueError(f"Manifest is not a list: {path}")
    return rows


def corrected_feature_path(row: Dict[str, object], root: Path) -> Path:
    """Resolve an official protocol row under an alternative feature root."""
    pid = str(row["pid"])
    if int(row["pid_numeric"]) > 0:
        name = str(row["custom_tracklet"])
    else:
        name = f"cam{int(row['camera_numeric'])}_official{int(row['official_index']):05d}"
    return root / pid / f"{name}.npy"


def override_feature_root(
    rows: List[Dict[str, object]], feature_root: str | None
) -> List[Dict[str, object]]:
    if feature_root is None:
        return rows
    root = Path(feature_root)
    output = []
    for row in rows:
        updated = dict(row)
        updated["feature_path"] = str(corrected_feature_path(row, root))
        output.append(updated)
    return output


def select_available(
    rows: List[Dict[str, object]], allow_missing: bool, label: str
) -> List[Dict[str, object]]:
    available = [row for row in rows if Path(str(row["feature_path"])).exists()]
    missing = len(rows) - len(available)
    if missing and not allow_missing:
        raise FileNotFoundError(
            f"{label} manifest has {missing} missing features; refusing a partial protocol"
        )
    if not available:
        raise ValueError(f"No available {label} features")
    return available


def embed_raw(
    rows: List[Dict[str, object]], view: str
) -> torch.Tensor:
    matrix = np.stack(
        [select_view(np.load(str(row["feature_path"])), view) for row in rows]
    )
    return F.normalize(torch.from_numpy(matrix).float(), p=2, dim=1)


def embed_projection(
    rows: List[Dict[str, object]],
    view: str,
    checkpoint_path: Path,
    device: torch.device,
    batch_size: int,
) -> torch.Tensor:
    paths = load_paths("configs/paths.json")
    add_parent_to_path(paths)
    from .retriever import ProjectionHead

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    expected_dim = VIEW_DIMS[view]
    if int(checkpoint["input_dim"]) != expected_dim:
        raise ValueError(
            f"Checkpoint input_dim={checkpoint['input_dim']}, expected {expected_dim}"
        )
    model = ProjectionHead(
        expected_dim,
        int(checkpoint.get("proj_dim", 512)),
        int(checkpoint["num_classes"]),
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    output = []
    with torch.no_grad():
        for offset in range(0, len(rows), batch_size):
            chunk = rows[offset : offset + batch_size]
            matrix = np.stack(
                [
                    select_view(np.load(str(row["feature_path"])), view)
                    for row in chunk
                ]
            )
            features, _ = model(torch.from_numpy(matrix).to(device))
            output.append(F.normalize(features, p=2, dim=1).cpu())
    return torch.cat(output, dim=0)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate frozen query/gallery manifests")
    parser.add_argument("--query-manifest", required=True)
    parser.add_argument("--gallery-manifest", required=True)
    parser.add_argument("--view", choices=sorted(VIEW_DIMS), required=True)
    parser.add_argument("--embedding", choices=("raw", "projection"), default="projection")
    parser.add_argument("--checkpoint")
    parser.add_argument(
        "--feature-root",
        help="Optional official feature root overriding paths stored in the manifests",
    )
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--allow-missing-gallery", action="store_true")
    parser.add_argument("--max-queries", type=int)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    query_all = override_feature_root(
        load_manifest(args.query_manifest), args.feature_root
    )
    gallery_all = override_feature_root(
        load_manifest(args.gallery_manifest), args.feature_root
    )
    query_rows = select_available(query_all, False, "query")
    gallery_rows = select_available(
        gallery_all, args.allow_missing_gallery, "gallery"
    )
    if args.max_queries:
        query_rows = query_rows[: args.max_queries]
    device = torch.device(
        f"cuda:{args.device}" if torch.cuda.is_available() else "cpu"
    )
    if args.embedding == "projection":
        if not args.checkpoint:
            parser.error("--checkpoint is required for projection embeddings")
        query_features = embed_projection(
            query_rows, args.view, Path(args.checkpoint), device, args.batch_size
        )
        gallery_features = embed_projection(
            gallery_rows, args.view, Path(args.checkpoint), device, args.batch_size
        )
    else:
        query_features = embed_raw(query_rows, args.view)
        gallery_features = embed_raw(gallery_rows, args.view)

    query_pids = [str(row["pid"]) for row in query_rows]
    query_cameras = [str(row["camera"]) for row in query_rows]
    gallery_pids = [str(row["pid"]) for row in gallery_rows]
    gallery_cameras = [str(row["camera"]) for row in gallery_rows]
    query_ids = [f"official:{row['official_index']}" for row in query_rows]
    metrics, details = evaluate_with_details(
        query_features,
        query_pids,
        query_cameras,
        gallery_features,
        gallery_pids,
        gallery_cameras,
        query_ids,
    )
    result = {
        "experiment": "manifest_retrieval",
        "protocol": {
            "query_manifest": str(Path(args.query_manifest).resolve()),
            "gallery_manifest": str(Path(args.gallery_manifest).resolve()),
            "query_manifest_count": len(query_all),
            "query_evaluated_count": len(query_rows),
            "gallery_manifest_count": len(gallery_all),
            "gallery_evaluated_count": len(gallery_rows),
            "partial_gallery": len(gallery_rows) != len(gallery_all),
            "feature_root_override": args.feature_root,
        },
        "embedding": args.embedding,
        "view": args.view,
        "checkpoint": args.checkpoint,
        "metrics": metrics,
        "per_query": details,
    }
    atomic_json_dump(result, args.output)
    print(json.dumps({**result["protocol"], **metrics}, indent=2))


if __name__ == "__main__":
    main()
