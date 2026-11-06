from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm

from .common import atomic_json_dump, set_seed
from .extract_sampler_features import sample_indices
from .mars_protocol import read_names


def load_manifest(path: Path) -> List[Dict[str, object]]:
    with open(path, "r", encoding="utf-8") as handle:
        rows = json.load(handle)
    if not isinstance(rows, list):
        raise ValueError(f"Manifest must be a list: {path}")
    return rows


def frame_names(row: Dict[str, object], names: Sequence[str]) -> List[str]:
    start = int(row["start_frame_index"]) - 1
    end = int(row["end_frame_index"])
    selected = list(names[start:end])
    if len(selected) != int(row["frame_count"]):
        raise ValueError(f"Frame-count mismatch for official row {row['official_index']}")
    return selected


def load_images(
    names: Sequence[str], indices: Sequence[int], bbox_root: Path
) -> List[Image.Image]:
    output = []
    for index in indices:
        name = names[int(index)]
        path = bbox_root / name[:4] / name
        if not path.exists():
            raise FileNotFoundError(path)
        output.append(Image.open(path).convert("RGB"))
    return output


class ResNetEncoder:
    def __init__(
        self,
        device: torch.device,
        checkpoint_path: Path | None = None,
        input_height: int = 256,
        input_width: int = 128,
    ):
        import torchvision.models as models

        if checkpoint_path is None:
            model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        else:
            from .train_reid_resnet import set_last_stride_one

            checkpoint = torch.load(checkpoint_path, map_location="cpu")
            model = models.resnet50(weights=None)
            if int(checkpoint.get("last_stride", 1)) == 1:
                set_last_stride_one(model)
            model.fc = nn.Identity()
            model.load_state_dict(checkpoint["backbone_state_dict"], strict=True)
            input_height = int(checkpoint.get("input_height", input_height))
            input_width = int(checkpoint.get("input_width", input_width))
        model.fc = nn.Identity()
        self.model = model.to(device).eval()
        self.device = device
        self.input_height = input_height
        self.input_width = input_width
        self.mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

    @torch.inference_mode()
    def frames(self, images: Sequence[Image.Image], batch_size: int = 64) -> np.ndarray:
        tensors = []
        for image in images:
            array = np.asarray(
                image.resize(
                    (self.input_width, self.input_height),
                    resample=Image.Resampling.BILINEAR,
                )
            )
            tensor = torch.from_numpy(array.copy()).float().permute(2, 0, 1) / 255.0
            tensors.append((tensor - self.mean) / self.std)
        outputs = []
        for offset in range(0, len(tensors), batch_size):
            batch = torch.stack(tensors[offset : offset + batch_size]).to(self.device)
            outputs.append(self.model(batch).flatten(1).cpu())
        return torch.cat(outputs).numpy().astype(np.float32)


def descriptor(annotation: Dict[str, object]) -> str:
    value = annotation["annotation"]
    if not isinstance(value, dict):
        raise ValueError("annotation field must be an object")
    parts = [str(value.get("description", "")).strip()]
    mappings = (
        ("gender_presentation", "Gender presentation"),
        ("top_color", "Top color"),
        ("top_type", "Top type"),
        ("sleeve_length", "Sleeve length"),
        ("bottom_color", "Bottom color"),
        ("bottom_type", "Bottom type"),
        ("shoe_color", "Shoe color"),
        ("shoe_type", "Shoe type"),
        ("hair_color", "Hair color"),
        ("hair_length", "Hair length"),
        ("backpack", "Backpack"),
        ("hat", "Hat"),
        ("glasses", "Glasses"),
        ("mask", "Mask"),
        ("carrying_item", "Carrying item"),
    )
    for key, label in mappings:
        item = str(value.get(key, "unknown")).strip().lower()
        if item and item != "unknown":
            parts.append(f"{label}: {item}.")
    return " ".join(part for part in parts if part)


@torch.inference_mode()
def encode_text(clip, tokenizer, text: str, device: torch.device) -> np.ndarray:
    tokens = tokenizer(
        [text], padding=True, truncation=True, max_length=77, return_tensors="pt"
    ).to(device)
    feature = clip.get_text_features(**tokens)
    return F.normalize(feature.float(), p=2, dim=1).squeeze(0).cpu().numpy()


def output_path(row: Dict[str, object], root: Path) -> Path:
    pid = str(row["pid"])
    if int(row["pid_numeric"]) > 0:
        name = str(row["custom_tracklet"])
    else:
        name = f"cam{int(row['camera_numeric'])}_official{int(row['official_index']):05d}"
    return root / pid / f"{name}.npy"


def extract_one(
    row: Dict[str, object],
    names: Sequence[str],
    annotation: Dict[str, object],
    bbox_root: Path,
    resnet: ResNetEncoder,
    clip,
    tokenizer,
    device: torch.device,
) -> tuple[np.ndarray, Dict[str, object]]:
    tracklet_names = frame_names(row, names)
    length = len(tracklet_names)
    global_indices = np.linspace(0, length - 1, 32).round().astype(int).tolist()
    action_specs = annotation["annotation"].get("actions", [])
    action_indices: List[List[int]] = []
    for action_index, action in enumerate(action_specs):
        start = int(round(float(action["start_fraction"]) * (length - 1)))
        end = int(round(float(action["end_fraction"]) * (length - 1)))
        indices, base = sample_indices(
            start,
            end,
            "adaptive_bidirectional",
            f"official:{row['subset']}:{row['official_index']}:{action_index}",
            3407,
        )
        action_indices.append(indices)
        action["sampled_indices"] = indices
        action["adaptive_base"] = base
    flat_indices = global_indices + [index for group in action_indices for index in group]
    images = load_images(tracklet_names, flat_indices, bbox_root)
    encoded = resnet.frames(images)
    global_feature = encoded[:32].mean(axis=0)
    if action_indices:
        offset = 32
        action_features = []
        for indices in action_indices:
            action_features.append(encoded[offset : offset + len(indices)].mean(axis=0))
            offset += len(indices)
        action_feature = np.stack(action_features).mean(axis=0)
    else:
        action_feature = np.zeros(2048, dtype=np.float32)
    text = descriptor(annotation)
    static_feature = encode_text(clip, tokenizer, text, device)
    feature = np.concatenate((global_feature, action_feature, static_feature)).astype(
        np.float32
    )
    if feature.shape != (4608,) or not np.isfinite(feature).all():
        raise ValueError(f"Invalid feature for official row {row['official_index']}")
    return feature, {
        "global_indices": global_indices,
        "actions": action_specs,
        "descriptor": text,
        "branch_norms": {
            "global": float(np.linalg.norm(global_feature)),
            "action": float(np.linalg.norm(action_feature)),
            "static": float(np.linalg.norm(static_feature)),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract leakage-free 4608-D features for an official MARS manifest"
    )
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--names-file", required=True)
    parser.add_argument("--bbox-root", required=True)
    parser.add_argument("--annotation-root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--clip-path", required=True)
    parser.add_argument("--resnet-checkpoint")
    parser.add_argument("--input-height", type=int, default=256)
    parser.add_argument("--input-width", type=int, default=128)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int)
    parser.add_argument("--max-tracklets", type=int)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for feature extraction")
    torch.cuda.set_device(args.device)
    set_seed(args.seed)
    device = torch.device(f"cuda:{args.device}")
    from transformers import CLIPModel, CLIPTokenizer

    resnet = ResNetEncoder(
        device,
        Path(args.resnet_checkpoint) if args.resnet_checkpoint else None,
        args.input_height,
        args.input_width,
    )
    clip = CLIPModel.from_pretrained(args.clip_path, local_files_only=True).to(device).eval()
    tokenizer = CLIPTokenizer.from_pretrained(args.clip_path, local_files_only=True)
    rows = load_manifest(Path(args.manifest))
    end = len(rows) if args.end is None else min(args.end, len(rows))
    rows = rows[args.start:end]
    if args.max_tracklets is not None:
        rows = rows[: args.max_tracklets]
    names = read_names(Path(args.names_file))
    bbox_root = Path(args.bbox_root)
    annotation_root = Path(args.annotation_root)
    destination_root = Path(args.output_root)
    records = []
    for row in tqdm(rows, desc="official features"):
        destination = output_path(row, destination_root)
        if destination.exists() and not args.force:
            records.append({"official_index": row["official_index"], "status": "existing"})
            continue
        annotation_path = (
            annotation_root
            / str(row["subset"])
            / f"official_{int(row['official_index']):05d}.json"
        )
        try:
            with open(annotation_path, "r", encoding="utf-8") as handle:
                annotation = json.load(handle)
            feature, detail = extract_one(
                row,
                names,
                annotation,
                bbox_root,
                resnet,
                clip,
                tokenizer,
                device,
            )
            destination.parent.mkdir(parents=True, exist_ok=True)
            np.save(destination, feature)
            records.append(
                {
                    "official_index": row["official_index"],
                    "status": "ok",
                    "output": str(destination),
                    **detail,
                }
            )
        except Exception as error:
            records.append(
                {"official_index": row["official_index"], "status": "error", "error": repr(error)}
            )
    meta_path = (
        destination_root
        / "metadata"
        / f"{Path(args.manifest).stem}_{args.start}_{end}.json"
    )
    atomic_json_dump(
        {
            "manifest": args.manifest,
            "sampler": "adaptive_bidirectional",
            "adaptive_base": "b=(L-1)^(1/7), no upper cap",
            "resnet_checkpoint": args.resnet_checkpoint,
            "input_height": resnet.input_height,
            "input_width": resnet.input_width,
            "records": records,
        },
        meta_path,
    )
    counts = {status: sum(row["status"] == status for row in records) for status in ("ok", "existing", "error")}
    print(json.dumps(counts, indent=2))


if __name__ == "__main__":
    main()
