"""
retriever.py
检索器：将4608维特征投影到低维空间，支持triplet + id + center loss训练
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
import glob
from typing import Dict, List, Tuple, Optional


# ─────────────────────────────────────────────
# 投影网络
# ─────────────────────────────────────────────
class ProjectionHead(nn.Module):
    """
    将4608维拼接特征投影到512维检索空间
    结构：4608 → 2048 → 1024 → 512，带BN+ReLU+Dropout
    """
    def __init__(self, input_dim: int = 4608, proj_dim: int = 512, num_classes: int = 625):
        super().__init__()
        self.proj_dim = proj_dim

        self.backbone = nn.Sequential(
            nn.Linear(input_dim, 2048),
            nn.BatchNorm1d(2048),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),

            nn.Linear(2048, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),

            nn.Linear(1024, proj_dim),
            nn.BatchNorm1d(proj_dim),
        )

        # ID分类头（用于ID loss）
        self.classifier = nn.Linear(proj_dim, num_classes)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            feat: L2归一化后的投影特征 (B, proj_dim)
            logits: 分类logits (B, num_classes)
        """
        feat = self.backbone(x)
        feat_norm = F.normalize(feat, p=2, dim=1)
        logits = self.classifier(feat)
        return feat_norm, logits


# ─────────────────────────────────────────────
# Losses
# ─────────────────────────────────────────────
class TripletLoss(nn.Module):
    """Batch hard triplet loss"""
    def __init__(self, margin: float = 0.3):
        super().__init__()
        self.margin = margin

    def forward(self, embeddings: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        dist_mat = torch.cdist(embeddings, embeddings, p=2)
        B = embeddings.size(0)
        labels = labels.view(-1, 1)
        mask_pos = (labels == labels.T).float()
        mask_neg = (labels != labels.T).float()
        eye = torch.eye(B, device=embeddings.device)
        mask_pos = mask_pos * (1 - eye)

        ap_dist = (dist_mat * mask_pos)
        ap_dist[mask_pos == 0] = 0
        hardest_pos, _ = ap_dist.max(dim=1)

        an_dist = dist_mat.clone()
        an_dist[mask_neg == 0] = float('inf')
        hardest_neg, _ = an_dist.min(dim=1)

        loss = F.relu(hardest_pos - hardest_neg + self.margin)
        return loss.mean()


class CircleLoss(nn.Module):
    """
    Circle Loss（数值稳定版本）
    参考：Circle Loss: A Unified Perspective of Pair Similarity Optimization (CVPR 2020)
    """
    def __init__(self, m: float = 0.25, gamma: float = 32.0):
        super().__init__()
        self.m       = m
        self.gamma   = gamma
        self.delta_p = 1 - m
        self.delta_n = m

    def forward(self, embeddings: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        embeddings = F.normalize(embeddings, p=2, dim=1)
        sim_mat    = torch.mm(embeddings, embeddings.T).clamp(-1, 1)  # (B, B)

        B   = labels.size(0)
        lab = labels.view(-1, 1)
        mask_pos = (lab == lab.T).float()
        mask_neg = (lab != lab.T).float()
        eye      = torch.eye(B, device=embeddings.device)
        mask_pos = mask_pos * (1 - eye)

        ap = torch.clamp_min(-sim_mat.detach() + 1 + self.m, min=0.0)
        an = torch.clamp_min( sim_mat.detach() + self.m,     min=0.0)

        logit_p = -ap * (sim_mat - self.delta_p) * self.gamma
        logit_n =  an * (sim_mat - self.delta_n) * self.gamma

        # 用softplus代替log(1+exp(a)*exp(b))，数值更稳定
        # log(1+exp(a+b)) = softplus(a+b)
        loss  = torch.tensor(0.0, device=embeddings.device)
        valid = 0
        for i in range(B):
            pos_mask = mask_pos[i].bool()
            neg_mask = mask_neg[i].bool()
            if pos_mask.sum() == 0:
                continue
            lp = logit_p[i][pos_mask]
            ln = logit_n[i][neg_mask]
            # 数值稳定的logsumexp
            lse_p = torch.logsumexp(lp, dim=0)
            lse_n = torch.logsumexp(ln, dim=0)
            # softplus(lse_n + lse_p) = log(1 + exp(lse_n + lse_p))
            loss  = loss + F.softplus(lse_n + lse_p)
            valid += 1

        return loss / max(valid, 1)


class CenterLoss(nn.Module):
    """Center loss：最小化类内特征到类中心的距离"""
    def __init__(self, num_classes: int, feat_dim: int, device: torch.device):
        super().__init__()
        self.centers = nn.Parameter(
            torch.randn(num_classes, feat_dim).to(device)
        )

    def forward(self, features: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        centers_batch = self.centers[labels]
        loss = F.mse_loss(features, centers_batch)
        return loss


# ─────────────────────────────────────────────
# Dataset
# ─────────────────────────────────────────────
class ReIDFeatureDataset(Dataset):
    """
    从extracted_features/train目录读取.npy特征文件
    目录结构: train/{pid}/{camX_trackY.npy}
    """
    def __init__(self, feature_root: str):
        self.samples = []   # (feature_path, pid_label)
        self.pid2label = {}

        pid_dirs = sorted(glob.glob(os.path.join(feature_root, '*')))
        for label_idx, pid_dir in enumerate(pid_dirs):
            pid = os.path.basename(pid_dir)
            self.pid2label[pid] = label_idx
            npy_files = glob.glob(os.path.join(pid_dir, '*.npy'))
            for npy_file in npy_files:
                self.samples.append((npy_file, label_idx))

        print(f"[Dataset] {len(self.samples)} tracklets, {len(self.pid2label)} identities")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        feat_path, label = self.samples[idx]
        feat = np.load(feat_path).astype(np.float32)
        # 确保是1D的4608维向量
        if feat.ndim > 1:
            feat = feat.flatten()
        return torch.tensor(feat), torch.tensor(label, dtype=torch.long)

    @property
    def num_classes(self):
        return len(self.pid2label)


# ─────────────────────────────────────────────
# 训练函数
# ─────────────────────────────────────────────
def train_retriever(
    feature_root: str,
    save_path: str,
    input_dim: int = 4608,
    proj_dim: int = 512,
    epochs: int = 60,
    batch_size: int = 64,
    lr: float = 3e-4,
    lambda_id: float = 1.0,
    lambda_center: float = 0.0005,
    lambda_triplet: float = 1.0,
    lambda_ap: float = 0.5,       # AP loss权重
    device_id: int = 0,
):
    device = torch.device(f'cuda:{device_id}' if torch.cuda.is_available() else 'cpu')
    print(f"[Retriever] Training on {device}")

    dataset = ReIDFeatureDataset(feature_root)
    num_classes = dataset.num_classes
    # PK采样：每个batch包含P个pid，每个pid取K个tracklet
    # 保证每个batch里有足够的正样本对，让circle loss有效
    from torch.utils.data import WeightedRandomSampler
    # 简单实现：增大batch_size，保证同pid样本概率
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True,
                        num_workers=4, drop_last=True)

    model = ProjectionHead(input_dim, proj_dim, num_classes).to(device)
    center_loss_fn  = CenterLoss(num_classes, proj_dim, device)
    triplet_loss_fn = TripletLoss(margin=0.3)
    circle_loss_fn  = CircleLoss(m=0.25, gamma=80.0)
    id_loss_fn      = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(
        list(model.parameters()) + list(center_loss_fn.parameters()),
        lr=lr, weight_decay=5e-4
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    os.makedirs(save_path, exist_ok=True)
    best_loss = float('inf')

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0

        for feats, labels in loader:
            feats = feats.to(device)
            labels = labels.to(device)

            feat_norm, logits = model(feats)

            l_triplet = triplet_loss_fn(feat_norm, labels)
            l_id      = id_loss_fn(logits, labels)
            l_center  = center_loss_fn(feat_norm, labels)
            l_circle  = circle_loss_fn(feat_norm, labels)

            loss = (lambda_triplet * l_triplet +
                    lambda_id      * l_id      +
                    lambda_center  * l_center  +
                    lambda_ap      * l_circle)

            optimizer.zero_grad()
            loss.backward()
            # center loss的梯度需要缩放
            for param in center_loss_fn.parameters():
                param.grad.data *= (1.0 / lambda_center) if lambda_center > 0 else 1.0
            optimizer.step()

            total_loss += loss.item()

        scheduler.step()
        avg_loss = total_loss / len(loader)

        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch+1}/{epochs}] Loss: {avg_loss:.4f}")

        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'proj_dim': proj_dim,
                'input_dim': input_dim,
                'num_classes': num_classes,
            }, os.path.join(save_path, 'retriever_best.pth'))

    print(f"[Retriever] Training done. Best loss: {best_loss:.4f}")
    return model


# ─────────────────────────────────────────────
# 推理工具
# ─────────────────────────────────────────────
def load_retriever(ckpt_path: str, device: torch.device) -> ProjectionHead:
    ckpt = torch.load(ckpt_path, map_location=device)
    model = ProjectionHead(
        input_dim=ckpt['input_dim'],
        proj_dim=ckpt['proj_dim'],
        num_classes=ckpt['num_classes'],
    ).to(device)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    print(f"[Retriever] Loaded from {ckpt_path}")
    return model


def build_gallery(
    feature_root: str,
    model: ProjectionHead,
    device: torch.device,
) -> Tuple[torch.Tensor, List[str], List[str], List[str]]:
    """
    从train特征目录构建gallery
    Returns:
        gallery_feats: (N, proj_dim) 归一化后的gallery特征
        gallery_pids: 每条tracklet的人物ID
        gallery_cams: 每条tracklet的摄像头ID
        gallery_paths: 每条tracklet的.npy文件路径（用于后续视频帧定位）
    """
    gallery_feats = []
    gallery_pids = []
    gallery_cams = []
    gallery_paths = []

    pid_dirs = sorted(glob.glob(os.path.join(feature_root, '*')))
    model.eval()

    with torch.no_grad():
        for pid_dir in pid_dirs:
            pid = os.path.basename(pid_dir)
            npy_files = glob.glob(os.path.join(pid_dir, '*.npy'))
            for npy_file in npy_files:
                feat = np.load(npy_file).astype(np.float32)
                if feat.ndim > 1:
                    feat = feat.flatten()
                feat_tensor = torch.tensor(feat).unsqueeze(0).to(device)
                feat_proj, _ = model(feat_tensor)
                gallery_feats.append(feat_proj.squeeze(0).cpu())

                # 从文件名解析cam信息，如cam2_track0003.npy
                fname = os.path.basename(npy_file)
                cam = fname.split('_')[0]  # e.g. cam2
                gallery_pids.append(pid)
                gallery_cams.append(cam)
                gallery_paths.append(npy_file)

    gallery_feats = torch.stack(gallery_feats, dim=0)  # (N, proj_dim)
    print(f"[Gallery] Built {len(gallery_pids)} tracklets from {len(pid_dirs)} identities")
    return gallery_feats, gallery_pids, gallery_cams, gallery_paths


def retrieve_topk(
    query_feat: torch.Tensor,
    gallery_feats: torch.Tensor,
    gallery_pids: List[str],
    gallery_cams: List[str],
    gallery_paths: List[str],
    query_pid: str,
    query_cam: str,
    topk: int = 10,
) -> Tuple[List[int], List[float], List[str], List[str]]:
    """
    余弦相似度检索，排除query自身（同pid同cam）
    Returns: top_indices, top_scores, top_pids, top_paths
    """
    q = F.normalize(query_feat.unsqueeze(0), p=2, dim=1)
    g = F.normalize(gallery_feats, p=2, dim=1)
    sims = (q @ g.T).squeeze(0)  # (N,)

    # 屏蔽同pid同cam的tracklet
    for i, (pid, cam) in enumerate(zip(gallery_pids, gallery_cams)):
        if pid == query_pid and cam == query_cam:
            sims[i] = -1.0

    top_indices = sims.argsort(descending=True)[:topk].tolist()
    top_scores = sims[top_indices].tolist()
    top_pids = [gallery_pids[i] for i in top_indices]
    top_paths = [gallery_paths[i] for i in top_indices]

    return top_indices, top_scores, top_pids, top_paths