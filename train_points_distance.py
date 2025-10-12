# train_points_distance.py
# GATMA — Point–Point distance regression with E(3) tests
# -------------------------------------------------------
# Requires: artificial_data.py, constants.py, gpu_transformer.py, gpu_building_blocks.py, batch_operations.py, multivector.py
# Device: CUDA if available, else CPU

import math
import os
import random
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, random_split

import constants
from artificial_data import create_data
from gpu_transformer import MVTransformer
from gpu_building_blocks import MVLinear

# ---------------------------
# Utility: reproducibility
# ---------------------------
def set_seed(seed: int = 42):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ============================================================
# (1) Encode Euclidean points (x,y,z) as PGA multivectors
#     Point embedding (PGA G3,0,1), consistent with GATr Tbl. 1:
#     P = e123 + x*e0e2e3 + y*e0e1e3 + z*e0e1e2
#     Indices from constants.components:
#       e0e1e2 -> 11, e0e1e3 -> 12, e1e2e3 -> 13, e0e2e3 -> 14
# ============================================================
IDX_E012  = constants.components.index('e0e1e2')   # 11  (z)
IDX_E013  = constants.components.index('e0e1e3')   # 12  (y)
IDX_E123  = constants.components.index('e1e2e3')   # 13  (1)
IDX_E023  = constants.components.index('e0e2e3')   # 14  (x)

def encode_point_batch(xyz: torch.Tensor) -> torch.Tensor:
    """
    xyz: [B, 3]  -> returns multivectors [B, 16]
    """
    B = xyz.shape[0]
    mv = torch.zeros(B, len(constants.components), dtype=xyz.dtype, device=xyz.device)
    mv[:, IDX_E123] = 1.0
    mv[:, IDX_E023] = xyz[:, 0]  # x
    mv[:, IDX_E013] = xyz[:, 1]  # y
    mv[:, IDX_E012] = xyz[:, 2]  # z
    return mv

def encode_pair_rows(rows: torch.Tensor) -> torch.Tensor:
    """
    rows: [B, 7] = [x1,y1,z1, x2,y2,z2, dist]
    returns tokens: [B, 2, 16]  (two point tokens per sample)
    """
    p1 = encode_point_batch(rows[:, 0:3])
    p2 = encode_point_batch(rows[:, 3:6])
    return torch.stack([p1, p2], dim=1)


# ============================================================
# (2) Model: MVTransformer backbone + scalar readout
#     We map multivectors -> multivectors, then a small equivariant
#     MVLinear to produce per-token scalars and average.
# ============================================================
class DistanceModel(nn.Module):
    def __init__(self, num_layers=4, num_heads=2, channels_per_atom=1):
        super().__init__()
        self.backbone = MVTransformer(num_layers=num_layers, num_heads=num_heads, channels_per_atom=channels_per_atom)
        self.readout  = MVLinear(in_channels=1, out_channels=1)  # produces [B, N, 16]; we’ll take scalar comp
    def forward(self, tokens):  # tokens: [B, N=2, 16]
        x = self.backbone(tokens)           # [B, 2, 16]
        y = self.readout(x)                 # [B, 2, 16]
        scalars = y[..., 0]                 # [B, 2]  (grade-0 channel)
        pred = scalars.mean(dim=1, keepdim=True)  # [B, 1]
        return pred.squeeze(-1)             # [B]


# ============================================================
# (3) Training / evaluation helpers
# ============================================================
@dataclass
class TrainConfig:
    data_size: int = 1000
    batch_size: int = 512
    epochs: int = 20
    lr: float = 3e-4
    weight_decay: float = 0.0
    seed: int = 42
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

def build_loaders(cfg: TrainConfig):
    full = create_data(cfg.data_size)  # [N, 7] rows
    # split 80/10/10
    n_total = full.shape[0]
    n_train = int(0.8 * n_total)
    n_val   = int(0.1 * n_total)
    n_test  = n_total - n_train - n_val

    ds_full = TensorDataset(full)
    train_ds, val_ds, test_ds = random_split(ds_full, [n_train, n_val, n_test], generator=torch.Generator().manual_seed(cfg.seed))

    def collate(batch):
        rows = torch.stack([b[0] for b in batch], dim=0)
        tokens = encode_pair_rows(rows)             # [B,2,16]
        target = rows[:, 6]                         # [B]
        return tokens, target

    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True,  collate_fn=collate)
    val_loader   = DataLoader(val_ds,   batch_size=cfg.batch_size, shuffle=False, collate_fn=collate)
    test_loader  = DataLoader(test_ds,  batch_size=cfg.batch_size, shuffle=False, collate_fn=collate)

    # Also return raw test rows for later transforms
    test_rows_all = torch.stack([r[0] for r in test_ds], dim=0)
    return train_loader, val_loader, test_loader, test_rows_all

@torch.no_grad()
def evaluate_mae(model, loader, device):
    model.eval()
    mae = 0.0
    n = 0
    for tokens, target in loader:
        tokens = tokens.to(device)
        target = target.to(device)
        pred = model(tokens)
        mae += (pred - target).abs().sum().item()
        n += target.numel()
    return mae / max(1, n)


# ============================================================
# (4) Translation of trivector points by (dx,dy,dz)
# ============================================================
def translate_tokens(tokens: torch.Tensor, dx=0.0, dy=0.0, dz=0.0) -> torch.Tensor:
    """
    tokens: [B, 2, 16] point trivectors
    Translation in PGA adds to the three e0ij coords.
    """
    out = tokens.clone()
    out[..., IDX_E023] += dx
    out[..., IDX_E013] += dy
    out[..., IDX_E012] += dz
    return out

# ============================================================
# (7) Rotation (about z-axis by 45 degrees) in trivector coords
#     x' =  x cos - y sin
#     y' =  x sin + y cos
#     z' =  z
# ============================================================
def rotate_tokens_z(tokens: torch.Tensor, degrees=45.0) -> torch.Tensor:
    theta = math.radians(degrees)
    c, s = math.cos(theta), math.sin(theta)
    out = tokens.clone()
    x = out[..., IDX_E023]
    y = out[..., IDX_E013]
    # new
    x_new = c * x - s * y
    y_new = s * x + c * y
    out[..., IDX_E023] = x_new
    out[..., IDX_E013] = y_new
    # z stays
    return out

# ============================================================
# (10) Reflection across the plane x=0 (the yz-plane)
#      x' = -x, y,z unchanged
# ============================================================
def reflect_tokens_yz(tokens: torch.Tensor) -> torch.Tensor:
    out = tokens.clone()
    out[..., IDX_E023] = -out[..., IDX_E023]
    return out


# ============================================================
# Main: puts all 12 labeled sections in order
# ============================================================
def main():
    cfg = TrainConfig()
    set_seed(cfg.seed)
    device = torch.device(cfg.device)

    # ---------------------------
    # Section 1) Encode to multivectors (done inside loaders)
    # ---------------------------
    train_loader, val_loader, test_loader, test_rows_all = build_loaders(cfg)

    # ---------------------------
    # Section 2) Train the model
    # ---------------------------
    model = DistanceModel(num_layers=4, num_heads=2, channels_per_atom=1).to(device)
    optim = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=cfg.epochs)
    loss_fn = nn.L1Loss()

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        for tokens, target in train_loader:
            tokens = tokens.to(device)
            target = target.to(device)
            pred = model(tokens)
            loss = loss_fn(pred, target)
            optim.zero_grad()
            loss.backward()
            optim.step()
        sched.step()

    # ---------------------------
    # Section 3) Print final MAE on test set
    # ---------------------------
    test_mae = evaluate_mae(model, test_loader, device)
    print(f"[3] Final test MAE: {test_mae:.6f}")

    # Prepare original test multivectors + ground truth
    with torch.no_grad():
        base_tokens = encode_pair_rows(test_rows_all).to(device)  # [T,2,16]
        gt = test_rows_all[:, 6].to(device)                      # [T]

    # ---------------------------
    # Section 4) Translate test_data by dx=dy=dz=5
    # ---------------------------
    trans_tokens = translate_tokens(base_tokens, dx=5.0, dy=5.0, dz=5.0)

    # ---------------------------
    # Section 5) Predict distances for trans_data
    # ---------------------------
    with torch.no_grad():
        pred_trans = model(trans_tokens)

    # ---------------------------
    # Section 6) |pred - gt| mean for translation
    # ---------------------------
    trans_abs_mean = (pred_trans - gt).abs().mean().item()
    print(f"[6] Translation equivariance MAE-to-GT: {trans_abs_mean:.6f}")

    # ---------------------------
    # Section 7) Rotate test_data by 45 degrees about z
    # ---------------------------
    rot_tokens = rotate_tokens_z(base_tokens, degrees=45.0)

    # ---------------------------
    # Section 8) Predict distances for rot_data
    # ---------------------------
    with torch.no_grad():
        pred_rot = model(rot_tokens)

    # ---------------------------
    # Section 9) |pred - gt| mean for rotation
    # ---------------------------
    rot_abs_mean = (pred_rot - gt).abs().mean().item()
    print(f"[9] Rotation equivariance MAE-to-GT: {rot_abs_mean:.6f}")

    # ---------------------------
    # Section 10) Reflect test_data across plane x=0
    # ---------------------------
    ref_tokens = reflect_tokens_yz(base_tokens)

    # ---------------------------
    # Section 11) Predict distances for ref_data
    # ---------------------------
    with torch.no_grad():
        pred_ref = model(ref_tokens)

    # ---------------------------
    # Section 12) |pred - gt| mean for reflection
    # ---------------------------
    ref_abs_mean = (pred_ref - gt).abs().mean().item()
    print(f"[12] Reflection equivariance MAE-to-GT: {ref_abs_mean:.6f}")


if __name__ == "__main__":
    main()
