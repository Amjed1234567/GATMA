# ============================================================
# GATMA — Simple plane–point distance training + invariance check
# Steps:
# (1)  Build original multivectors from (a,b,c,d,x,y,z,dist)
# (2)  Train the model on originals
# (3)  Print final MAE on originals
# (4)  Make translated data (dx=dy=dz=5)
# (5)  Predict on translated data
# (6)  Print mean |pred_trans - ground_truth|
# (7)  Make rotated data (angle=45° about Z)
# (8)  Predict on rotated data
# (9)  Print mean |pred_rot - ground_truth|
# (10) Make reflected data (fixed mirror plane)
# (11) Predict on reflected data
# (12) Print mean |pred_ref - ground_truth|
# ============================================================

import math
import random
from typing import Tuple

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

# --- project modules (already in your repo) ---
import constants
from artificial_data import create_data        # -> [N,8] rows: [a,b,c,d,x,y,z,dist]
from gpu_transformer import MVTransformer
from multivector import Multivector

# --------------------
# Hyperparams (simple)
# --------------------
SEED        = 0
DATA_SIZE   = 20000
BATCH_SIZE  = 256
EPOCHS      = 25
LR          = 1e-2
PLANE_NORMALIZE = True
CKPT_PATH   = "gatma_plane_point_best.pt"

# ============================================================
# (Helper) Encoding: Euclidean plane/point -> PGA multivectors
# ============================================================

def encode_plane_mv(a,b,c,d, comps) -> torch.Tensor:
    # Plane (a,b,c,d) -> vector: d*e0 + a*e1 + b*e2 + c*e3
    mv = torch.zeros(16)
    mv[comps.index('e0')] = float(d)
    mv[comps.index('e1')] = float(a)
    mv[comps.index('e2')] = float(b)
    mv[comps.index('e3')] = float(c)
    return mv

def encode_point_mv(x,y,z, comps) -> torch.Tensor:
    # Point (x,y,z) -> trivector: x*e0e2e3 + y*e0e1e3 + z*e0e1e2 + 1*e1e2e3
    mv = torch.zeros(16)
    mv[comps.index('e0e2e3')] = float(x)  # e023
    mv[comps.index('e0e1e3')] = float(y)  # e013
    mv[comps.index('e0e1e2')] = float(z)  # e012
    mv[comps.index('e1e2e3')] = 1.0       # e123
    return mv

def rows_to_tokens(rows: torch.Tensor, comps, normalize_plane=True) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    rows: [N,8] -> [a,b,c,d,x,y,z,dist]
    returns:
      tokens:  [N, 2, 16]  (plane_token, point_token)
      targets: [N]
    """
    N = rows.size(0)
    toks = torch.zeros(N, 2, 16)
    targets = rows[:,7].clone()
    for i in range(N):
        a,b,c,d,x,y,z,_ = rows[i].tolist()
        if normalize_plane:
            n = math.sqrt(a*a + b*b + c*c) + 1e-12
            a,b,c,d = a/n, b/n, c/n, d/n
        toks[i,0,:] = encode_plane_mv(a,b,c,d, comps)
        toks[i,1,:] = encode_point_mv(x,y,z, comps)
    return toks, targets

# ============================================================
# (Helper) Simple invariant readout head (grade-0 only)
# ============================================================

class InvariantReadout(nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(1.0))
        self.bias  = nn.Parameter(torch.tensor(0.0))
        self.idx_scalar = constants.components.index('1')
    def forward(self, x):           # x: [B, N, 16]
        s = x[..., self.idx_scalar] # [B, N]
        s = s.mean(dim=1)           # reduce over tokens (plane+point)
        return s * self.scale + self.bias

class GATMAPlanePointModel(nn.Module):
    def __init__(self, num_layers=4, num_heads=4, channels_per_atom=1):
        super().__init__()
        self.backbone = MVTransformer(
            num_layers=num_layers,
            num_heads=num_heads,
            channels_per_atom=channels_per_atom,
        )
        self.readout = InvariantReadout()
    def forward(self, x):           # x: [B, 2, 16]
        y_mv = self.backbone(x)
        return self.readout(y_mv)

# ============================================================
# (Helper) Rigid motions (fixed ones for eval)
# ============================================================

def translate_tokens_cpu(toks_cpu, dx, dy, dz):
    B = toks_cpu.size(0)
    out = []
    for i in range(B):
        pair = []
        for j in range(2):
            M = Multivector(toks_cpu[i,j,:])
            Mt = M.translate(dx, dy, dz)
            pair.append(Mt.coefficients)
        out.append(torch.stack(pair, dim=0))
    return torch.stack(out, dim=0)

def make_fixed_rotor_z(deg=45.0):
    theta = math.radians(deg)
    comps = constants.components
    R = torch.zeros(16)
    R[comps.index('1')]   = math.cos(theta/2.0)
    R[comps.index('e1e2')] = math.sin(theta/2.0)  # rotor about z
    return Multivector(R)

def rotate_tokens_cpu(toks_cpu, rotor: Multivector):
    B = toks_cpu.size(0)
    out = []
    for i in range(B):
        pair = []
        for j in range(2):
            M = Multivector(toks_cpu[i,j,:])
            Mr = M.rotate(rotor)   # R M R^{-1}
            pair.append(Mr.coefficients)
        out.append(torch.stack(pair, dim=0))
    return torch.stack(out, dim=0)

def make_fixed_mirror_plane():
    """
    Fixed mirror Π = d*e0 + a*e1 + b*e2 + c*e3 (unit normal).
    Here: mirror across plane x + 2y + 3z + d = 0 with d = 1.5
    """
    comps = constants.components
    mv = torch.zeros(16)
    a,b,c = 1.0, 2.0, 3.0
    n = math.sqrt(a*a + b*b + c*c)
    a,b,c = a/n, b/n, c/n
    d = 1.5
    mv[comps.index('e0')] = d
    mv[comps.index('e1')] = a
    mv[comps.index('e2')] = b
    mv[comps.index('e3')] = c
    return Multivector(mv)

def reflect_tokens_cpu(toks_cpu, mirror: Multivector):
    B = toks_cpu.size(0)
    out = []
    for i in range(B):
        pair = []
        for j in range(2):
            M = Multivector(toks_cpu[i,j,:])
            Mr = M.reflect(mirror)  # -Π M Π^{-1}
            pair.append(Mr.coefficients)
        out.append(torch.stack(pair, dim=0))
    return torch.stack(out, dim=0)

# ============================================================
# (Helper) MAE on a token set
# ============================================================

@torch.no_grad()
def mae_on_tokens(model, toks: torch.Tensor, targets: torch.Tensor, device) -> float:
    model.eval()
    preds = model(toks.to(device).float()).view(-1)
    return (preds - targets.to(device).float()).abs().mean().item()

# ============================================================
# Main
# ============================================================

def main():
    # seeds & device
    torch.manual_seed(SEED)
    random.seed(SEED)
    try:
        torch.set_float32_matmul_precision("high")
    except Exception:
        pass
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    comps = constants.components

    # --------------------------------------------------------
    # (1) Build ORIGINAL multivectors
    # --------------------------------------------------------
    full = create_data(DATA_SIZE)              # [N,8]
    toks_all, y_all = rows_to_tokens(full, comps, normalize_plane=PLANE_NORMALIZE)

    # simple split: 90% train, 10% val
    N = toks_all.size(0)
    n_val = max(int(0.1 * N), 1)
    idx = torch.randperm(N)
    train_idx, val_idx = idx[n_val:], idx[:n_val]

    toks_train, y_train = toks_all[train_idx], y_all[train_idx]
    toks_val,   y_val   = toks_all[val_idx],   y_all[val_idx]

    train_loader = DataLoader(TensorDataset(toks_train, y_train),
                              batch_size=BATCH_SIZE, shuffle=True, drop_last=True)

    # --------------------------------------------------------
    # (2) Train the model
    # --------------------------------------------------------
    model = GATMAPlanePointModel(num_layers=4, num_heads=4, channels_per_atom=1).to(device)
    opt   = torch.optim.AdamW(model.parameters(), lr=LR)
    crit  = nn.L1Loss()

    for epoch in range(1, EPOCHS+1):
        model.train()
        running = 0.0
        n = 0
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            opt.zero_grad(set_to_none=True)
            pred = model(xb.float()).view(-1)
            loss = crit(pred, yb.float())
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            opt.step()
            running += loss.item() * yb.numel()
            n += yb.numel()
        if epoch % 5 == 0 or epoch == 1:
            print(f"Epoch {epoch:03d} | train_loss≈{running/max(n,1):.6f}")

    # save (optional), keep in memory for now
    torch.save(model.state_dict(), CKPT_PATH)

    # --------------------------------------------------------
    # (3) Final MAE on ORIGINALS
    # --------------------------------------------------------
    mae_orig = mae_on_tokens(model, toks_val, y_val, device)
    print(f"(3) FINAL MAE on originals: {mae_orig:.6f}")

    # --------------------------------------------------------
    # (4) Make TRANSLATED data (dx=dy=dz=5)  -> trans_data
    # --------------------------------------------------------
    toks_val_cpu = toks_val.detach().cpu()
    trans_data = translate_tokens_cpu(toks_val_cpu, dx=5.0, dy=5.0, dz=5.0)

    # --------------------------------------------------------
    # (5) Predict on trans_data
    # --------------------------------------------------------
    # (6) Print mean |pred_trans - ground_truth|
    mae_trans = mae_on_tokens(model, trans_data, y_val, device)
    print(f"(6) MAE on translated (dx=dy=dz=5): {mae_trans:.6f}")

    # --------------------------------------------------------
    # (7) Make ROTATED data (angle=45° about Z)  -> rot_data
    # --------------------------------------------------------
    Rz45 = make_fixed_rotor_z(deg=45.0)
    rot_data = rotate_tokens_cpu(toks_val_cpu, Rz45)

    # --------------------------------------------------------
    # (8) Predict on rot_data
    # --------------------------------------------------------
    # (9) Print mean |pred_rot - ground_truth|
    mae_rot = mae_on_tokens(model, rot_data, y_val, device)
    print(f"(9) MAE on rotated (z, 45 deg): {mae_rot:.6f}")

    # --------------------------------------------------------
    # (10) Make REFLECTED data (fixed mirror)   -> ref_data
    # --------------------------------------------------------
    mirror = make_fixed_mirror_plane()
    ref_data = reflect_tokens_cpu(toks_val_cpu, mirror)

    # --------------------------------------------------------
    # (11) Predict on ref_data
    # --------------------------------------------------------
    # (12) Print mean |pred_ref - ground_truth|
    mae_ref = mae_on_tokens(model, ref_data, y_val, device)
    print(f"(12) MAE on reflected (fixed plane): {mae_ref:.6f}")

if __name__ == "__main__":
    main()
