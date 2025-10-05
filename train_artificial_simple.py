# ============================================================
# GATMA — Simple plane–point distance training + test invariance check (v2)
# Dataset split: 80% train / 10% val / 10% test
# Steps:
# (1)  Build original multivectors
# (2)  Train the model on the train split (monitor val)
# (3)  Print FINAL MAE on TEST ORIGINALS
# (4)  Make TRANSLATED test set (dx=dy=dz=5)      -> trans_data
# (5)  Predict on trans_data
# (6)  Print mean |pred_trans - ground_truth|
# (7)  Make ROTATED test set (45° about Z)        -> rot_data
# (8)  Predict on rot_data
# (9)  Print mean |pred_rot - ground_truth|
# (10) Make REFLECTED test set (fixed plane)      -> ref_data
# (11) Predict on ref_data
# (12) Print mean |pred_ref - ground_truth|
# ============================================================

import math
import random
from typing import Tuple

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

# --- project modules  ---
import constants
from artificial_data import create_data        # -> [N,8] rows: [a,b,c,d,x,y,z,dist]
from gpu_transformer import MVTransformer
from multivector import Multivector

# --------------------
# Hyperparams
# --------------------
SEED        = 0
DATA_SIZE   = 20000
BATCH_SIZE  = 256
EPOCHS      = 100
LR          = 1e-2
PLANE_NORMALIZE = True
CKPT_PATH   = "gatma_plane_point_best.pt"
CLIP_NORM   = 1.0

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
# (Helper) Invariant readout head (grade-0 only)
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
# (Helper) Rigid motions (fixed ones for test eval)
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
    R[comps.index('1')]    = math.cos(theta/2.0)
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
    Fixed mirror = d*e0 + a*e1 + b*e2 + c*e3 (unit normal).
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
            Mr = M.reflect(mirror)  # -R M R^{-1}
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

    # Split: 80 / 10 / 10
    N = toks_all.size(0)
    n_train = max(int(0.8 * N), 1)
    n_val   = max(int(0.1 * N), 1)
    n_test  = N - n_train - n_val
    idx = torch.randperm(N)
    train_idx = idx[:n_train]
    val_idx   = idx[n_train:n_train+n_val]
    test_idx  = idx[n_train+n_val:]

    toks_train, y_train = toks_all[train_idx], y_all[train_idx]
    toks_val,   y_val   = toks_all[val_idx],   y_all[val_idx]
    toks_test,  y_test  = toks_all[test_idx],  y_all[test_idx]

    train_loader = DataLoader(TensorDataset(toks_train, y_train),
                              batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    val_loader   = DataLoader(TensorDataset(toks_val, y_val),
                              batch_size=BATCH_SIZE, shuffle=False, drop_last=False)

    # --------------------------------------------------------
    # (2) Train the model
    # --------------------------------------------------------
    model = GATMAPlanePointModel(num_layers=4, num_heads=4, channels_per_atom=1).to(device)
    opt   = torch.optim.AdamW(model.parameters(), lr=LR)
    crit  = nn.L1Loss()

    best_val = float("inf")
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
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=CLIP_NORM)
            opt.step()
            running += loss.item() * yb.numel()
            n += yb.numel()

        # quick val MAE
        val_mae = mae_on_tokens(model, toks_val, y_val, device)
        if val_mae < best_val:
            best_val = val_mae
            torch.save(model.state_dict(), CKPT_PATH)

        if epoch % 5 == 0 or epoch == 1:
            print(f"Epoch {epoch:03d} | train_loss≈{running/max(n,1):.6f} | val_MAE={val_mae:.6f}")

    # load best on val before final test
    model.load_state_dict(torch.load(CKPT_PATH, map_location=device))

    # --------------------------------------------------------
    # (3) FINAL MAE on TEST ORIGINALS
    # --------------------------------------------------------
    mae_test_orig = mae_on_tokens(model, toks_test, y_test, device)
    print(f"(3) FINAL MAE on TEST originals: {mae_test_orig:.6f}")

    # --------------------------------------------------------
    # (4) Make TRANSLATED test set (dx=dy=dz=5)  -> trans_data
    # --------------------------------------------------------
    toks_test_cpu = toks_test.detach().cpu()
    trans_data = translate_tokens_cpu(toks_test_cpu, dx=5.0, dy=5.0, dz=5.0)

    # --------------------------------------------------------
    # (5) Predict on trans_data
    # --------------------------------------------------------
    # (6) Print mean |pred_trans - ground_truth|
    mae_trans = mae_on_tokens(model, trans_data, y_test, device)
    print(f"(6) MAE on TEST translated (dx=dy=dz=5): {mae_trans:.6f}")

    # --------------------------------------------------------
    # (7) Make ROTATED test set (angle=45° about Z)  -> rot_data
    # --------------------------------------------------------
    Rz45 = make_fixed_rotor_z(deg=45.0)
    rot_data = rotate_tokens_cpu(toks_test_cpu, Rz45)

    # --------------------------------------------------------
    # (8) Predict on rot_data
    # --------------------------------------------------------
    # (9) Print mean |pred_rot - ground_truth|
    mae_rot = mae_on_tokens(model, rot_data, y_test, device)
    print(f"(9) MAE on TEST rotated (z, 45 deg): {mae_rot:.6f}")

    # --------------------------------------------------------
    # (10) Make REFLECTED test set (fixed plane)   -> ref_data
    # --------------------------------------------------------
    mirror = make_fixed_mirror_plane()
    ref_data = reflect_tokens_cpu(toks_test_cpu, mirror)

    # --------------------------------------------------------
    # (11) Predict on ref_data
    # --------------------------------------------------------
    # (12) Print mean |pred_ref - ground_truth|
    mae_ref = mae_on_tokens(model, ref_data, y_test, device)
    print(f"(12) MAE on TEST reflected (fixed plane): {mae_ref:.6f}")

if __name__ == "__main__":
    main()
