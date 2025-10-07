# ============================================================
# GATMA — Plane–Point Distance: Training + Test Invariance (v2 + Option A)
# ------------------------------------------------------------
# Toggle invariance training (augmentation + consistency):
#   ENABLE_INVAR_TRAINING = True  -> train with rotate/translate/reflect + consistency
#   ENABLE_INVAR_TRAINING = False -> plain training on originals only
# ============================================================

import math
import random
from typing import Tuple

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

# --- project modules ---
import constants
from artificial_data import create_data        # -> [N,8] rows: [a,b,c,d,x,y,z,dist]
from gpu_transformer import MVTransformer
from multivector import Multivector

import time
import os

USE_WANDB = True  # flip to False to disable logging 
if USE_WANDB:
    import wandb


# --------------------
# (A) Global config
# --------------------
SEED        = 0
DATA_SIZE   = 20000
BATCH_SIZE  = 256
EPOCHS      = 25
LR          = 2e-4
PLANE_NORMALIZE = True
CKPT_PATH   = "gatma_plane_point_best_5.pt"
CLIP_NORM   = 1.0

# --- Option A toggle ---
ENABLE_INVAR_TRAINING = True     # <--- set False to compare plain training
LAMBDA_INV            = 0.6      # Control the equivariance. 

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
# (Helper) Model with invariant readout (grade-0 only)
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
# (Helper) Rigid motions (train-time + test-time)
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

def make_random_rotor_z(max_deg=180.0):
    deg = random.uniform(-max_deg, max_deg)
    return make_fixed_rotor_z(deg=deg)

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

def rotate_translate_tokens_cpu(toks_cpu):
    R = make_random_rotor_z()
    tx, ty, tz = [random.uniform(-5.0, 5.0) for _ in range(3)]
    toks_rot = rotate_tokens_cpu(toks_cpu, R)
    toks_rt  = translate_tokens_cpu(toks_rot, tx, ty, tz)
    return toks_rt

# ============================================================
# (Helper) Metric: MAE on a token set
# ============================================================

@torch.no_grad()
def mae_on_tokens(model, toks: torch.Tensor, targets: torch.Tensor, device) -> float:
    model.eval()
    preds = model(toks.to(device).float()).view(-1)
    return (preds - targets.to(device).float()).abs().mean().item()

# ============================================================
# (1) Build ORIGINAL multivectors
# ============================================================

def build_originals():
    comps = constants.components
    full = create_data(DATA_SIZE)              # [N,8]
    toks_all, y_all = rows_to_tokens(full, comps, normalize_plane=PLANE_NORMALIZE)
    return toks_all, y_all

# ============================================================
# (2) Train the model (80/10/10 split) — WITH optional invariance training
# ============================================================

def train_model(toks_all, y_all, device):
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
                              batch_size=BATCH_SIZE, shuffle=True, drop_last=True, pin_memory=True)
    val_loader   = DataLoader(TensorDataset(toks_val, y_val),
                              batch_size=BATCH_SIZE, shuffle=False, drop_last=False, pin_memory=True)

    model = GATMAPlanePointModel(num_layers=4, num_heads=4, channels_per_atom=1).to(device)
    opt   = torch.optim.AdamW(model.parameters(), lr=LR)
    #crit  = nn.L1Loss()
    crit = nn.SmoothL1Loss(beta=1.0)

    
    if USE_WANDB:
        wandb.watch(model, log="gradients", log_freq=100, log_graph=False)   

    best_val = float("inf")
    for epoch in range(1, EPOCHS+1):
        epoch_t0 = time.time()
        model.train()
        # Reuse the SAME mirror plane for this entire epoch
        fixed_mirror = make_fixed_mirror_plane()
        running = 0.0
        n = 0
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)

            opt.zero_grad(set_to_none=True)

            # ----- original -----
            pred = model(xb.float()).view(-1)
            loss_main = crit(pred, yb.float())

            if ENABLE_INVAR_TRAINING:
                # ----- rotated+translated (same batch) -----
                toks_cpu = xb.detach().cpu()
                toks_rt  = rotate_translate_tokens_cpu(toks_cpu).to(device)
                pred_rt  = model(toks_rt.float()).view(-1)

                # ----- fixed reflection (same mirror as test) -----
                toks_ref = reflect_tokens_cpu(toks_cpu, fixed_mirror).to(device)
                pred_ref = model(toks_ref.float()).view(-1)

                # ----- invariance losses -----
                loss_aug  = crit(pred_rt,  yb.float()) + crit(pred_ref, yb.float())
                #loss_cons = LAMBDA_INV * ((pred_rt - pred).abs().mean() +
                                          #(pred_ref - pred).abs().mean())
                progress = epoch / EPOCHS
                lam = LAMBDA_INV * progress     # 0 → LAMBDA_INV
                loss_cons = lam * ((pred_rt - pred).abs().mean() + (pred_ref - pred).abs().mean())

                loss = loss_main + loss_aug + loss_cons
            else:
                loss = loss_main

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
            
            if USE_WANDB:
                # Log checkpoint as an artifact (best on val)
                art = wandb.Artifact("gatma_plane_point_best", type="model")
                art.add_file(CKPT_PATH)
                wandb.log_artifact(art)


        if epoch % 5 == 0 or epoch == 1 or epoch == EPOCHS:
            tag = "INV" if ENABLE_INVAR_TRAINING else "PLAIN"
            print(f"[{tag}] Epoch {epoch:03d} | train_loss≈{running/max(n,1):.6f} | val_MAE={val_mae:.6f}")
            epoch_time = time.time() - epoch_t0

            if USE_WANDB:
                wandb.log({
                    "epoch": epoch,
                    "train/loss_epoch_mean": running / max(n, 1),
                    "val/mae": val_mae,
                    "time/epoch_seconds": epoch_time,
                    "tag": "INV" if ENABLE_INVAR_TRAINING else "PLAIN",
                }, step=epoch)

            
            
    # load best on val before final test
    model.load_state_dict(torch.load(CKPT_PATH, map_location=device))
    return model, toks_train, y_train, toks_val, y_val, toks_test, y_test

# ============================================================
# (3) Print FINAL MAE on TEST originals
# ============================================================

def step3_final_mae_test_originals(model, toks_test, y_test, device):
    mae_test_orig = mae_on_tokens(model, toks_test, y_test, device)
    print(f"(3) FINAL MAE on TEST originals: {mae_test_orig:.6f}")
    if 'wandb' in globals() and USE_WANDB:
        wandb.log({"test/mae_orig": mae_test_orig})


# ============================================================
# (4) Make TRANSLATED test set (dx=dy=dz=5)  -> trans_data
# (5) Predict on trans_data
# (6) Print mean |pred_trans - ground_truth|
# ============================================================

def steps4_6_translated(model, toks_test, y_test, device):
    toks_test_cpu = toks_test.detach().cpu()
    trans_data = translate_tokens_cpu(toks_test_cpu, dx=5.0, dy=5.0, dz=5.0)
    mae_trans = mae_on_tokens(model, trans_data, y_test, device)
    print(f"(6) MAE on TEST translated (dx=dy=dz=5): {mae_trans:.6f}")
    if 'wandb' in globals() and USE_WANDB:
        wandb.log({"test/mae_translated": mae_trans})


# ============================================================
# (7) Make ROTATED test set (45° about Z)  -> rot_data
# (8) Predict on rot_data
# (9) Print mean |pred_rot - ground_truth|
# ============================================================

def steps7_9_rotated(model, toks_test, y_test, device):
    toks_test_cpu = toks_test.detach().cpu()
    Rz45 = make_fixed_rotor_z(deg=45.0)
    rot_data = rotate_tokens_cpu(toks_test_cpu, Rz45)
    mae_rot = mae_on_tokens(model, rot_data, y_test, device)
    print(f"(9) MAE on TEST rotated (z, 45 deg): {mae_rot:.6f}")
    if 'wandb' in globals() and USE_WANDB:
        wandb.log({"test/mae_rotated": mae_rot})


# ============================================================
# (10) Make REFLECTED test set (fixed plane) -> ref_data
# (11) Predict on ref_data
# (12) Print mean |pred_ref - ground_truth|
# ============================================================

def steps10_12_reflected(model, toks_test, y_test, device):
    toks_test_cpu = toks_test.detach().cpu()
    mirror = make_fixed_mirror_plane()
    ref_data = reflect_tokens_cpu(toks_test_cpu, mirror)
    mae_ref = mae_on_tokens(model, ref_data, y_test, device)
    print(f"(12) MAE on TEST reflected (fixed plane): {mae_ref:.6f}")
    if 'wandb' in globals() and USE_WANDB:
        wandb.log({"test/mae_reflected": mae_ref})


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
    
    # --- Weights & Biases init ---
    if USE_WANDB:
        wandb.init(
            project=os.environ.get("WANDB_PROJECT", "GATMA"),
            entity=os.environ.get("WANDB_ENTITY", None),  
            config={
                "seed": SEED,
                "data_size": DATA_SIZE,
                "batch_size": BATCH_SIZE,
                "epochs": EPOCHS,
                "lr": LR,
                "plane_normalize": PLANE_NORMALIZE,
                "clip_norm": CLIP_NORM,
                "enable_invar_training": ENABLE_INVAR_TRAINING,
                "lambda_inv": LAMBDA_INV,
                "model": {
                    "num_layers": 4,
                    "num_heads": 4,
                    "channels_per_atom": 1,
                },
            },
            name="gatma-plane-point-invariant",  
            notes="Training with rotate/translate/reflect + consistency" if ENABLE_INVAR_TRAINING else "Plain training",
        )    

    # (1) originals
    toks_all, y_all = build_originals()

    # (2) train (optionally with invariance augmentation + consistency)
    model, toks_train, y_train, toks_val, y_val, toks_test, y_test = train_model(toks_all, y_all, device)

    # (3)
    step3_final_mae_test_originals(model, toks_test, y_test, device)

    # (4) (5) (6)
    steps4_6_translated(model, toks_test, y_test, device)

    # (7) (8) (9)
    steps7_9_rotated(model, toks_test, y_test, device)

    # (10) (11) (12)
    steps10_12_reflected(model, toks_test, y_test, device)
    
    if USE_WANDB:
        wandb.finish()

if __name__ == "__main__":
    main()
