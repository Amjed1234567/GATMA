import math
import random
from typing import Tuple

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader, random_split

import wandb  # Weights & Biases

# =========================
# Hyperparameters (edit here)
# =========================
DATA_SIZE        = 20000
BATCH_SIZE       = 256
EPOCHS           = 30
LR               = 3e-4
VAL_FRAC         = 0.10
TEST_FRAC        = 0.10
SEED             = 0
PLANE_NORMALIZE  = True
CLIP_NORM        = 1.0

PROJECT_NAME     = "GATMA-plane-point"
RUN_NAME         = None
WANDB_WATCH      = False
TAGS             = ["artificial", "PGA", "distance"]
NOTES            = "Plane(point) distance sanity-check with MVTransformer."

# ---- precision control ----
USE_BF16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()

# faster matmul on Ampere+
try:
    torch.set_float32_matmul_precision("high")
except Exception:
    pass

# ---------------------------------------------------------------------
from artificial_data import create_data        # returns [N,8] with [a,b,c,d,x,y,z,dist]
import constants
from gpu_transformer import MVTransformer

# ---------------------------------------------------------------------
# Multivector encoders (PGA G3,0,1)
# Plane (a,b,c,d) -> vector:  d*e0 + a*e1 + b*e2 + c*e3
# Point (x,y,z)   -> trivector: x*e0e2e3 + y*e0e1e3 + z*e0e1e2 + 1*e1e2e3
def encode_plane_mv(a,b,c,d, comps) -> torch.Tensor:
    mv = torch.zeros(16)
    mv[comps.index('e0')] = float(d)
    mv[comps.index('e1')] = float(a)
    mv[comps.index('e2')] = float(b)
    mv[comps.index('e3')] = float(c)
    return mv

def encode_point_mv(x,y,z, comps) -> torch.Tensor:
    mv = torch.zeros(16)
    mv[comps.index('e0e2e3')] = float(x)  # e023
    mv[comps.index('e0e1e3')] = float(y)  # e013
    mv[comps.index('e0e1e2')] = float(z)  # e012
    mv[comps.index('e1e2e3')] = 1.0       # e123
    return mv

def rows_to_tokens(batch_rows: torch.Tensor, comps, normalize_plane=True) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    batch_rows: [B, 8] with [a,b,c,d,x,y,z,dist]
    returns:
      tokens [B, 2, 16]   -> (plane_token, point_token)
      targets [B]         -> dist
    """
    B = batch_rows.size(0)
    toks = torch.zeros(B, 2, 16)
    targets = batch_rows[:, 7].clone()

    for i in range(B):
        a,b,c,d,x,y,z,_ = batch_rows[i].tolist()
        if normalize_plane:
            norm = math.sqrt(a*a + b*b + c*c) + 1e-12
            a, b, c, d = a/norm, b/norm, c/norm, d/norm
        toks[i,0,:] = encode_plane_mv(a,b,c,d, comps)
        toks[i,1,:] = encode_point_mv(x,y,z, comps)

    return toks, targets


# --- Readout head -----------------------------------------------------
class ScalarReadout(nn.Module):
    """ Map [B, N, 16] -> scalar via linear+GELU, mean over tokens, MLP """
    def __init__(self, hidden=64):
        super().__init__()
        self.proj = nn.Linear(16, hidden)
        self.act = nn.GELU()
        self.mlp = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, x):  # x: [B, N, 16]
        h = self.act(self.proj(x))     # [B, N, H]
        h = h.mean(dim=1)              # [B, H]
        out = self.mlp(h).squeeze(-1)  # [B]
        return out


class GATMAPlanePointModel(nn.Module):
    """
    Wrap MVTransformer -> scalar head.
    Expects input tokens: [B, 2, 16]  (plane, point)
    """
    def __init__(self, num_layers=4, num_heads=4, channels_per_atom=1):
        super().__init__()
        self.backbone = MVTransformer(
            num_layers=num_layers,
            num_heads=num_heads,
            channels_per_atom=channels_per_atom,
        )
        self.readout = ScalarReadout(hidden=64)

    def forward(self, x):  # [B, N=2, 16]
        y_mv = self.backbone(x)   # returns [B, N*(channels_per_atom), 16]
        return self.readout(y_mv)


# --- Eval -------------------------------------------------------------
def evaluate(model, loader, device):
    model.eval()
    mse, mae, n = 0.0, 0.0, 0
    with torch.no_grad():
        for toks, targets in loader:
            toks = toks.to(device)
            targets = targets.to(device)
            
            if next(model.parameters()).dtype == torch.bfloat16:
                toks = toks.to(torch.bfloat16)
                
            preds = model(toks)
            diff = preds - targets
            mse += torch.sum(diff * diff).item()
            mae += torch.sum(diff.abs()).item()
            n += targets.numel()
    rmse = math.sqrt(mse / n) if n > 0 else float("nan")
    mae  = (mae / n) if n > 0 else float("nan")
    return rmse, mae


# --- Main -------------------------------------------------------------
def main():
    # seeds & device
    torch.manual_seed(SEED)
    random.seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    # W&B init
    run = wandb.init(
        project=PROJECT_NAME,
        name=RUN_NAME,
        notes=NOTES,
        tags=TAGS,
        config={
            "data_size": DATA_SIZE,
            "batch_size": BATCH_SIZE,
            "epochs": EPOCHS,
            "lr": LR,
            "val_frac": VAL_FRAC,
            "test_frac": TEST_FRAC,
            "seed": SEED,
            "plane_normalize": PLANE_NORMALIZE,
            "clip_norm": CLIP_NORM,
            "model": {"num_layers": 4, "num_heads": 4, "channels_per_atom": 1},
        },
        settings=wandb.Settings(start_method="fork")
    )

    comps = constants.components

    # data
    full = create_data(DATA_SIZE)             # [N, 8]
    toks, targets = rows_to_tokens(full, comps, normalize_plane=PLANE_NORMALIZE)
    dataset = TensorDataset(toks, targets)

    # 3-way split: 80/10/10
    N = len(dataset)
    n_val  = int(N * VAL_FRAC)
    n_test = int(N * TEST_FRAC)
    n_train = N - n_val - n_test
    g = torch.Generator().manual_seed(SEED)
    train_set, val_set, test_set = random_split(dataset, [n_train, n_val, n_test], generator=g)

    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    val_loader   = DataLoader(val_set,   batch_size=BATCH_SIZE, shuffle=False)
    test_loader  = DataLoader(test_set,  batch_size=BATCH_SIZE, shuffle=False)

    # model
    model = GATMAPlanePointModel(
        num_layers=4,
        num_heads=4,
        channels_per_atom=1,
    ).to(device)

    if USE_BF16:
        model = model.to(torch.bfloat16)


    if WANDB_WATCH:
        wandb.watch(model, log="all", log_freq=50)

    opt   = torch.optim.AdamW(model.parameters(), lr=LR)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
    crit  = nn.L1Loss()  # MAE suits a distance target

    best_val = float("inf")
    ckpt_path = "gatma_plane_point_best.pt"

    # -------- training loop --------
    for epoch in range(1, EPOCHS + 1):
        model.train()
        running_loss = 0.0
        n_obs = 0

        for toks_b, targets_b in train_loader:
            toks_b = toks_b.to(device)
            targets_b = targets_b.to(device)

            if USE_BF16:
                toks_b = toks_b.to(torch.bfloat16)

            pred  = model(toks_b)

            # compute loss in float32 for numerical stability
            loss  = crit(pred.float(), targets_b.float())

            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=CLIP_NORM)
            opt.step()

            running_loss += loss.item() * targets_b.numel()
            n_obs += targets_b.numel()

        sched.step()
        train_mae = running_loss / max(n_obs, 1)
        val_rmse, val_mae = evaluate(model, val_loader, device)
        current_lr = sched.get_last_lr()[0]

        print(f"Epoch {epoch:03d} | train_mae={train_mae:.6f} | val_mae={val_mae:.6f} | val_rmse={val_rmse:.6f} | lr={current_lr:.2e}")

        wandb.log({
            "train/mae": train_mae,
            "val/mae": val_mae,
            "val/rmse": val_rmse,
            "lr": current_lr,
            "epoch": epoch,
        })

        # save & upload best
        if val_mae < best_val:
            best_val = val_mae
            torch.save(model.state_dict(), ckpt_path)
            wandb.summary["best_val_mae"] = best_val

            art = wandb.Artifact("gatma_plane_point_model", type="model")
            art.add_file(ckpt_path)
            wandb.log_artifact(art)

    # -------- reload best before testing --------
    map_location = device if device.type == "cpu" else {"cuda:0": "cuda:0"}
    state = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state)
    model.to(device)
    model.eval()

    # final test evaluation (best model)
    test_rmse, test_mae = evaluate(model, test_loader, device)
    print("Final test set results: RMSE={:.6f}, MAE={:.6f}".format(test_rmse, test_mae))
    wandb.summary["test_mae"] = test_mae
    wandb.summary["test_rmse"] = test_rmse

    print("Best val MAE:", best_val)
    run.finish()


if __name__ == "__main__":
    main()
