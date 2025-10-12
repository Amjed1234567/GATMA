import math, os, random
from dataclasses import dataclass
import torch, torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, random_split

# W&B check.
try:
    import wandb
    _WANDB_AVAILABLE = True
except Exception:
    _WANDB_AVAILABLE = False

import constants
from artificial_data import create_data
from gpu_transformer import MVTransformer
from gpu_building_blocks import MVLinear

# ---------------------------
# Reproducibility
# ---------------------------
def set_seed(seed: int = 42):
    random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

# ==== PGA point embedding indices ====
IDX_E012  = constants.components.index('e0e1e2')
IDX_E013  = constants.components.index('e0e1e3')
IDX_E123  = constants.components.index('e1e2e3')
IDX_E023  = constants.components.index('e0e2e3')

def encode_point_batch(xyz: torch.Tensor) -> torch.Tensor:
    B = xyz.shape[0]
    mv = torch.zeros(B, len(constants.components), dtype=xyz.dtype, device=xyz.device)
    mv[:, IDX_E123] = 1.0
    mv[:, IDX_E023] = xyz[:, 0]
    mv[:, IDX_E013] = xyz[:, 1]
    mv[:, IDX_E012] = xyz[:, 2]
    return mv

def encode_pair_rows(rows: torch.Tensor) -> torch.Tensor:
    p1 = encode_point_batch(rows[:, 0:3])
    p2 = encode_point_batch(rows[:, 3:6])
    return torch.stack([p1, p2], dim=1)

# ==== Model ====
class DistanceModel(nn.Module):
    def __init__(self, num_layers=4, num_heads=2, channels_per_atom=1):
        super().__init__()
        self.backbone = MVTransformer(num_layers=num_layers, num_heads=num_heads, channels_per_atom=channels_per_atom)
        self.readout  = MVLinear(in_channels=1, out_channels=1)
    def forward(self, tokens):
        x = self.backbone(tokens)
        y = self.readout(x)
        scalars = y[..., 0]
        return scalars.mean(dim=1)

# ==== Config: read LR / WD from environment ====
def _as_float(env_name: str, default: float) -> float:
    v = os.getenv(env_name, str(default))
    try:
        return float(v)
    except ValueError:
        raise ValueError(f"Env var {env_name}='{v}' is not a valid float.")

@dataclass
class TrainConfig:
    data_size: int = int(os.getenv("DATA_SIZE", "50000"))
    batch_size: int = int(os.getenv("BATCH_SIZE", "256"))
    epochs: int = int(os.getenv("EPOCHS", "100"))
    lr: float = _as_float("LR", 3e-4)                    
    weight_decay: float = _as_float("WEIGHT_DECAY", 0.0) 
    seed: int = int(os.getenv("SEED", "42"))
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    save_dir: str = os.getenv("CKPT_DIR", "checkpoints")

def build_loaders(cfg: TrainConfig):
    full = create_data(cfg.data_size)
    n_total = full.shape[0]
    n_train = int(0.8 * n_total); n_val = int(0.1 * n_total); n_test = n_total - n_train - n_val
    ds_full = TensorDataset(full)
    gen = torch.Generator().manual_seed(cfg.seed)
    train_ds, val_ds, test_ds = random_split(ds_full, [n_train, n_val, n_test], generator=gen)

    def collate(batch):
        rows = torch.stack([b[0] for b in batch], dim=0)
        tokens = encode_pair_rows(rows)
        target = rows[:, 6]
        return tokens, target

    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True,  collate_fn=collate)
    val_loader   = DataLoader(val_ds,   batch_size=cfg.batch_size, shuffle=False, collate_fn=collate)
    test_loader  = DataLoader(test_ds,  batch_size=cfg.batch_size, shuffle=False, collate_fn=collate)
    test_rows_all = torch.stack([r[0] for r in test_ds], dim=0)
    return train_loader, val_loader, test_loader, test_rows_all

@torch.no_grad()
def evaluate_mae(model, loader, device):
    model.eval(); mae = 0.0; n = 0
    for tokens, target in loader:
        tokens = tokens.to(device); target = target.to(device)
        pred = model(tokens)
        mae += (pred - target).abs().sum().item()
        n += target.numel()
    return mae / max(1, n)

# ==== Transforms ====
def translate_tokens(tokens, dx=0.0, dy=0.0, dz=0.0):
    out = tokens.clone()
    out[..., IDX_E023] += dx; out[..., IDX_E013] += dy; out[..., IDX_E012] += dz
    return out

def rotate_tokens_z(tokens, degrees=45.0):
    import math
    theta = math.radians(degrees); c, s = math.cos(theta), math.sin(theta)
    out = tokens.clone()
    x = out[..., IDX_E023]; y = out[..., IDX_E013]
    out[..., IDX_E023] = c * x - s * y
    out[..., IDX_E013] = s * x + c * y
    return out

def reflect_tokens_yz(tokens):
    out = tokens.clone()
    out[..., IDX_E023] = -out[..., IDX_E023]
    return out

def main():
    cfg = TrainConfig()
    set_seed(cfg.seed)
    device = torch.device(cfg.device)
    os.makedirs(cfg.save_dir, exist_ok=True)

    # W&B setup: only if env looks configured and package available
    use_wandb = _WANDB_AVAILABLE and os.getenv("WANDB_PROJECT")
    if use_wandb:
        run_name = os.getenv("WANDB_RUN_NAME", f"LR={cfg.lr}_WD={cfg.weight_decay}")
        wandb.init(project=os.getenv("WANDB_PROJECT"),
                   entity=os.getenv("WANDB_ENTITY"),
                   name=run_name,
                   config={"lr": cfg.lr, "weight_decay": cfg.weight_decay,
                           "epochs": cfg.epochs, "batch_size": cfg.batch_size,
                           "data_size": cfg.data_size})
        wandb.define_metric("epoch")
        wandb.define_metric("train/loss", step_metric="epoch")
        wandb.define_metric("val/mae", step_metric="epoch")

    # ---------------------------
    # 1) Encode happens inside loaders
    # ---------------------------
    train_loader, val_loader, test_loader, test_rows_all = build_loaders(cfg)
    
    # --- compute train target mean/std for standardization ---
    def _collect_targets(loader, device):
        ys = []
        for _, y in loader:
            ys.append(y)
        return torch.cat(ys, dim=0)
    

    y_train_all = _collect_targets(train_loader, device=torch.device("cpu"))
    t_mean = y_train_all.mean().item()
    t_std  = y_train_all.std(unbiased=False).item() or 1.0  # avoid div-by-zero
    print(f"[info] target mean={t_mean:.4f}, std={t_std:.4f}")

    # ---------------------------
    # 2) Train
    # ---------------------------
    model = DistanceModel(num_layers=6, num_heads=2, channels_per_atom=4).to(device)
    optim = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    loss_fn = nn.L1Loss()

    # before training loop — warmup + cosine schedule
    warmup_epochs = max(3, int(0.05 * cfg.epochs))
    total_epochs  = cfg.epochs

    def lr_lambda(ep):
        if ep < warmup_epochs:
            return (ep + 1) / warmup_epochs
        # cosine from 1.0 down to 0 over the rest
        progress = (ep - warmup_epochs) / max(1, total_epochs - warmup_epochs)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    sched = torch.optim.lr_scheduler.LambdaLR(optim, lr_lambda=lr_lambda)


    best_val = float("inf"); best_path = None
    print(f"Training with LR={cfg.lr} WD={cfg.weight_decay}")

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        running = 0.0; seen = 0
        for tokens, target in train_loader:
            tokens = tokens.to(device); target = target.to(device)
            pred = model(tokens)
            loss = loss_fn((pred - t_mean)/t_std, (target - t_mean)/t_std)
            optim.zero_grad() 
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optim.step()
            running += loss.item() * target.numel(); seen += target.numel()
        sched.step()
        train_loss = running / max(1, seen)

        val_mae = evaluate_mae(model, val_loader, device)
        if use_wandb:
            wandb.log({"epoch": epoch, "train/loss": train_loss, "val/mae": val_mae})

        # save best-by-val: file name includes LR/WD from env
        if val_mae < best_val:
            best_val = val_mae
            # keep env strings to be readable in filenames
            lr_str = os.getenv("LR", str(cfg.lr))
            wd_str = os.getenv("WEIGHT_DECAY", str(cfg.weight_decay))
            best_path = os.path.join(cfg.save_dir, f"gatma_points_best_LR={lr_str}_WD={wd_str}.pt")
            torch.save({"model": model.state_dict(),
                        "cfg": cfg.__dict__,
                        "val_mae": best_val}, best_path)

    # finish W&B
    if use_wandb:
        wandb.summary["best_val_mae"] = best_val
        if best_path: wandb.summary["ckpt_path"] = best_path

    # ---------------------------
    # Section 3) Print final MAE on test set (standardized)
    # ---------------------------
    with torch.no_grad():
        mae = 0.0
        n = 0
        for tokens, target in test_loader:
            tokens = tokens.to(device)
            target = target.to(device)
            pred = model(tokens)

            # denormalize prediction
            pred = (pred - t_mean) / t_std   # ← same normalization used in training loss
            pred = pred * t_std + t_mean     # optional: restores original units (keeps code clear)

            mae += (pred - target).abs().sum().item()
            n += target.numel()
    test_mae = mae / max(1, n)
    print(f"[3] Final test MAE: {test_mae:.6f}")


    with torch.no_grad():
        base_tokens = encode_pair_rows(test_rows_all).to(device)
        gt = test_rows_all[:, 6].to(device)

    # 4) Translate
    trans_tokens = translate_tokens(base_tokens, dx=5.0, dy=5.0, dz=5.0)

    # 5) Predict (translation)
    with torch.no_grad():
        pred_trans = model(trans_tokens)
    # 6) Mean |pred-gt|
    trans_abs_mean = (pred_trans - gt).abs().mean().item()
    print(f"[6] Translation equivariance MAE-to-GT: {trans_abs_mean:.6f}")

    # 7) Rotate 45 deg
    rot_tokens = rotate_tokens_z(base_tokens, degrees=45.0)

    # 8) Predict (rotation)
    with torch.no_grad():
        pred_rot = model(rot_tokens)
    # 9) Mean |pred-gt|
    rot_abs_mean = (pred_rot - gt).abs().mean().item()
    print(f"[9] Rotation equivariance MAE-to-GT: {rot_abs_mean:.6f}")

    # 10) Reflect across x=0
    ref_tokens = reflect_tokens_yz(base_tokens)

    # 11) Predict (reflection)
    with torch.no_grad():
        pred_ref = model(ref_tokens)
    # 12) Mean |pred-gt|
    ref_abs_mean = (pred_ref - gt).abs().mean().item()
    print(f"[12] Reflection equivariance MAE-to-GT: {ref_abs_mean:.6f}")

    # Log finals to W&B too
    if use_wandb:
        wandb.log({
            "test/mae": test_mae,
            "equiv/translation_mae_to_gt": trans_abs_mean,
            "equiv/rotation_mae_to_gt": rot_abs_mean,
            "equiv/reflection_mae_to_gt": ref_abs_mean
        })
        wandb.finish()

if __name__ == "__main__":
    main()
