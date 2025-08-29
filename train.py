import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from gpu_transformer import MVTransformer
import wandb
from torch_geometric.datasets import QM9
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch import amp 
import constants

# -------------------------
# Device & basic prints
# -------------------------
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(torch.__version__)
print("Using device:", device)

# From https://docs.pytorch.org/docs/stable/amp.html
use_amp = (device.type == "cuda")
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.set_float32_matmul_precision("high")

# -------------------------
# Encoding
# -------------------------

def atom_to_multivector(position, atomic_number):
    mv = torch.zeros(16)
    mv[constants.components.index('e0e1e2')] = position[0]
    mv[constants.components.index('e0e1e3')] = position[1]
    mv[constants.components.index('e0e2e3')] = position[2]
    mv[constants.components.index('e1e2e3')] = 1.0
    mv[constants.components.index('1')] = atomic_number
    return mv

# Choose a QM9 target (0 = dipole moment)
TARGET_IDX = 0


# -------------------------
# Dataset wrapper
# -------------------------
class MVQM9Data(torch.utils.data.Dataset):
    def __init__(self, root='data/QM9', target_idx: int = TARGET_IDX, max_atoms: int = 29):
        self.dataset = QM9(root)
        self.target_idx = target_idx
        self.max_atoms = max_atoms

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        molecule = self.dataset[idx]
        positions = molecule.pos
        atomic_numbers = molecule.z

        # Flatten y and select one scalar target
        y = molecule.y
        if y.dim() > 1:
            y = y.view(-1)
        target = y[self.target_idx].to(torch.float32)  # shape: []

        # Encode atoms -> 16-d multivectors
        multivectors = []
        for pos, z in zip(positions, atomic_numbers):
            mv = atom_to_multivector(pos, z)
            multivectors.append(mv)

        padded = torch.zeros(self.max_atoms, 16, dtype=torch.float32)
        for i in range(min(self.max_atoms, len(multivectors))):
            padded[i] = multivectors[i]

        return padded, target


# From https://docs.pytorch.org/docs/stable/amp.html#gradient-scaling
scaler = amp.GradScaler('cuda', enabled=use_amp)


# -------------------------
# Padding, pooling and masking - Because some atoms are just zeros.
# -------------------------
def pooled_pred(outputs, inputs, model):
    mask = (inputs.abs().sum(dim=2) > 0).float()         # [B,N]
    if getattr(model, "channels_per_atom", 1) > 1:
        mask = mask.repeat_interleave(model.channels_per_atom, dim=1)  # [B,N*n]
    atom_feats = outputs[:, :, 0]                         # [B,N or N*n]
    sum_mask = mask.sum(dim=1).clamp_min(1.0)
    return (atom_feats * mask).sum(dim=1) / sum_mask


# -------------------------
# Eval helper
# -------------------------
@torch.no_grad()
def evaluate(model, loader, criterion):
    model.eval()
    total_mae = 0.0
    total_norm = 0.0
    n_batches = 0
    for inputs, targets in loader:
        inputs  = inputs.to(device, non_blocking=True)
        targets = targets.to(device).view(-1).to(torch.float32)

        with amp.autocast('cuda', enabled=use_amp, dtype=torch.bfloat16):
            outputs = model(inputs)
            preds   = pooled_pred(outputs, inputs, model)

        # Unnormalized MAE (human-readable)
        mae  = criterion(preds, targets).item()

        # Normalized L1 (matches training scale)
        norm_targets = (targets - y_mean) / y_std
        norm_preds   = (preds   - y_mean.to(device)) / y_std.to(device)
        norm = criterion(norm_preds, norm_targets).item()

        total_mae  += mae
        total_norm += norm
        n_batches  += 1

    return (total_mae / max(n_batches, 1), total_norm / max(n_batches, 1))


# -------------------------
# Train loop with validation & checkpoint
# -------------------------
def train(model, train_loader, val_loader, criterion, optimizer, num_epochs=20, ckpt_path="best_model.pth"):
    os.environ["WANDB_DEBUG"] = "true"
    os.environ["WANDB_CONSOLE"] = "wrap"

    wandb.init(
        project="G301-Transformer",
        config={
            "epochs": num_epochs,
            "batch_size": train_loader.batch_size,
            "lr": optimizer.param_groups[0]['lr'],
            "model": "MVTransformer",
            "split": {"train": len(train_loader.dataset), "val": len(val_loader.dataset)}
        }
    )

    model.to(device)
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=5, min_lr=1e-6)
    ema_alpha = 0.9
    ema_val = None
    best_val = float("inf")

    for epoch in range(1, num_epochs + 1):
        model.train()
        running = 0.0
        
        for batch_idx, (inputs, targets) in enumerate(train_loader):
            inputs = inputs.to(device, non_blocking=True)
            targets = targets.to(device).view(-1).to(torch.float32)

            optimizer.zero_grad(set_to_none=True)

            # Forward & loss under autocast
            with amp.autocast('cuda', enabled=use_amp, dtype=torch.bfloat16):
                outputs = model(inputs)
                preds = pooled_pred(outputs, inputs, model)
                norm_targets = (targets - y_mean) / y_std
                norm_preds   = (preds - y_mean.to(device)) / y_std.to(device)
                loss = criterion(norm_preds, norm_targets)

            # Backprop & step via GradScaler (no-op on CPU)
            scaler.scale(loss).backward()
            # --- Gradient clipping ---
            scaler.unscale_(optimizer)  # unscale before clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()

            running += loss.item()

            wandb.log({
                "train/batch_loss": loss.item(),
                "epoch": epoch,
                "batch": batch_idx,
            })
                        
        train_loss = running / max(len(train_loader), 1)

        val_mae, val_norm = evaluate(model, val_loader, criterion)

        # scheduler on normalized (consistent scale)
        if ema_val is None:
            ema_val = val_norm
        else:
            ema_val = ema_alpha * ema_val + (1.0 - ema_alpha) * val_norm
        scheduler.step(ema_val)

        current_lr = scheduler.get_last_lr()[0]
        print(f"Epoch {epoch:03d} | train_norm={train_loss:.6f} | val_mae={val_mae:.6f} | "
            f"val_norm={val_norm:.6f} | val_norm_ema={ema_val:.6f} | lr={current_lr:.6e}")

        wandb.log({
            "train/epoch_norm": train_loss,
            "val/epoch_mae": val_mae,
            "val/epoch_norm": val_norm,
            "val/epoch_norm_ema": ema_val,
            "lr": current_lr,
            "epoch": epoch
        })

    # save best checkpoint
    if val_mae < best_val:
        best_val = val_mae
        torch.save(model.state_dict(), ckpt_path)
        wandb.log({"model/best_val": best_val})


def main():
    # -------------------------
    # Model & dataset init
    # -------------------------
    model = MVTransformer(num_layers=4, num_heads=3, channels_per_atom=64)
    full_dataset = MVQM9Data()
    print(f"Full QM9 size reported by wrapper: {len(full_dataset)}")

    # Fixed split sizes
    TRAIN_N, VAL_N, TEST_N = 110_000, 10_000, 10_831
    g = torch.Generator().manual_seed(0)
    train_set, val_set, test_set = random_split(full_dataset, [TRAIN_N, VAL_N, TEST_N], generator=g)
    print(f"Train: {len(train_set)}  Val: {len(val_set)}  Test: {len(test_set)}")

    # Dataloaders
    BATCH_SIZE, NUM_WORKERS = 100, 4
    PIN = (device.type == 'cuda')
    PERSIST = PIN and NUM_WORKERS > 0
    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=NUM_WORKERS, pin_memory=PIN,
                              persistent_workers=PERSIST if NUM_WORKERS > 0 else False,
                              drop_last=True)
    val_loader   = DataLoader(val_set,   batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=NUM_WORKERS, pin_memory=PIN,
                              persistent_workers=PERSIST if NUM_WORKERS > 0 else False)
    test_loader  = DataLoader(test_set,  batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=NUM_WORKERS, pin_memory=PIN,
                              persistent_workers=PERSIST if NUM_WORKERS > 0 else False)
    #print(f"Steps per epoch (train): {(len(train_set) + BATCH_SIZE - 1)//BATCH_SIZE}, batch_size={BATCH_SIZE}")
    # This is used if drop_last=True.
    steps_per_epoch = len(train_set) // BATCH_SIZE
    print(f"Steps per epoch (train): {steps_per_epoch}, batch_size={BATCH_SIZE}")

    
    # Normalization (compute on CPU, no workers)
    with torch.no_grad():
        ys = []
        for xb, yb in DataLoader(train_set, batch_size=2048, shuffle=False):
            ys.append(yb)
        ys = torch.cat(ys).float()
    global y_mean, y_std
    y_mean = ys.mean()
    y_std  = ys.std().clamp_min(1e-8)
    print(f"y_mean={y_mean.item():.6f}, y_std={y_std.item():.6f}")

    # Loss & optimizer & scaler
    criterion = nn.L1Loss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-3, weight_decay=1e-2)
    #scaler = amp.GradScaler(device_type="cuda", enabled=use_amp)

    # >>> QUICK 1 EPOCH SANITY RUN WITH TIMING (TEMPORARY) <<<
    import time
    print("\n[Sanity check] Running 1 quick epoch for timing...")
    model.to(device)
    model.train()
    start_time = time.time()
    running_loss = 0.0
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        inputs = inputs.to(device, non_blocking=True)
        targets = targets.to(device).view(-1).to(torch.float32)
        optimizer.zero_grad(set_to_none=True)
        with amp.autocast('cuda', enabled=use_amp, dtype=torch.bfloat16):
            outputs = model(inputs)
            preds = pooled_pred(outputs, inputs, model)
            loss = criterion(preds, targets)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()

    epoch_time = time.time() - start_time
    avg_loss = running_loss / max(len(train_loader), 1)
    print(f"[Sanity check] Epoch time: {epoch_time:.2f} sec | Avg loss: {avg_loss:.6f}")
    # >>> END PROBE BLOCK <<<

    # Train, then test
    train(model, train_loader, val_loader, criterion, optimizer, num_epochs=95, ckpt_path="best_model.pth")
    if os.path.isfile("best_model.pth"):
        model.load_state_dict(torch.load("best_model.pth", map_location=device))
    test_loss = evaluate(model.to(device), test_loader, criterion)
    print(f"[TEST] MAE: {test_loss:.6f}")
    wandb.log({"test/mae": test_loss})
    torch.save(model.state_dict(), 'gatma_model_final.pth')

if __name__ == "__main__":
    # From https://docs.pytorch.org/docs/stable/multiprocessing.html
    import torch.multiprocessing as mp
    mp.set_start_method("spawn", force=True)
    main()
