import os
import gc; gc.collect() # from https://docs.python.org/3/library/gc.html
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
# This part is only for dipole moment - TARGET_IDX = 0.
# Padding, pooling and masking - Because some atoms are just zeros.
# -------------------------
def dipole_pred(outputs, inputs, model):
    """
    Compute dipole prediction as:
      For each atom:
        sum_vector  = sum over channels of vector part (e0,e1,e2,e3) of MV
        sum_scalar  = sum over channels of scalar part (1) of MV
        sum_atom    = sum_vector + sum_scalar * X   (X = atom coords from trivector inputs)
      Then sum over atoms -> vector V, and prediction = ||V||_2.

    Args:
        outputs: [B, N*C, 16]  (or [B, N, 16] if C=1)
        inputs:  [B, N, 16]    (padded atoms)
        model:   MVTransformer (for channels_per_atom)

    Returns:
        preds: [B] dipole magnitudes
    """
    B, N, D = inputs.shape
    C = getattr(model, "channels_per_atom", 1)

    # Reshape outputs back to per-atom, per-channel
    if C > 1:
        outputs = outputs.view(B, N, C, D)  # [B, N, C, 16]
    else:
        outputs = outputs.unsqueeze(2)       # [B, N, 1, 16]

    # Indices for components
    comps = constants.components
    idx_scalar = comps.index('1')
    idx_e0 = comps.index('e0')
    idx_e1 = comps.index('e1'); idx_e2 = comps.index('e2'); idx_e3 = comps.index('e3')
    idx_x  = comps.index('e0e1e2')
    idx_y  = comps.index('e0e1e3')
    idx_z  = comps.index('e0e2e3')
    idx_w  = comps.index('e1e2e3')

    # Sum over channels: scalar and vector parts
    sum_scalar = outputs[..., idx_scalar].sum(dim=2)                   # [B, N]
    sum_vector = torch.stack([
        outputs[..., idx_e0].sum(dim=2),
        outputs[..., idx_e1].sum(dim=2),
        outputs[..., idx_e2].sum(dim=2),
        outputs[..., idx_e3].sum(dim=2),
    ], dim=-1)                                                         # [B, N, 3]

    # Extract atom coordinates X from the input trivector components
    X = torch.stack([inputs[..., idx_x], inputs[..., idx_y], inputs[..., idx_z]], dim=-1)  # [B,N,3]
    w = inputs[..., idx_w].unsqueeze(-1)  # [B,N,1]
    
    # Safe dehomogenization
    X = X / torch.clamp(w, min=1e-8)
    
    # Add a 4th component (zero) to X to match sum_vector
    X_4d = torch.cat([X, torch.zeros_like(X[..., :1])], dim=-1)  # [B, N, 4]
    # Per-atom contribution, then sum over (real) atoms
    sum_atom = sum_vector + sum_scalar.unsqueeze(-1) * X_4d      # [B, N, 4]

    # Mask out padded atoms
    atom_mask = (inputs.abs().sum(dim=2) > 0).float().unsqueeze(-1)    # [B, N, 1]
    V = (sum_atom * atom_mask).sum(dim=1)                               # [B, 3]

    # Dipole magnitude
    return torch.linalg.norm(V, dim=-1)                                 # [B]



# -------------------------
# Padding, pooling and masking - Because some atoms are just zeros.
# -------------------------
def pooled_pred(outputs, inputs, model):
    # Special dipole construction only when predicting QM9 target 0
    if TARGET_IDX == 0:
        return dipole_pred(outputs, inputs, model)
    
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

        # Unnormalized MAE.
        mae  = criterion(preds, targets).item()

        # Normalized L1.
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
def train(model, train_loader, val_loader, criterion, optimizer,
          num_epochs=20, ckpt_path="best_model.pth",
          early_stop_patience=20):
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

    best_val_mae = float("inf")
    best_epoch = -1
    epochs_no_improve = 0

    for epoch in range(1, num_epochs + 1):
        model.train()
        running = 0.0

        for batch_idx, (inputs, targets) in enumerate(train_loader):
            inputs = inputs.to(device, non_blocking=True)
            targets = targets.to(device).view(-1).to(torch.float32)

            optimizer.zero_grad(set_to_none=True)
            with amp.autocast('cuda', enabled=use_amp, dtype=torch.bfloat16):
                outputs = model(inputs)
                preds = pooled_pred(outputs, inputs, model)
                # train on normalized scale 
                norm_targets = (targets - y_mean) / y_std
                norm_preds   = (preds   - y_mean.to(device)) / y_std.to(device)
                loss = criterion(norm_preds, norm_targets)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()

            running += loss.item()
            wandb.log({"train/batch_loss": loss.item(), "epoch": epoch, "batch": batch_idx})

        train_loss = running / max(len(train_loader), 1)

        # ----- validation -----
        val_mae, val_norm = evaluate(model, val_loader, criterion)  # returns (mae, norm) 

        # scheduler step on smoothed val_norm
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
        """
        # ----- early-stopping on val_mae & best checkpoint inside loop -----
        if val_mae < best_val_mae:
            best_val_mae = val_mae
            best_epoch = epoch
            epochs_no_improve = 0
            torch.save(model.state_dict(), ckpt_path)
            wandb.log({"model/best_val_mae": best_val_mae, "model/best_epoch": best_epoch})
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= early_stop_patience:
                print(f"Early stopping at epoch {epoch} (best val_mae={best_val_mae:.6f} at epoch {best_epoch})")
                break
        """

def main():
    # -------------------------
    # Model & dataset init
    # -------------------------
    model = MVTransformer(num_layers=2, num_heads=4, channels_per_atom=128)
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
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-3, weight_decay=1e-4)
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

        running_loss += loss.item()
        
    epoch_time = time.time() - start_time
    avg_loss = running_loss / max(len(train_loader), 1)
    print(f"[Sanity check] Epoch time: {epoch_time:.2f} sec | Avg loss: {avg_loss:.6f}")
    
    # Free up any leftovers from the quick run
    del outputs, preds, loss    
    torch.cuda.empty_cache()
    # >>> END PROBE BLOCK <<<

    # Train, then test
    train(model, train_loader, val_loader, criterion, optimizer, num_epochs=50, ckpt_path="best_model.pth")
    if os.path.isfile("best_model.pth"):
        model.load_state_dict(torch.load("best_model.pth", map_location=device))
    # Please note that test_mae is the raw mean absolute error (MAE) 
    # in the original physical units of the target.
    # test_norm is the normalized MAE, computed after subtracting the dataset mean (y_mean) 
    # and dividing by the dataset standard deviation (y_std). For for comparing 
    # across different targets.        
    test_mae, test_norm = evaluate(model.to(device), test_loader, criterion)
    print(f"[TEST] MAE: {test_mae:.6f} | Norm: {test_norm:.6f}")
    wandb.log({"test/mae": test_mae, "test/norm": test_norm})
    torch.save(model.state_dict(), 'gatma_model_final.pth')
    
    # Clean up before saving
    del test_loader, val_loader, train_loader
    del outputs, preds, loss
    gc.collect() # From https://docs.python.org/3/library/gc.html
    torch.cuda.empty_cache()


if __name__ == "__main__":
    # From https://docs.pytorch.org/docs/stable/multiprocessing.html
    import torch.multiprocessing as mp
    mp.set_start_method("spawn", force=True)
    main()
