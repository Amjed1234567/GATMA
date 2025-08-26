import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from gpu_transformer import MVTransformer
import wandb
from torch_geometric.datasets import QM9
from torch.optim.lr_scheduler import ReduceLROnPlateau

from multivector import Multivector
import constants

# -------------------------
# Device & basic prints
# -------------------------
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(torch.__version__)
print("Using device:", device)

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


# -------------------------
# Model & dataset init
# -------------------------
model = MVTransformer(num_layers=3, num_heads=3, channels_per_atom=3)
full_dataset = MVQM9Data()

print(f"Full QM9 size reported by wrapper: {len(full_dataset)}")  # should be 130,831

# -------------------------
# Fixed split sizes (train/val/test)
# -------------------------
TRAIN_N = 110_000
VAL_N   = 10_000
TEST_N  = 10_831  # 110000 + 10000 + 10831 = 130831

assert TRAIN_N + VAL_N + TEST_N == len(full_dataset), \
    f"Split sizes ({TRAIN_N}+{VAL_N}+{TEST_N}) must equal dataset size {len(full_dataset)}"

# Reproducible split
g = torch.Generator().manual_seed(0)
train_set, val_set, test_set = random_split(full_dataset, [TRAIN_N, VAL_N, TEST_N], generator=g)

print(f"Train: {len(train_set)}  Val: {len(val_set)}  Test: {len(test_set)}")

# -------------------------
# Dataloaders
# -------------------------
BATCH_SIZE = 100
NUM_WORKERS = 4
PIN = (device.type == 'cuda')
PERSIST = PIN and NUM_WORKERS > 0

train_loader = DataLoader(
    train_set,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=NUM_WORKERS,
    pin_memory=PIN,
    persistent_workers=PERSIST if NUM_WORKERS > 0 else False,
)

val_loader = DataLoader(
    val_set,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=PIN,
    persistent_workers=PERSIST if NUM_WORKERS > 0 else False,
)

test_loader = DataLoader(
    test_set,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=PIN,
    persistent_workers=PERSIST if NUM_WORKERS > 0 else False,
)

print(f"Steps per epoch (train): {(len(train_set) + BATCH_SIZE - 1)//BATCH_SIZE}, batch_size={BATCH_SIZE}")

# -------------------------
# Normalization
# -------------------------
# --- compute train target mean/std for normalization ---
with torch.no_grad():
    ys = []
    for xb, yb in DataLoader(train_set, batch_size=2048, shuffle=False):
        ys.append(yb)
    ys = torch.cat(ys).float()
    y_mean = ys.mean()
    y_std  = ys.std().clamp_min(1e-8)  # avoid div-by-zero

print(f"y_mean={y_mean.item():.6f}, y_std={y_std.item():.6f}")

 
# -------------------------
# Loss & optimizer
# -------------------------
criterion = nn.L1Loss() # This is the MAE. 
#criterion = nn.MSELoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=5e-3, weight_decay=1e-2)


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
    total_loss = 0.0
    n_batches = 0
    for inputs, targets in loader:
        inputs = inputs.to(device, non_blocking=True)
        targets = targets.to(device).view(-1).to(torch.float32)
        outputs = model(inputs)                     # [B, max_atoms, ...]
        """
        # build mask from inputs (padded rows are all zeros)
        mask = (inputs.abs().sum(dim=2) > 0).float()   # [B, max_atoms]
        atom_feats = outputs[:, :, 0]                  # [B, max_atoms]

        # masked mean (safe divide)
        sum_mask = mask.sum(dim=1).clamp_min(1.0)
        preds = (atom_feats * mask).sum(dim=1) / sum_mask
        """
        preds = pooled_pred(outputs, inputs, model)
        # convert preds back to original scale before computing MSE
        preds_unscaled = preds
        
        loss = criterion(preds_unscaled, targets)

        total_loss += loss.item()
        n_batches += 1
    return total_loss / max(n_batches, 1)


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
            outputs = model(inputs)
            """
            # build mask from inputs (padded rows are all zeros)
            mask = (inputs.abs().sum(dim=2) > 0).float()   # [B, max_atoms]
            atom_feats = outputs[:, :, 0]                  # [B, max_atoms]

            # masked mean (safe divide)
            sum_mask = mask.sum(dim=1).clamp_min(1.0)
            
            preds = (atom_feats * mask).sum(dim=1) / sum_mask
            """
            preds = pooled_pred(outputs, inputs, model)

            # preds is still in *raw* scale: we predict raw then normalize for loss
            norm_targets = (targets - y_mean) / y_std
            norm_preds   = (preds - y_mean.to(device)) / y_std.to(device)
            loss = criterion(norm_preds, norm_targets)

            loss.backward()
            optimizer.step()

            running += loss.item()

            wandb.log({
                "train/batch_loss": loss.item(),
                "epoch": epoch,
                "batch": batch_idx,
            })

        train_loss = running / max(len(train_loader), 1)
        val_loss = evaluate(model, val_loader, criterion)
        
        # --- EMA of validation loss for LR scheduling (alpha = 0.9) ---
        if ema_val is None:
            ema_val = val_loss
        else:
            ema_val = ema_alpha * ema_val + (1.0 - ema_alpha) * val_loss

        # Step the scheduler using the EMA-smoothed validation loss
        scheduler.step(ema_val)

        # Report both raw and EMA val losses + current LR
        current_lr = scheduler.get_last_lr()[0]  # list -> float
        print(f"Epoch {epoch:03d} | train_loss={train_loss:.6f} | val_loss={val_loss:.6f} | "
            f"val_ema={ema_val:.6f} | lr={current_lr:.6e}")

        wandb.log({
            "train/epoch_loss": train_loss,
            "val/epoch_loss": val_loss,
            "val/epoch_loss_ema": ema_val,
            "lr": current_lr,
            "epoch": epoch
        })


        # save best checkpoint
        if val_loss < best_val:
            best_val = val_loss
            torch.save(model.state_dict(), ckpt_path)
            wandb.log({"model/best_val": best_val})


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
    outputs = model(inputs)
    preds = outputs[:, :, 0].sum(dim=1)
    loss = criterion(preds, targets)
    loss.backward()
    optimizer.step()

    running_loss += loss.item()

epoch_time = time.time() - start_time
avg_loss = running_loss / max(len(train_loader), 1)

print(f"[Sanity check] Epoch time: {epoch_time:.2f} sec | Avg loss: {avg_loss:.6f}")
# >>> END PROBE BLOCK <<<


# -------------------------
# Run training, then test
# -------------------------
train(model, train_loader, val_loader, criterion, optimizer, num_epochs=100, ckpt_path="best_model.pth")

# Load best checkpoint before testing (optional but recommended)
if os.path.isfile("best_model.pth"):
    model.load_state_dict(torch.load("best_model.pth", map_location=device))

test_loss = evaluate(model.to(device), test_loader, criterion)
print(f"[TEST] MAE: {test_loss:.6f}")
wandb.log({"test/mae": test_loss})

# Save final weights too (separate from best val)
torch.save(model.state_dict(), 'gatma_model_final.pth')
