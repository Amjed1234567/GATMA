import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from gpu_transformer import MVTransformer
# Reuse train(), evaluate(), device, and the save/load helpers
import train_0  

def main():
    # -------------------------
    # Model & dataset init
    # -------------------------
    model = MVTransformer(num_layers=2, num_heads=4, channels_per_atom=64)
    ckpt_model_path = "gatma_model_final.pth"
    # full state (model + opt + sched + scaler + epoch + stats)
    full_ckpt_path  = "gatma_checkpoint.pth"  

    # Dataset and splits (same seed/sizes as the first run)
    full_dataset = train_0.MVQM9Data()
    TRAIN_N, VAL_N, TEST_N = 110_000, 10_000, 10_831
    g = torch.Generator().manual_seed(0)
    train_set, val_set, test_set = random_split(full_dataset, [TRAIN_N, VAL_N, TEST_N], generator=g)

    BATCH_SIZE, NUM_WORKERS = 32, 4
    PIN = (train_0.device.type == 'cuda')
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

    # Loss, optimizer, scheduler (created before loading so we can load their state)
    criterion = nn.L1Loss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=5, min_lr=1e-6)

    # --- Load checkpoint if available (true resume); else fall back to model-only and recompute stats ---
    start_epoch = 0
    if os.path.isfile(full_ckpt_path):
        print(f"Loading full training state from {full_ckpt_path}...")
        start_epoch, y_mean_ckpt, y_std_ckpt = train_0.load_checkpoint(
            full_ckpt_path, model, optimizer, scheduler, train_0.scaler, map_location=train_0.device
        )
        # Restore normalization globals for train/evaluate
        if y_mean_ckpt is not None and y_std_ckpt is not None:
            train_0.y_mean = y_mean_ckpt
            train_0.y_std  = y_std_ckpt
            print(f"[Resume] y_mean={train_0.y_mean.item():.6f}, y_std={train_0.y_std.item():.6f}")
        else:
            # safety: recompute if missing
            with torch.no_grad():
                ys = []
                for xb, yb in DataLoader(train_set, batch_size=2048, shuffle=False):
                    ys.append(yb)
                ys = torch.cat(ys).float()
            train_0.y_mean = ys.mean()
            train_0.y_std  = ys.std().clamp_min(1e-8)
            print(f"[Recomputed] y_mean={train_0.y_mean.item():.6f}, y_std={train_0.y_std.item():.6f}")
    else:
        # Fallback path: load model-only weights and recompute normalization (old behavior)
        if os.path.isfile(ckpt_model_path):
            print(f"Loading model weights from {ckpt_model_path}...")
            model.load_state_dict(torch.load(ckpt_model_path, map_location=train_0.device))
        else:
            raise FileNotFoundError(f"Neither {full_ckpt_path} nor {ckpt_model_path} found!")

        with torch.no_grad():
            ys = []
            for xb, yb in DataLoader(train_set, batch_size=2048, shuffle=False):
                ys.append(yb)
            ys = torch.cat(ys).float()
        train_0.y_mean = ys.mean()
        train_0.y_std  = ys.std().clamp_min(1e-8)
        print(f"[Fresh] y_mean={train_0.y_mean.item():.6f}, y_std={train_0.y_std.item():.6f}")

    # Continue training for another 80 epochs
    train_0.train(model, train_loader, val_loader, criterion, optimizer, scheduler,
                  num_epochs=80, start_epoch=start_epoch)

    # Evaluate
    test_mae, test_norm = train_0.evaluate(model.to(train_0.device), test_loader, criterion)
    print(f"[TEST after more training] MAE: {test_mae:.6f} | Norm: {test_norm:.6f}")

    # Save model-only (for compatibility) and full checkpoint (true resume for the next round)
    torch.save(model.state_dict(), ckpt_model_path)
    train_0.save_checkpoint(
        full_ckpt_path,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        scaler=train_0.scaler,
        epoch=start_epoch + 80,   # cumulative epochs trained so far
        y_mean=train_0.y_mean,
        y_std=train_0.y_std,
    )

if __name__ == "__main__":
    import torch.multiprocessing as mp
    mp.set_start_method("spawn", force=True)
    main()
