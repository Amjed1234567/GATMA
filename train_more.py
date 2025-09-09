import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from gpu_transformer import MVTransformer
from train_0 import MVQM9Data, evaluate, train, device
import os

def main():
    # -------------------------
    # Model & dataset init
    # -------------------------
    model = MVTransformer(num_layers=2, num_heads=4, channels_per_atom=64)
    ckpt_path = "gatma_model_final.pth"

    if os.path.isfile(ckpt_path):
        print(f"Loading checkpoint from {ckpt_path}...")
        model.load_state_dict(torch.load(ckpt_path, map_location=device))
    else:
        raise FileNotFoundError(f"{ckpt_path} not found!")

    # Dataset and splits
    full_dataset = MVQM9Data()
    TRAIN_N, VAL_N, TEST_N = 110_000, 10_000, 10_831
    g = torch.Generator().manual_seed(0)
    train_set, val_set, test_set = random_split(full_dataset, [TRAIN_N, VAL_N, TEST_N], generator=g)

    BATCH_SIZE, NUM_WORKERS = 32, 4
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

    # Recompute normalization (y_mean, y_std)
    with torch.no_grad():
        ys = []
        for xb, yb in DataLoader(train_set, batch_size=2048, shuffle=False):
            ys.append(yb)
        ys = torch.cat(ys).float()
    import train_0
    train_0.y_mean = ys.mean()
    train_0.y_std  = ys.std().clamp_min(1e-8)
    print(f"y_mean={train_0.y_mean.item():.6f}, y_std={train_0.y_std.item():.6f}")

    # Loss & optimizer
    criterion = nn.L1Loss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-3, weight_decay=1e-4)

    # Continue training
    train(model, train_loader, val_loader, criterion, optimizer,
          num_epochs=80, ckpt_path="best_model_more.pth")

    test_mae, test_norm = evaluate(model.to(device), test_loader, criterion)
    print(f"[TEST after more training] MAE: {test_mae:.6f} | Norm: {test_norm:.6f}")

    # Save again for chaining
    torch.save(model.state_dict(), "gatma_model_final.pth")

if __name__ == "__main__":
    import torch.multiprocessing as mp
    mp.set_start_method("spawn", force=True)
    main()
