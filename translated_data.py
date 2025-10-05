import math
import torch
import torch.nn as nn

# Repo modules (same as training / rotation check)
import constants
from artificial_data import create_data
from multivector import Multivector
from gpu_transformer import MVTransformer

# ---- Encoders & wrapper (identical to training) ----------------------
def encode_plane_mv(a,b,c,d, comps):
    mv = torch.zeros(16)
    mv[comps.index('e0')] = float(d)
    mv[comps.index('e1')] = float(a)
    mv[comps.index('e2')] = float(b)
    mv[comps.index('e3')] = float(c)
    return mv

def encode_point_mv(x,y,z, comps):
    mv = torch.zeros(16)
    mv[comps.index('e0e2e3')] = float(x)  # e023
    mv[comps.index('e0e1e3')] = float(y)  # e013
    mv[comps.index('e0e1e2')] = float(z)  # e012
    mv[comps.index('e1e2e3')] = 1.0       # e123
    return mv

def rows_to_tokens(batch_rows, comps, normalize_plane=True):
    B = batch_rows.size(0)
    toks = torch.zeros(B, 2, 16)
    targets = batch_rows[:, 7].clone()
    for i in range(B):
        a,b,c,d,x,y,z,_ = batch_rows[i].tolist()
        if normalize_plane:
            n = math.sqrt(a*a + b*b + c*c) + 1e-12
            a,b,c,d = a/n, b/n, c/n, d/n
        toks[i,0,:] = encode_plane_mv(a,b,c,d, comps)
        toks[i,1,:] = encode_point_mv(x,y,z, comps)
    return toks, targets

class ScalarReadout(nn.Module):
    def __init__(self, hidden=64):
        super().__init__()
        self.proj = nn.Linear(16, hidden)
        self.act  = nn.GELU()
        self.mlp  = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, 1),
        )
    def forward(self, x):           # [B, N, 16]
        h = self.act(self.proj(x))  # [B, N, H]
        h = h.mean(dim=1)           # [B, H]
        out = self.mlp(h).squeeze(-1)
        return out

class GATMAPlanePointModel(nn.Module):
    def __init__(self, num_layers=4, num_heads=4, channels_per_atom=1):
        super().__init__()
        self.backbone = MVTransformer(
            num_layers=num_layers, num_heads=num_heads,
            channels_per_atom=channels_per_atom,
        )
        self.readout = ScalarReadout(hidden=64)
    def forward(self, x):  # x: [B, 2, 16]
        y_mv = self.backbone(x)
        return self.readout(y_mv)

# ---- Translate both tokens in each pair by the SAME vector -----------
def translate_tokens(toks, dx=5.0, dy=5.0, dz=5.0):
    """
    toks: [B, 2, 16] on any device. Returns translated tokens on CPU.
    Uses Multivector.translate (sandwich with a translator versor).
    """
    toks_cpu = toks.detach().cpu()
    out = []
    for i in range(toks_cpu.size(0)):
        pair = []
        for j in range(2):
            M = Multivector(toks_cpu[i, j, :])
            Mt = M.translate(dx, dy, dz)  # T M T*
            pair.append(Mt.coefficients)
        out.append(torch.stack(pair, dim=0))
    return torch.stack(out, dim=0)  # [B, 2, 16]

# ---- Main -------------------------------------------------------------
def main():
    try:
        torch.set_float32_matmul_precision("high")
    except Exception:
        pass

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    comps = constants.components

    # 1) Data and tokens (same generator as training)
    full = create_data(4096)  # [N,8] = [a,b,c,d,x,y,z,dist]
    toks, targets = rows_to_tokens(full, comps, normalize_plane=True)

    # 2) Model & weights (must match training)
    model = GATMAPlanePointModel(num_layers=4, num_heads=4, channels_per_atom=1).to(device)
    state = torch.load("gatma_plane_point_best.pt", map_location=device)
    model.load_state_dict(state)
    model.eval()

    toks = toks.to(device)
    targets = targets.to(device)

    # 3) Predictions on original tokens
    with torch.no_grad():
        preds_orig = model(toks.float())

    # 4) Translate both plane & point by t = (5, 5, 5) for every pair
    toks_tr = translate_tokens(toks, dx=5.0, dy=5.0, dz=5.0).to(device)

    # 5) Predictions on translated tokens
    with torch.no_grad():
        preds_tr = model(toks_tr.float())

    # 6) Compare
    diffs = (preds_tr - preds_orig).abs()
    print(f"Δ(pred after translation vs orig): mean={diffs.mean().item():.6e}, max={diffs.max().item():.6e}")

    mae_orig = (preds_orig - targets).abs().mean().item()
    mae_tr   = (preds_tr   - targets).abs().mean().item()
    print(f"MAE vs GT: orig={mae_orig:.6f} | translated={mae_tr:.6f}")

    for i in range(5):
        print(f"[{i:03d}] gt={targets[i].item():.5f} | "
              f"pred_orig={preds_orig[i].item():.5f} | "
              f"pred_tr={preds_tr[i].item():.5f} | "
              f"Δ={abs(preds_tr[i].item()-preds_orig[i].item()):.3e}")

if __name__ == "__main__":
    main()
