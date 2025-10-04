import math
import torch
import torch.nn as nn

# Our repo modules
import constants
from gpu_transformer import MVTransformer
from artificial_data import create_data
from multivector import Multivector

# -----------------------
# Same encoders as training
# -----------------------
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

# -----------------------
# Readout head + wrapper (matches training)
# -----------------------
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
            num_layers=num_layers,
            num_heads=num_heads,
            channels_per_atom=channels_per_atom,
        )
        self.readout = ScalarReadout(hidden=64)
    def forward(self, x):  # x: [B, N=2, 16]
        y_mv = self.backbone(x)
        return self.readout(y_mv)

# -----------------------
# Build a PGA rotor: rotate by 'deg' about a principal axis
# axis in {'x','y','z'} -> bivectors e23, e31, e12 respectively
# -----------------------
def make_rotor(deg=45.0, axis='z'):
    theta = math.radians(deg)
    R = torch.zeros(16)
    R[constants.components.index('1')] = math.cos(theta/2.0)
    if axis == 'x':
        R[constants.components.index('e2e3')] = math.sin(theta/2.0)  # e23
    elif axis == 'y':
        R[constants.components.index('e1e3')] = -math.sin(theta/2.0) # e13 = -e31
    else:  # 'z'
        R[constants.components.index('e1e2')] = math.sin(theta/2.0)  # e12
    return Multivector(R)

# -----------------------
# Apply same rotor to both tokens in each pair
# toks: [B, 2, 16]
# -----------------------
def rotate_tokens(toks, rotor):
    comps = []
    for i in range(toks.size(0)):
        pair = []
        for j in range(2):
            M = Multivector(toks[i,j,:])
            Mr = M.rotate(rotor)    # sandwich R M R^{-1}
            pair.append(Mr.coefficients)
        comps.append(torch.stack(pair, dim=0))
    return torch.stack(comps, dim=0)

# -----------------------
# Main
# -----------------------
def main():
    torch.set_float32_matmul_precision("high")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    comps = constants.components

    # 1) Data (same generator used for training)
    full = create_data(4096)  # any size is fine for this check
    toks, targets = rows_to_tokens(full, comps, normalize_plane=True)

    # 2) Model and weights
    model = GATMAPlanePointModel(num_layers=4, num_heads=4, channels_per_atom=1).to(device)
    state = torch.load("gatma_plane_point_best.pt", map_location=device)
    model.load_state_dict(state)
    model.eval()

    toks = toks.to(device)
    targets = targets.to(device)

    # 3) Forward on original tokens
    with torch.no_grad():
        preds_orig = model(toks.float())

    # 4) Build a 45° rotor about z, rotate both plane & point of every pair
    R = make_rotor(deg=45.0, axis='z')
    toks_rot = rotate_tokens(toks.cpu(), R).to(device)

    # 5) Forward on rotated tokens
    with torch.no_grad():
        preds_rot = model(toks_rot.float())

    # 6) Compare
    diffs = (preds_rot - preds_orig).abs()
    print(f"Δ(pred after rot vs orig): mean={diffs.mean().item():.6e}, max={diffs.max().item():.6e}")
    # Optional: also compare to ground-truth distance (should match before & after)
    mae_orig = (preds_orig - targets).abs().mean().item()
    mae_rot  = (preds_rot  - targets).abs().mean().item()
    print(f"MAE vs GT: orig={mae_orig:.6f} | rotated={mae_rot:.6f}")

    # Show a few examples
    for i in range(5):
        print(f"[{i:03d}] gt={targets[i].item():.5f} | pred_orig={preds_orig[i].item():.5f} | "
              f"pred_rot={preds_rot[i].item():.5f} | Δ={abs(preds_rot[i].item()-preds_orig[i].item()):.3e}")

if __name__ == "__main__":
    main()
