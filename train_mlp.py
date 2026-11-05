"""Step 2: time-delay MLP model of the DPA_200MHz power amplifier.

Same split, same NMSE/ACPR definitions and same evaluation samples as fit_baselines.py,
so the numbers can be put in one table.

Run from the repo root:   python train_mlp.py
Outputs: results/mlp_comparison.csv, results/all_models_comparison.csv, figures/mlp_loss_M*.png
"""
from pathlib import Path
import copy
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

# ---------------------------------------------------------------- config
DEPTHS = (1, 3, 5)       # memory depths (number of input samples in the window)
HIDDEN = 32              # neurons per hidden layer
EPOCHS = 200
PATIENCE = 20            # early-stopping patience (epochs without val improvement)
BATCH = 256
LR = 1e-3
SEEDS = (0, 1, 2)
FS = 800e6               # ASSUMPTION (same as fit_baselines.py) - confirm against dataset docs
BW = 200e6
# ----------------------------------------------------------------------

root = Path(__file__).resolve().parent
df = pd.read_csv(root / "datasets" / "dpa_200.csv")
x = df["I_in"].to_numpy() + 1j * df["Q_in"].to_numpy()
y = df["I_out"].to_numpy() + 1j * df["Q_out"].to_numpy()
assert len(x) == len(y) == 38400

sl_train, sl_val, sl_test = slice(0, 23040), slice(23040, 30720), slice(30720, None)
x_tr, y_tr = x[sl_train], y[sl_train]
x_va, y_va = x[sl_val], y[sl_val]
x_te, y_te = x[sl_test], y[sl_test]
MAX_M = max(DEPTHS)


def nmse_db(actual, pred):
    return 10 * np.log10(np.sum(np.abs(actual - pred) ** 2) / np.sum(np.abs(actual) ** 2))


def acpr_db(sig, fs=FS, bw=BW):
    seg = 2048
    n = (len(sig) // seg) * seg
    win = np.hanning(seg)
    psd = np.mean([np.abs(np.fft.fftshift(np.fft.fft(sig[i:i + seg] * win))) ** 2
                   for i in range(0, n, seg)], axis=0)
    f = np.fft.fftshift(np.fft.fftfreq(seg, 1 / fs))
    main = psd[np.abs(f) <= bw / 2].sum()
    lo = psd[(f >= -1.5 * bw) & (f < -0.5 * bw)].sum()
    hi = psd[(f > 0.5 * bw) & (f <= 1.5 * bw)].sum()
    return 10 * np.log10(max(lo, hi) / main)


# ------------------------------------------------------------ data prep
# Scale so the network sees values around 1. Inverted before every metric.
SX = np.max(np.abs(x_tr))
SY = np.max(np.abs(y_tr))
# Linear gain (in scaled units) - the network learns only what the linear part misses.
xs_tr, ys_tr = x_tr / SX, y_tr / SY
G = np.vdot(xs_tr, ys_tr) / np.vdot(xs_tr, xs_tr)


def window_features(sig_scaled, M):
    """Row k <-> sample index k+M-1 (same alignment as mp_features in fit_baselines.py).
    Columns: for delay m = 0..M-1 -> [I, Q, |x|] of sample (k+M-1-m)."""
    N = len(sig_scaled)
    cols = []
    for m in range(M):
        d = sig_scaled[M - 1 - m:N - m]
        cols += [d.real, d.imag, np.abs(d)]
    return np.column_stack(cols).astype(np.float32)


def make_set(x_split, y_split, M):
    X = window_features(x_split / SX, M)
    cur = (x_split / SX)[M - 1:]                    # current sample, for the linear part
    resid = y_split[M - 1:] / SY - G * cur          # what the network must learn
    T = np.column_stack([resid.real, resid.imag]).astype(np.float32)
    return X, T, cur


class DelayMLP(nn.Module):
    def __init__(self, n_in, hidden=HIDDEN):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_in, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
            nn.Linear(hidden, 2),
        )

    def forward(self, v):
        return self.net(v)


def predict(model, X, cur):
    """Full PA output estimate in ORIGINAL units, length len(X)."""
    model.eval()
    with torch.no_grad():
        r = model(torch.from_numpy(X)).numpy()
    return (G * cur + (r[:, 0] + 1j * r[:, 1])) * SY


def train_one(M, seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    Xtr, Ttr, _ = make_set(x_tr, y_tr, M)
    Xva, Tva, _ = make_set(x_va, y_va, M)
    Xtr_t, Ttr_t = torch.from_numpy(Xtr), torch.from_numpy(Ttr)
    Xva_t, Tva_t = torch.from_numpy(Xva), torch.from_numpy(Tva)

    model = DelayMLP(Xtr.shape[1])
    n_params = sum(p.numel() for p in model.parameters())
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=8)
    loss_fn = nn.MSELoss()

    best_val, best_state, bad = float("inf"), None, 0
    hist = {"train": [], "val": []}
    for epoch in range(EPOCHS):
        model.train()
        perm = torch.randperm(len(Xtr_t))
        tot = 0.0
        for i in range(0, len(perm), BATCH):
            idx = perm[i:i + BATCH]
            opt.zero_grad()
            loss = loss_fn(model(Xtr_t[idx]), Ttr_t[idx])
            loss.backward()
            opt.step()
            tot += loss.item() * len(idx)
        model.eval()
        with torch.no_grad():
            val = loss_fn(model(Xva_t), Tva_t).item()
        hist["train"].append(tot / len(perm))
        hist["val"].append(val)
        sched.step(val)
        if val < best_val - 1e-9:
            best_val, best_state, bad = val, copy.deepcopy(model.state_dict()), 0
        else:
            bad += 1
            if bad >= PATIENCE:
                break
    model.load_state_dict(best_state)
    return model, n_params, best_val, hist


def evaluate(model, M):
    """Score on the SAME samples as the baselines: drop first MAX_M-1 samples of each split."""
    out = {}
    for name, xs, ys in (("val", x_va, y_va), ("test", x_te, y_te)):
        X, _, cur = make_set(xs, ys, M)
        pred = predict(model, X, cur)[MAX_M - M:]
        target = ys[MAX_M - 1:]
        out[name] = (nmse_db(target, pred), acpr_db(pred))
    return out


if __name__ == "__main__":
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    (root / "results").mkdir(exist_ok=True)
    (root / "figures").mkdir(exist_ok=True)
    rows = []
    for M in DEPTHS:
        runs = []
        for seed in SEEDS:
            model, n_params, best_val, hist = train_one(M, seed)
            ev = evaluate(model, M)
            runs.append({"seed": seed, "val_nmse": ev["val"][0], "test_nmse": ev["test"][0],
                         "test_acpr": ev["test"][1], "hist": hist})
            print(f"M={M} seed={seed} params={n_params} "
                  f"val NMSE={ev['val'][0]:.2f} dB  test NMSE={ev['test'][0]:.2f} dB  "
                  f"test ACPR={ev['test'][1]:.2f} dB  epochs={len(hist['val'])}")
        test = np.array([r["test_nmse"] for r in runs])
        best = min(runs, key=lambda r: r["val_nmse"])     # choose seed by VALIDATION only
        rows.append({
            "Model": f"MLP (M={M})", "Coefficients": n_params,
            "Test_NMSE_dB": best["test_nmse"],
            "Test_NMSE_mean_dB": test.mean(), "Test_NMSE_std_dB": test.std(),
            "Test_ACPR_dB": best["test_acpr"],
        })
        plt.figure(figsize=(6, 4))
        plt.plot(best["hist"]["train"], label="train")
        plt.plot(best["hist"]["val"], label="validation")
        plt.yscale("log"); plt.xlabel("epoch"); plt.ylabel("MSE (scaled units)")
        plt.title(f"MLP training curve, M={M} (best-validation seed)")
        plt.legend(); plt.grid(alpha=0.3); plt.tight_layout()
        plt.savefig(root / "figures" / f"mlp_loss_M{M}.png", dpi=200)
        plt.close()

    mlp_df = pd.DataFrame(rows)
    mlp_df.to_csv(root / "results" / "mlp_comparison.csv", index=False)
    print("\n", mlp_df.to_string(index=False))

    base = root / "results" / "baseline_comparison.csv"
    if base.exists():
        allm = pd.concat([pd.read_csv(base), mlp_df], ignore_index=True)
        allm.to_csv(root / "results" / "all_models_comparison.csv", index=False)
        print("\nCOMBINED TABLE\n", allm.to_string(index=False))
