"""Step 3: comparison figures for the PA-modelling project (DPA_200MHz).

Run from the repo root AFTER fit_baselines/baselines_v2.py and train_mlp.py:
    python make_figures.py

Makes (in figures/):
  fig_am_am.png         AM/AM: measured vs memory polynomial vs MLP (M=5)
  fig_am_pm.png         AM/PM: same comparison
  fig_spectrum.png      Output power spectrum: measured vs models
  fig_nmse_vs_params.png  accuracy vs model size (all models)
  fig_nmse_vs_depth.png   NMSE vs memory depth (polynomial M=1..9, MLP from CSV)
The MLP (M=5) is retrained here exactly as in train_mlp.py (same seeds, best seed by
validation). If PyTorch is missing, the MLP curves are skipped with a warning.
"""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

root = Path(__file__).resolve().parent
(root / "figures").mkdir(exist_ok=True)

ORDERS = (1, 3, 5)
FS, BW = 800e6, 200e6          # same ASSUMPTION as the other scripts
MP_DEPTHS = (1, 3, 5, 7, 9)
BEST_M = 5                     # depth shown in the AM/AM, AM/PM and spectrum figures
DROP = max(MP_DEPTHS) - 1      # common samples dropped so every model is scored identically

df = pd.read_csv(root / "datasets" / "dpa_200.csv")
x = df["I_in"].to_numpy() + 1j * df["Q_in"].to_numpy()
y = df["I_out"].to_numpy() + 1j * df["Q_out"].to_numpy()
x_tr, y_tr = x[:23040], y[:23040]
x_te, y_te = x[30720:], y[30720:]


def nmse_db(a, p):
    return 10 * np.log10(np.sum(np.abs(a - p) ** 2) / np.sum(np.abs(a) ** 2))


def mp_features(sig, memory, orders=ORDERS):
    N = len(sig)
    cols = []
    for m in range(memory):
        d = sig[memory - 1 - m:N - m]
        for p in orders:
            cols.append(d * np.abs(d) ** (p - 1))
    return np.column_stack(cols)


def mp_predict_test(M):
    c, *_ = np.linalg.lstsq(mp_features(x_tr, M), y_tr[M - 1:], rcond=None)
    return (mp_features(x_te, M) @ c)[DROP - (M - 1):], len(c)


def psd(sig, seg=2048):
    n = (len(sig) // seg) * seg
    w = np.hanning(seg)
    p = np.mean([np.abs(np.fft.fftshift(np.fft.fft(sig[i:i + seg] * w))) ** 2
                 for i in range(0, n, seg)], axis=0)
    return np.fft.fftshift(np.fft.fftfreq(seg, 1 / FS)) / 1e6, p   # MHz


# ------------------------------------------------ models on identical test samples
x_c = x_te[DROP:]
y_c = y_te[DROP:]
models = {}                                   # name -> predicted output
mp_pred, mp_params = mp_predict_test(BEST_M)
models[f"Memory poly (M={BEST_M})"] = mp_pred

mlp_ok = True
try:
    import train_mlp as T           # same split/scaling/model as step 2 (main block is guarded)
    best = None
    for seed in T.SEEDS:
        model, n_params, best_val, _ = T.train_one(BEST_M, seed)
        if best is None or best_val < best[0]:
            best = (best_val, model, n_params)
    _, model, mlp_params = best
    X, _, cur = T.make_set(x_te, y_te, BEST_M)
    mlp_full = T.predict(model, X, cur)                     # starts at sample BEST_M-1
    models[f"MLP (M={BEST_M})"] = mlp_full[DROP - (BEST_M - 1):]
except ImportError as e:
    mlp_ok = False
    print("WARNING: PyTorch/train_mlp not available -> MLP curves skipped:", e)

colors = {f"Memory poly (M={BEST_M})": "tab:blue", f"MLP (M={BEST_M})": "tab:red"}
for k, v in models.items():
    print(f"{k}: NMSE on common test samples = {nmse_db(y_c, v):.2f} dB")

# ------------------------------------------------ AM/AM and AM/PM
amp_in = np.abs(x_c)
idx = np.arange(0, len(x_c), 2)                # light thinning for readability
for tag, fn, ylabel in (
        ("am_am", lambda s, xi: np.abs(s), "|output| (normalised units)"),
        ("am_pm", lambda s, xi: np.degrees(np.angle(s * np.conj(xi))), "phase shift (deg)")):
    fig, axes = plt.subplots(1, len(models), figsize=(5.2 * len(models), 4.2), sharey=True, squeeze=False)
    for ax, (name, pred) in zip(axes[0], models.items()):
        ax.scatter(amp_in[idx], fn(y_c, x_c)[idx], s=3, alpha=0.25, color="gray", label="measured PA")
        ax.scatter(amp_in[idx], fn(pred, x_c)[idx], s=3, alpha=0.25, color=colors[name], label=name)
        ax.set_xlabel("|input| (normalised)")
        ax.grid(alpha=0.3)
        ax.legend(markerscale=4, loc="best")
    axes[0][0].set_ylabel(ylabel)
    fig.suptitle(f"{tag.upper().replace('_', '/')} - DPA_200MHz, test set")
    fig.tight_layout()
    fig.savefig(root / "figures" / f"fig_{tag}.png", dpi=200)
    plt.close(fig)

# ------------------------------------------------ spectrum
f, p_meas = psd(y_c)
norm = p_meas.max()
plt.figure(figsize=(7.5, 4.5))
plt.plot(f, 10 * np.log10(p_meas / norm), color="gray", lw=2, label="measured PA output")
for name, pred in models.items():
    plt.plot(f, 10 * np.log10(psd(pred)[1] / norm), color=colors[name], lw=1, label=name)
plt.axvline(-BW / 2e6, ls=":", color="k"); plt.axvline(BW / 2e6, ls=":", color="k")
plt.xlabel("frequency offset (MHz)"); plt.ylabel("power spectral density (dB, rel. peak)")
plt.title("Output spectrum (dotted = signal band edges)  [assumes fs = 800 MHz]")
plt.grid(alpha=0.3); plt.legend(); plt.tight_layout()
plt.savefig(root / "figures" / "fig_spectrum.png", dpi=200); plt.close()

# ------------------------------------------------ accuracy vs size (from result CSVs)
res = root / "results" / "all_models_comparison.csv"
if res.exists():
    t = pd.read_csv(res).dropna(subset=["Test_NMSE_dB"])
    t = t[t["Coefficients"] > 0]
    plt.figure(figsize=(7, 4.5))
    for _, r in t.iterrows():
        is_mlp = r["Model"].startswith("MLP")
        plt.scatter(r["Coefficients"], r["Test_NMSE_dB"], s=60,
                    color="tab:red" if is_mlp else "tab:blue")
        plt.annotate(r["Model"], (r["Coefficients"], r["Test_NMSE_dB"]),
                     textcoords="offset points", xytext=(5, 5), fontsize=8)
    plt.xscale("log"); plt.xlabel("number of parameters / coefficients (log)")
    plt.ylabel("test NMSE (dB)  - lower is better")
    plt.title("Accuracy vs model size (blue: polynomial, red: MLP)")
    plt.grid(alpha=0.3, which="both"); plt.tight_layout()
    plt.savefig(root / "figures" / "fig_nmse_vs_params.png", dpi=200); plt.close()
else:
    print("results/all_models_comparison.csv not found -> skipped fig_nmse_vs_params.png")

# ------------------------------------------------ NMSE vs memory depth
mp_curve = []
for M in MP_DEPTHS:
    pred, _ = mp_predict_test(M)
    mp_curve.append(nmse_db(y_c, pred))
print("Polynomial NMSE vs depth:", dict(zip(MP_DEPTHS, np.round(mp_curve, 2))))
plt.figure(figsize=(6.5, 4.3))
plt.plot(MP_DEPTHS, mp_curve, "o-", color="tab:blue", label="memory polynomial")
mlp_csv = root / "results" / "mlp_comparison.csv"
if mlp_csv.exists():
    m = pd.read_csv(mlp_csv)
    depths = [int(s.split("M=")[1].rstrip(")")) for s in m["Model"]]
    plt.errorbar(depths, m["Test_NMSE_mean_dB"], yerr=m["Test_NMSE_std_dB"], fmt="s-",
                 color="tab:red", capsize=3, label="MLP (mean ± std over 3 seeds)")
plt.xlabel("memory depth M"); plt.ylabel("test NMSE (dB)")
plt.title("Effect of memory depth"); plt.xticks(MP_DEPTHS); plt.grid(alpha=0.3); plt.legend()
plt.tight_layout(); plt.savefig(root / "figures" / "fig_nmse_vs_depth.png", dpi=200); plt.close()

print("\nDone. Figures written to figures/:", sorted(p.name for p in (root / "figures").glob("fig_*.png")))
