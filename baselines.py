"""Baseline PA models on DPA_200MHz: linear, memoryless polynomial, memory polynomial.

Model selection (memory depth) uses the validation split only;
the final comparison is reported on the test split, on a common set of samples.
"""
from pathlib import Path
import numpy as np
import pandas as pd

# ---------------------------------------------------------------- config
ORDERS = (1, 3, 5)
DEPTHS = (1, 3, 5)
SELECTED_M = 3          # overwritten below by best validation depth
FS = 800e6              # ASSUMPTION: sampling rate for the 200 MHz signal (OpenDPD convention). Confirm.
BW = 200e6              # signal bandwidth
# ----------------------------------------------------------------------

root = Path(__file__).resolve().parent
df = pd.read_csv(root / "datasets" / "dpa_200.csv")
x = df["I_in"].to_numpy() + 1j * df["Q_in"].to_numpy()
y = df["I_out"].to_numpy() + 1j * df["Q_out"].to_numpy()
assert len(x) == len(y) == 38400

# OpenDPD Studio contiguous split (60/20/20)
sl_train, sl_val, sl_test = slice(0, 23040), slice(23040, 30720), slice(30720, None)
x_tr, y_tr = x[sl_train], y[sl_train]
x_va, y_va = x[sl_val], y[sl_val]
x_te, y_te = x[sl_test], y[sl_test]


def nmse_db(actual, pred):
    return 10 * np.log10(np.sum(np.abs(actual - pred) ** 2) / np.sum(np.abs(actual) ** 2))


def acpr_db(sig, fs=FS, bw=BW):
    """Worst-case adjacent-channel power ratio (dB) from a Welch-style PSD."""
    seg = 2048
    n = (len(sig) // seg) * seg
    win = np.hanning(seg)
    psd = np.mean(
        [np.abs(np.fft.fftshift(np.fft.fft(sig[i:i + seg] * win))) ** 2
         for i in range(0, n, seg)], axis=0)
    f = np.fft.fftshift(np.fft.fftfreq(seg, 1 / fs))
    main = psd[np.abs(f) <= bw / 2].sum()
    lo = psd[(f >= -1.5 * bw) & (f < -0.5 * bw)].sum()
    hi = psd[(f > 0.5 * bw) & (f <= 1.5 * bw)].sum()
    return 10 * np.log10(max(lo, hi) / main)


def mp_features(sig, memory, orders=ORDERS):
    """Row k corresponds to sample index k+memory-1 of `sig` (uses sig[k+memory-1-m], m=0..memory-1)."""
    N = len(sig)
    cols = []
    for m in range(memory):
        d = sig[memory - 1 - m:N - m]
        for p in orders:
            cols.append(d * np.abs(d) ** (p - 1))
    return np.column_stack(cols)


def fit_mp(sig, tgt, memory):
    coeff, *_ = np.linalg.lstsq(mp_features(sig, memory), tgt[memory - 1:], rcond=None)
    return coeff


def predict_mp(sig, coeff, memory):
    return mp_features(sig, memory) @ coeff


# --------------------------------------------- memory-depth selection (validation)
max_m = max(DEPTHS)
rows = []
coeffs = {}
for M in DEPTHS:
    c = fit_mp(x_tr, y_tr, M)
    coeffs[M] = c
    # score on the SAME validation samples for every depth (drop first max_m-1)
    pred = predict_mp(x_va, c, M)[max_m - M:]
    val = nmse_db(y_va[max_m - 1:], pred)
    rows.append({"Depth_M": M, "Coefficients": len(c), "Val_NMSE_dB": val})
depth_df = pd.DataFrame(rows)
best_M = int(depth_df.loc[depth_df["Val_NMSE_dB"].idxmin(), "Depth_M"])
print("MEMORY DEPTH (validation)\n", depth_df.to_string(index=False))
print("Selected depth:", best_M)

# --------------------------------------------- final test comparison (common samples)
y_t = y_te[max_m - 1:]
G = np.vdot(x_tr, y_tr) / np.vdot(x_tr, x_tr)
results = [("Linear", 1, G * x_te[max_m - 1:])]
results.append(("Memoryless poly", len(ORDERS),
                (mp_features(x_te, 1) @ coeffs[1])[max_m - 1:]))
for M in DEPTHS:
    if M == 1:
        continue
    results.append((f"Memory poly (M={M})", len(coeffs[M]),
                    predict_mp(x_te, coeffs[M], M)[max_m - M:]))

summary = pd.DataFrame({
    "Model": [r[0] for r in results],
    "Coefficients": [r[1] for r in results],
    "Test_NMSE_dB": [nmse_db(y_t, r[2]) for r in results],
    "Test_ACPR_dB": [acpr_db(r[2]) for r in results],
})
summary.loc[len(summary)] = ["Measured PA output (reference)", 0, np.nan, acpr_db(y_t)]
print("\nTEST COMPARISON (identical samples)\n", summary.to_string(index=False))

out = root / "results"
out.mkdir(exist_ok=True)
summary.to_csv(out / "baseline_comparison.csv", index=False)
depth_df.to_csv(out / "memory_depth_validation.csv", index=False)
print("\nSaved results/baseline_comparison.csv and results/memory_depth_validation.csv")
