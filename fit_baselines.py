
from pathlib import Path
import numpy as np
import pandas as pd

# 1. Load dataset
root = Path(__file__).resolve().parent
df = pd.read_csv(root / "datasets" / "dpa_200.csv")

x = df["I_in"].to_numpy() + 1j * df["Q_in"].to_numpy()
y = df["I_out"].to_numpy() + 1j * df["Q_out"].to_numpy()

assert len(x) == len(y) == 38400

# 2. Preserve the OpenDPD Studio contiguous split
x_train, y_train = x[:23040], y[:23040]
x_val, y_val = x[23040:30720], y[23040:30720]
x_test, y_test = x[30720:], y[30720:]


def nmse_db(actual, predicted):
    error_power = np.sum(np.abs(actual - predicted) ** 2)
    signal_power = np.sum(np.abs(actual) ** 2)
    return 10 * np.log10(error_power / signal_power)


# 3. Linear least-squares baseline
G = np.vdot(x_train, y_train) / np.vdot(x_train, x_train)

print("Estimated complex gain:", G)

linear_pred = G * x_test
print("Linear test NMSE:", nmse_db(y_test, linear_pred), "dB")


# 4. Memory Polynomial feature construction
def mp_features(signal, memory=3, orders=(1, 3, 5)):
    # Each row uses current and previous input samples.
    # First (memory-1) rows do not have enough history.
    N = len(signal)
    features = []

    for m in range(memory):
        delayed = signal[memory - 1 - m:N - m]

        for p in orders:
            features.append(delayed * np.abs(delayed) ** (p - 1))

    return np.column_stack(features)


# 5. Fit MP coefficients using training samples
M = 3
Phi_train = mp_features(x_train, memory=M)

# Align targets with rows of the feature matrix
target_train = y_train[M - 1:]

coefficients, residuals, rank, singular_values = np.linalg.lstsq(
    Phi_train, target_train, rcond=None
)

# 6. Evaluate the MP on test samples
Phi_test = mp_features(x_test, memory=M)
target_test = y_test[M - 1:]

mp_pred = Phi_test @ coefficients

print("Memory Polynomial test NMSE:",
      nmse_db(target_test, mp_pred), "dB")

print("MP complex coefficients:", len(coefficients))
print("MP matrix rank:", rank)

# 7. Compare both models on the same test indices
linear_pred_aligned = G * x_test[M - 1:]

print("\nFINAL COMPARISON")
print("Linear:", nmse_db(target_test, linear_pred_aligned), "dB")
print("MP:", nmse_db(target_test, mp_pred), "dB")

# 8. Save results
results_dir = root / "results"
results_dir.mkdir(exist_ok=True)

summary = pd.DataFrame({
    "Model": ["Linear", "Memory Polynomial"],
    "Test_NMSE_dB": [
        nmse_db(target_test, linear_pred_aligned),
        nmse_db(target_test, mp_pred)
    ]
})

summary.to_csv(results_dir / "baseline_comparison.csv", index=False)
print("\nResults saved to results/baseline_comparison.csv")

# Memoryless polynomial baseline
Phi_memoryless = mp_features(
    x_train,
    memory=1,
    orders=(1, 3, 5)
)

coeff_memoryless, *_ = np.linalg.lstsq(
    Phi_memoryless,
    y_train,
    rcond=None
)

Phi_test_memoryless = mp_features(
    x_test,
    memory=1,
    orders=(1, 3, 5)
)

pred_memoryless = Phi_test_memoryless @ coeff_memoryless

print(
    "Memoryless Polynomial NMSE:",
    nmse_db(y_test, pred_memoryless),
    "dB"
)
# Memoryless polynomial baseline
Phi_memoryless = mp_features(
    x_train,
    memory=1,
    orders=(1, 3, 5)
)

coeff_memoryless, *_ = np.linalg.lstsq(
    Phi_memoryless,
    y_train,
    rcond=None
)

Phi_test_memoryless = mp_features(
    x_test,
    memory=1,
    orders=(1, 3, 5)
)

pred_memoryless = Phi_test_memoryless @ coeff_memoryless

print(
    "Memoryless Polynomial NMSE:",
    nmse_db(y_test, pred_memoryless),
    "dB"
)

print("\nMEMORY DEPTH EXPERIMENT")

for M in [1, 3, 5]:

    # Training
    Phi_train = mp_features(
        x_train,
        memory=M,
        orders=(1, 3, 5)
    )

    target_train = y_train[M-1:]

    coeff, *_ = np.linalg.lstsq(
        Phi_train,
        target_train,
        rcond=None
    )

    # Validation
    Phi_val = mp_features(
        x_val,
        memory=M,
        orders=(1, 3, 5)
    )

    target_val = y_val[M-1:]

    prediction = Phi_val @ coeff

    score = nmse_db(target_val, prediction)

    print(
        f"Memory={M}, "
        f"Validation NMSE={score:.4f} dB, "
        f"Coefficients={len(coeff)}"
    )