import pandas as pd
from pathlib import Path

data_path = Path(__file__).resolve().parent / "datasets" / "dpa_200.csv"

df = pd.read_csv(data_path)

print("Dataset shape:", df.shape)
print("\nColumn names:")
print(df.columns.tolist())

print("\nFirst 5 rows:")
print(df.head())

print("\nMissing values:")
print(df.isnull().sum())

#visualizing the given data
import numpy as np
import matplotlib.pyplot as plt

# Reconstruct complex input and output signals
x = df["I_in"].to_numpy() + 1j * df["Q_in"].to_numpy()
y = df["I_out"].to_numpy() + 1j * df["Q_out"].to_numpy()

# Calculate amplitudes
input_amp = np.abs(x)
output_amp = np.abs(y)

# Plot AM/AM relationship
plt.figure(figsize=(8, 5))

plt.scatter(
    input_amp,
    output_amp,
    s=2,
    alpha=0.25,
    color="steelblue"
)

plt.xlabel("Input Signal Magnitude |x[n]| (normalized)")
plt.ylabel("Output Signal Magnitude |y[n]| (normalized units)")

plt.title("AM/AM Characteristic of DPA_200MHz Power Amplifier")

plt.grid(True, alpha=0.4)
plt.tight_layout()

figures_dir = Path(__file__).resolve().parent / "figures"
figures_dir.mkdir(parents=True, exist_ok=True)

plt.savefig(
    figures_dir / "AM_AM_DPA_200MHz.png",
    dpi=300,
    bbox_inches="tight"
)