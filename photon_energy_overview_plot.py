import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

# 1. Path Configuration
specification = "photonEnergies_Ru_v3"
datapath = os.path.join(
    "C:\\Users\\Samuel_Uni\\psi-sim-dmr-nanocatalysts\\Images_and_Data",
    specification,
)

final_filename = f"{specification}_analysis.csv"
final_datapath = f"C:\\Users\\Samuel_Uni\\psi-sim-dmr-nanocatalysts\\Images_and_Data\\{final_filename}"

raw_inputs = []
raw_readbacks = []

# 2. Extract Data from all CSV files
csv_files = [f for f in os.listdir(datapath) if f.endswith(".csv")]

for file in csv_files:
    filepath = os.path.join(datapath, file)
    try:
        data = np.loadtxt(filepath, delimiter=",", skiprows=2)
        if data.ndim == 1:
            data = data.reshape(1, -1)

        # Col 0: Readback Energy, Col 1: Input Energy
        valid = ~np.isnan(data[:, 0]) & ~np.isnan(data[:, 1])
        raw_readbacks.extend(data[valid, 0])
        raw_inputs.extend(data[valid, 1])
    except Exception as e:
        print(f"Error reading {file}: {e}")

raw_inputs = np.array(raw_inputs)
raw_readbacks = np.array(raw_readbacks)

# 3. Combine into DataFrame & Calculate Group Statistics (Mean, Std, SEM)
df = pd.DataFrame({"Input": raw_inputs, "Readback": raw_readbacks})

df_grouped = (
    df.groupby("Input")["Readback"]
    .agg(mean="mean", std="std", sem="sem", count="count")
    .reset_index()
)

# 4. Fit linear regression on the averages
res = stats.linregress(df_grouped["Input"], df_grouped["mean"])
fit_line = res.slope * df_grouped["Input"] + res.intercept

# 5. Calculate average error bar across all energy steps
avg_error_std = df_grouped["std"].fillna(0).mean()
avg_error_sem = df_grouped["sem"].fillna(0).mean()

# 6. Plot raw points, averages with error bars, and fit
plt.figure(figsize=(9, 5))

# Raw data scatter
plt.scatter(
    df["Input"],
    df["Readback"],
    color="lightgray",
    alpha=0.5,
    label="Raw Readbacks",
)

# Average per energy level with Error Bars (±1 Std Dev)
plt.errorbar(
    df_grouped["Input"],
    df_grouped["mean"],
    yerr=df_grouped["std"],
    fmt="o",
    color="blue",
    ecolor="lightblue",
    elinewidth=1.5,
    capsize=3,
    markersize=5,
    label=f"Average ±1 Std Dev (Avg: ±{avg_error_std:.4f} eV)",
)

# Linear Regression line
plt.plot(
    df_grouped["Input"],
    fit_line,
    "r--",
    linewidth=2,
    label=f"Fit: y = {res.slope:.4f}x + {res.intercept:.4f}, R² = {res.rvalue**2:.4f}",
)

plt.xlabel("Input Energy (eV)")
plt.xlim(459, 467)
plt.ylim(455,470)
plt.ylabel("Readback Energy (eV)")
plt.title("Photon Energy Analysis with Error Bars")
plt.legend()
plt.grid(True, linestyle="--", alpha=0.6)
plt.savefig(
    os.path.join(datapath, f"Photon_Energy_Analysis_{specification}.png"),
    dpi=300,
)
plt.show()

# 7. Statistics & Error Output
print(f"Processed Raw Points: {len(raw_inputs)}")
print(f"Unique Input Energies: {len(df_grouped)}")
print(f"Average Error Bar (Standard Deviation): ±{avg_error_std:.6f} eV")
print(f"Average Error Bar (Standard Error - SEM): ±{avg_error_sem:.6f} eV")
print(f"Slope (Gain Error): {res.slope:.6f}")
print(f"Intercept (Constant Offset): {res.intercept:.6f} eV")
print(f"R-squared (R^2): {res.rvalue**2:.6f}")
print(f"Standard Error of Estimate: {res.stderr:.6f}")