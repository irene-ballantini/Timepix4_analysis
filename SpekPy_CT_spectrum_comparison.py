# Script for the empirical energy calibration

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os 
import glob
from natsort import natsorted
from pathlib import Path
from scipy.signal import savgol_filter

# To Change according to what you want to analize
cluster_size = "all"  # possibilities: 1, 2, 3, 4, 5, 234, 2345, "all"

offset = 0.0  # 0.0 or 0.5 or 1.0

####  Normalized SpekPy Spectrum

# Read Excel file
nome_file_SpekPy = 'SpekPy_CT_spectrum_comparison/spettro_SpekPy_40kV_1mA_1.5mmAl.xlsx'
df = pd.read_excel(
    nome_file_SpekPy, skiprows=3, usecols=[0, 1], header=None, names=['energy', 'fluence']
)

energy_spek = df['energy'].values
fluence_spek = df['fluence'].values

idx_max_spek = np.argmax(fluence_spek)
x_max_spek = energy_spek[idx_max_spek]

#### Non-calibrated CT spectrum

input_folder = Path("CLUSTERED_IMAGES/TOT/series_16bit/60sec")
counts_per_energy_bin_folder = input_folder / f"counts_per_energy_bin_TOT_{cluster_size}clustersize"
output_folder = Path(f"SpekPy_CT_spectrum_comparison/not_linear_calibration/offset_plus{offset}")

output_folder.mkdir(parents=True, exist_ok=True)

files = glob.glob(os.path.join(counts_per_energy_bin_folder, "*.raw"))
sorted_files = natsorted(files)

counts_per_bin = []
for file_path in sorted_files:
    data = np.fromfile(file_path, dtype=np.uint16).reshape((448, 512))
    ROI = data[2:446, 2:500]
    counts_per_bin.append(np.sum(ROI))

x = np.arange(1, len(counts_per_bin)*100, 100) 

counts_per_bin = np.array(counts_per_bin, dtype=float)

# ==========================================
# 3-POINT QUADRATIC CALIBRATION (TOT -> keV)
# ==========================================

threshold_noise = 0.01 * np.max(counts_per_bin)
valid_indices = np.where(counts_per_bin > threshold_noise)[0]

# 1 - Find TOT_zero (Low-energy point -> 10 keV)
idx_start_tot = valid_indices[0]
TOT_zero = x[idx_start_tot]
E_zero = 10.0  # Soglia imposta a 10 keV

# 2 - Find TOT_max (Peak)
idx_max_tot = np.argmax(counts_per_bin)
TOT_max = x[idx_max_tot]
E_max = x_max_spek  # keV del picco SpekPy (~25.5 keV)

# 3 - Find TOT inflection point
counts_smooth = savgol_filter(counts_per_bin, window_length=9, polyorder=3)
dy = np.gradient(counts_smooth, x)

# TOT inflection point (Minimum of the derivative after the peak)
idx_flesso_relativo = np.argmin(dy[idx_max_tot:])
idx_flesso_tot = idx_max_tot + idx_flesso_relativo

# Couple x_f, y_f and slope m
x_f = x[idx_flesso_tot]  # TOT at the inflection point
y_f = counts_smooth[idx_flesso_tot]  # Counts at the inflection points
m = dy[idx_flesso_tot] # Slope (derivative) at the inflection point

# Calculate the intercept of the tangent at y = 0
# $ y - y_f = m \cdot (x - x_f) $
# $ 0 - y_f = m \cdot (x_intercept - x_f) $  ->  $ x_intercept = x_f - \frac{y_f}{m} $
TOT_intercept = x_f - (y_f / m)

print(f"Cluster size: {cluster_size}")
print(f'Inflection point at TOT = {x_f:.1f}')
print(f'Tangent intercept (estimated cutoff) at TOT = {TOT_intercept:.1f}')

TOT_cutoff_tangente = TOT_intercept
E_cutoff = 40.0

# --- VECTORS FOR THE FIT ---
TOT_points = np.array([TOT_zero, TOT_max, TOT_cutoff_tangente])
E_points = np.array([E_zero, E_max, E_cutoff])

# 2nd degree fit: E = a*TOT^2 + b*TOT + c
coeffs = np.polyfit(TOT_points, E_points, deg=2)

print(f"Points used for the calibration:")
print(f"  - Low-energy tail: TOT = {TOT_zero}  --> {E_zero} keV")
print(f"  - Peak:  TOT = {TOT_max}  --> {E_max:.2f} keV")
print(f"  - High-energy tail:   TOT = {TOT_cutoff_tangente} --> {E_cutoff} keV")

# --- APPLICATION OF CALIBRATION ---
idx_cutoff_tot = valid_indices[-1]
x_tot_trimmed = x[: idx_cutoff_tot + 1]
tot_norm = counts_per_bin[: idx_cutoff_tot + 1] / np.max(
    counts_per_bin[: idx_cutoff_tot + 1]
)

# Convert TOT to keV using a second-degree polynomial
E_tot_calibrated = (
    coeffs[0] * (x_tot_trimmed**2) + coeffs[1] * x_tot_trimmed + coeffs[2]
) + offset

# ==========================================
# QUANTITATIVE VALIDATION OF CALIBRATION
# ==========================================

# CHECK OF THE TANGENT CUT-OFF AT THE INFLECTION POINT
# We calculate the inflection point and the tangent directly on the spectrum, which is already calibrated in keV
counts_smooth_cal = savgol_filter(tot_norm, window_length=9, polyorder=3)
dy_cal = np.gradient(counts_smooth_cal, E_tot_calibrated)

# Peak and inflection index on the calibrated axis
idx_max_cal = np.argmax(counts_smooth_cal)
idx_flesso_cal = idx_max_cal + np.argmin(dy_cal[idx_max_cal:])

# Coordinates at the inflection point (in keV and amplitude) and slope
E_f = E_tot_calibrated[idx_flesso_cal]
y_f_cal = counts_smooth_cal[idx_flesso_cal]
m_cal = dy_cal[idx_flesso_cal]

# Calculation of the intercept of the tangent line at y = 0
E_cutoff_tangente = E_f - (y_f_cal / m_cal)

print(
    f"\nCut-off estimated via the tangent at the inflection point: {E_cutoff_tangente:.2f} keV"
)

if 39.5 <= E_cutoff_tangente <= 40.5:
  print("Cut-off result: EXCELLENT (falls within the 40.0 ± 0.5 keV) range")
else:
  print(
      f"Cut-off result: WARNING (Deviation of"
      f" {abs(E_cutoff_tangente - 40.0):.2f} keV compared to 40.0 keV)"
  )

print(coeffs[0])

# Coefficients found by the fit: coeffs = [a, b, c]
a, b, c = coeffs

# Convert a single TOT value
tot_test = [2501]  # example of TOT value

tot_test = np.array(tot_test)
energy_keV = a * (tot_test**2) + b * tot_test + c + offset

y_values = np.interp(tot_test, x_tot_trimmed, tot_norm)

#print(f"Offset {offset}: TOT = {tot_test} corresponds to {energy_keV:.2f} keV")

# ==========================================
#    COMPARISON AND OVERLAY PLOTS
# ==========================================

plt.figure(figsize=(10, 6))

# Plot SpekPy
plt.plot(
    energy_spek,
    fluence_spek,
    label="SpekPy (40 kVp)",
    color="crimson",
    linewidth=2,
)

# Calibrated CT spectrum plot 
plt.plot(
    E_tot_calibrated,
    tot_norm,
    label="Spectrum CT (Tangent Calibrated TOT -> keV)",
    color="dodgerblue",
    linestyle="--",
    linewidth=2,
)

plt.scatter(energy_keV, y_values, color="black", marker="x", label="Test Points (TOT)")

# Add text lable near the points
for tot_val, e_val, y_val in zip(tot_test, energy_keV, y_values):
    plt.annotate(
        f"TOT {tot_val}\n({e_val:.1f} keV)",
        (e_val, y_val),
        textcoords="offset points",
        xytext=(0, 25),
        ha="center",
        fontsize=9,
        fontweight="bold",
        arrowprops=dict(arrowstyle="->", color="black", lw=0.8),
    )

# Graphics
plt.title(
    f"Comparison normalized spectra (SpekPy vs CT Experimental) [cluster: {cluster_size}]",
    fontsize=14,
)
plt.xlabel("Energy (keV)", fontsize=12)
plt.ylabel("Normalized Fluence (A.U.)", fontsize=12)
plt.xlim(0, 45)
plt.ylim(0, 1.15)
plt.grid(True, linestyle="--", alpha=0.6)
plt.legend(fontsize=9, loc="lower left")
#plt.legend(loc="best")
#plt.tight_layout()

# Save and show
os.makedirs(output_folder, exist_ok=True)
plt.savefig(os.path.join(output_folder, f"Spectrum_Overlay_Comparison_{cluster_size}clustersize_not_linear_plus{offset}_check.png"))
plt.show()