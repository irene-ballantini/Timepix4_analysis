import numpy as np
import matplotlib.pyplot as plt
import glob
from natsort import natsorted
import os

from scipy.optimize import curve_fit
from sklearn.metrics import r2_score

def mean(img):
    img_mean = np.mean(img[2:446,2:510])
    return img_mean

def linear_func(x, m, q):
    return m * x + q

width, height = 512, 448

# Done for not clustered images with/without calibration, and for clustered images (TOT) )with/without calibration
# path changes depending on the type of images you want to analyze
path_input = "CLUSTERED_IMAGES/TOT/calibrated_series_400V_TOT/*.raw"
output_folder = "plots/"

os.makedirs(output_folder, exist_ok=True)

files = glob.glob(path_input)
sorted_files = natsorted(files)

print(sorted_files)

signal = np.empty(len(files))
time = np.array([10, 20, 30 , 40, 50, 60, 70, 80])

for i, file_path in enumerate(sorted_files):
    img = np.fromfile(file_path, dtype=np.uint16).reshape((height, width))
    signal[i] = mean(img)

# Comment or decomment if you want to leave or remove the 70 sec (outlier)
#signal = np.delete(signal, 6)  
#time = np.delete(time, 6)

# --- LINEAR FIT ---
# popt contains the optimal parameters [m, q]
# pcov is the covariance matrix
popt, pcov = curve_fit(linear_func, time, signal)
m, q = popt

# Errors on paramenters (standard deviation)
perr = np.sqrt(np.diag(pcov))

# Calculation of the estimated signal from the fit and the R^2 coefficient
signal_pred = linear_func(time, *popt)
r_squared = r2_score(signal, signal_pred)

print(len(time), len(signal))
plt.figure(figsize=(8, 5))
plt.scatter(time, signal, s=15, color = "red", label="Data points")
# Fit (generated on a denser grid to make the line more fluid)
time_fit = np.linspace(min(time), max(time), 100)
plt.plot(time_fit, linear_func(time_fit, *popt), color="blue", linestyle="--", linewidth=1,
         label=rf"Linear Fit: $y = ({m:.2f} \pm {perr[0]:.2f})x + ({q:.2f} \pm {perr[1]:.2f})$" + f"\n$R^2$ = {r_squared:.4f}")
plt.title("Linearity")
plt.xlabel("Exposure time [s]")
plt.ylabel("Signal (mean pixel value)")
plt.grid(True, alpha=0.5, linestyle='--')
plt.legend()
plt.savefig(os.path.join(output_folder, "linearity_calibrated_clustered.png"))
plt.show()

# Print the results
print(f"Slope (m): {m:.4f} ± {perr[0]:.4f}")
print(f"Intercept (q): {q:.4f} ± {perr[1]:.4f}")
print(f"Determination coefficient (R^2): {r_squared:.6f}")
