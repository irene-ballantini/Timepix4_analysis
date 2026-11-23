# Calculation and visualization of SNR for uncalibrated and calibrated images
# Comparison with theoretical trend

# You can change the mode by writing "TOT" or "CH" in the mode variable below

import numpy as np
import matplotlib.pyplot as plt
import glob
from natsort import natsorted

from scipy.optimize import curve_fit
from sklearn.metrics import r2_score

import os
from pathlib import Path

import pandas as pd


def mean(img):
    img_mean = np.mean(img[2:446,2:510])
    return img_mean

def std(img):
    img_std = np.std(img[2:446,2:510])
    return img_std

def img_snr(img):
    SNR = mean(img)/np.std(img[2:446,2:510])
    return SNR

width, height = 512, 448

# Decomment the mode you want to analyze (CH or TOT)
#mode = "CH"  
mode = "TOT" 

path_input_no_cal = f"calibrazione_14lug2026_CLUSTERING/{mode}/image_raw_16bit_{mode}/*.raw"
path_input_cal = f"CLUSTERED_IMAGES/{mode}/calibrated_series_400V_{mode}/*.raw"

files_no_cal = glob.glob(path_input_no_cal)
sorted_files_no_cal = natsorted(files_no_cal)

files_cal = glob.glob(path_input_cal)
sorted_files_cal = natsorted(files_cal)

SNR_no_cal = np.empty(len(files_no_cal))
N_no_cal = np.empty(len(files_no_cal))
std_no_cal = np.empty(len(files_no_cal))

SNR_cal = np.empty(len(files_cal))
N_cal = np.empty(len(files_cal))
std_cal = np.empty(len(files_no_cal))

for i, file_path in enumerate(sorted_files_no_cal):
    img = np.fromfile(file_path, dtype=np.uint16).reshape((height, width))
    N_no_cal[i] = mean(img)
    std_no_cal[i] = std(img)
    SNR_no_cal[i] = img_snr(img)

for i, file_path in enumerate(sorted_files_cal):
    img = np.fromfile(file_path, dtype=np.uint16).reshape((height, width))
    N_cal[i] = mean(img)
    std_cal[i] = std(img)
    SNR_cal[i] = img_snr(img)

# Found that 70 sec got let counts than expected (outlier) -> so removed in case
N_no_cal = np.delete(N_no_cal,6)
SNR_no_cal = np.delete(SNR_no_cal,6)
N_cal = np.delete(N_cal,6)
SNR_cal = np.delete(SNR_cal,6)

print(f"SNR no calibration: {SNR_no_cal}")
print(f"SNR with calibration: {SNR_cal}")

def fun(N, a, exp):
    return a*N**(exp)

def fit(N, SNR, color, label_prefix, scatter_label):
    popt, pcov = curve_fit(fun, N, SNR, p0=[1,0.5])
    a_opt, exp_opt = popt    #popt contains the optimal values

    SNR_fit = fun(N, *popt)

    # R^2
    r2 = r2_score(SNR, SNR_fit)

    errore_a = np.sqrt(pcov[0, 0])    #or  ->  errore_a, errore_exp = np.sqrt(np.diag(pcov))
    errore_exp = np.sqrt(pcov[1, 1]) 

    plt.scatter(N, SNR, color=color, label=scatter_label, s=10)
    ###########################################    
    N_line = np.linspace(N.min(), N.max(), 300)
    SNR_line = fun(N_line, *popt)
    ###########################################
    label_testo = (rf'{label_prefix}: $y = ({a_opt:.2f} \pm {errore_a:.2f}) \cdot '
               rf'x^{{{exp_opt:.2f} \pm {errore_exp:.2f}}}$'
                   f'\n'
                   rf'$(R^2 = {r2: .4f})$')
    plt.plot(N_line, SNR_line, color=color, linestyle=":", label=label_testo)

    plt.xlabel('Average counts per pixel on ROI (counts)')
    plt.ylabel('Mean/devst')
    plt.title(f'SNR on clustered images ({mode})')
    #plt.legend(loc='best', fontsize='small')
    plt.legend(fontsize="small", loc=(0.48,0.15))
    plt.grid(True, linestyle='--', alpha=0.6)


plt.figure(figsize=(9, 6))
n_range = np.linspace(min(N_no_cal.min(), N_cal.min()), max(N_no_cal.max(), N_cal.max()), 100)

# Theoretical curve
theoretical_y = np.sqrt(n_range)

# Standard deviation derived from the Poisson propagation (sigma = 0.5)
sigma_band = 0.5 


plt.plot(n_range, theoretical_y, color='black', label=r'Theoretical Reference ($\sqrt{N}$)', alpha=0.7)
#plt.plot(N_cal, np.sqrt(N_cal), '--', color='gray', label=r'Theoretical trend $(\sqrt{N})$')
fit(N_cal, SNR_cal,"red", "Power (calibrated image)", "Calibrated image")
fit(N_no_cal, SNR_no_cal,"blue", "Power (original image)", "Original image")
plt.legend(loc="upper left", fontsize="small")

plt.savefig(f"CLUSTERED_IMAGES/{mode}/calibrated_series_400V_{mode}/SNR_calibrated_vs_original_clustering_{mode}_no70sec.png")
plt.show()

# DataFrame creation with pandas
df_data = pd.DataFrame({
    'N_no_cal': N_no_cal,
    'SNR_no_cal': SNR_no_cal,
    'N_cal': N_cal,
    'SNR_cal': SNR_cal
})

# Defining the output path for the Excel file
excel_path = f"CLUSTERED_IMAGES/{mode}/calibrated_series_400V_{mode}/SNR_data_{mode}.xlsx"

# Saving in Excel
df_data.to_excel(excel_path, index=False)

print(f"Data saved in: {excel_path}")


