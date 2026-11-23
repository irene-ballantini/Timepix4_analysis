# This script calculates the DQE for different shutter times and put them in the same plot, to compare them
# Done both for calibrated and uncalibrated images

from scipy.fft import fftn, fftshift
import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.signal import windows
from scipy.signal import savgol_filter
from MTF_module import MTFCalculator    #custom class implemented in the script MTF_module.py

n_seconds = [10, 30, 50]
calibration = True # Set to True for DQE calculation with calibration, False for without calibration

if calibration == True:
    DQE_folder = 'CLUSTERED_IMAGES/TOT/DQE/DQE_cal/'          # Folder for DQE calculation with calibration
    label = " - calibrated images"
elif calibration == False:
    DQE_folder = 'CLUSTERED_IMAGES/TOT/DQE/DQE_no_cal/'      # Folder for DQE calculation without calibration    
    label = " - not calibrated images"
else:
    raise ValueError("Invalid value for 'calibration'. Please set it to True or False.")

output_folder_FT = os.path.join(DQE_folder, 'FT_domain/')
mtf_folder = os.path.join(DQE_folder, 'MTF_plot/')
nps_folder = os.path.join(DQE_folder, 'NPS_plots/')
dqe_folder = os.path.join(DQE_folder, 'DQE_plots/')

os.makedirs(output_folder_FT, exist_ok=True)
os.makedirs(mtf_folder, exist_ok = True)
os.makedirs(nps_folder, exist_ok = True)
os.makedirs(dqe_folder, exist_ok = True)

#######   MTF   #######

input_mtf_file = 'tif_images/data_driven_MTF_lead_50kV_5uA_300s_collimator_32bit_float.raw'

# Inizialize the class
tool = MTFCalculator()

freq_lp_mm, module_norm = tool.calculate_mtf(input_mtf_file, x_range=(414, 420), y_range=(205, 215))

output_path_MTF = os.path.join(mtf_folder, "MTF_TimePix.png")

fig_mtf, ax_mtf = plt.subplots()
ax_mtf.plot(freq_lp_mm, module_norm)
ax_mtf.set_title("Modulation Transfer Function (MTF)")
ax_mtf.set_xlabel("Spatial Frequency (lp/mm)")
ax_mtf.set_ylim(0,1)
ax_mtf.set_ylabel("MTF")
ax_mtf.grid(True)
fig_mtf.savefig(output_path_MTF)
plt.close(fig_mtf)

# Initialize the figure and axes for NPS and DQE plots
fig_nps, ax_nps = plt.subplots()
fig_dqe, ax_dqe = plt.subplots()

# Substitute with the dimensions
width, height = 512, 448 
roi_size = 256
pixel_size = 0.055 # mm
# Calculation of the center of the image
cy, cx = height // 2, width // 2
# Calcultation of the limits (start:end)
y_start, y_end = cy - roi_size // 2, cy + roi_size // 2
x_start, x_end = cx - roi_size // 2, cx + roi_size // 2

# Generate the correct NPS frequency axis (127 points, excluding zero)
# Create the complete axis from 0 to Nyquist (0.5 / pixel_size) with 128 points
freq_nps_full = np.linspace(0, 0.5 / pixel_size, roi_size // 2)
# Cut the first point (zero) to have the 127 points corresponding to nps_smooth
freq_nps = freq_nps_full[1:]  # Length: 127
# Interpolate the MTF values on NPS frequencies 
mtf_resampled = np.interp(freq_nps, freq_lp_mm, module_norm)  # Length: 127


for time in n_seconds:
    if calibration == True:
        input_file = f"CLUSTERED_IMAGES/TOT/calibrated_series_400V_TOT/cal_series_{time}sec.raw"   #images calibrated through correction matrix
    elif calibration == False:
        input_file = f"calibrazione_14lug2026_CLUSTERING/TOT/image_raw_16bit_TOT/calibrazione_BIAS_400V_40kV_4uA_{time}sec_clustering_TOT.raw"  # images not calibrated throuh correction matrix
    else:
        raise ValueError("Invalid value for 'calibration'. Please set it to True or False.")

    file_basename, _ = os.path.splitext(os.path.basename(input_file))
    print(file_basename)    

    data = np.fromfile(input_file, dtype=np.uint16)
    image = data.reshape((height, width))   

    # Extraction of the ROI
    roi = image[y_start:y_end, x_start:x_end]

    
    roi_FT = fftn(roi)

    #Shift the zero frequency to the center
    roi_FT_shifted = fftshift(roi_FT)
    magnitude_spectrum = np.abs(roi_FT_shifted)

    # for the NPS use module^2
    squared_module = magnitude_spectrum**2

    output_path_FFT = os.path.join(output_folder_FT, "FT_squared_module_" + file_basename + ".raw")
    squared_module.astype(np.float32).tofile(output_path_FFT)

    ######   NPS   #####
    input_file_nps = output_path_FFT

    center = roi_size//2
    width_nps = 10    

    data_nps = np.fromfile(input_file_nps, dtype=np.float32)
    image_nps = data_nps.reshape((roi_size, roi_size))
    roi_mean_squared = (np.mean(roi))**2

    #image_nps = image_nps/(roi_size**4)  # without all the constants for normalization

    image_nps = image_nps/(roi_mean_squared*roi_size**2)*(pixel_size)**2 

    nps_profile_raw = np.mean(image_nps[center-width_nps : center, center: ], axis=0)  # with "center-width_nps : center" the slicing stops at center-1, so the zero frequency component is excluded (the central row)

    nps_profile = nps_profile_raw[1:]  # Exclude the zero frequency component (the central column (pixel in this case)), to avoid this line you can write [center-width_nps : center, center+1: ] above

    # window_length: number of samples (must be odd)
    # polyorder: order of the polynomial (usually 2 or 3)
    nps_smooth = savgol_filter(nps_profile, window_length=11, polyorder=3)

    ##########   CALCULATING THE FLUX   ##########
    
    # flux = counts/mm^2

    roi_dim = pixel_size*roi_size
    flux = np.sum(roi)/(roi_dim**2)

    #######   DQE   #######
    dqe = mtf_resampled**2/(nps_smooth*flux)

    ax_nps.plot(freq_nps, nps_smooth, label=f"{time} s")
    ax_nps.set_yscale("log")
    ax_nps.set_ylim(10**(-6), 10**(-3))
    ax_nps.set_xlim(0, None)
    ax_dqe.plot(freq_nps, dqe, label=f"{time} s")
    ax_dqe.set_ylim(0, None)
    ax_dqe.set_xlim(0, None)

output_path_NPS = os.path.join(nps_folder, f"NPS_comparison{label}.png")
ax_nps.set_title(f"Noise Power Spectrum Comparison {label}")
ax_nps.set_ylabel("NPS")
ax_nps.set_xlabel("Spatial Frequency (lp/mm)")
ax_nps.grid(True)
ax_nps.legend()
fig_nps.savefig(output_path_NPS)

output_path_DQE = os.path.join(dqe_folder, f"DQE_comparison{label}.png")
ax_dqe.set_title(f"DQE Comparison {label}")
ax_dqe.set_xlabel("Spatial Frequency (lp/mm)")
ax_dqe.set_ylabel("DQE")
ax_dqe.grid(True)
ax_dqe.legend()
fig_dqe.savefig(output_path_DQE)

plt.show()