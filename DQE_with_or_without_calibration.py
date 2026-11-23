"""
This code imports MTFCalculator from MTF_module.py, from which MTF of the 
specific detector used can be retrieved and used for the DQE calculation

There for the NPS calculation, the code uses the summed image coming from the clustering conversion. 
This image comes from a n_seconds acquisition

"""

from scipy.fft import fftn, fftshift
import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.signal import windows
from scipy.signal import savgol_filter
from MTF_module import MTFCalculator

n_seconds = 50
calibration = True  # Set to True for DQE calculation with calibration, False for without calibration

if calibration == True:
    DQE_folder = 'CLUSTERED_IMAGES/TOT/DQE/DQE_cal/'          # Folder for DQE calculation with calibration
    input_file = f"CLUSTERED_IMAGES/TOT/calibrated_series_400V_TOT/cal_series_{n_seconds}sec.raw"   #images calibrated through correction matrix
    label = " - calibrated images"
elif calibration == False:
    DQE_folder = 'CLUSTERED_IMAGES/TOT/DQE/DQE_no_cal/'      # Folder for DQE calculation without calibration
    input_file = f"calibrazione_14lug2026_CLUSTERING/TOT/image_raw_16bit_TOT/calibrazione_BIAS_400V_40kV_4uA_{n_seconds}sec_clustering_TOT.raw"  # images not calibrated throuh correction matrix
    label = " - not calibrated images"
else:
    raise ValueError("Invalid value for 'calibration'. Please set it to True or False.")

output_folder_FT = os.path.join(DQE_folder, 'FT_domain/')
mtf_folder = os.path.join(DQE_folder, 'MTF_plot/')
nps_folder = os.path.join(DQE_folder, 'NPS_plots/')
dqe_folder = os.path.join(DQE_folder, 'DQE_plots/')

plot_title = label + " (" + str(n_seconds) + " seconds acquisition)"    #TO CHANGE DEPENDING ON THE IMAGE: if you forget to change it, look at the name of the file


file_basename, _ = os.path.splitext(os.path.basename(input_file))
print(file_basename)

os.makedirs(output_folder_FT, exist_ok=True)
os.makedirs(mtf_folder, exist_ok = True)
os.makedirs(nps_folder, exist_ok = True)
os.makedirs(dqe_folder, exist_ok = True)

# Substitute with the dimensions
width, height = 512, 448 

#######   MTF   #######

input_mtf_file = 'tif_images/data_driven_MTF_lead_50kV_5uA_300s_collimator_32bit_float.raw'

# Inizialize the class
tool = MTFCalculator()
# x_range=(414, 420), y_range=(205, 215)
freq_lp_mm, module_norm = tool.calculate_mtf(input_mtf_file, x_range=(414, 420), y_range=(205, 220))

output_path_MTF = os.path.join(mtf_folder, "MTF_TimePix.png")

plt.figure(num=1)
plt.plot(freq_lp_mm, module_norm)
plt.title("Modulation Transfer Function (MTF)")
plt.xlabel("Spatial Frequency (lp/mm)")
plt.ylim(0,1.1)
plt.xlim(0, 10)
plt.ylabel("MTF")
plt.grid(True)
plt.savefig(output_path_MTF)

### NPS ###

data = np.fromfile(input_file, dtype=np.uint16)
image = data.reshape((height, width))

roi_size = 256

# Calculation of the center of the image
cy, cx = height // 2, width // 2

# Calcultation of the limits (start:end)
y_start, y_end = cy - roi_size // 2, cy + roi_size // 2
x_start, x_end = cx - roi_size // 2, cx + roi_size // 2

# ROI extraction
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

pixel_size = 0.055 # mm

data_nps = np.fromfile(input_file_nps, dtype=np.float32)
image_nps = data_nps.reshape((roi_size, roi_size))
roi_mean_squared = (np.mean(roi))**2

#image_nps = image_nps/(roi_size**4)  # without all the constants for normalization

image_nps = image_nps/(roi_mean_squared*roi_size**2)*(pixel_size)**2 

nps_profile_raw = np.mean(image_nps[center-width_nps : center, center: ], axis=0)  # with "center-width_nps : center" the slicing stops at center-1, so the zero frequency component is excluded (the central row)

nps_profile = nps_profile_raw[1:]  # Exclude the zero frequency component (the central column (pixel in this case)), to avoid this line you can write [center-width_nps : center, center+1: ] above

# window_length: number of samples (must be odd)
# polyorder: order of the polynomial (usually 2 or 3)
nps_smooth = savgol_filter(nps_profile, window_length=15, polyorder=3)

# Generate the correct NPS frequency axis (127 points, excluding zero)
# Create the complete axis from 0 to Nyquist (0.5 / pixel_size) with 128 points
freq_nps_full = np.linspace(0, 0.5 / pixel_size, roi_size // 2)
# Cut the first point (zero) to have the 127 points corresponding to nps_smooth
freq_nps = freq_nps_full[1:]  # Length: 127

output_path_NPS = os.path.join(nps_folder, "NPS_" + file_basename + ".png")

plt.figure(num=2)
plt.plot(freq_nps, nps_smooth)
plt.yscale("log")
plt.ylim(10**(-6), 10**(-3))
plt.title("Noise Power Spectrum" + plot_title)
plt.ylabel("NPS")
plt.xlabel("Spatial Frequency (lp/mm)")
plt.xlim(0, 10)
plt.grid(True)
plt.savefig(output_path_NPS)

##########   CALCULATING THE FLUX   ##########

# flux = counts/mm^2

roi_dim = pixel_size*roi_size
flux = np.sum(roi)/(roi_dim**2)
print("Flux:", flux)

# Interpolate the MTF values on NPS frequencies 
mtf_resampled = np.interp(freq_nps, freq_lp_mm, module_norm)  # Length: 127

dqe = mtf_resampled**2/(nps_smooth*flux)

output_path_DQE = os.path.join(dqe_folder, "DQE_" + file_basename + ".png")

plt.figure(num=3)
plt.plot(freq_nps, dqe)
plt.title("DQE" + plot_title)
plt.xlabel("Spatial Frequency (lp/mm)")
plt.ylabel("DQE")
plt.xlim(0, 10)
plt.ylim(0, 1.01) 
plt.grid(True)
plt.savefig(output_path_DQE)
plt.show()