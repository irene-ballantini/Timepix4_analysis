# Calibration using a list containing the seconds of the acquisition times of the calibration series
# This list is then used to calculate the arbitrary unit proportional to the time, which is necessary for the calibration

# This list is named seconds

import numpy as np
import matplotlib.pyplot as plt
import glob
from natsort import natsorted
from pathlib import Path

#####  CORRECTION MATRIX

# Substitute with the dimensions
width, height = 512, 448 

bin_folder = "upper_bin"

path_input = Path("CLUSTERED_IMAGES/TOT")
path_output_cal = path_input / "series_16bit/"
path_output_proj = path_input / "calibrated_bins" / bin_folder / "projections"

path_output_cal.mkdir(parents=True, exist_ok=True)
path_output_proj.mkdir(parents=True, exist_ok=True)

series = f"*sec/{bin_folder}/projectionbak.sdt"
sorted_files = natsorted(list(path_output_cal.glob(series)))

normalized_accumulator = np.zeros((height, width), dtype=np.float64)

seconds = [10, 20, 40, 60, 80, 100]      # <----------------------- LIST OF SECONDS TO CHANGE ACCORDING TO THE ACQUISITIONS TIMES OF THE SERIES

for i,file_path in enumerate(sorted_files):
    print(file_path)
    img = np.fromfile(file_path, dtype=np.uint16).reshape((height, width))
    # Normalize the image by 'i'
    # add to the accumulator
    var_prop_to_sec = seconds[i]/10
    print(var_prop_to_sec)
    normalized_accumulator += img.astype(np.float64)/var_prop_to_sec

# Calculate mean response 
mean_response = normalized_accumulator / len(sorted_files)

global_mean = np.mean(mean_response)

# Correction matrix
correction_matrix = np.ones((height, width), dtype=np.float64)
mask = mean_response > 0
correction_matrix[mask] = global_mean / mean_response[mask]

print(correction_matrix)


for i, file_path in enumerate(sorted_files, start=1):
    img = np.fromfile(file_path, dtype=np.uint16).reshape((height, width))
    img_corrected = img.astype(np.float64) * correction_matrix
    img_calibrated= np.clip(img_corrected, 0, 65535).astype(np.uint16)
    seconds_folder = file_path.parents[1].name 
    path_output_cal_sec = path_output_cal / seconds_folder / bin_folder / "calibrated_proj/"
    path_output_cal_sec.mkdir(parents=True, exist_ok=True)
    img_calibrated.tofile(path_output_cal_sec / file_path.name)
    


###  CALIBRATED IMAGES

###  Applying the Correction matrix to the images for tomographic reconstruction

path_input_tomo = path_input / bin_folder
sorted_tomo_files = natsorted(list(path_input_tomo.glob("projection_*.sdt")))

for i,file_path in enumerate(sorted_tomo_files):
    img_raw= np.fromfile(file_path, dtype=np.uint16).reshape((height, width))

    img_corrected = img_raw.astype(np.float64) * correction_matrix

    img_calibrated= np.clip(img_corrected, 0, 65535).astype(np.uint16)
    filename = f'projection_{i}.sdt'
    img_calibrated.tofile( path_output_proj / filename)

         
            
        

