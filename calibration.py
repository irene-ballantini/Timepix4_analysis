# Calibration for not clustered images
# First script dedicated to the calibration

import numpy as np
import glob
from natsort import natsorted
from pathlib import Path

#####  CORRECTION MATRIX

# Substitute with the dimensions
width, height = 512, 448 

path_input = "calibrazione_14lug2026/image_raw_16bit/*.raw"   #input path for the calibration series
path_output_cal = Path("calibrated_series_400V")
path_output_proj = Path("projections")

path_output_cal.mkdir(parents=True, exist_ok=True)
path_output_proj.mkdir(parents=True, exist_ok=True)

files = glob.glob(path_input)
sorted_files = natsorted(files)

normalized_accumulator = np.zeros((height, width), dtype=np.float64)

seconds = [10, 20, 40, 60, 80, 100]      # <----------------------- LIST OF SECONDS TO CHANGE ACCORDING TO THE ACQUISITION TIMES OF THE SERIES

for i,file_path in enumerate(sorted_files):
    img = np.fromfile(file_path, dtype=np.uint16).reshape((height, width))
    # Normalize the image by a variable proportional to the acquisition time
    # add to the accumulator
    var_prop_to_sec = seconds[i]/10
    normalized_accumulator += img.astype(np.float64)/var_prop_to_sec   # matrix 

# Calculate mean response 
mean_response = normalized_accumulator / len(files)     # mean response matrix of the pixels 

global_mean = np.mean(mean_response)

# Correction matrix
correction_matrix = np.ones((height, width), dtype=np.float64)
mask = mean_response > 0
correction_matrix[mask] = global_mean / mean_response[mask]

print(correction_matrix)


for i, file_path in enumerate(sorted_files):
    img = np.fromfile(file_path, dtype=np.uint16).reshape((height, width))
    img_corrected = img.astype(np.float64) * correction_matrix
    img_calibrated= img_corrected.astype(np.uint16)
    filename_cal = f"cal_series_{seconds[i]}sec.raw"
    img_calibrated.tofile(path_output_cal / filename_cal)


###  CALIBRATED IMAGES

###  Applying the Correction matrix to the images for tomographic reconstruction

path_input = "image_raw_16bit/*.raw"

files = glob.glob(path_input)
sorted_files = natsorted(files)

for i,file_path in enumerate(sorted_files):
    img_raw= np.fromfile(file_path, dtype=np.uint16).reshape((height, width))

    img_corrected = img_raw.astype(np.float64) * correction_matrix

    img_calibrated= img_corrected.astype(np.uint16)
    filename = 'projection_'+str(i)+'.sdt'
    img_calibrated.tofile(path_output_proj / filename)

         
            
        

