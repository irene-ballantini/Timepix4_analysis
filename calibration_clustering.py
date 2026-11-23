# Script to do the calibration of the summed images and generate the projections for tomographic reconstruction

import numpy as np
import matplotlib.pyplot as plt
import glob
from natsort import natsorted
from pathlib import Path

#####  CORRECTION MATRIX

# Substitute with the detector dimensions
width, height = 512, 448 

path_input = "calibrazione_14lug2026_CLUSTERING/TOT/image_raw_16bit_TOT/*.raw"     # path to the images of the calibration series
path_output_cal = Path("CLUSTERED_IMAGES/TOT/calibrated_series_400V_TOT")
path_output_proj = Path("CLUSTERED_IMAGES/TOT/summed/projections")

path_output_cal.mkdir(parents=True, exist_ok=True)
path_output_proj.mkdir(parents=True, exist_ok=True)

files = glob.glob(path_input) 
sorted_files = natsorted(files)

normalized_accumulator = np.zeros((height, width), dtype=np.float64)

for i,file_path in enumerate(sorted_files, start=1):
    img = np.fromfile(file_path, dtype=np.uint16).reshape((height, width))
    # Normalize the image by 'i'
    # add to the accumulator
    normalized_accumulator += img.astype(np.float64)/i

# Calculate mean response 
mean_response = normalized_accumulator / len(files)

global_mean = np.mean(mean_response)

# Correction matrix
correction_matrix = np.ones((height, width), dtype=np.float64)
mask = mean_response > 0
correction_matrix[mask] = global_mean / mean_response[mask]

print(correction_matrix)


for i, file_path in enumerate(sorted_files, start=1):
    img = np.fromfile(file_path, dtype=np.uint16).reshape((height, width))
    img_corrected = img.astype(np.float64) * correction_matrix
    img_calibrated= img_corrected.astype(np.uint16)
    filename_cal = f"cal_series_{i * 10}sec.raw"
    img_calibrated.tofile(path_output_cal / filename_cal)
    


###  CALIBRATED IMAGES

###  Applying the Correction matrix to the images for tomographic reconstruction

path_input = "CLUSTERED_IMAGES/TOT/summed/*.raw"

files = glob.glob(path_input)
sorted_files = natsorted(files)

for i,file_path in enumerate(sorted_files):
    img_raw= np.fromfile(file_path, dtype=np.uint16).reshape((height, width))

    img_corrected = img_raw.astype(np.float64) * correction_matrix

    img_calibrated= img_corrected.astype(np.uint16)
    filename = 'projection_'+str(i)+'.sdt'
    img_calibrated.tofile( path_output_proj / filename)

         
            
        

