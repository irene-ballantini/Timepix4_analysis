## Copy from external hard disk the binned images (all energy bins) converting them from 64 bit to 16 bit 
# and saving them as .raw files in the correct angle folder

# This script is done for the data of the tomography

import tifffile as tiff
import numpy as np
import glob
from natsort import natsorted
from pathlib import Path

# Done for CT folder
base_dir = "D:/TIMEPIX/CT_Perline_16lug2026/TOT"

path_input_tot = f"{base_dir}/angle*/datatiff/ct_data_*.tiff"
path_output_tot = Path('CLUSTERED_IMAGES/TOT')

# Find the files 
files = glob.glob(path_input_tot)
sorted_files = natsorted(files)

print(f"Found {len(sorted_files)} files to elaborate.")

for file_path_str in sorted_files:
    file_path = Path(file_path_str)

    # file_path.parents[1] takes the angle folder
    folder_angle = file_path.parents[1].name  
    
    print(f"\nProcessing: {folder_angle}")

    current_output_dir = path_output_tot / folder_angle
    current_output_dir.mkdir(parents=True, exist_ok=True)
    
    # Read and elaborate the image
    img_64 = tiff.imread(file_path)
    img_16 = img_64.astype(np.uint16)
    
    # Save the RAW file in the correct angle folder
    # Use the original filename (.tiff) and change the extension in .raw
    output_name = file_path.with_suffix('.raw').name
    output_file_path = current_output_dir / output_name
    
    img_16.tofile(output_file_path)
    print(f"Converted: {file_path.name} -> {output_file_path}")

print("Processing completed successfully!")

