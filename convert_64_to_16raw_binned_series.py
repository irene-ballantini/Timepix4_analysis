# Convert binned images of the calibration series from 64-bit .tiff image to 16-bit .raw image

import tifffile as tiff
import numpy as np
import glob
from natsort import natsorted
from pathlib import Path

# Done for CH folder
path_input_tot = "calibration_CT_perline_40kV_4uA_CLUSTERING/*sec/run_0/outputdata_ToT/datatiff/results_*.tiff"
path_output_tot = Path('CLUSTERED_IMAGES/TOT/series_16bit')

# Find the files 
files = glob.glob(path_input_tot)
sorted_files = natsorted(files)

print(f"Found {len(sorted_files)} files to elaborate.")

for file_path_str in sorted_files:
    file_path = Path(file_path_str)

    # file_path.parents[3] takes the sec folder
    folder_sec = file_path.parents[3].name  

    print(f"\nProcessing: {folder_sec}")
    
    current_output_dir = path_output_tot / folder_sec
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

