## Copy from external hard disk the summed images converting them from 64 bit to 16 bit and saving them as .raw files in the correct angle folder

import tifffile as tiff
import numpy as np
import glob
from natsort import natsorted
from pathlib import Path

# Done for CH folder
base_dir = "D:/TIMEPIX/CT_Perline_16lug2026/TOT"

path_input_tot = f"{base_dir}/angle*/datatiff/summed/summed_img.tiff"
path_output_tot = Path('CLUSTERED_IMAGES/TOT/summed')

path_output_tot.mkdir(parents=True, exist_ok=True)

# Find the files 
files = glob.glob(path_input_tot)
sorted_files = natsorted(files)

print(f"Found {len(sorted_files)} files to elaborate.")

for file_path_str in sorted_files:
    file_path = Path(file_path_str)

    # file_path.parents[2] takes the angle folder (three levels above the .tiff file)
    folder_angle = file_path.parents[2].name  
    
    # Read and elaborate the image
    img_64 = tiff.imread(file_path)
    img_16 = img_64.astype(np.uint16)
    
    # Save the RAW file in the correct angle folder
    # Usiamo il nome originale del file (.tiff) e cambiamo l'estensione in .raw
    output_name = f"summed_img_{folder_angle}.raw"
    output_file_path = path_output_tot / output_name
    
    img_16.tofile(output_file_path)
    print(f"Converted: {file_path.name} -> {output_file_path}")

print("Processing completed successfully!")
