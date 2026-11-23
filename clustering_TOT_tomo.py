# Reading files and summing counts related to the same energy bin (TOT files) discarding or not some clustersizes
# In other words the script summes together all the images belonging to the same ToT bin

# TO ADAPT TO THE NEW DATA

# If the clustering setting are changed make sure the initial configuration settings below are changed accordingly

import os 
import glob
import tifffile as tiff
import numpy as np
from natsort import natsorted
from pathlib import Path

# Initial configuration
in_keV = int(1)
fin_keV = int(5000 + 1)   # N_files = 250 

en_bins = np.arange(in_keV, fin_keV, 100)
print(f"Energy bins in ToT: {en_bins}")
step = 100

input_folder = Path("CLUSTERED_IMAGES/TOT/series_16bit/60sec")
counts_per_energy_bin_folder = input_folder / "counts_per_energy_bin_TOT_234cluster"

os.makedirs(counts_per_energy_bin_folder, exist_ok=True)

# Summing counts related to the same energy bin

all_files = glob.glob(os.path.join(input_folder, "*.raw"))

# Filtra via quelli che finiscono con 1.raw o 5.raw
files_raw = [
    f for f in all_files 
    if not (f.endswith("1.raw") or f.endswith("5.raw"))
]

sorted_files_raw = natsorted(files_raw)

print(f"Found {len(files_raw)} files to elaborate.")

filename_prefix = "results_"

for start_en in en_bins:
    end_en = start_en + step
    # 1. Initialize the sum to None for each new bin
    sum_data = None
    
    # Prefix to filter files belonging to the current energy bin
    current_bin_prefix = f"{filename_prefix}{start_en}-{end_en}ToT_"
    print(f"Searching for the files of the bin: {current_bin_prefix}") # Debug

    for file_path in sorted_files_raw:
        file_name = os.path.basename(file_path)
        # Check if the file belongs to the current energy bin
        if file_name.startswith(current_bin_prefix):
            data = np.fromfile(file_path, dtype=np.uint16)

            if sum_data is None:
                # Create an array at the first match
                sum_data = np.zeros_like(data, dtype=np.uint32)  
            
            # Sum the counts
            sum_data += data
    
    if sum_data is not None:
        output_path = os.path.join(counts_per_energy_bin_folder, f"sum_{start_en}_{end_en}ToT.raw")
        # Riporta a uint16 prima di salvare se necessario
        sum_data.astype(np.uint16).tofile(output_path)
        print(f"--- Saved: sum_{start_en}_{end_en}ToT.raw ---")
