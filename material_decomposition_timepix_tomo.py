# This script creates the summed images for each energy bin
# To produce images of different energy bins.

# It sums all the binned images of the tomography related to a specific ToT interval by looking at the labels of the files 
# To change the ToT intevals and so the energy interval, you can change the label of the bins below (e.g., bin_label_1)
# You can also create more labels for the bins

# This script was written to obtain the images of different energy bins to perform material decomposition later

import os
from collections import defaultdict
from pathlib import Path
import numpy as np
from natsort import natsorted

# Support function to sum and save images for each angle
def process_and_sum(files_dict, output_dir):
    for i, (angle_name, files) in enumerate(files_dict.items()):
        if not files:
            continue

        sum_data = None
        for file in files:
            data = np.fromfile(file, dtype=np.uint16)
            if sum_data is None:
                sum_data = np.zeros_like(data, dtype=np.uint32)
            #print(f"Processing {file} for angle {angle_name}")
            sum_data += data

        output_file = output_dir / f"projection_{i}.sdt"
        sum_data.astype(np.uint16).tofile(output_file)
        print(f"Saved summed image for {angle_name} to {output_file}")

# Root folder containing the "angle*" subfolders
base_folder = Path("CLUSTERED_IMAGES/TOT")

# Output folder definition
bin_dir_1 = base_folder / "mat_dec_en_bins"/ "bin_one"
bin_dir_2 = base_folder / "mat_dec_en_bins"/ "bin_two"
bin_dir_3 = base_folder / "mat_dec_en_bins"/ "bin_three"

bin_dir_1.mkdir(parents=True, exist_ok=True)
bin_dir_2.mkdir(parents=True, exist_ok=True)
bin_dir_3.mkdir(parents=True, exist_ok=True)

# Bin labels 
bin_label_1 = {"_801-901", "_901-1001", "_1001-1101", "_1101-1201" }
bin_label_2 = {"_1201-1301", "_1301-1401", "_1401-1501", "_1501-1601"}
bin_label_3 = {"_1601-1701", "_1701-1801", "_1801-1901", "_1901-2001"}

# Find all .raw files in the subfolders "angle*"
all_raw_files = list(base_folder.glob("angle*/*.raw"))

all_raw_files_sorted = natsorted(all_raw_files)

# Group files by angle: { "angle_0": [file1, file2], ... }
files_by_angle_one = defaultdict(list)
files_by_angle_two = defaultdict(list)
files_by_angle_three = defaultdict(list)

for file_path in all_raw_files_sorted:
    angle_name = file_path.parent.name  # Es. "angle_001"
    filename = file_path.name

    if any(label in filename for label in bin_label_1):
        files_by_angle_one[angle_name].append(file_path)
    elif any(label in filename for label in bin_label_2):
            files_by_angle_two[angle_name].append(file_path)
    elif any(label in filename for label in bin_label_3):
        files_by_angle_three[angle_name].append(file_path)
    

# Execution for both bins
process_and_sum(files_by_angle_one, bin_dir_1)
process_and_sum(files_by_angle_two, bin_dir_2)
process_and_sum(files_by_angle_three, bin_dir_3)