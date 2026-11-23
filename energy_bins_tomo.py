# This script creates the summed images for each energy bin (lower and upper for example) 
# To produce images of different energy bins.

# It sums all the binned images of the tomography related to a specific ToT interval by looking at the labels of the files 
# To change the ToT intevals and so the energy interval, you can change the label of the bins below (here bin_label_lower, bin_label_central, bin_label_upper)
# You can also create more labels for the bins

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
lower_bin_dir = base_folder / "lower_bin"
central_bin_dir = base_folder / "central_bin"
upper_bin_dir = base_folder / "upper_bin"
lower_bin_dir.mkdir(parents=True, exist_ok=True)
central_bin_dir.mkdir(parents=True, exist_ok=True)
upper_bin_dir.mkdir(parents=True, exist_ok=True)

# Bin labels 
bin_label_lower = {"_301", "_401", "_501", "_601"}
bin_label_central = {"_701", "_801", "_901", "_1001", "_1101", "_1201"}
bin_label_upper = {"_1301", "_1401", "_1501", "_1601"}

# Find all .raw files in the subfolders "angle*"
all_raw_files = list(base_folder.glob("angle*/*.raw"))

all_raw_files_sorted = natsorted(all_raw_files)

# Group files by angle: { "angle_0": [file1, file2], ... }
files_by_angle_lower = defaultdict(list)
files_by_angle_central = defaultdict(list)
files_by_angle_upper = defaultdict(list)

for file_path in all_raw_files_sorted:
    angle_name = file_path.parent.name  # Es. "angle_001"
    filename = file_path.name

    if any(label in filename for label in bin_label_lower):
        files_by_angle_lower[angle_name].append(file_path)
    elif any(label in filename for label in bin_label_central):
            files_by_angle_central[angle_name].append(file_path)
    elif any(label in filename for label in bin_label_upper):
        files_by_angle_upper[angle_name].append(file_path)

# Execution for both bins
process_and_sum(files_by_angle_lower, lower_bin_dir)
process_and_sum(files_by_angle_central, central_bin_dir)
process_and_sum(files_by_angle_upper, upper_bin_dir)