# This script sums all the binned images of the calibration series related to a specific ToT interval by looking at the labels of the files 
# To change the ToT intevals and so the energy interval, you can change the label of the bins below (here bin_label_lower, bin_label_central, bin_label_upper)
# You can also create more labels for the bins

import os
from collections import defaultdict
from pathlib import Path
import numpy as np
from natsort import natsorted

# Support function to sum and save images for each angle
def process_and_sum(files_dict, bin_type):
    for sec_dir_path, files in files_dict.items():
        if not files:
            continue

        sum_data = None
        for file in files:
            data = np.fromfile(file, dtype=np.uint16)
            if sum_data is None:
                sum_data = np.zeros_like(data, dtype=np.uint32)
            sum_data += data

        # Output folder definition
        target_dir = sec_dir_path / bin_type
        target_dir.mkdir(parents=True, exist_ok=True)

        output_file = target_dir / "projectionbak.sdt"
        sum_data.astype(np.uint16).tofile(output_file)
        print(f"Saved summed image for {sec_dir_path.name} ({bin_type}) to {output_file}")

# Root folder containing the "*sec" subfolders
base_folder = Path("CLUSTERED_IMAGES/TOT/series_16bit")

# Bin labels 
bin_label_lower = {"_301", "_401", "_501", "_601"}
bin_label_central = {"_701", "_801", "_901", "_1001", "_1101", "_1201"}
bin_label_upper = {"_1301", "_1401", "_1501", "_1601"}

# Find all .raw files in the subfolders "*sec"
all_raw_files = list(base_folder.glob("*sec/*.raw"))

all_raw_files_sorted = natsorted(all_raw_files)

# Group files by seconds: { "60sec": [file1, file2], ... }
files_by_seconds_lower = defaultdict(list)
files_by_seconds_central = defaultdict(list)
files_by_seconds_upper = defaultdict(list)

for file_path in all_raw_files_sorted:
    sec_dir_path = file_path.parent # Es. "60sec"
    filename = file_path.name

    if any(label in filename for label in bin_label_lower):
        files_by_seconds_lower[sec_dir_path].append(file_path)
    elif any(label in filename for label in bin_label_central):
        files_by_seconds_central[sec_dir_path].append(file_path)
    elif any(label in filename for label in bin_label_upper):
        files_by_seconds_upper[sec_dir_path].append(file_path)

# Execution for both bins
process_and_sum(files_by_seconds_lower, "lower_bin")
process_and_sum(files_by_seconds_central, "central_bin")
process_and_sum(files_by_seconds_upper, "upper_bin")

