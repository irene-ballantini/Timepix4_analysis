# This script sums all the binned images of the calibration series related to a specific ToT interval by looking at the labels of the files 
# To change the ToT intevals and so the energy interval, you can change the label of the bins below (e.g., bin_label_1)
# You can also create more labels for the bins

# This script was written to obtain the images of different energy bins to perform material decomposition later

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
bin_label_1 = {"_801-901", "_901-1001", "_1001-1101", "_1101-1201" }
bin_label_2 = {"_1201-1301", "_1301-1401", "_1401-1501", "_1501-1601"}
bin_label_3 = {"_1601-1701", "_1701-1801", "_1801-1901", "_1901-2001"}

# Find all .raw files in the subfolders "*sec"
all_raw_files = list(base_folder.glob("*sec/*.raw"))

all_raw_files_sorted = natsorted(all_raw_files)

# Group files by seconds: { "60sec": [file1, file2], ... }
files_by_seconds_one = defaultdict(list)
files_by_seconds_two = defaultdict(list)
files_by_seconds_three = defaultdict(list)

for file_path in all_raw_files_sorted:
    sec_dir_path = file_path.parent # Es. "60sec"
    filename = file_path.name

    if any(label in filename for label in bin_label_1):
        files_by_seconds_one[sec_dir_path].append(file_path)
    elif any(label in filename for label in bin_label_2):
        files_by_seconds_two[sec_dir_path].append(file_path)
    elif any(label in filename for label in bin_label_3):
        files_by_seconds_three[sec_dir_path].append(file_path)

# Execution for both bins
process_and_sum(files_by_seconds_one, "bin_one")
process_and_sum(files_by_seconds_two, "bin_two")
process_and_sum(files_by_seconds_three, "bin_three")

