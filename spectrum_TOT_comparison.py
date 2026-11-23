# This script generates the plot containing the spectra from clustersizes 2,3,4 to compare them in the same plot.

import os 
import glob
#import tifffile as tiff
import numpy as np
from natsort import natsorted
import matplotlib.pyplot as plt
from pathlib import Path

input_folder = Path("CLUSTERED_IMAGES/TOT/angle0")
output_folder = input_folder / "spectra_TOT"

os.makedirs(output_folder, exist_ok=True)

files_2 = glob.glob(os.path.join(input_folder, "*2.raw"))
sorted_files_2 = natsorted(files_2)

files_3 = glob.glob(os.path.join(input_folder, "*3.raw"))
sorted_files_3 = natsorted(files_3)

files_4 = glob.glob(os.path.join(input_folder, "*4.raw"))
sorted_files_4 = natsorted(files_4)

#print(sorted_files_2, sorted_files_3, sorted_files_4)

counts_per_bin_2 = []
for file_path in sorted_files_2:
    data = np.fromfile(file_path, dtype=np.uint16).reshape((448, 512))
    ROI = data[30:200, 30:200]
    counts_per_bin_2.append(np.sum(ROI))

counts_per_bin_3 = []
for file_path in sorted_files_3:
    data = np.fromfile(file_path, dtype=np.uint16).reshape((448, 512))
    ROI = data[30:200, 30:200]
    counts_per_bin_3.append(np.sum(ROI))

counts_per_bin_4 = []
for file_path in sorted_files_4:
    ROI = data[30:200, 30:200]
    data = np.fromfile(file_path, dtype=np.uint16).reshape((448, 512))
    counts_per_bin_4.append(np.sum(ROI))

x = np.arange(1, len(counts_per_bin_2)*100+1, 100)
#x = np.arange(len(counts_per_bin_2))

plt.figure(figsize=(10, 6))
plt.plot(x,counts_per_bin_2, color='dodgerblue', label='Cluster size 2')
plt.plot(x, counts_per_bin_3, color='orange', label='Cluster size 3')
plt.plot(x, counts_per_bin_4, color='green', label='Cluster size 4')
plt.xlabel('Energy bin (TOT)')
plt.ylabel('Counts')
plt.ylim(bottom=0)
plt.legend()
plt.grid(True)
plt.title('Counts per Energy Bin (TOT)')
plt.savefig(os.path.join(output_folder, "spectrum_TOT_comparison.png"))
plt.show()