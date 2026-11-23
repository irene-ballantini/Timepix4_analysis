# This script generates the spectra from the images of each ToT/Energy bin

# According to the clustering_TOT scripts you could have images of these bins containing all the clustersizes or a selected group of clustersizes

from pathlib import Path
import os
import glob
import numpy as np
from natsort import natsorted
import matplotlib.pyplot as plt
import pandas as pd


input_folder = Path("CLUSTERED_IMAGES/TOT/series_16bit/60sec")
counts_per_energy_bin_folder = input_folder / "counts_per_energy_bin_TOT"
#output_folder = Path("SpekPy_CT_spectrum_comparison")

files = glob.glob(os.path.join(counts_per_energy_bin_folder, "*.raw"))
sorted_files = natsorted(files)

print(f"Found {len(files)} files to elaborate.")


counts_per_bin = []
for file_path in sorted_files:
    data = np.fromfile(file_path, dtype=np.uint16).reshape((448, 512))
    ROI = data[30:200, 30:200]
    counts_per_bin.append(np.sum(ROI))

x = np.arange(1, len(counts_per_bin)*100, 100) 

counts_per_bin = np.array(counts_per_bin)

dati = {'Energy bin (TOT)': x, 'Counts': counts_per_bin}

# Create the DataFrame
df = pd.DataFrame(dati)

output_path = os.path.join(counts_per_energy_bin_folder, 'counts_per_energy_bin.xlsx')
# Save in Excel
df.to_excel(output_path, index=False)

plt.figure(figsize=(10, 6))
plt.plot(x, counts_per_bin, color='dodgerblue', label="ROI")
plt.xlabel('Energy bin (TOT)')
plt.ylabel('Counts')
plt.ylim(bottom=0)
plt.xlim(0, None)
plt.title('Counts per Energy Bin (TOT)')
plt.legend()
plt.grid(True)
plt.show()