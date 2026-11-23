# Convert 64-bit TIFF summed images to 16-bit and save as raw files

import tifffile as tiff
import numpy as np
import glob
from natsort import natsorted
from pathlib import Path

path_input = "calibration_CT_perline_40kV_4uA_CLUSTERING/*sec/run_0/outputdata_ToT/datatiff/summed/summed_img.tiff"
path_output = Path('calibrazione_14lug2026_CLUSTERING/TOT/image_raw_16bit_TOT')

path_output.mkdir(parents=True, exist_ok=True)

files = glob.glob(path_input)
sorted_files = natsorted(files)

print(f"Found {len(files)} files to elaborate.")


for i, file_path in enumerate(sorted_files):
    img_64 = tiff.imread(file_path)
    img_16 =img_64.astype(np.uint16)
    #tiff.imwrite('image_tif_16bit/image_'+str(i)+'.tif', img_16)   # saving TIFF format
    img_16.tofile(path_output/f'calibrazione_BIAS_400V_40kV_4uA_{(i+1)*10}sec_clustering_TOT.raw')          # saving raw format


