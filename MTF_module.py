# MTF module for the MTF calculation

import numpy as np
from scipy.fft import fft, fftfreq

class MTFCalculator:
    def __init__(self, width=512, height=448, pixel_size=0.055):     #default values that can be changed when calling the module
        """
        Configuration for a specific detector
        """
        self.width = width
        self.height = height
        self.pixel_size = pixel_size

    def calculate_mtf(self, file_path, x_range, y_range, roi_size=256):
        # Loading data
        data = np.fromfile(file_path, dtype=np.float32)
        image = data.reshape((self.height, self.width))
        
        # ROI selection and ESF calculation (Edge Spread Function)
        roi = image[y_range[0]:y_range[1], x_range[0]:x_range[1]]
        esf = np.mean(roi, axis=0)
        
        # LSF (Line Spread Function) via derivative
        lsf = np.gradient(esf)
         

        # Zero Padding to improve spectral resolution
        array = np.zeros(roi_size,  dtype=np.float32)
        center = roi_size // 2
        start_idx = center - len(lsf) // 2     # This works for both even and odd values ​​of len(lsf)
        end_idx = start_idx + len(lsf)
        array[start_idx:end_idx] = lsf
        
        # FFT e Normalization
        mtf_complex = fft(array)
        mtf_half = np.abs(mtf_complex[:roi_size // 2])
        mtf_norm = mtf_half / mtf_half[0]
        
        # Frequency axis calculation (lp/mm)
        freq_lp_mm = fftfreq(
            roi_size,
            d=self.pixel_size
        )[:roi_size // 2]

                
        return freq_lp_mm, mtf_norm