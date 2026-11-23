# Material decomposition script

import matplotlib.pyplot as plt
from pathlib import Path
import tifffile
import numpy as np
from scipy.ndimage import gaussian_filter

def mat_dec_area():
    # ---------------------------------------------------------------------------------------------
    # MATERIAL DECOMPOSITION BASE ALGORITHM
    # proportions of base pigments w1, w2 are calculated with linear system direct matrix solution
    # all scalar values
    # ---------------------------------------------------------------------------------------------
    
    # Configuration of the graphic style (white backgroud, lines/text black)
    # Matplotlib does it by default but we can make sure of this:
    plt.rcParams['figure.facecolor'] = 'white'
    plt.rcParams['text.color'] = 'black'
    plt.rcParams['axes.labelcolor'] = 'black'
    
    # In Python, for images in greyscale the colormap used is 'gray' 
    # directly when the image is shown
        
    path = Path(r"CLUSTERED_IMAGES/TOT/calibrated_bins/cal_mat_dec_bins")
    
    print(f"Data folder: {path}")
    
    # Image E1
    filename1 = path / "bin_one" / "projections" / "recXZ_ok_305.sdt"

    width, height = 512, 448
    
    # Read
    img1 = np.fromfile(filename1, dtype=np.float32).reshape((height, width))
    # ---------------------------------------------------------------------------------------------
    # EXTRACTION OF ROI1
    # ---------------------------------------------------------------------------------------------
    x_start, y_start = 230, 370
    dx, dy = 30, 380

    # In Python NumPy slicing is used: [y_start : y_start+dy, x_start : x_start+dx]
    roi1 = img1[y_start : y_start + dy, x_start : x_start + dx]

    # ---------------------------------------------------------------------------------------------
    # DISPLAY EXTRACTED ROI1
    # ---------------------------------------------------------------------------------------------
    # Create a figure defining the dimentions in inches (es. 9x5.3 pollici)
    plt.figure(num = 1, figsize=(9, 5.3))
        
    # If image upside down, remove origin='lower'.
    plt.imshow(img1, cmap='gray')
    
    plt.title("ROI1")
    plt.axis('off')  # Hides the axes 
    
    # ---------------------------------------------------------------------------------------------
    # Image E2
    filename2 = path / "bin_two" / "projections" / "recXZ_ok_305.sdt"
    
    # Read
    img2 = np.fromfile(filename2, dtype=np.float32).reshape((height, width))
    roi2 = img2[y_start : y_start + dy, x_start : x_start + dx]

    # ---------------------------------------------------------------------------------------------
    # EXTRACTION OF ROI2
    # ---------------------------------------------------------------------------------------------
    plt.figure(num = 2, figsize=(9, 5.3))
    plt.imshow(img2, cmap='gray')    
    plt.title("ROI2")
    plt.axis('off') 

    # ---------------------------------------------------------------------------------------------
    # Image E3
    filename3 = path / "bin_three" / "projections" / "recXZ_ok_305.sdt"
    
    # Read
    img3 = np.fromfile(filename3, dtype=np.float32).reshape((height, width))
    roi3 = img3[y_start : y_start + dy, x_start : x_start + dx]

    # ---------------------------------------------------------------------------------------------
    # EXTRACTION OF ROI3
    # ---------------------------------------------------------------------------------------------
    plt.figure(num = 3, figsize=(9, 5.3))
    plt.imshow(img3, cmap='gray')    
    plt.title("ROI3")
    plt.axis('off') 
    
    # Apply a slight smoothing to reduce pixel-by-pixel noise.
    img1_smooth = gaussian_filter(img1, sigma=1.0)
    img2_smooth = gaussian_filter(img2, sigma=1.0)
    img3_smooth = gaussian_filter(img3, sigma=1.0)
    
    # -------------------------------------------------------------------------------------------
    # BASE MATERIALS
    # Material 1 = Bright bead
    # Material 2 = Medium Bead
    # Material 3 = Low bead
    
    # MATRICE DEI COEFFICIENTI 
    # Righe: [Bin 1, Bin 2, Bin 3]
    # Colonne: [Materiale 1, Materiale 2, Materiale 3]
    A = np.array([
        [5.00, 2.74, 1.94],  # Bin 1
        [4.31, 2.05, 1.24],  # Bin 2
        [3.31, 1.51, 0.82]   # Bin 3
    ])
    
    dy_roi, dx_roi = img1.shape

    # Pack the three ROIs into a matrix with shape (N_pixel, 3)
    B_pixels = np.stack([img1_smooth.ravel(), img2_smooth.ravel(), img3_smooth.ravel()], axis=-1)

    lam = 0.01  # If the images still appear too blurry or rigid, try lowering the value (e.g., 0.01) or increasing it (e.g., 1.0).
    I = np.eye(A.shape[1])
    
    term = A.T @ A + lam * I
    W_flat = np.linalg.solve(term, A.T @ B_pixels.T).T

    # Restore the images to their original 2D format (height, width, 3 materials)
    W_stack = W_flat.reshape((dy_roi, dx_roi, 3))

    # Extract the individual abundance maps for the 3 materials
    W1 = W_stack[:, :, 0]
    W2 = W_stack[:, :, 1]
    W3 = W_stack[:, :, 2]

    # Cleanup: we zero out negative values ​​caused by noise using np.clip
    W1 = np.clip(W1, 0, None)
    W2 = np.clip(W2, 0, None)
    W3 = np.clip(W3, 0, None)

    # Threshold 
    perc = 0.2

    W1[W1 < perc * W1.max()] = 0
    W2[W2 < perc * W2.max()] = 0
    W3[W3 < perc * W3.max()] = 0

    # ---------------------------------------------------------------------------------------------
    # VIEWING THE RESULTS (e.g., subplot 2x2 o 2x3)
    # ---------------------------------------------------------------------------------------------
    fig, axs = plt.subplots(2, 2, figsize=(10, 8))
    
    axs[0, 0].imshow(img1, cmap='gray')
    axs[0, 0].set_title("Bin 1")
    axs[0, 0].axis('off')

    axs[0, 1].imshow(W1, cmap='gray')
    axs[0, 1].set_title("Material 1 (Bright bead)")
    axs[0, 1].axis('off')

    axs[1, 0].imshow(W2, cmap='gray')
    axs[1, 0].set_title("Material 2 (Medium bead)")
    axs[1, 0].axis('off')

    axs[1, 1].imshow(W3, cmap='gray')
    axs[1, 1].set_title("Material 3 (Low bead)")
    axs[1, 1].axis('off')

    fig.tight_layout()
    plt.savefig(path / "Material_decomposition_maps_threshold.png")
    plt.show()

def plot_attenuation_spectra():
    # 1. define the centers of the three energy bins (keV)
    energies = np.array([25.15, 30.40, 35.20])
    
    # The semi-width of each bin (for the horizontal error bars)
    # Calculated as (E_max - E_min) / 2
    energy_widths = np.array([2.75, 2.50, 2.30]) 

    # 2. The mean values ​​of the coefficients for the three materials (from the matrices)
    bright_bead = np.array([5.00, 4.31, 3.31])
    medium_bead = np.array([2.74, 2.05, 1.51])
    low_bead    = np.array([1.94, 1.24, 0.82])

    bright_bead_err = np.array([0.57, 0.32, 0.28])
    medium_bead_err = np.array([0.20, 0.06, 0.02])
    low_bead_err   = np.array([0.19, 0.09, 0.07])

    # 3. Graph creation
    plt.figure(figsize=(8, 6))

    # Usiamo plt.errorbar: xerr mostra l'intervallo energetico del bin
    plt.errorbar(energies, bright_bead, xerr=energy_widths, yerr=bright_bead_err, fmt='-o', capsize=4, label='Bright Bead')
    plt.errorbar(energies, medium_bead, xerr=energy_widths, yerr=medium_bead_err, fmt='-s', capsize=4, label='Medium Bead')
    plt.errorbar(energies, low_bead,    xerr=energy_widths, yerr=low_bead_err, fmt='-^', capsize=4, label='Low Bead')

    # Plot
    plt.title("Attenuation coefficient vs Energy (Energy bins)", fontsize=12)
    plt.xlabel("Energy (keV)", fontsize=11)
    plt.ylabel("Mean attenuation coefficient (Gray Value)", fontsize=11)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()
    
    plt.tight_layout()
    output_path = Path(r"CLUSTERED_IMAGES/TOT/calibrated_bins/cal_mat_dec_bins/Attenuation_coeff_vs_energy_mat_dec.png") 
    plt.savefig(output_path)
    plt.show()

# To execute the function (corresponds to compile and execute PRO in IDL)
if __name__ == "__main__":
    mat_dec_area()
    plot_attenuation_spectra()

