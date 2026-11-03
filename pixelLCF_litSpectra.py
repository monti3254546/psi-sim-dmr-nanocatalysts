from matplotlib import lines
import numpy as np
import pandas as pd
from scipy.interpolate import CubicSpline
from scipy.optimize import curve_fit
import os
import tifffile as tiff
import scipy as sp
import matplotlib.pyplot as plt
import copy


# REFERENCE IMPORT
referencePath = "C:\\Users\\Samuel_Uni\\psi-sim-dmr-nanocatalysts\\joannaRefs\\Reference_EELS_spectra.csv"

# Import of csv file spectra
df = pd.read_csv(referencePath, sep=';', header=0)

ni_spectra = {
    "Ni": (0, 1),
    "Ni_O": (2, 3),
    "Ni_(OH)2": (4, 5),
}

ru_spectra = {
    "Ru": (6, 7),
    "Ru_O": (8, 9),
}


# --- DOUBLE ARCTAN BACKGROUND MODEL ---

def double_arctan_function(E, a3, E0_3, w3, a2, E0_2, w2, b, c):
    """
    Two-step Arctan background model for EELS L3 and L2 edges:
      a3, E0_3, w3 : step amplitude, inflection energy, and width for L3
      a2, E0_2, w2 : step amplitude, inflection energy, and width for L2
      b, c         : linear slope and baseline offset
    """
    step_L3 = a3 * (0.5 + np.arctan((E - E0_3) / w3) / np.pi)
    step_L2 = a2 * (0.5 + np.arctan((E - E0_2) / w2) / np.pi)
    return step_L3 + step_L2 + b * E + c


def fit_and_subtract_double_arctan(energy, intensity, l3_range=(850, 856), l2_range=(867, 873)):
    """
    Fits double-arctan step background using only non-peak points (peak masking),
    returning individual arctan curves, total background, and corrected spectrum.
    """
    # 1. Mask out L3 and L2 white-line peaks during parameter optimization
    peak_mask = ((energy >= l3_range[0]) & (energy <= l3_range[1])) | \
                ((energy >= l2_range[0]) & (energy <= l2_range[1]))
    fit_mask = ~peak_mask

    if np.sum(fit_mask) < 8:
        fit_mask = np.ones_like(energy, dtype=bool)

    e_fit = energy[fit_mask]
    i_fit = intensity[fit_mask]

    # 2. Initial parameter guesses & bounds
    # [a3, E0_3, w3, a2, E0_2, w2, b, c]
    p0 = [
        0.05,                           # a3 initial guess
        (l3_range[0] + l3_range[1]) / 2, # E0_3 center (e.g. 853 eV)
        1.5,                            # w3 width
        0.02,                           # a2 initial guess
        (l2_range[0] + l2_range[1]) / 2, # E0_2 center (e.g. 870 eV)
        1.5,                            # w2 width
        0.0,                            # b slope
        np.min(intensity)               # c baseline offset
    ]

    lower_bounds = [0.0, l3_range[0], 0.2, 0.0, l2_range[0], 0.2, -np.inf, -np.inf]
    upper_bounds = [np.inf, l3_range[1], 10.0, np.inf, l2_range[1], 10.0, np.inf, np.inf]

    # Fit parameters using non-peak regions only
    popt, _ = curve_fit(
        double_arctan_function, e_fit, i_fit, p0=p0, bounds=(lower_bounds, upper_bounds), maxfev=10000
    )

    a3, E0_3, w3, a2, E0_2, w2, b, c = popt

    # 3. Calculate individual background components across full energy grid
    arctan_L3_curve = a3 * (0.5 + np.arctan((energy - E0_3) / w3) / np.pi) + c
    arctan_L2_curve = a2 * (0.5 + np.arctan((energy - E0_2) / w2) / np.pi)
    total_bg = double_arctan_function(energy, *popt)

    # 4. Subtract continuum background step and enforce non-negativity
    intensity_corrected = np.maximum(intensity - total_bg, 0.0)
    intensity_corrected=intensity_corrected/np.max(intensity_corrected)  #normalize to 1

    return intensity_corrected, total_bg, arctan_L3_curve, arctan_L2_curve, popt


def process_and_align_spectra(
    df, spectra_map, e_min=None, e_max=None, n_points=500
):
    """
    Extracts and resamples reference spectra onto a unified energy grid.
    """
    raw_data = {}
    
    for name, (e_col, i_col) in spectra_map.items():
        e = pd.to_numeric(df.iloc[:, e_col], errors='coerce').dropna().values
        i = pd.to_numeric(df.iloc[:, i_col], errors='coerce').dropna().values
        raw_data[name] = (e, i)
        
    if e_min is None:
        e_min = max([e.min() for e, _ in raw_data.values()])
    if e_max is None:
        e_max = min([e.max() for e, _ in raw_data.values()])

    target_grid = np.linspace(e_min, e_max, n_points)
    processed_spectra = {}
    
    for name, (e, i) in raw_data.items():
        spline = CubicSpline(e, i, extrapolate=False)
        i_resampled = spline(target_grid)
        processed_spectra[name] = i_resampled
        

    return target_grid, processed_spectra


# --- EXECUTION ---

target_grid, aligned_spectra = process_and_align_spectra(
    df=df,
    spectra_map=ni_spectra,
    n_points=600
)

# Apply double-arctan fit and visualization to target reference spectrum
target_spectrum = "Ni"
if target_spectrum in aligned_spectra:
    raw_signal = aligned_spectra[target_spectrum]
    
    corrected_signal, total_bg, arctan_L3, arctan_L2, popt = fit_and_subtract_double_arctan(
        target_grid, raw_signal, l3_range=(850, 856), l2_range=(867, 873)
    )
plt.tight_layout()
savepath = "C:\\Users\\Samuel_Uni\\psi-sim-dmr-nanocatalysts\\joannaRefs\\double_arctan_fit.png"
plt.savefig(savepath, dpi=300)
plt.show()

# Align all of the spectra to the same energy grid


# Correct the background of the spectra by subtracting the minimum value from each spectrum


# Import of txt file spectra
# reffiles = sorted(os.listdir(referencePath))
# first=True
# run = "ni"
# refs = []
# ref_energies = []
# ref_names = []
# for i in reffiles:
#     if i.endswith(".txt") and run in i:
#         ref = np.loadtxt(
#             os.path.join(referencePath, i),
#             delimiter=",",
#             skiprows=0
#         )
# #12.8 ev shift for ni spectra, Ru spectra need to be checked again
#         ref_energies.append(ref[:, 0] + 12.8)  #energy grid. shift systematic error of 12.5 eV
#         refAppend = ref[:, 1]   #spectra
#         refAppend -= np.min(refAppend)
#         #refAppend /= np.max(refAppend)
#         refs.append(refAppend)
#         ref_names.append(i)
#         print(i)


# for i in range(len(refs)):
#     refs[i] = sp.ndimage.gaussian_filter(refs[i], sigma = 45)   #sigma = 45 seems to yield a similar degree of detail to the data --> best for fitting
# #     plt.plot(ref_energies[i], refs[i])
# # plt.show()


# #interpolate to 0.2 eV grid, aligned to a .0 grid (i.e. 1, 1.2, 1.4, 1.6, ...)
# for i in range(len(refs)):  #E_ref is the energy grid of the ref spectrum, ref the spectrum itself
#     newAxisStart = round(min(ref_energies[i]) * 5) / 5
#     newAxisEnd = round(max(ref_energies[i]) * 5) / 5
#     newAxis = np.arange(newAxisStart, newAxisEnd, 0.2)
#     ref_interp = np.interp(
#         newAxis, #resample refs to 0.2 eV steps
#         ref_energies[i],
#         refs[i]
#     )
#     refs[i] = ref_interp
#     ref_energies[i] = newAxis

# refEnergyRanges = []   #use this to prevent shifting peaks out of the spectrum
# for i in range(len(ref_energies)):  #loop through the energy axis of all ref spectra
#     refEnergyRanges.append([min(ref_energies[i]), max(ref_energies[i])])
# refEnergyRanges = np.array(refEnergyRanges)

# lowerBoundRefs = np.min(refEnergyRanges)
# upperBoundRefs = np.max(refEnergyRanges)

# globRefEnergy = np.arange(0, 1000, 0.2)  #USE THIS FROM NOW ON, ITS GLOBAL FOR ALL REFERENCE SPECTRA
# print(globRefEnergy)
# for i in range(len(refs)):
#     padLeft = int(round((min(ref_energies[i]) - 0) * 5))
#     padRight = int(round((999.8 - max(ref_energies[i])) * 5))
#     refs[i] = np.pad(refs[i], (padLeft, padRight), mode = "constant", constant_values = 0)


# #DATA IMPORT
# #Data path for Moritz
# #datapath = "/Users/moritz/Library/Mobile Documents/com~apple~CloudDocs/Studium/Semester 5/Schlussprojekt/20260918_Ni_Ru_AbsorptionSpectrum v5-real/"
# #Data path for Samuel
# datapath= "E:\\20260918_Ni_Ru_AbsorptionSpectrum v5-real\\"
# #Path for new LCF data
# LCF_datapath="C:\\Users\\Samuel_Uni\\psi-sim-dmr-nanocatalysts\\LCF_results_new"
# datafolders = [
#     name for name in os.listdir(datapath)
#     if os.path.isdir(os.path.join(datapath, name))
# ]
# # Specify your single target folder name here (must exist inside `datapath`)
# # target_folder = "E:\\20260918_Ni_Ru_AbsorptionSpectrum v5-real\\20260918_1779_AbsortionSpectrum"  

# # datafolders = [target_folder]
# for i in range(len(datafolders)):
#     sample_id = datafolders[i][9:13]
#     try:
#         path = os.path.join(datapath, datafolders[i])
#         tiffpath = os.path.join(path + "/background-corrected")
#         datafiles = sorted(os.listdir(tiffpath))
#     except FileNotFoundError:
#         print("no background correction yet, continuing")
#         continue
#     if len(datafiles) < 5:
#         print("scan aborted, continuing")
#         continue

#     data = []
#     energies = []


#     for j in range(len(datafiles)):
#         #datafiles[j].startswith('._") required on Windows
#         if not datafiles[j].startswith("._") and (datafiles[j].endswith(".tif") or datafiles[j].endswith(".tiff")):
#             data.append(
#                 tiff.imread(
#                     os.path.join(tiffpath, datafiles[j])
#                 ).astype(float)
#             )
#             energies.append(float(datafiles[j][-10:-5]))


#     data = np.array(data)
#     energies = np.array(energies)
#     if run == "ni" and np.any(energies < 600): continue
#     elif run == "ru" and np.any(energies > 600): continue

#     mask = ((globRefEnergy >= np.min(energies) - 0.0001) & (globRefEnergy <= np.max(energies) + 0.0001))  #discard any ref spectra regions that are not in our data
#     refs_temp = np.column_stack(refs)[mask, :]
    
#     energy_peaks_range=[(463.5, 471.5), (485.5, 492.5), (864, 872), (882, 887)]
#     # 1. Compute mask ONCE outside the spatial loops (start with False)
#     mask = np.zeros_like(energies, dtype=bool)
#     for low, high in energy_peaks_range:
#         mask |= ((energies >= low) & (energies <= high))

#     # 2. Slice reference spectra ONCE to match masked dimensions
#     refs_masked = refs_temp[mask, :]
#     # Initialize output array
#     coeffs = np.zeros((len(ref_names), data.shape[1], data.shape[2]))

#     # 3. Perform LCF loop over spatial pixels
#     for m in range(data.shape[1]):
#         for n in range(data.shape[2]):
#         # Extract and mask target pixel spectrum
#             target_masked = data[mask, m, n]
        
#             # Non-negative least squares fit
#             coefficients, _ = sp.optimize.nnls(refs_masked, target_masked)
        
#             # Store coefficients
#             coeffs[:, m, n] = coefficients

#             #fitted = refs @ coefficients
#             #plt.plot(energies, target)
#             #plt.plot(globRefEnergy, fitted)
#             #print(coefficients)
#             #plt.show()
            
#     # Calculate ratio maps
#     for k in range(coeffs.shape[0]):
#         coeffs[k, :, :] = np.clip(coeffs[k, :, :], 0, np.percentile(coeffs[k, :, :], 99.9))
#         #plt.imsave(datapath + f"oxideMaps/{sample_id}_map_{ref_names[k]}.png", coeffs[k, :, :])
    
#     #simple clipping mask for ni2+ maps to make it human readable
#     #coeffs[0,:,:]=np.clip(coeffs[0,:,:],np.max(coeffs[0,:,:])*0.1,np.max(coeffs[0,:,:]*0.9))
#     k_0=0#coefficient 0
#     k_1=1#coefficient 1
#     k_2=2#coefficient 2
#     # if first:
#     #         print("select coefficients to plot")
#     #         input(k_1)
#     #         input(k_2)
#     #         first=False
#     #If this idea does not work, just set values below a certain threshold to 0.
#     #coeffs[0,:,:][coeffs[0,:,:]<np.max(coeffs[0,:,:])*0.025]=0 #try 0.05 or 0.1 
#     #ni1
#     coeffs[k_0,:,:][coeffs[k_0,:,:]<np.max(coeffs[k_0,:,:])*0.025]=0
#     #ni2
#     coeffs[k_1,:,:][coeffs[k_1,:,:]<np.max(coeffs[k_1,:,:])*0.025]=0
#     #ni3
#     coeffs[k_2,:,:][coeffs[k_2,:,:]<np.max(coeffs[k_2,:,:])*0.025]=0

#     #coeffs[0,:,:][(coeffs[0,:,:] < p10) | (coeffs[0,:,:] > p90)] = 0
#     sum_coeff=coeffs[k_1,:,:]+coeffs[k_2,:,:]
#     #would it be required to include coefficient 0 
#     #plotting ratios by dividing through the sum of coefficients from both maps
#     #masking based on total signal
#     # Check how many pixels have non-zero values in BOTH coefficients simultaneously
#     overlap_pixels = np.zeros([coeffs.shape[1],coeffs.shape[2]])
#     overlap_pixels[:,:][((coeffs[k_1, :, :] > 0) & (coeffs[k_2, :, :] > 0))] = 1
#     #ratio_k_0 = np.divide(coeffs[k_0,:,:], sum_coeff, out=np.zeros_like(sum_coeff), where=sum_coeff != 0)
#     ratio_k_1 = np.divide(coeffs[k_1,:,:], sum_coeff, out=np.zeros_like(sum_coeff), where=sum_coeff != 0)
#     ratio_k_2 = np.divide(coeffs[k_2,:,:], sum_coeff, out=np.zeros_like(sum_coeff), where=sum_coeff != 0)

#     #plotting ratios by using logartihmic scale
#     #ni_ratio = np.log10(coeffs[1, :, :] / coeffs[2, :, :])  # ni 3 over ni2

#     # fig, axes = plt.subplots(1, 3, figsize=(12, 4))
#     # for i in range(3):
#     #    im = axes[i].imshow(maps[i])
#     #    axes[i].set_title(titles[i])
#     #    fig.colorbar(im, ax=axes[i], fraction=0.046, pad=0.04)

#     # plt.tight_layout()
#     # plt.close()

#     # # #For Mac
#     # plt.imsave(LCF_datapath + f"{sample_id}_map_ni2.png", coeffs[0, :, :])
#     # plt.imsave(LCF_datapath + f"{sample_id}_map_ni3.png", coeffs[1, :, :])
#     # plt.imsave(LCF_datapath + f"{sample_id}_map_ni2-3.png", ni_ratio)
#     # #For Windows
    
#     #for multiple plots
#     fig, ax=plt.subplots(2,3,figsize=(12,4))
#     im1=ax[0,0].imshow(coeffs[k_0,:,:])
#     ax[0,0].title.set_text(f"Ni[{k_0+1}+]")
#     im2=ax[0,1].imshow(coeffs[k_1,:,:])
#     ax[0,1].title.set_text(f"Ni[{k_1+1}+]")
#     im3=ax[0,2].imshow(coeffs[k_2,:,:])
#     ax[0,2].title.set_text(f"Ni[{k_2+1}+]")
#     im4=ax[1,0].imshow(ratio_k_1)
#     ax[1,0].title.set_text(f"Ni[{k_1+1}+] / coefficients 2,3")
#     im5=ax[1,1].imshow(ratio_k_2)
#     ax[1,1].title.set_text(f"Ni[{k_2+1}+] / coefficients 2,3")
#     im6=ax[1,2].imshow(overlap_pixels)
#     ax[1,2].title.set_text(f"Overlapping pixels of {k_1+1} and {k_2+1}")
#     fig.suptitle(f"{sample_id}")
#     #im3 needs to be replaced afterwards
#     fig.colorbar(im1,ax=ax[0,0],fraction=0.046,pad=0.04)
#     fig.colorbar(im2,ax=ax[0,1],fraction=0.046,pad=0.04)
#     fig.colorbar(im3,ax=ax[0,2],fraction=0.046,pad=0.04)
#     fig.colorbar(im4,ax=ax[1,0],fraction=0.046,pad=0.04)
#     fig.colorbar(im5,ax=ax[1,1],fraction=0.046,pad=0.04)
#     fig.colorbar(im6,ax=ax[1,2],fraction=0.046,pad=0.04)
#     #for single plot
#     # fig,ax=plt.subplots(figsize=(4,4))
#     # ax.imshow(coeffs[0,:,:])
#     # ax.title.set_text(f"{sample_id} (Ni2+)")
#     # fig.colorbar(ax.imshow(coeffs[0,:,:]),ax=ax,fraction=0.046,pad=0.04)
#     plt.tight_layout()
#     plt.savefig(LCF_datapath + f"\\corrected_maps_n2_n3\\{sample_id}.png",dpi=300)
#     #plt.imsave(LCF_datapath + f"\\mask_setup\\{sample_id}_map_ni2.png", coeffs[0, :, :])
#     # plt.imsave(LCF_datapath + f"\\initial_setup\\{sample_id}_map_ni3.png", coeffs[1, :, :])
#     # plt.imsave(LCF_datapath + f"\\initial_setup\\{sample_id}_map_ni2-3.png", ni_ratio)