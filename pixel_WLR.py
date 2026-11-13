"""
White Line Ratios:
Moritz hat die Kalibration begonnen zu implementieren.
Die L2+L3 Peak-Struktur werden White Lines genannt.
Ratios der Integrale bilden.

Noch nicht fertig. Ergänzt & Leistung verbessert mit Gemini.
(Stand: 04.11.2026)
"""
import numpy as np
import os
import tifffile as tiff
import scipy as sp
import matplotlib.pyplot as plt
import copy
import pandas as pd

# user
username = 'Tim'    # 'Moritz' or 'Tim'

# -------------------- #
#  REFERENCE & EXPERIMENTAL DATA IMPORT
# -------------------- #

if username == 'Moritz':
    referencePath = "/Users/moritz/Library/Mobile Documents/com~apple~CloudDocs/Studium/Semester 5/Schlussprojekt/psi-sim-dmr-nanocatalysts/activeRefs/"

elif username == 'Tim':
    referencePath = "activeRefs/"
    # raw string to prevent unicodeescape syntax error
    datapath = r"C:\Users\Tim Elsener ETHZ\OneDrive\Dokumente\04_Studium\2_Vorlesungen\Projekte & Praktika\PSI-Projekt (Sem5)\Datenexport USB-Stick 30.10.2026"
    datafolders = [
            name for name in os.listdir(datapath)
            if os.path.isdir(os.path.join(datapath, name))
        ]

# extracting reference file data
reffiles = sorted(os.listdir(referencePath))

# importing CTM4XAS reference data from 'activeRefs' folder
run = "ni"
refs = []           # intensities
ref_energies = []   # energies
ref_names = []      # names
for i in reffiles:
    if i.endswith(".txt") and run in i: # only 'ru' or 'ni' .txt files
        #print(i)
        ref = np.loadtxt(
            os.path.join(referencePath, i),
            delimiter=",",
            skiprows=0
        )

        ref_energies.append(np.array(ref[:, 0]))
        refAppend = ref[:, 1]   #spectra
        refAppend -= np.min(refAppend)  # set minimum value to zero
        refAppend = np.array(refAppend)
        refs.append(refAppend / np.max(refAppend))  # normalise by maximum value
        ref_names.append(i)

# Load the single CSV file containing all 3 spectra
csv_path = os.path.join(referencePath, "Reference_EELS_spectra.csv")
df = pd.read_csv(csv_path, delimiter=";")

num_cols = df.shape[1]  # number of columns

# Loop over each spectrum pair (Cols 0&1, 2&3, 4&5)  for ni in "Reference_EELS_spectra.csv"
for i in range(2, num_cols - 1, 2):
    pair_data = df.iloc[:, [i, i+1]].dropna().to_numpy()
    
    ref_energies.append(np.array(pair_data[:, 0]))
    
    refAppend = pair_data[:, 1]  # spectra
    refAppend -= np.min(refAppend) # set minimum value to zero
    refAppend = np.array(refAppend)
    refs.append(refAppend / np.max(refAppend))  # normalise by maximum value
    
    ref_names.append(df.columns[i+1])  # Uses column name as label

# -------------------- #
# CALCULATING REFERENCE WLRs
# -------------------- #
#TODO: understand what this does & in what formats it is saved

# separation energies: divide the data spectra in two halves for separate integration later
sepE_ni = 864
sepE_ru = 479

sepE = sepE_ni if run == "ni" else sepE_ru
#lines = [(852, 864), (870, 877)]    # old conditions?
lines = [(463.5, 471.5), (485.5, 492.5), (864, 872), (882, 887)]

def eels_atan_fit(x, a, c, e):
    return a*np.arctan((x - 855)) + c * np.arctan((x - 872)) + e

for i in range(len(refs)):  # 'refs' is a list containing the reference spectra's intensities
    energies = ref_energies[i]
    mask = np.ones_like(energies, dtype=bool)
    for low, high in lines:
        mask &= ~((energies >= low) & (energies <= high))
    popt, pcov = sp.optimize.curve_fit(eels_atan_fit, energies[mask], refs[i][mask])
    if i < 2:
        plt.plot(energies, refs[i] - eels_atan_fit(energies, *popt), label=ref_names[i])
    else:
        plt.plot(energies, refs[i], label=ref_names[i])
    #plt.plot(energies, eels_atan_fit(energies, *popt))
    #plt.plot(energies, refs[i])

plt.axvline(x=sepE, color='r', linestyle='--', label='Seperation Energy')
plt.xlabel('Energy in eV')
plt.ylabel('Intensity (arbitrary units)')
plt.legend()
plt.grid()
plt.title('Reference spectra')
print('\nPlotted reference spectra.')
plt.show()

ox_states = []
wlrs = []
for i in range(len(refs)):  # remember: "refs" contains the intensities
    # Riemann sums
    #TODO: Adapt/remove the lower and upper boundaries (850, 874.5). I simply added them to compare 4 different Ni2+ ref spectra more easily.
    edge3 = np.sum(refs[i][(850 < ref_energies[i]) & (ref_energies[i] < sepE)])
    edge2 = np.sum(refs[i][(sepE < ref_energies[i]) & (ref_energies[i] < 874.5)])
    WLR = edge3 / edge2
    ox_states.append(int(ref_names[i][2]))
    wlrs.append(WLR)

plt.scatter(ox_states, WLR, label='WLR ' + ref_names[i])
plt.xlabel('Oxidation state')
plt.ylabel('White line ratio (L3/L2)')
plt.legend()
plt.grid()
plt.title(f'Reference spectra WLRs using {sepE} as separation energy')
print('\nPlotted WLRs.')
plt.show()



# -------------------- #
# PROCESSING EXPERIMENTAL DATA
# -------------------- #

#DONE: Replace "datafolders", "datapath", and "cut_regions" with something reasonable (see "data-analysis.py")
#DONE: Import data spectra without np.abs() from Moritz's OneDrive (we switched from iCloud)
#TODO: Test run with only one experimental data stack
#DONE: Speed up nested for m... & for n... loop (see answer Gemini Pro)
#TODO: Compare WLR with references (linear interpolation between oxidation states)
#TODO: Make a chemical map showing the WLR, assigning certain WLR values distinct oxidation states
#TODO: Do this for all experiments (all stacks)
#TODO: Make a nice overview (plots) for all experiments comparing it to other approaches?<

# dictionary containing special conditions
cut_regions = {
    1776: [("above", 490)],
    1792: [("below", 852), ("above", 891)],
    1768: [("above", 889)],
    1780: [("above", 889)],
    1761: [("between", 474, 479)],
    1787: [("between", 869.2, 872)],
    1781: [("point", 866.8)],
    1756: [("between", 867.8, 868.8)],
    1782: [("point", 863)]
}


for i in range(len(datafolders)):   # all folders, e.g. "20260918_1750_AbsortionSpectrum"

    i=35 #TODO: remove this so that all experimental stacks are investigated.
    print(f'\nProcessing experiment {datafolders[i][9:13]}...\n') # experiment number

    try:
        path = os.path.join(datapath, datafolders[i])
        tiffpath = os.path.join(path, "background-corrected")
        
        # Filter for valid, non-hidden TIFF files immediately
        datafiles = sorted([
            f for f in os.listdir(tiffpath)
            if not f.startswith(".") and f.lower().endswith((".tif", ".tiff"))
        ])
    except FileNotFoundError:
        print("no drift correction yet, continuing")
        continue
        
    if len(datafiles) < 5:
        print("scan aborted, continuing")
        continue

    # Since datafiles is already strictly valid TIFFs, the loop becomes much cleaner (see Gemini):
    data = [
        tiff.imread(os.path.join(tiffpath, filename)).astype(float)
        for filename in datafiles
    ]


    data = np.array(data)   # np array containing the extracted tiff image information. Each element corresponds to one tiff file.

    energies = np.loadtxt(path + "/scan_1.txt", delimiter=";", skiprows=1, usecols=0)
    keithley = np.loadtxt(path + "/scan_1.txt", delimiter=";", skiprows=1, usecols=4)
    #print(data.shape)
    spec_raw = np.sum(data, axis = (1, 2))
    
    data /= keithley[:, None, None]
    energies_full = energies.copy()

    sample_id = int(datafolders[i][9:13])
    gapMask = np.ones_like(energies, dtype=bool)
    trim_mask = np.ones_like(energies, dtype=bool)

    # making sure that the correct seperation energy is applied
    is_ni = np.nanmean(energies) > 600
    sepE = sepE_ni if is_ni else sepE_ru
    
    for condition in cut_regions.get(sample_id, []):    # get all special conditions for certain experiments that show particularly noisy pixels
        if condition[0] == "above":
            trim_mask &= (energies <= condition[1])

        elif condition[0] == "below":
            trim_mask &= (energies >= condition[1])

        elif condition[0] == "between":
            _, low, high = condition
            gapMask &= ~((energies >= low) & (energies <= high))

        elif condition[0] == "point":
            gapMask &= ~np.isclose(energies, condition[1], atol=1e-6)

    energies = energies[trim_mask]
    gapMask = gapMask[trim_mask]

    # mask out the actual L3 and L2 peaks
    mask = np.ones_like(energies, dtype=bool)
    for low, high in lines:
        mask &= ~((energies >= low) & (energies <= high))

    # Note: This is Moritz's nested for m/for n... loop. Takes long to compiled, thus replaced with Gemini suggestion below.
    """

    data_patched = np.zeros((energies.shape[0], data.shape[1], data.shape[2]))
    data_corr = np.zeros((energies.shape[0], data.shape[1], data.shape[2])) #background corrected data
    
    # :param data_corr: 3-dimensional np array. Format: np.array((energy axis, pixel y, pixel x))
    

    # pixel-by-pixel gap petching
    for m in range(data.shape[1]):
        #print(m)
        for n in range(data.shape[2]):
            pxspec = data[:, m, n].copy()   # 1D energy spectrum
            pxspec = pxspec[trim_mask]

            pxspec_patched = pxspec.copy()

            invalid_idx = np.where(~gapMask)[0]
            valid_idx = np.where(gapMask)[0]

            if len(invalid_idx) > 0 and len(valid_idx) > 0: #check whether there is a gap and whether there are valid points to calculate the fit from
                for idx in invalid_idx:
                    # Grab up to 4 valid neighboring points on left and right
                    left = valid_idx[valid_idx < idx][-4:]
                    right = valid_idx[valid_idx > idx][:4]
                    neighbors = np.concatenate([left, right])

                    # smoothly extrapolate the missing datapoints using a 3rd-degree polynomial fit
                    if len(neighbors) > 3:
                        # Fit a cubic local baseline through surrounding valid data
                        coeffs = np.polyfit(energies[neighbors], pxspec[neighbors], deg=3)
                        pxspec_patched[idx] = np.polyval(coeffs, energies[idx])
            data_patched[:, m, n] = pxspec_patched
            pxspec = pxspec_patched.copy()

            # background fitting using a 5th-degree polynomial fit
            bgfit = np.polyfit(energies[mask], pxspec[mask], 5)
            pxspec_corr = pxspec - np.polyval(bgfit, energies)  # subtract 5th-degree polynomial fit from the entire spectrum

            data_corr[:, m, n] = pxspec_corr
            #plt.plot(energies, pxspec_corr)
            #plt.plot(energies, pxspec)
            #plt.plot(energies, np.polyval(bgfit, energies))
            #plt.title(f"Sample {sample_id, m, n}")
            #plt.show()

    data_corr[:, :, :10] = 0    #many broken pixels there
    invalidDataMask = np.any(data_patched == 0, axis = 0)  #1 for all invalid pixels. PROBLEM: when using data, we havened done any interpolation yet, so there are gaps causing zeros. if we use data_corr, the zero areas are no longer zero due to background subtraction. we have to use an intermediate stage inbetween
    validDataMask = ~invalidDataMask    #1 for all valid pixels
    validDataMask = sp.ndimage.binary_fill_holes(validDataMask)
    data_corr[:, ~validDataMask] = 0
    """


    # -------------------- #
    # Vectorised patching & 5th-degree polynomial (Gemini)
    # -------------------- #

    # 1. Vectorized gap patching across all pixels simultaneously
    # (Preserves data_patched for the downstream invalidDataMask check)
    data_patched = data.copy()
    if not np.all(gapMask):
        interpolator = sp.interpolate.interp1d(
            energies[gapMask],
            data_patched[gapMask, :, :],
            kind="cubic",
            axis=0,
            fill_value="extrapolate",
        )
        data_patched[~gapMask, :, :] = interpolator(energies[~gapMask])

    # 2. Vectorized 5th-degree polynomial background subtraction
    n_energies, height, width = data_patched.shape
    data_flat = data_patched.reshape(n_energies, height * width)

    # Normalize energies to [-1, 1] to prevent numerical ill-conditioning from E^5 (e.g. 870^5)
    e_min, e_max = energies.min(), energies.max()
    e_norm = 2.0 * (energies - e_min) / (e_max - e_min) - 1.0

    # Solve the 5th-degree polynomial fit on the masked baseline for ALL pixels in one matrix step
    X_masked = np.vander(e_norm[mask], 6)
    coeffs, _, _, _ = np.linalg.lstsq(X_masked, data_flat[mask, :], rcond=None)

    # Evaluate the background polynomial across all energies and subtract
    bg_flat = np.vander(e_norm, 6) @ coeffs
    data_corr = data_patched - bg_flat.reshape(n_energies, height, width)

    #data_corr = np.abs(data_corr)  #setting all negative values to zero is bad for the WLR integration purpose!

    #map = np.zeros((data_corr.shape[1], data_corr.shape[2]))
    #for m in range(data_corr.shape[1]):
    #    print(m)
    #    for n in range(data_corr.shape[2]):
    #        if np.any(data_corr[:, m, n] < -2):
    #            map[m, n] = 1
    #            plt.plot(energies, data[:, m, n], label = str(m) + ", " + str(n))
    #            #plt.plot(energies, np.clip(data_corr[:, m, n], 0, None), linestyle = "--")
    #plt.legend()
    #plt.show()
    #plt.imshow(map)
    #plt.show()
    
    
    """
    out_dir = path + "/background-corrected/"
    os.makedirs(out_dir, exist_ok=True)
    stack_to_save = data_corr.astype('float32')
    for i, frame in enumerate(stack_to_save):
        pass
        #tiff.imwrite(out_dir + f"{sample_id}_b-corr_{i:03d}_{energies[i]:.1f}.tiff", frame)
    """

    # -------------------- #
    #   CALCULATING WLRs
    # -------------------- #
    
    #TODO: Why are there negative WLRs? (negative integrals?!)
    #TODO: Why are the WLRs so small? (reference WLR is 1.35)
    #TODO: Play with parameters (masks). Large area is NaN right now; do we want that?
    #TODO: Why is a single pixel's spectrum so messy (i. e. pixel (50, 50) as tested below)
    #TODO: Why do both wlr_map imshow plots look different?

    total_signal = np.nansum(data_corr, axis=0) # sets invalid NaN (not number values) to zero
    sample_mask = total_signal > (0.1 * np.nanmax(total_signal))   # ignore unphysical signals which are show less than 10 % intensity with respect to the maximum intensity
    # note: sample_mask is a 2D Boolean array (mask) containing True and False values.

    edge3_map = np.nansum(data_corr[energies < sepE, :, :], axis=0) # Riemann Sum
    edge2_map = np.nansum(data_corr[energies > sepE, :, :], axis=0) # Riemann Sum

    wlr_map = np.full(edge3_map.shape, np.nan, dtype=float) # same shape and data type
    valid = sample_mask & (edge2_map > 0)   # note: The "&" operator performs an element-by-element logical AND between those two grids.
    wlr_map[valid] = edge3_map[valid] / edge2_map[valid]    # white line ratio calculation

    vmin, vmax = np.nanpercentile(wlr_map, [2, 98]) # ignoring the bottom 2 percentile and the top 2 percentile
    plt.imshow(wlr_map, vmin=vmin, vmax=vmax, cmap="viridis")
    plt.colorbar(label="White Line Ratio (L3/L2)")
    plt.show()

    
    """
    edge3_map = np.sum(data_corr[energies < sepE, :, :], axis=0)
    edge2_map = np.sum(data_corr[energies > sepE, :, :], axis=0)
    

    # Avoid division by zero on masked/empty pixels
    wlr_map = np.divide(
        edge3_map, 
        edge2_map, 
        out=np.zeros_like(edge3_map, dtype=float), 
        where=edge2_map != 0
    )
    """

    # testing purposes only
    
    px, py = (360, 235) # that's where a ni particle is in experiment 
    plt.title(f'Energy @ spectrum pixel {px, py}, Experiment {datafolders[i][9:13]}')
    plt.plot(energies, data_corr[:, px, py])    # arbitrary pixel (20, 20)
    plt.show()
    """
    print(f'\nEnergies {energies}')
    print(f'\nEdge high map {edge2_map}')
    print(f'\nEdge low map {edge3_map}')
    print(f'\nHigh map how many non-zero? {len(np.where(edge2_map > 0))}')
    print(f'\nLow map how many non-zero? {len(np.where(edge3_map > 0))}')
    print(f'\nHighest and lowest WLR values on the map: {np.max(wlr_map), np.min(wlr_map)}')
    """

    print("Energies range:", energies.min(), "to", energies.max(), "| sepE:", sepE)
    print("Non-zero pixels in data_corr:", np.count_nonzero(data_corr))
    print("wlr_map dtype:", wlr_map.dtype, "| NaNs:", np.isnan(wlr_map).sum())
    print("wlr_map min / median / 99th / max:",
        np.nanmin(wlr_map),
        np.nanmedian(wlr_map),
        np.nanpercentile(wlr_map, 99),
        np.nanmax(wlr_map))

    plt.title(f'WLR map: Experiment {datafolders[i][9:13]}')
    plt.imshow(wlr_map)
    plt.colorbar()
    print('\nPlotted WLR map.')
    plt.show()


    # -------------------- #
    # PLOTTING EXPERIMENTAL & REFERENCE WLRs
    # -------------------- #

    break   #TODO: remove this so that all experimental stacks are investigated.

    """
    # average spectrum over all pixels
    newspec = np.mean(data_corr, axis = (1, 2))
    newspec /= np.max(newspec)
    newspec2 = sp.signal.savgol_filter(newspec, 11, 3)  # Savitzky-Golay filter to smoothen it out
    plt.plot(energies, newspec, label = "processed")
    plt.plot(energies, newspec2, label = "processed filtered")
    plt.plot(energies_full, spec_raw / np.max(spec_raw), label = "raw")
    plt.legend()
    plt.show()
    #plt.savefig(datapath + f"{sample_id}_image2.png", dpi=300)

    image = np.sum(data, axis = 0)
    image[:, :10] = 0
    image /= np.max(image)
    plt.imshow(image)
    plt.show()
    #isNi = np.all(energies > 600)
    """

print('\nDone.')