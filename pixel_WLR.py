"""
White Line Ratios:
Moritz hat die Kalibration begonnen zu implementieren.
Die L2+L3 Peak-Struktur werden White Lines genannt.
Ratios der Integrale bilden.

Noch nicht fertig.
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
#  REFERENCE & DATA IMPORT
# -------------------- #

if username == 'Moritz':
    referencePath = "/Users/moritz/Library/Mobile Documents/com~apple~CloudDocs/Studium/Semester 5/Schlussprojekt/psi-sim-dmr-nanocatalysts/activeRefs/"

elif username == 'Tim':
    referencePath = "activeRefs/"
    datapath = "C:\Users\Tim Elsener ETHZ\OneDrive\Dokumente\04_Studium\2_Vorlesungen\Projekte & Praktika\PSI-Projekt (Sem5)\Datenexport USB-Stick 30.10.2026"
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

# Loop over each spectrum pair (Cols 0&1, 2&3, 4&5)  for ni in "Reference_EELS_spectra.csv"
for i in range(2, 6, 2):
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

for i in range(len(refs)):
    energies = ref_energies[i]
    mask = np.ones_like(energies, dtype=bool)
    for low, high in lines:
        mask &= ~((energies >= low) & (energies <= high))
    popt, pcov = sp.optimize.curve_fit(eels_atan_fit, energies[mask], refs[i][mask])
    if i < 2:
        plt.plot(energies, refs[i] - eels_atan_fit(energies, *popt))
    else:
        plt.plot(energies, refs[i])
    #plt.plot(energies, eels_atan_fit(energies, *popt))
    #plt.plot(energies, refs[i])
plt.show()

ox_states = []
wlrs = []
for i in range(len(refs)):  # remember: "refs" contains the intensities
    # Riemann sums
    edge3 = np.sum(refs[i][ref_energies[i] > sepE])
    edge2 = np.sum(refs[i][ref_energies[i] < sepE])
    WLR = edge3 / edge2
    ox_states.append(int(ref_names[i][2]))
    wlrs.appens(WLR)

plt.scatter(ox_states, WLR)
plt.show()


# -------------------- #
# IMPORTING EXPERIMENTAL DATA & PROCESSING
# -------------------- #
#DONE: Replace "datafolders", "datapath", and "cut_regions" with something reasonable (see "data-analysis.py")
#TODO: Import data spectra without np.abs() from Moritz's OneDrive (we switched from iCloud)
#TODO: Test run with only one experimental data stack
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


# -------------------- #
# DATA PREPARATION & CALCULATING WLRs
# -------------------- #

for i in range(len(datafolders)):   # all folders, e.g. "20260918_1750_AbsortionSpectrum"
    #print(datafolders[i][9:13])


    try:
        path = os.path.join(datapath, datafolders[i])
        tiffpath = os.path.join(path + "/background-corrected")
        datafiles = sorted(os.listdir(tiffpath))    # list containing the tiff stack
    except FileNotFoundError:
        print("no drift correction yet, continuing")
        continue
    if len(datafiles) < 5:
        print("scan aborted, continuing")
        continue

    data = []

    for j in range(len(datafiles)): # go through all datafiles in one tiff stack
        if datafiles[j].endswith(".tif") or datafiles[j].endswith(".tiff"):
            data.append(
                tiff.imread(
                    os.path.join(tiffpath, datafiles[j])
                ).astype(float)
            )


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

    data_patched = np.zeros((energies.shape[0], data.shape[1], data.shape[2]))
    data_corr = np.zeros((energies.shape[0], data.shape[1], data.shape[2])) #background corrected data

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

    # calculate white line ratios (WLRs)


    # -------------------- #
    # PLOTTING EXPERIMENTAL & REFERENCE WLRs
    # -------------------- #


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