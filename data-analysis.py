import numpy as np
import pandas as pd
import tifffile as tiff
import os
import scipy as sp
import matplotlib.pyplot as plt



### IMPORT DATA
datapath = "/Users/moritz/Library/Mobile Documents/com~apple~CloudDocs/Studium/Semester 5/Schlussprojekt/20260918_Ni_Ru_AbsorptionSpectrum v5-real/"

datafolders = [
    name for name in os.listdir(datapath)
    if os.path.isdir(os.path.join(datapath, name))
]
lines = [(463.5, 471.5), (485.5, 492.5), (864, 872), (882, 887)]

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



# CSV file
roi_path = os.path.join("/Users/moritz/Library/Mobile Documents/com~apple~CloudDocs/Studium/Semester 5/Schlussprojekt/roi_bounds.csv")


# Create figure with one plot
fig, ax = plt.subplots(figsize=(6,6))

for i in range(len(datafolders)):
    print(datafolders[i][9:13])
    try:
        path = os.path.join(datapath, datafolders[i])
        tiffpath = os.path.join(path + "/drift-correction")
        datafiles = sorted(os.listdir(tiffpath))
    except FileNotFoundError:
        print("no drift correction yet, continuing")
        continue
    if len(datafiles) < 5:
        print("scan aborted, continuing")
        continue

    data = []

    for j in range(len(datafiles)):
        if datafiles[j].endswith(".tif") or datafiles[j].endswith(".tiff"):
            data.append(
                tiff.imread(
                    os.path.join(tiffpath, datafiles[j])
                ).astype(float)
            )


    data = np.array(data)

    energies = np.loadtxt(path + "/scan_1.txt", delimiter=";", skiprows=1, usecols=0)
    keithley = np.loadtxt(path + "/scan_1.txt", delimiter=";", skiprows=1, usecols=4)
    print(data.shape)
    spec_raw = np.sum(data, axis = (1, 2))
    
    data /= keithley[:, None, None]
    spec_norm = np.sum(data, axis = (1, 2))
    spec_norm /= np.max(spec_norm)
    plt.plot(energies.copy(), spec_norm.copy())


    energies_full = energies.copy()
    spec_norm_full = spec_norm.copy()

    sample_id = int(datafolders[i][9:13])
    gapMask = np.ones_like(energies, dtype=bool)
    trim_mask = np.ones_like(energies, dtype=bool)
    for condition in cut_regions.get(sample_id, []):
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
    spec_norm = spec_norm[trim_mask]
    gapMask = gapMask[trim_mask]

    spec_patched = spec_norm.copy()

    invalid_idx = np.where(~gapMask)[0]
    valid_idx = np.where(gapMask)[0]

    if len(invalid_idx) > 0 and len(valid_idx) > 0: #check whether there is a gap and whether there are valid points to calculate the fit from
        for idx in invalid_idx:
            # Grab up to 4 valid neighboring points on left and right
            left = valid_idx[valid_idx < idx][-4:]
            right = valid_idx[valid_idx > idx][:4]
            neighbors = np.concatenate([left, right])
            
            if len(neighbors) > 3:
                # Fit a cubic local baseline through surrounding valid data
                coeffs = np.polyfit(energies[neighbors], spec_norm[neighbors], deg=3)
                spec_patched[idx] = np.polyval(coeffs, energies[idx])
    spec_norm = spec_patched.copy()

    spec_filter = sp.signal.savgol_filter(spec_norm, 11, 3)

    mask = np.ones_like(energies, dtype=bool)
    for low, high in lines:
        mask &= ~((energies >= low) & (energies <= high))

    bgfit = np.polyfit(energies[mask], spec_norm[mask], 5)

    plt.plot(energies, spec_filter)
    plt.plot(energies, bgfit[0] * energies ** 5 + bgfit[1] * energies ** 4 + bgfit[2] * energies ** 3 + bgfit[3] * energies ** 2 + bgfit[4] * energies + bgfit[5])
    
    plt.plot(energies_full, keithley / np.max(keithley))
    

    spec_corr = spec_filter - (bgfit[0] * energies ** 5 + bgfit[1] * energies ** 4 + bgfit[2] * energies ** 3 + bgfit[3] * energies ** 2 + bgfit[4] * energies + bgfit[5])
    plt.plot(energies, spec_corr)
    plt.title(f"Sample {sample_id}")
    plt.savefig(datapath + f"{sample_id}_image.png", dpi=300)

    image = np.sum(data, axis = 0)
    image[:, :10] = 0
    image /= np.max(image)
    plt.imshow(image)
    plt.close()

    isNi = np.all(energies > 600)
    


