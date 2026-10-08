import numpy as np
import pandas as pd
import tifffile as tiff
import os
import random
import scipy as sp
import matplotlib.pyplot as plt


### IMPORT DATA
datapath = "/Users/moritz/Library/Mobile Documents/com~apple~CloudDocs/Studium/Semester 5/Schlussprojekt/20260918_Ni_Ru_AbsorptionSpectrum v5-real/"
#problem: in 1776, there is a gap, but also a different issue that causes not being able to bridge the gap by interpolation cause there is very few data behind
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
    energies_full = energies.copy()

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
    gapMask = gapMask[trim_mask]

    mask = np.ones_like(energies, dtype=bool)
    for low, high in lines:
        mask &= ~((energies >= low) & (energies <= high))

    data_patched = np.zeros((energies.shape[0], data.shape[1], data.shape[2]))
    data_corr = np.zeros((energies.shape[0], data.shape[1], data.shape[2])) #background corrected data
    for m in range(data.shape[1]):
        print(m)
        for n in range(data.shape[2]):
            pxspec = data[:, m, n].copy()
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
                    
                    if len(neighbors) > 3:
                        # Fit a cubic local baseline through surrounding valid data
                        coeffs = np.polyfit(energies[neighbors], pxspec[neighbors], deg=3)
                        pxspec_patched[idx] = np.polyval(coeffs, energies[idx])
            data_patched[:, m, n] = pxspec_patched
            pxspec = pxspec_patched.copy()

            bgfit = np.polyfit(energies[mask], pxspec[mask], 5)
            pxspec_corr = pxspec - np.polyval(bgfit, energies)

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
    data_corr = np.abs(data_corr)  #set all negative values to zero
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
    
    
    
    out_dir = path + "/background-corrected/"
    os.makedirs(out_dir, exist_ok=True)
    stack_to_save = data_corr.astype('float32')
    for i, frame in enumerate(stack_to_save):
        pass
        #tiff.imwrite(out_dir + f"{sample_id}_b-corr_{i:03d}_{energies[i]:.1f}.tiff", frame)


    newspec = np.mean(data_corr, axis = (1, 2))
    newspec /= np.max(newspec)
    newspec2 = sp.signal.savgol_filter(newspec, 11, 3)
    plt.plot(energies, newspec, label = "processed")
    plt.plot(energies, newspec2, label = "processed filtered")
    plt.plot(energies_full, spec_raw / np.max(spec_raw), label = "raw")
    plt.legend()
    #plt.savefig(datapath + f"{sample_id}_image2.png", dpi=300)
    plt.close()

    #image = np.sum(data, axis = 0)
    #image[:, :10] = 0
    #image /= np.max(image)
    #plt.imshow(image)
    #plt.close()
    #isNi = np.all(energies > 600)
    


