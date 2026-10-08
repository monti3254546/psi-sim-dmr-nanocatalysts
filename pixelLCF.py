import numpy as np
import os
import tifffile as tiff
import scipy as sp
import matplotlib.pyplot as plt
import copy


#REFERENCE IMPORT

referencePath = "/Users/moritz/Library/Mobile Documents/com~apple~CloudDocs/Studium/Semester 5/Schlussprojekt/psi-sim-dmr-nanocatalysts/activeRefs/"

reffiles = sorted(os.listdir(referencePath))

run = "ni"
refs = []
ref_energies = []
ref_names = []
for i in reffiles:
    if i.endswith(".txt") and run in i:
        ref = np.loadtxt(
            os.path.join(referencePath, i),
            delimiter=",",
            skiprows=0
        )

        ref_energies.append(ref[:, 0] + 12.8)  #energy grid. shift systematic error of 12.5 eV
        refAppend = ref[:, 1]   #spectra
        refAppend -= np.min(refAppend)
        #refAppend /= np.max(refAppend)
        refs.append(refAppend)
        ref_names.append(i)
        print(i)


for i in range(len(refs)):
    refs[i] = sp.ndimage.gaussian_filter(refs[i], sigma = 45)   #sigma = 45 seems to yield a similar degree of detail to the data --> best for fitting
    plt.plot(ref_energies[i], refs[i])
plt.show()


#interpolate to 0.2 eV grid, aligned to a .0 grid (i.e. 1, 1.2, 1.4, 1.6, ...)
for i in range(len(refs)):  #E_ref is the energy grid of the ref spectrum, ref the spectrum itself
    newAxisStart = round(min(ref_energies[i]) * 5) / 5
    newAxisEnd = round(max(ref_energies[i]) * 5) / 5
    newAxis = np.arange(newAxisStart, newAxisEnd, 0.2)
    ref_interp = np.interp(
        newAxis, #resample refs to 0.2 eV steps
        ref_energies[i],
        refs[i]
    )
    refs[i] = ref_interp
    ref_energies[i] = newAxis

refEnergyRanges = []   #use this to prevent shifting peaks out of the spectrum
for i in range(len(ref_energies)):  #loop through the energy axis of all ref spectra
    refEnergyRanges.append([min(ref_energies[i]), max(ref_energies[i])])
refEnergyRanges = np.array(refEnergyRanges)

lowerBoundRefs = np.min(refEnergyRanges)
upperBoundRefs = np.max(refEnergyRanges)

globRefEnergy = np.arange(0, 1000, 0.2)  #USE THIS FROM NOW ON, ITS GLOBAL FOR ALL REFERENCE SPECTRA
print(globRefEnergy)
for i in range(len(refs)):
    padLeft = int(round((min(ref_energies[i]) - 0) * 5))
    padRight = int(round((999.8 - max(ref_energies[i])) * 5))
    refs[i] = np.pad(refs[i], (padLeft, padRight), mode = "constant", constant_values = 0)


#DATA IMPORT
datapath = "/Users/moritz/Library/Mobile Documents/com~apple~CloudDocs/Studium/Semester 5/Schlussprojekt/20260918_Ni_Ru_AbsorptionSpectrum v5-real/"
datafolders = [
    name for name in os.listdir(datapath)
    if os.path.isdir(os.path.join(datapath, name))
]


for i in range(len(datafolders)):
    sample_id = datafolders[i][9:13]
    try:
        path = os.path.join(datapath, datafolders[i])
        tiffpath = os.path.join(path + "/background-corrected")
        datafiles = sorted(os.listdir(tiffpath))
    except FileNotFoundError:
        print("no background correction yet, continuing")
        continue
    if len(datafiles) < 5:
        print("scan aborted, continuing")
        continue

    data = []
    energies = []

    for j in range(len(datafiles)):
        if datafiles[j].endswith(".tif") or datafiles[j].endswith(".tiff"):
            data.append(
                tiff.imread(
                    os.path.join(tiffpath, datafiles[j])
                ).astype(float)
            )
            energies.append(float(datafiles[j][-10:-5]))


    data = np.array(data)
    energies = np.array(energies)
    if run == "ni" and np.any(energies < 600): continue
    elif run == "ru" and np.any(energies > 600): continue

    mask = ((globRefEnergy >= np.min(energies) - 0.0001) & (globRefEnergy <= np.max(energies) + 0.0001))  #discard any ref spectra regions that are not in our data
    refs_temp = np.column_stack(refs)[mask, :]
    globRefEnergy_temp = globRefEnergy[mask]
    
    
    coeffs = np.zeros((len(ref_names), data.shape[1], data.shape[2]))
    for m in range(data.shape[1]):
        for n in range(data.shape[2]):

            target = data[:, m, n]

            # Non-negative least squares
            coefficients, residual_norm = sp.optimize.nnls(
                refs_temp,
                target
            )
            coeffs[:, m, n] = coefficients


            #fitted = refs @ coefficients
            #plt.plot(energies, target)
            #plt.plot(globRefEnergy, fitted)
            #print(coefficients)
            #plt.show()
            
    # Calculate ratio maps
    for k in range(coeffs.shape[0]):
        coeffs[k, :, :] = np.clip(coeffs[k, :, :], 0, np.percentile(coeffs[k, :, :], 99.9))
        #plt.imsave(datapath + f"oxideMaps/{sample_id}_map_{ref_names[k]}.png", coeffs[k, :, :])

    #ni_ratio = coeffs[0, :, :] / coeffs[1, :, :]    # ni 2 over ni3
    #ni_ratio = np.nan_to_num(ni_ratio)
    #ni_ratio = np.clip(ni_ratio, 0, 10)
#
    ## Plot Coeff 0, Coeff 1, and Ratio side-by-side
    #titles = ["Coeff 0 (Ni2+)", "Coeff 1 (Ni3+)", "Ni2+ / Ni3+ Ratio"]
    #maps = [coeffs[0, :, :], coeffs[1, :, :], ni_ratio]
#
    #fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    #for i in range(3):
    #    im = axes[i].imshow(maps[i])
    #    axes[i].set_title(titles[i])
    #    fig.colorbar(im, ax=axes[i], fraction=0.046, pad=0.04)
#
    #plt.tight_layout()
    #plt.close()

    #plt.imsave(datapath + f"{sample_id}_map_ni2.png", coeffs[0, :, :])
    #plt.imsave(datapath + f"{sample_id}_map_ni3.png", coeffs[1, :, :])
    #plt.imsave(datapath + f"{sample_id}_map_ni2-3.png", ni_ratio)