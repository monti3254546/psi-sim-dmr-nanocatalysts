import os
import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
import tifffile as tiff

index = 9   #1 to 10, both including. this dictates what sample to analyze
#Mac Moritz
datapath = "/Users/moritz/Library/Mobile Documents/com~apple~CloudDocs/Studium/Semester 5/Schlussprojekt/20230908_23136_AbsortionSpectrum/" + str(index) + "/"
referencePath = "/Users/moritz/Library/Mobile Documents/com~apple~CloudDocs/Studium/Semester 5/Schlussprojekt/psi-sim-dmr-nanocatalysts/"
#Windows Samuel
#datapath="C:\\Users\\Samuel_Uni\\Documents\\Uni\\Semester V\\PSI_projekt\\XAS_Plots\\" + str(index) + "\\"
energies = np.loadtxt(datapath[:-2] + "AbsortionSpectrum_23136_" + str(index).zfill(2) + ".csv", delimiter=",", skiprows=1, usecols=0)



def lcf(target, refs):
    
    coefficients, residual_norm = sp.optimize.nnls(refs, target)
    fitted = refs @ coefficients
    return coefficients, fitted

def globSpecLCF():
    globSpectrum -= np.min(globSpectrum)
    globSpectrum /= np.max(globSpectrum)

    coeffs, fitted_spectrum = lcf(globSpectrum, refs)
    print(coeffs)
    plt.plot(energies, globSpectrum / np.max(globSpectrum), label = "global")
    for i in range(len(refs)):
        plt.plot(energies, refs[i] / np.max(refs), label = "ref" + str(i+1))
    plt.plot(energies, fitted_spectrum / np.max(fitted_spectrum), label = "fitted")
    plt.legend()
    plt.show()




### IMPORT DATA

data = []

datafiles = sorted(os.listdir(datapath))

for i in datafiles:
    if i.endswith(".tif"):
        data.append(
            tiff.imread(
                os.path.join(datapath, i)
            ).astype(float)
        )

data = np.array(data)
data[np.isnan(data)] = 0


### IMPORT REFS

refs = []
ref_energies = []

reffiles = sorted(os.listdir(referencePath))

for i in reffiles:

    if i.endswith(".txt") and "fit" in i:

        ref = np.loadtxt(
            os.path.join(referencePath, i),
            delimiter=" ",
            skiprows=1
        )

        ref_energies.append(ref[:, 0])
        refs.append(ref[:, 1])


print("number of refs loaded:", len(refs))


### FIND COMMON ENERGY RANGE

lower_energy = max(
    energies.min(),
    *[E.min() for E in ref_energies]
)

upper_energy = min(
    energies.max(),
    *[E.max() for E in ref_energies]
)

print("common energy range:", lower_energy, "-", upper_energy)


### SELECT TARGET ENERGY POINTS IN COMMON RANGE

energy_mask = (
    (energies >= lower_energy) &
    (energies <= upper_energy)
)

fit_energies = energies[energy_mask]



### INTERPOLATE REFERENCES ONTO TARGET ENERGY GRID

refs_interp = []

for E_ref, ref in zip(ref_energies, refs):

    ref_interp = np.interp(
        fit_energies,
        E_ref,
        ref
    )

    refs_interp.append(ref_interp)

refs_interp = np.column_stack(refs_interp)

print("reference matrix:", refs_interp.shape)
print("target energy points:", len(fit_energies))


### LCF FUNCTION

def lcf(target, refs_interp):


    max_target = np.max(target)

    if max_target > 0:
        target = target / max_target

    # Normalize references using the same common scale
    refs_fit = refs_interp.copy()

    refs_fit -= np.min(refs_fit, axis=0)

    max_refs = np.max(refs_fit)

    if max_refs > 0:
        refs_fit /= max_refs

    # Non-negative least squares
    coefficients, residual_norm = sp.optimize.nnls(
        refs_fit,
        target
    )

    fitted = refs_fit @ coefficients

    return coefficients, fitted


### GLOBAL SPECTRUM

globSpectrum = np.sum(data, axis=(1, 2))
plt.plot(energies, globSpectrum)
globSpectrum = globSpectrum[energy_mask]
globSpectrum -= np.min(globSpectrum)
globSpectrum /= np.max(globSpectrum)

for i in range(len(refs)):
    plt.plot(
        fit_energies,
        refs_interp[:, i],
        label=f"ref {i+1} interpolated"
    )
plt.show()

coeffs, fitted_spectrum = lcf(
    globSpectrum,
    refs_interp
)


print("global coefficients:", coeffs)

## PLOT GLOBAL FIT
plt.plot(
    fit_energies,
    globSpectrum,
    label="global"
)
for i in range(refs_interp.shape[1]):
    plt.plot(
        fit_energies,
        refs_interp[:, i],
        label="ref " + str(i + 1)
    )

plt.plot(
    fit_energies,
    fitted_spectrum,
    label="fitted"
)
plt.xlabel("Energy")
plt.ylabel("Intensity")
plt.legend()
plt.show()


### PIXEL-BY-PIXEL LCF

coeff_ratio = np.full(
    data.shape[1:],
    np.nan,
    dtype=float
)

for i in range(data.shape[1]):

    for j in range(data.shape[2]):

        coeffs, _ = lcf(
            data[:, i, j],
            refs_interp
        )

        if coeffs[0] > 0:
            coeff_ratio[i, j] = coeffs[1] / coeffs[0]


### PLOT COEFFICIENT RATIO

plt.imshow(
    coeff_ratio,
    cmap="inferno"
)

plt.colorbar(label="Reference 2 / Reference 1")

plt.show()

#maxRefs = np.max(np.column_stack(refs))
#for i in range(len(refs)):
#    refs[i] -= np.min(refs[i])
#refs = np.column_stack(refs)
#refs /= maxRefs


#globSpectrum = np.sum(data, axis=(1, 2))

#print("target:", globSpectrum.shape, np.nanmin(globSpectrum), np.nanmax(globSpectrum))
#print("refs:", [r.shape for r in refs])
#print("ref ranges:", [(np.nanmin(r), np.nanmax(r)) for r in refs])
#print("NaNs target:", np.any(np.isnan(globSpectrum)))
#print("NaNs refs:", [np.any(np.isnan(r)) for r in refs])

#norm_data = data / np.max(data)
#coeff_ratio = np.zeros_like(data[0])
#for i in range(coeff_ratio.shape[0]):
#    for j in range(coeff_ratio.shape[1]):
#        coeffs, _ = lcf(norm_data[:, i, j], refs)
#        coeff_ratio[i, j] = coeffs[1] / coeffs[0]
#plt.imshow(coeff_ratio, cmap = "inferno")
#plt.show()

#image L3 and L2 peak intensities
#L3_image = np.sum(data[39:90], axis = 0) / 51
#L2_image = np.sum(data[135:199], axis = 0) / 64
#Select area of interest
# L3_image = L3_image[100:200, 100:200]
# L2_image = L2_image[100:200, 100:200]

#plt.imshow(L3_image, cmap = "gray")
#plt.show()
#plt.imshow(L2_image, cmap = "gray")
#plt.show()

#relative_absorption =np.nan_to_num(np.divide(L3_image, L2_image), nan=0)
#plt.imshow(relative_absorption, cmap="gray")
#plt.show()

#Plot single spectrum
# plt.plot(energies, globSpectrum)
# plt.xlabel("Energy (eV)")
# plt.ylabel("Intensity")
# plt.show()
#backgroundSpectrum = np.sum(data[:, 16:116, 58:149], axis=(1, 2))  #ONLY APPLIES TO SAMPLE 9, INDICES MIGHT HAVE TO BE CHANGED FOR OTHER SAMPLES
#structureSpectrum = np.sum(data[:, 193:227, 300:327], axis=(1, 2))
# Export the spectrum to a text file
#np.savetxt(
#     os.path.join(referencePath, "background_spectrum.txt"),
#     np.column_stack((energies, backgroundSpectrum)),
#     header="Energy (eV)\tIntensity",
#     fmt="%f\t%f",
#)
#Plot multiple spectra
#Load reference spectrum into folder
#ref_spectrum = np.loadtxt(os.path.join(referencePath + "refSpectrum_1.txt"), delimiter="\t", skiprows=1)



##plt.plot(ref_spectrum[:, 0], ref_spectrum[:, 1], label="Reference Spectrum")
#plt.plot(energies, backgroundSpectrum, label="Measured Spectrum")
#plt.xlabel("Energy (eV)")
#plt.ylabel("Intensity")
#plt.legend()
#plt.show()


