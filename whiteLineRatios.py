import os
import matplotlib.pyplot as plt
import numpy as np
import tifffile as tiff

index = 9   #1 to 10, both including. this dictates what sample to analyze
#Mac Moritz
#path = "/Users/moritz/Library/Mobile Documents/com~apple~CloudDocs/Studium/Semester 5/Schlussprojekt/20230908_23136_AbsortionSpectrum/" + str(index) + "/"
#Windows Samuel
path="C:\\Users\\Samuel_Uni\\Documents\\Uni\\Semester V\\PSI_projekt\\XAS_Plots\\" + str(index) + "\\"
energies = np.loadtxt(path[:-2] + "AbsortionSpectrum_23136_" + str(index).zfill(2) + ".csv", delimiter=",", skiprows=1, usecols=0)

data = []
files = sorted(os.listdir(path))
for i in files:
    if i.endswith(".tif"):
        data.append(tiff.imread(os.path.join(path, i)).astype(float))
data = np.array(data)
globSpectrum = np.sum(data, axis=(1, 2))
L3_image = np.sum(data[39:90], axis = 0) / 51
L2_image = np.sum(data[135:199], axis = 0) / 64
#Select area of interest
# L3_image = L3_image[100:200, 100:200]
# L2_image = L2_image[100:200, 100:200]

#plt.imshow(L3_image, cmap = "gray")
#plt.show()
#plt.imshow(L2_image, cmap = "gray")
#plt.show()

relative_absorption =np.nan_to_num(np.divide(L3_image, L2_image), nan=0)

#plt.imshow(relative_absorption, cmap="gray")
#plt.show()

#Plot single spectrum
# plt.plot(energies, globSpectrum)
# plt.xlabel("Energy (eV)")
# plt.ylabel("Intensity")
# plt.show()

# # Export the spectrum to a text file
# np.savetxt(
#     os.path.join(path[:-3], "globSpectrum.txt"),
#     np.column_stack((energies, globSpectrum)),
#     header="Energy (eV)\tIntensity",
#     fmt="%f\t%f",
# )
#Plot multiple spectra
# Load reference spectrum into folder
ref_spectrum = np.loadtxt(os.path.join(path[:-3], "refSpectrum_1.txt"), delimiter="\t", skiprows=1)
plt.plot(ref_spectrum[:, 0], ref_spectrum[:, 1], label="Reference Spectrum")
plt.plot(energies, globSpectrum, label="Measured Spectrum")
plt.xlabel("Energy (eV)")
plt.ylabel("Intensity")
plt.legend()
plt.show()


