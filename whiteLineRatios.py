import numpy as np
import os
import tifffile as tiff
import matplotlib.pyplot as plt

#Mac Moritz
#path = "/Users/moritz/Library/Mobile Documents/com~apple~CloudDocs/Studium/Semester 5/Schlussprojekt/20230908_23136_AbsortionSpectrum/5/"
#Windows Samuel
#path="C:\\Users\\Samuel_Uni\\Documents\\Uni\\Semester V\\PSI_projekt\\XAS_Plots\\10"
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

plt.imshow(L3_image, cmap = "gray")
plt.show()
plt.imshow(L2_image, cmap = "gray")
plt.show()
plt.imshow(L3_image / L2_image, cmap = "gray")
plt.show()
#change x-axis of the spectrum to energy values
start_energy=700
end_energy=744
energy=np.linspace(start_energy, end_energy, len(globSpectrum))
plt.plot(energy, globSpectrum)
plt.xlabel("Energy (eV)")
plt.ylabel("Intensity")
plt.show()
