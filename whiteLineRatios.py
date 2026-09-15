import numpy as np
import os
import tifffile as tiff
import matplotlib.pyplot as plt



path = "/Users/moritz/Library/Mobile Documents/com~apple~CloudDocs/Studium/Semester 5/Schlussprojekt/20230908_23136_AbsortionSpectrum/5/"
data = []
files = sorted(os.listdir(path))
for i in files:
    if i.endswith(".tif"):
        data.append(tiff.imread(os.path.join(path, i)).astype(float))
data = np.array(data)
globSpectrum = np.sum(data, axis=(1, 2))
L3_image = np.sum(data[39:90], axis = 0) / 51
L2_image = np.sum(data[135:199], axis = 0) / 64

plt.imshow(L3_image, cmap = "gray")
plt.show()
plt.imshow(L2_image, cmap = "gray")
plt.show()
plt.imshow(L3_image / L2_image, cmap = "gray")
plt.show()
plt.plot(globSpectrum)
plt.show()