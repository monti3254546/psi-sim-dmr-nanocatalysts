import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colormaps
import os
import tifffile as tiff


colors = colormaps["tab20"].colors
ru_counter = 0
ni_counter = 0

### IMPORT DATA
datapath = "/Users/moritz/Library/Mobile Documents/com~apple~CloudDocs/Studium/Semester 5/Schlussprojekt/monochromator/"

datafolders = [
    name for name in os.listdir(datapath)
    if os.path.isdir(os.path.join(datapath, name))
]

# Create figure with two plots
fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(14, 6))

for i in range(len(datafolders)):
    path = os.path.join(datapath, datafolders[i])
    datafiles = sorted(os.listdir(path))

    if len(datafiles) < 5:
        print("scan aborted, continuing")
        continue

    data = []

    for j in range(len(datafiles)):
        if datafiles[j].endswith(".tiff"):
            data.append(
                tiff.imread(
                    os.path.join(path, datafiles[j])
                ).astype(float)
            )

    data = np.array(data)

    print(path)

    energies = np.loadtxt(
        path + "/scan_1.txt",
        delimiter=";",
        skiprows=1,
        usecols=0
    )

    roi1_data = data[:, 235:235+83, 88:88+92]
    roi2_data = data[:, 339:339+83, 347:347+92]

    roiSpectrum1 = np.sum(roi1_data, axis=(1, 2))
    roiSpectrum2 = np.sum(roi2_data, axis=(1, 2))

    # Select plot based on energy range
    useRu = False
    if np.all(energies < 600):
        ax = ax_left
        useRu = True
        ru_counter += 1
    else:
        ax = ax_right
        ni_counter += 1

    ax.plot(
        energies,
        roiSpectrum1,
        label="roiNi-" + datafolders[i][9:13], 
        color=colors[ru_counter if useRu else ni_counter], 
        linewidth = 2
    )

    ax.plot(
        energies,
        roiSpectrum2,
        label="roiBG-" + datafolders[i][9:13], 
        color=colors[ru_counter if useRu else ni_counter], 
        linestyle = ":", 
        linewidth = 2
    )

ax_left.set_title("Ruthenium")
ax_left.set_xlabel("Energy (eV)")
ax_left.set_ylabel("Intensity")
ax_left.legend()

ax_right.set_title("Nickel")
ax_right.set_xlabel("Energy (eV)")
ax_right.set_ylabel("Intensity")
ax_right.legend()

plt.tight_layout()


plt.show()