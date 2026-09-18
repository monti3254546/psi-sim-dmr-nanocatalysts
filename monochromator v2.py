import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colormaps
import os
import tifffile as tiff


colors = colormaps["tab20"].colors
counter = 0

### SELECT ENERGY RANGE
#energy_min = 450    #Ru
#energy_max = 510

energy_min = 855    #Ni
energy_max = 885

roi_BG = {
    
}

### IMPORT DATA
datapath = "/Users/moritz/Library/Mobile Documents/com~apple~CloudDocs/Studium/Semester 5/Schlussprojekt/monochromator v2/"

datafolders = [
    name for name in os.listdir(datapath)
    if os.path.isdir(os.path.join(datapath, name))
]

# Create figure with one plot
fig, ax = plt.subplots(figsize=(10, 6))

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

    # Only use scans within the selected energy range
    if np.all(energies < energy_min) or np.all(energies > energy_max):
        print("scan outside selected energy range, continuing")
        continue

    roi1_data = data[:, 235:235+83, 88:88+92]
    roi2_data = data[:, 339:339+83, 347:347+92]

    roiSpectrum1 = np.sum(roi1_data, axis=(1, 2))
    roiSpectrum2 = np.sum(roi2_data, axis=(1, 2))

    # Select only the requested energy range
    energy_mask = (energies >= energy_min) & (energies <= energy_max)

    energies_plot = energies[energy_mask]
    roiSpectrum1_plot = roiSpectrum1[energy_mask]
    roiSpectrum2_plot = roiSpectrum2[energy_mask]

    ax.plot(
        energies_plot,
        roiSpectrum1_plot,
        label="roiNi-" + datafolders[i][9:13],
        color=colors[counter],
        linewidth=2
    )

    ax.plot(
        energies_plot,
        roiSpectrum2_plot,
        label="roiBG-" + datafolders[i][9:13],
        color=colors[counter],
        linestyle=":",
        linewidth=2
    )

    counter += 1


ax.set_title("Monochromator")
ax.set_xlabel("Energy (eV)")
ax.set_ylabel("Intensity")
ax.set_xlim(energy_min, energy_max)
ax.legend()

plt.tight_layout()

plt.savefig(
    path + "monochromator-debugging.png",
    dpi=300
)

plt.show()