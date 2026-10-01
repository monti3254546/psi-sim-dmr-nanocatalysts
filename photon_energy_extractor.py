#extract photon energy from TIFF file headers and txt files and save to CSV
import csv
import os
import re
import numpy as np
import tifffile as tiff

index = "1737"  # Example index, replace with actual index as needed
#datapath  Windows Samuel
datapath = (
    "C:\\Users\\Samuel_Uni\\psi-sim-dmr-nanocatalysts\\20260918_"
    + index
    + "_AbsortionSpectrum"
)
#Name and path for final CSV
final_filename="AS_" + index + "_photonEnergies.csv"
final_datapath="C:\\Users\\Samuel_Uni\\psi-sim-dmr-nanocatalysts"

# 1. Read photon energy from the TXT file
photon_energy_txt = np.loadtxt(
    datapath + "\\scan_1.txt", delimiter=";", skiprows=1, usecols=0
)

# 2. Extract photon energy directly from TIFF file headers
photon_energy_tfiles = []
tiff_files = sorted(
    [f for f in os.listdir(datapath) if f.lower().endswith((".tiff", ".tif"))]
)


for file in tiff_files:
    filepath = os.path.join(datapath, file)
    with tiff.TiffFile(filepath) as tif:
        page = tif.pages[0]
        energy = None

        # Search across all header tags for energy strings (e.g., 'Photon Energy = 530.5')
        header_content = " ".join([str(tag.value) for tag in page.tags.values()])
        match = re.search(
            r"(?:photon\s*)?energy\s*[:=]\s*([\d\.]+)",
            header_content,
            re.IGNORECASE,
        )

        if match:
            energy = float(match.group(1))

        photon_energy_tfiles.append(energy)

# 3. extract intensity values from Tiff files and save to CSV
intensity_values = []
tiff_files = sorted(
    [f for f in os.listdir(datapath) if f.lower().endswith((".tiff", ".tif"))]
)

# 4. Save extracted TIFF energy and TXT energy directly to CSV
csv_path = os.path.join(final_datapath, final_filename)

with open(csv_path, mode="w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["Energy_TIFF", "Energy_TXT"])

    for e_tiff, e_txt in zip(photon_energy_tfiles, photon_energy_txt):
        writer.writerow([e_tiff, e_txt])

print(f"Saved extracted energies to: {csv_path}")