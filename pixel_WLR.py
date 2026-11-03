"""
White Line Ratios:
Moritz hat die Kalibration begonnen zu implementieren.
Die L2+L3 Peak-Struktur werden White Lines genannt.
Ratios der Integrale bilden.

Noch nicht fertig.
(Stand: 27.10.2026)
"""
import numpy as np
import os
import tifffile as tiff
import scipy as sp
import matplotlib.pyplot as plt
import copy
import pandas as pd

# user
username = 'Moritz'    # 'Moritz' or 'Tim'


#REFERENCE IMPORT

if username == 'Moritz':
    referencePath = "/Users/moritz/Library/Mobile Documents/com~apple~CloudDocs/Studium/Semester 5/Schlussprojekt/psi-sim-dmr-nanocatalysts/activeRefs/"

elif username == 'Tim':
    referencePath = None


reffiles = sorted(os.listdir(referencePath))

<<<<<<< HEAD
run = "ni"
refs = []
ref_energies = []
ref_names = []
=======
run = "ru"  # 'ru' or 'ni' (?)
refs = []   # spectra
ref_energies = []   # energies
ref_names = []  # file names

>>>>>>> c85ee5d4d10853b7305374a2bc0592e71d1af72a
for i in reffiles:
    if i.endswith(".txt") and run in i: # only 'ru' or 'ni' .txt files
        print(i)
        ref = np.loadtxt(
            os.path.join(referencePath, i),
            delimiter=",",
            skiprows=0
        )

        ref_energies.append(np.array(ref[:, 0]))
        refAppend = ref[:, 1]   #spectra
<<<<<<< HEAD
        refAppend -= np.min(refAppend)
        refAppend = np.array(refAppend)
        refs.append(refAppend / np.max(refAppend))
=======
        refAppend -= np.min(refAppend)  # rough baseline correction (set minimum to zero)
        refs.append(np.array(refAppend))
>>>>>>> c85ee5d4d10853b7305374a2bc0592e71d1af72a
        ref_names.append(i)

# Load the single CSV file containing all 3 spectra
csv_path = os.path.join(referencePath, "Reference_EELS_spectra.csv")
df = pd.read_csv(csv_path, delimiter=";")

<<<<<<< HEAD
# Loop over each spectrum pair (Cols 0&1, 2&3, 4&5)
for i in range(2, 6, 2):
    pair_data = df.iloc[:, [i, i+1]].dropna().to_numpy()
    
    ref_energies.append(np.array(pair_data[:, 0]))
    
    refAppend = pair_data[:, 1]  # spectra
    refAppend -= np.min(refAppend)
    refAppend = np.array(refAppend)
    refs.append(refAppend / np.max(refAppend))
    
    ref_names.append(df.columns[i+1])  # Uses column name as label


sepE_ni = 864
=======
# separating the spectrum into L3- and L2-edge regions
sepE_ni = 865
>>>>>>> c85ee5d4d10853b7305374a2bc0592e71d1af72a
sepE_ru = 479

sepE = sepE_ni if run == "ni" else sepE_ru
lines = [(852, 864), (870, 877)]

def eels_atan_fit(x, a, c, e):
    return a*np.arctan((x - 855)) + c * np.arctan((x - 872)) + e

for i in range(len(refs)):
    energies = ref_energies[i]
    mask = np.ones_like(energies, dtype=bool)
    for low, high in lines:
        mask &= ~((energies >= low) & (energies <= high))
    popt, pcov = sp.optimize.curve_fit(eels_atan_fit, energies[mask], refs[i][mask])
    if i < 2:
        plt.plot(energies, refs[i] - eels_atan_fit(energies, *popt))
    else:
        plt.plot(energies, refs[i])
    #plt.plot(energies, eels_atan_fit(energies, *popt))
    #plt.plot(energies, refs[i])
plt.show()

ox_states = []
wlrs = []
for i in range(len(refs)):
    # Riemann sums
    edge3 = np.sum(refs[i][ref_energies[i] > sepE])
    edge2 = np.sum(refs[i][ref_energies[i] < sepE])
    WLR = edge3 / edge2
    ox_states.append(int(ref_names[i][2]))
    wlrs.appens(WLR)

plt.scatter(ox_states, WLR)
plt.show()




