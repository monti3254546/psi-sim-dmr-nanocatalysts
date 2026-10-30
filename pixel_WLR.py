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

# user
username = 'Moritz'    # 'Moritz' or 'Tim'


#REFERENCE IMPORT

if username == 'Moritz':
    referencePath = "/Users/moritz/Library/Mobile Documents/com~apple~CloudDocs/Studium/Semester 5/Schlussprojekt/psi-sim-dmr-nanocatalysts/activeRefs/"

elif username == 'Tim':
    referencePath = None


reffiles = sorted(os.listdir(referencePath))

run = "ru"  # 'ru' or 'ni' (?)
refs = []   # spectra
ref_energies = []   # energies
ref_names = []  # file names

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
        refAppend -= np.min(refAppend)  # rough baseline correction (set minimum to zero)
        refs.append(np.array(refAppend))
        ref_names.append(i)


# separating the spectrum into L3- and L2-edge regions
sepE_ni = 865
sepE_ru = 479

sepE = sepE_ni if run == "ni" else sepE_ru

for i in range(len(refs)):
    # Riemann sums
    edge3 = np.sum(refs[i][ref_energies[i] > sepE])
    edge2 = np.sum(refs[i][ref_energies[i] < sepE])
    WLR = edge3 / edge2
    ox_state = int(ref_names[i][2])
    plt.plot(ref_energies[i], refs[i])
    #plt.scatter(WLR, ox_state, label = str(ox_state))
    plt.legend()
plt.show()




