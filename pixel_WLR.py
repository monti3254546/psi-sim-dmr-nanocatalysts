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


#REFERENCE IMPORT

referencePath = "/Users/moritz/Library/Mobile Documents/com~apple~CloudDocs/Studium/Semester 5/Schlussprojekt/psi-sim-dmr-nanocatalysts/activeRefs/"

reffiles = sorted(os.listdir(referencePath))

run = "ru"
refs = []
ref_energies = []
ref_names = []
for i in reffiles:
    if i.endswith(".txt") and run in i:
        print(i)
        ref = np.loadtxt(
            os.path.join(referencePath, i),
            delimiter=",",
            skiprows=0
        )

        ref_energies.append(np.array(ref[:, 0]))
        refAppend = ref[:, 1]   #spectra
        refAppend -= np.min(refAppend)
        refs.append(np.array(refAppend))
        ref_names.append(i)


sepE_ni = 865
sepE_ru = 479

sepE = sepE_ni if run == "ni" else sepE_ru

for i in range(len(refs)):
    edge3 = np.sum(refs[i][ref_energies[i] > sepE])
    edge2 = np.sum(refs[i][ref_energies[i] < sepE])
    WLR = edge3 / edge2
    ox_state = int(ref_names[i][2])
    plt.plot(ref_energies[i], refs[i])
    #plt.scatter(WLR, ox_state, label = str(ox_state))
    plt.legend()
plt.show()




