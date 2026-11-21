"""
White Line Ratios:
Moritz hat die Kalibration begonnen zu implementieren.
Die L2+L3 Peak-Struktur werden White Lines genannt.
Ratios der Integrale bilden.

Noch nicht fertig. Ergänzt & Leistung verbessert mit Gemini.
(Stand: 04.11.2026)
"""
import numpy as np
import os
import tifffile as tiff
import scipy as sp
import matplotlib.pyplot as plt
import copy
import pandas as pd

# user
username = 'Tim'    # 'Moritz' or 'Tim'


# Multiple image plot
def multiplot(maps, title=None):
    """
    :param maps: List containing elements [wlr_map, file_name]
    :param title: Overall plot title (also used for saving the image). Facultative.
    """
    fig = plt.figure(figsize=(10, 7))

    for i, map in enumerate(maps):
        # see: https://www.geeksforgeeks.org/python/how-to-display-multiple-images-in-one-figure-correctly-in-matplotlib/
        plt.subplot(2, (len(maps)+1)//2, i+1)  # 2 rows, columns, i-th position
        plt.imshow(map[0], cmap = "PiYG")  
        plt.axis('off')  # Hide the axis labels
        plt.title(map[1], fontsize=8)

    if title != None:
        fig.suptitle(title)
        plt.savefig('WLR_images\\' + title + ' multiplot')
    else:
        plt.savefig('WLR_images\\' + map[1] + ' multiplot')

    plt.show()

# -------------------- #
#  REFERENCE & EXPERIMENTAL DATA IMPORT
# -------------------- #

if username == 'Moritz':
    referencePath = "/Users/moritz/Library/Mobile Documents/com~apple~CloudDocs/Studium/Semester 5/Schlussprojekt/psi-sim-dmr-nanocatalysts/activeRefs/"

elif username == 'Tim':
    referencePath = "activeRefs/"
    # raw string to prevent unicodeescape syntax error
    datapath = r"C:\Users\Tim Elsener ETHZ\OneDrive\Dokumente\04_Studium\2_Vorlesungen\Projekte & Praktika\PSI-Projekt (Sem5)\Datenexport OneDrive Moritz 11.11.2026"
    datafolders = [
            name for name in os.listdir(datapath)
            if os.path.isdir(os.path.join(datapath, name))
        ]

# extracting reference file data
reffiles = sorted(os.listdir(referencePath))

# importing CTM4XAS reference data from 'activeRefs' folder
run = "ni"
refs = []           # intensities
ref_energies = []   # energies
ref_names = []      # names
for i in reffiles:
    if i.endswith(".txt") and run in i: # only 'ru' or 'ni' .txt files
        #print(i)
        ref = np.loadtxt(
            os.path.join(referencePath, i),
            delimiter=",",
            skiprows=0
        )

        ref_energies.append(np.array(ref[:, 0]))
        refAppend = ref[:, 1]   #spectra
        refAppend -= np.min(refAppend)  # set minimum value to zero
        refAppend = np.array(refAppend)
        refs.append(refAppend / np.max(refAppend))  # normalise by maximum value
        ref_names.append(i)

# Load the single CSV file containing all 3 spectra
csv_path = os.path.join(referencePath, "Reference_EELS_spectra.csv")
df = pd.read_csv(csv_path, delimiter=";")

num_cols = df.shape[1]  # number of columns

# Loop over each spectrum pair (Cols 0&1, 2&3, 4&5)  for ni in "Reference_EELS_spectra.csv"
for i in range(2, num_cols - 1, 2):
    pair_data = df.iloc[:, [i, i+1]].dropna().to_numpy()
    
    ref_energies.append(np.array(pair_data[:, 0]))
    
    refAppend = pair_data[:, 1]  # spectra
    refAppend -= np.min(refAppend) # set minimum value to zero
    refAppend = np.array(refAppend)
    refs.append(refAppend / np.max(refAppend))  # normalise by maximum value
    
    ref_names.append(df.columns[i+1])  # Uses column name as label



# -------------------- #
# CALCULATING REFERENCE WLRs
# -------------------- #
#DONE: understand what this does & in what formats it is saved

# separation energies: divide the data spectra in two halves for separate integration later
sepE_ni = 864 + 14  # we have a systematic energy shift of about 14 eV.
sepE_ru = 479 + 14  # dito

sepE = sepE_ni if run == "ni" else sepE_ru
sepE_ref = sepE - 14
#lines = [(852, 864), (870, 877)]    # old conditions?
lines = [(463.5, 471.5), (485.5, 492.5), (864, 872), (882, 887)]

def eels_atan_fit(x, a, c, e):
    return a*np.arctan((x - 855)) + c * np.arctan((x - 872)) + e

for i in range(len(refs)):  # 'refs' is a list containing the reference spectra's intensities
    energies = ref_energies[i]
    mask = np.ones_like(energies, dtype=bool)
    for low, high in lines:
        mask &= ~((energies >= low) & (energies <= high))
    popt, pcov = sp.optimize.curve_fit(eels_atan_fit, energies[mask], refs[i][mask])
    if i < 2:
        plt.plot(energies, refs[i] - eels_atan_fit(energies, *popt), label=ref_names[i])
    else:
        plt.plot(energies, refs[i], label=ref_names[i])
    #plt.plot(energies, eels_atan_fit(energies, *popt))
    #plt.plot(energies, refs[i])

plt.axvline(x=sepE_ref, color='r', linestyle='--', label='Seperation Energy')
plt.xlabel('Energy in eV')
plt.ylabel('Intensity (arbitrary units)')
plt.legend()
plt.grid()
plt.title('Reference spectra')
print('\nPlotted reference spectra.')
plt.savefig(f'WLR_images\\Reference spectra', dpi=200)
plt.show()

ox_states = []
wlrs = []
for i in range(len(refs)):  # remember: "refs" contains the intensities
    # Riemann sums
    #DONE: Adapt/remove the lower and upper boundaries (850, 874.5). I simply added them to compare 4 different Ni2+ ref spectra more easily.
    edge3 = np.sum(refs[i][ref_energies[i] < sepE_ref])
    edge2 = np.sum(refs[i][sepE_ref < ref_energies[i]])
    WLR = edge3 / edge2
    ox_states.append(int(ref_names[i][2]))
    wlrs.append(WLR)

plt.grid()
plt.scatter(ox_states, WLR, label='WLR ' + ref_names[i])
plt.xlabel('Oxidation state')
plt.ylabel('White line ratio (L3/L2)')
plt.legend()
plt.title(f'Reference spectra WLRs using {sepE_ref} as separation energy')
print('\nPlotted WLRs.')
plt.savefig(f'WLR_images\\Reference spectra WLRs')
plt.show()



# -------------------- #
# PROCESSING EXPERIMENTAL DATA
# -------------------- #

#DONE: Replace "datafolders", "datapath", and "cut_regions" with something reasonable (see "data-analysis.py")
#DONE: Import data spectra without np.abs() from Moritz's OneDrive (we switched from iCloud)
#DONE: Test run with only one experimental data stack
#DONE: Speed up nested for m... & for n... loop (see answer Gemini Pro)
#TODO: Compare WLR with references (linear interpolation between oxidation states)
    #TODO: Do we have a good Ni1 and Ni0 spectrum?
#TODO: Make a chemical map showing the WLR, assigning certain WLR values distinct oxidation states
#TODO: Do this for all experiments (all stacks)
#TODO: Make a nice overview (plots) for all experiments comparing it to other approaches?

# dictionary containing special conditions
cut_regions = {
    1776: [("above", 490)],
    1792: [("below", 852), ("above", 891)],
    1768: [("above", 889)],
    1780: [("above", 889)],
    1761: [("between", 474, 479)],
    1787: [("between", 869.2, 872)],
    1781: [("point", 866.8)],
    1756: [("between", 867.8, 868.8)],
    1782: [("point", 863)]
}

# used for comparing all maps across later on in one large multiplot
wlr_maps = []   # each element's format: [wlr_map, vmin, vmax, str(sample_id)]

buggyexeriments = [
    '1768',
    '1776',
    '1780',
    '1792'
]


for i in range(len(datafolders)):   # all folders, e.g. "20260918_1750_AbsortionSpectrum"
    sample_id = str(datafolders[i][9:13])
    if sample_id in buggyexeriments: # various experiments raise ValueError for keithley (line 230)
        continue
    
    print(f'\n\nProcessing experiment {sample_id}...\n') # experiment number

    try:
        path = os.path.join(datapath, datafolders[i])
        tiffpath = os.path.join(path, "background-corrected")
        
        # Filter for valid, non-hidden TIFF files immediately
        datafiles = sorted([
            f for f in os.listdir(tiffpath)
            if not f.startswith(".") and f.lower().endswith((".tif", ".tiff"))
        ])
    except FileNotFoundError:
        print("no drift correction yet, continuing")
        continue
        
    if len(datafiles) < 5:
        print("scan aborted, continuing")
        continue

    # Since datafiles is already strictly valid TIFFs, the loop becomes much cleaner (see Gemini):
    data = [
        tiff.imread(os.path.join(tiffpath, filename)).astype(float)
        for filename in datafiles
    ]


    data_corr = np.array(data)   # np array containing the extracted tiff image information. Each element corresponds to one tiff file.
    energies = np.loadtxt(path + "/scan_1.txt", delimiter=";", skiprows=1, usecols=0)
    # Note: Moritz already corrected the data, thus all background correction, Keithley normalisation, masking, 5th degree polynomial fitting etc. is abundant!



    # -------------------- #
    #   CALCULATING WLRs
    # -------------------- #
    
    #DONE: Why are there negative WLRs? (negative integrals, where )
    #TODO: Why are the WLRs so small? (reference WLR is 1.35)
    #TODO: Play with parameters (masks). Large area is NaN right now; do we want that?
    #TODO: Why is a single pixel's spectrum so messy (i. e. pixel (50, 50) as tested below)
    #TODO: Why do both wlr_map imshow plots look different?
    #TODO: Why is there such a large energy shift between the reference spectrum and the experimental data? I. e. compare "Spectra 1757..." with the reference spectrum. The sepE's don't align; in the refrence spectrum, it should be 864, but in the experimental spectrum, 879 is more convenient separation. Why?!

    #TODO: edge3 und edge 2 maps: plot them seperately.

    total_signal = np.nansum(data_corr, axis=0) # sets invalid NaN (not number values) to zero
    sample_mask = total_signal > (0.1 * np.nanmax(total_signal))   # ignore unphysical signals which are show less than 10 % intensity with respect to the maximum intensity
    # note: sample_mask is a 2D Boolean array (mask) containing True and False values.

    edge3_map = np.nansum(data_corr[energies < sepE, :, :], axis=0) # Riemann Sum
    edge2_map = np.nansum(data_corr[energies > sepE, :, :], axis=0) # Riemann Sum


    # 3 map comparison multiplot: L3/L2, edge3_map, edge2_map
    fig = plt.figure(figsize=(21, 7))

    wlr_map = np.full(edge3_map.shape, np.nan, dtype=float) # same shape and data type
    valid = sample_mask & (edge2_map > 0)   # note: The "&" operator performs an element-by-element logical AND between those two grids.
    wlr_map[valid] = edge3_map[valid] / edge2_map[valid]    # white line ratio calculation

    vmin, vmax = np.nanpercentile(wlr_map, [5, 95]) # ignoring the bottom 2 percentile and the top 2 percentile
    plt.subplot(1, 3, 1)  # rows, columns, position
    plt.imshow(wlr_map, vmin=vmin, vmax=vmax, cmap="viridis")
    plt.colorbar()
    plt.axis('off')  # Hide the axis labels
    plt.title(f'White Line Ratio (L3/L2)')

    vmin, vmax = np.nanpercentile(edge3_map, [5, 95]) # ignoring the bottom 2 percentile and the top 2 percentile
    plt.subplot(1, 3, 2)  # rows, columns, position
    plt.imshow(edge3_map, vmin=vmin, vmax=vmax, cmap="viridis")
    plt.colorbar()
    plt.axis('off')  # Hide the axis labels
    plt.title(f'L3 integral')

    vmin, vmax = np.nanpercentile(edge2_map, [5, 95]) # ignoring the bottom 2 percentile and the top 2 percentile
    plt.subplot(1, 3, 3)  # rows, columns, position
    plt.imshow(edge2_map, vmin=vmin, vmax=vmax, cmap="viridis")
    plt.colorbar()
    plt.axis('off')  # Hide the axis labels
    plt.title(f'L2 integral')   


    plt.suptitle(f'Comparing WLR to L3 and L2 maps (Nr. {sample_id})')
    plt.tight_layout()  # prevent overlapping of axes, titles etc.
    plt.savefig(f'WLR_images\\{sample_id} WLR L3 L2 comparison')
    #plt.show()
    plt.close()  # Frees memory and prevents it from popping up later when plt.show() is called

    wlr_maps.append([wlr_map, vmin, vmax, str(sample_id)])

    
    # multi plotting various point spectra
    """
    fig = plt.figure(figsize=(15, 7))

    plist = [   # that's where some ni particles are in experiment 
        (355, 245),
        (257, 254),
        (211, 131),
        (107, 17)
    ]

    for j, (px, py) in enumerate(plist):
        plt.subplot(2, 2, j+1)  # rows, columns, position
        plt.title(f'Spectrum @ pixel {px, py}')
        plt.plot(energies, data_corr[:, px, py])    # arbitrary pixel (20, 20)
        plt.axvline(x=sepE, color='r', linestyle='--')

    plt.suptitle(f'Experiment {sample_id}')
    plt.savefig(f'WLR_images\\Spectra {sample_id} comparison multiplot')
    #plt.show()
    """

    
    # debugging: energy distribution across the map (Gemini)
    """
    print("Energies range:", energies.min(), "to", energies.max(), "| sepE:", sepE)
    print("Non-zero pixels in data_corr:", np.count_nonzero(data_corr))
    print("wlr_map dtype:", wlr_map.dtype, "| NaNs:", np.isnan(wlr_map).sum())
    print("wlr_map min / median / 99th / max:",
        np.nanmin(wlr_map),
        np.nanmedian(wlr_map),
        np.nanpercentile(wlr_map, 99),
        np.nanmax(wlr_map))
    """



    # -------------------- #
    # PLOTTING EXPERIMENTAL & REFERENCE WLRs
    # -------------------- #


    # average spectrum over all pixels
    """
    newspec = np.mean(data_corr, axis = (1, 2))
    newspec /= np.max(newspec)
    newspec2 = sp.signal.savgol_filter(newspec, 11, 3)  # Savitzky-Golay filter to smoothen it out
    plt.plot(energies, newspec, label = "processed")
    plt.plot(energies, newspec2, label = "processed filtered")
    plt.plot(energies_full, spec_raw / np.max(spec_raw), label = "raw")
    plt.legend()
    plt.show()
    #plt.savefig(datapath + f"{sample_id}_image2.png", dpi=300)

    image = np.sum(data, axis = 0)
    image[:, :10] = 0
    image /= np.max(image)
    plt.imshow(image)
    plt.show()
    #isNi = np.all(energies > 600)
    """

    print(f'\nlength WLR maps: {len(wlr_maps)}')


# comparing all WLR maps in one large multiplot
fig = plt.figure(figsize=(20, 15))

for i, m in enumerate(wlr_maps):  # wlr_maps elements' format: [wlr_map, vmin, vmax, str(sample_id)]
    # see: https://www.geeksforgeeks.org/python/how-to-display-multiple-images-in-one-figure-correctly-in-matplotlib/
    plt.subplot(8, (len(wlr_maps)+7)//8, i+1)  # 8 rows, columns, i-th position
    plt.imshow(m[0], vmin=m[1], vmax=m[2], cmap="viridis")
    plt.colorbar()
    plt.axis('off')  # Hide the axis labels
    plt.title(m[3], fontsize=8)

plt.tight_layout()
plt.suptitle('All WLR maps: L3/L2, 5-95 percentiles')
plt.savefig('WLR_images\\All WLR maps multiplot 5-95', dpi=500)
plt.show()

print('\nDone.')