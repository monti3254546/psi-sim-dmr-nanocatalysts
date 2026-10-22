from matplotlib import lines
import numpy as np
import os
import tifffile as tiff
import scipy as sp
import matplotlib.pyplot as plt
import copy


#REFERENCE IMPORT
#Moritz Path
#referencePath = "/Users/moritz/Library/Mobile Documents/com~apple~CloudDocs/Studium/Semester 5/Schlussprojekt/psi-sim-dmr-nanocatalysts/activeRefs/"
#Samuel path
referencePath = "C:\\Users\\Samuel_Uni\\psi-sim-dmr-nanocatalysts\\activeRefs"

reffiles = sorted(os.listdir(referencePath))
first=True
run = "ni"
refs = []
ref_energies = []
ref_names = []
for i in reffiles:
    if i.endswith(".txt") and run in i:
        ref = np.loadtxt(
            os.path.join(referencePath, i),
            delimiter=",",
            skiprows=0
        )
#12.8 ev shift for ni spectra, Ru spectra need to be checked again
        ref_energies.append(ref[:, 0] + 12.8)  #energy grid. shift systematic error of 12.5 eV
        refAppend = ref[:, 1]   #spectra
        refAppend -= np.min(refAppend)
        #refAppend /= np.max(refAppend)
        refs.append(refAppend)
        ref_names.append(i)
        print(i)


for i in range(len(refs)):
    refs[i] = sp.ndimage.gaussian_filter(refs[i], sigma = 45)   #sigma = 45 seems to yield a similar degree of detail to the data --> best for fitting
#     plt.plot(ref_energies[i], refs[i])
# plt.show()


#interpolate to 0.2 eV grid, aligned to a .0 grid (i.e. 1, 1.2, 1.4, 1.6, ...)
for i in range(len(refs)):  #E_ref is the energy grid of the ref spectrum, ref the spectrum itself
    newAxisStart = round(min(ref_energies[i]) * 5) / 5
    newAxisEnd = round(max(ref_energies[i]) * 5) / 5
    newAxis = np.arange(newAxisStart, newAxisEnd, 0.2)
    ref_interp = np.interp(
        newAxis, #resample refs to 0.2 eV steps
        ref_energies[i],
        refs[i]
    )
    refs[i] = ref_interp
    ref_energies[i] = newAxis

refEnergyRanges = []   #use this to prevent shifting peaks out of the spectrum
for i in range(len(ref_energies)):  #loop through the energy axis of all ref spectra
    refEnergyRanges.append([min(ref_energies[i]), max(ref_energies[i])])
refEnergyRanges = np.array(refEnergyRanges)

lowerBoundRefs = np.min(refEnergyRanges)
upperBoundRefs = np.max(refEnergyRanges)

globRefEnergy = np.arange(0, 1000, 0.2)  #USE THIS FROM NOW ON, ITS GLOBAL FOR ALL REFERENCE SPECTRA
print(globRefEnergy)
for i in range(len(refs)):
    padLeft = int(round((min(ref_energies[i]) - 0) * 5))
    padRight = int(round((999.8 - max(ref_energies[i])) * 5))
    refs[i] = np.pad(refs[i], (padLeft, padRight), mode = "constant", constant_values = 0)


#DATA IMPORT
#Data path for Moritz
#datapath = "/Users/moritz/Library/Mobile Documents/com~apple~CloudDocs/Studium/Semester 5/Schlussprojekt/20260918_Ni_Ru_AbsorptionSpectrum v5-real/"
#Data path for Samuel
datapath= "E:\\20260918_Ni_Ru_AbsorptionSpectrum v5-real"
#Path for new LCF data
LCF_datapath="C:\\Users\\Samuel_Uni\\psi-sim-dmr-nanocatalysts\\LCF_results_new"
datafolders = [
    name for name in os.listdir(datapath)
    if os.path.isdir(os.path.join(datapath, name))
]
for i in range(len(datafolders)):
    sample_id = datafolders[i][9:13]
    try:
        path = os.path.join(datapath, datafolders[i])
        tiffpath = os.path.join(path + "/background-corrected")
        datafiles = sorted(os.listdir(tiffpath))
    except FileNotFoundError:
        print("no background correction yet, continuing")
        continue
    if len(datafiles) < 5:
        print("scan aborted, continuing")
        continue

    data = []
    energies = []


    for j in range(len(datafiles)):
        #not datafiles[j].startswith('._") required on Windows
        if not datafiles[j].startswith("._") and (datafiles[j].endswith(".tif") or datafiles[j].endswith(".tiff")):
            data.append(
                tiff.imread(
                    os.path.join(tiffpath, datafiles[j])
                ).astype(float)
            )
            energies.append(float(datafiles[j][-10:-5]))


    data = np.array(data)
    energies = np.array(energies)
    if run == "ni" and np.any(energies < 600): continue
    elif run == "ru" and np.any(energies > 600): continue

    mask = ((globRefEnergy >= np.min(energies) - 0.0001) & (globRefEnergy <= np.max(energies) + 0.0001))  #discard any ref spectra regions that are not in our data
    refs_temp = np.column_stack(refs)[mask, :]
    
    energy_peaks_range=[(463.5, 471.5), (485.5, 492.5), (864, 872), (882, 887)]
    # 1. Compute mask ONCE outside the spatial loops (start with False)
    mask = np.zeros_like(energies, dtype=bool)
    for low, high in energy_peaks_range:
        mask |= ((energies >= low) & (energies <= high))

    # 2. Slice reference spectra ONCE to match masked dimensions
    refs_masked = refs_temp[mask, :]

    # Initialize output array
    coeffs = np.zeros((len(ref_names), data.shape[1], data.shape[2]))

    # 3. Perform LCF loop over spatial pixels
    for m in range(data.shape[1]):
        for n in range(data.shape[2]):
        # Extract and mask target pixel spectrum
            target_masked = data[mask, m, n]
        
            # Non-negative least squares fit
            coefficients, _ = sp.optimize.nnls(refs_masked, target_masked)
        
            # Store coefficients
            coeffs[:, m, n] = coefficients


            #fitted = refs @ coefficients
            #plt.plot(energies, target)
            #plt.plot(globRefEnergy, fitted)
            #print(coefficients)
            #plt.show()
            
    # Calculate ratio maps
    for k in range(coeffs.shape[0]):
        coeffs[k, :, :] = np.clip(coeffs[k, :, :], 0, np.percentile(coeffs[k, :, :], 99.9))
        #plt.imsave(datapath + f"oxideMaps/{sample_id}_map_{ref_names[k]}.png", coeffs[k, :, :])
    
    #simple clipping mask for ni2+ maps to make it human readable
    #coeffs[0,:,:]=np.clip(coeffs[0,:,:],np.max(coeffs[0,:,:])*0.1,np.max(coeffs[0,:,:]*0.9))

    #If this idea does not work, just set values below a certain threshold to 0.
    coeffs[0,:,:][coeffs[0,:,:]<np.max(coeffs[0,:,:])*0.025]=0 #try 0.05 or 0.1 
    coeffs[1,:,:][coeffs[1,:,:]<np.max(coeffs[1,:,:])*0.025]=0

    #try working with percentiles
    #p10 = np.percentile(coeffs[0,:,:], 10)
    #p90 = np.percentile(coeffs[0,:,:], 90)
    # Set values OUTSIDE the 10th-90th range to 0
    #coeffs[0,:,:][(coeffs[0,:,:] < p10) | (coeffs[0,:,:] > p90)] = 0
    
    #plotting ratios by dividing thorugh the sum of coefficients from both maps
    # ni_ratio_3=np.where(coeffs[1,:,:]!=0,coeffs[1, :, :] / (coeffs[0, :, :] + coeffs[1, :, :]),0)  # ni 3 ratio over total
    # ni_ratio_2=np.where(coeffs[0,:,:]!=0,coeffs[0, :, :] / (coeffs[0, :, :] + coeffs[1, :, :]),0)  # ni 2 ratio over total
    
    #masking based on total signal
    # Check how many pixels have non-zero values in BOTH coefficients simultaneously
    overlap_pixels = np.zeros([coeffs.shape[1],coeffs.shape[2]])
    overlap_pixels[:,:][((coeffs[0, :, :] > 0) & (coeffs[1, :, :] > 0))] = 1
    tot_signal=coeffs[0,:,:]+coeffs[1,:,:]
    ni_ratio_3=np.divide(coeffs[1,:,:],tot_signal,out=np.zeros_like(tot_signal),where=tot_signal!=0)
    ni_ratio_2=np.divide(coeffs[0,:,:],tot_signal,out=np.zeros_like(tot_signal),where=tot_signal!=0)
    #plotting ratios by using logartihmic scale
    #ni_ratio = np.log10(coeffs[0, :, :] / coeffs[1, :, :])  # ni 2 over ni3

    # fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    # for i in range(3):
    #    im = axes[i].imshow(maps[i])
    #    axes[i].set_title(titles[i])
    #    fig.colorbar(im, ax=axes[i], fraction=0.046, pad=0.04)

    # plt.tight_layout()
    # plt.close()

    # # #For Mac
    # plt.imsave(LCF_datapath + f"{sample_id}_map_ni2.png", coeffs[0, :, :])
    # plt.imsave(LCF_datapath + f"{sample_id}_map_ni3.png", coeffs[1, :, :])
    # plt.imsave(LCF_datapath + f"{sample_id}_map_ni2-3.png", ni_ratio)
    # #For Windows
    if first:
        print("Press enter if you checked file destination")
        input()
        first=False
    #for multiple plots
    fig, ax=plt.subplots(1,5,figsize=(12,4))
    im1=ax[0].imshow(coeffs[0,:,:])
    ax[0].title.set_text("Ni2+")
    im2=ax[1].imshow(coeffs[1,:,:])
    ax[1].title.set_text("Ni3+")
    im3=ax[2].imshow(ni_ratio_3)
    ax[2].title.set_text("Ni3+/sum coefficients")
    im4=ax[3].imshow(ni_ratio_2)
    ax[3].title.set_text("Ni2+/sum coefficients")
    im5=ax[4].imshow(overlap_pixels)
    ax[4].title.set_text("Overlapping pixels")
    fig.suptitle(f"{sample_id}")
    fig.colorbar(im1,ax=ax[0],fraction=0.046,pad=0.04)
    fig.colorbar(im2,ax=ax[1],fraction=0.046,pad=0.04)
    fig.colorbar(im3,ax=ax[2],fraction=0.046,pad=0.04)
    fig.colorbar(im4,ax=ax[3],fraction=0.046,pad=0.04)
    fig.colorbar(im5,ax=ax[4],fraction=0.046,pad=0.04)
    #for single plot
    # fig,ax=plt.subplots(figsize=(4,4))
    # ax.imshow(coeffs[0,:,:])
    # ax.title.set_text(f"{sample_id} (Ni2+)")
    # fig.colorbar(ax.imshow(coeffs[0,:,:]),ax=ax,fraction=0.046,pad=0.04)
    plt.tight_layout()

    plt.savefig(LCF_datapath + f"\\mask_peaks\\{sample_id}_maps.png",dpi=300)
    #plt.imsave(LCF_datapath + f"\\mask_setup\\{sample_id}_map_ni2.png", coeffs[0, :, :])
    # plt.imsave(LCF_datapath + f"\\initial_setup\\{sample_id}_map_ni3.png", coeffs[1, :, :])
    # plt.imsave(LCF_datapath + f"\\initial_setup\\{sample_id}_map_ni2-3.png", ni_ratio)