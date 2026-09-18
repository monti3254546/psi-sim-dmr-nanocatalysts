"""
Plots two spectra next to each other:
#1 Intensity vs. energies (theoretical input values for monochromator)
#2 Intensity vs. energies (actual output = what the monochromator used)

Script by Samuel Sottrovisch, 17.09.2026
Modified by Tim Elsener, using Gemini Flash 3.6 for help on 18.09.2026
"""
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats


# Load data
nexp = '1737_normalisation'     # number of the experiment. Note that file names need to be identical!
element = 'ru'  # element (Ni or Ru)
mono_data=np.loadtxt('Images\\Plots comparison Ru t=1415\\mono_'+element+nexp+'.csv', delimiter=',',skiprows=5)
input_data=np.loadtxt('Images\\Plots comparison Ru t=1415\\input_'+element+nexp+'.csv', delimiter=',',skiprows=5)

x_input = input_data[:,0]
y_mono = mono_data[:,0]

# Linear Regression
# Returns slope (m), intercept (c), r-value, p-value, and standard error
slope, intercept, r_value, p_value, std_err = stats.linregress(x_input, y_mono)
r_squared = r_value ** 2

# Generate points for the trendline
x_trend = np.linspace(x_input.min(), x_input.max(), 100)
y_trend = slope * x_trend + intercept

# Multi-panel figure
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
plt.suptitle('Spectra Comparison for Experiment nr. '+nexp+' ('+element+')')

# First panel
ax1.plot(mono_data[:,0], mono_data[:,1], label='Mono Spectrum (readout)', color='orange')
ax1.plot(input_data[:,0], input_data[:,1], label='Input Spectrum', color='blue')
ax1.set_xlabel("Energy in eV")
ax1.set_ylabel("Intensity [arbitrary units]")
ax1.legend()
ax1.grid(True, linestyle=':', alpha=0.6)

# Second panel
ax2.plot(x_input, y_mono, label='Readout vs. Input', color='violet')
#ax2.scatter(x_input, y_mono, color='red', label='Linear Regression', zorder=3)
ax2.plot(
    x_trend, 
    y_trend, 
    color='black', 
    linestyle='--', 
    label=f'Fit: y = {slope:.3f}x + {intercept:.3f}\n$R^2$ = {r_squared:.4f}'
)
ax2.set_xlabel("Input energy range (theoretical) in eV")
ax2.set_ylabel("Readout energy values (monochromator) in eV")
ax2.legend()
ax2.grid(True, linestyle=':', alpha=0.6)


# Prevent overlapping titles and axis labels
plt.tight_layout()

plt.savefig('Images\\Plots comparison Ru t=1415\\Comparison_Spectras_'+element+'-'+nexp+'.png', dpi=300)
plt.show()