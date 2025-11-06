# %%
import numpy as np
import pandas as pd

from bioio import BioImage
import bioio_czi

import skimage.io as io
from skimage.filters import threshold_otsu
from skimage.exposure import histogram, cumulative_distribution

from scipy.optimize import curve_fit
from scipy.optimize import brentq
from scipy import stats

from matplotlib import pyplot as plt


from IPython.display import clear_output

import time

from scipy.optimize import curve_fit
from pathlib import Path

from natsort import natsorted


import seaborn as sns

import scipy.signal as signal

import os


# %% [markdown]
# # mStayGold photobleaching analysis
# ## Data description
# * Here we compare three mStayGold variants: `mSG-BJ` (mStayGold-BJ), `mSG-B` (mStayGold-B), and `mSG-J` (mStayGold-J). As a reference, we also include `mNG` (mNeonGreen), a standard in the field.
# * mRNA encoding each variant was injected into zebrafish embryos (1-cell-stage). We co-injected `mSc3` (mScarlet3) with each injection to control for variation in injection volume.
# * For each variant, we acquired following data in triplicates:
#     * Background images (shot noise) in the green channel (488 nm) and red channel (561 nm). They will be used to calculate the offset for each imaging channel and are specific for the detector & optical set up.
#     * Single frames in the green (mSG, 488 nm at 3% power) and red channel (mSc3, 561 nm at 0.5% power). They will be used to compare absolute brightness of the mSG variants, normalized by the signal intensity of mSc3. They were acquired in triplicates over 21 embryos.
#     * bleaching time series (800 continuous frames) in the green channel (mSG, 488 nm) at:
#         * 1% power: 0.53 W/cm^2
#         * 3% power: 1.9 W/cm^2
#         * 6% power: 3.9 W/cm^2
#         * 12% power: 5.9 W/cm^2
#     * Total, per variant, we have 3 dual-color z-stacks and 9 bleaching time series.
#
# ## Analysis
# * Background correction: Determine the noise threshold by averaging the background images and subtracting it from the data.
# * Average frame intensity of mSG over time
# * Fit exponential decay to the data, ignoring first and last percentiles
# * Calculate the half-time of the decay based on the fitted curve
#
#

# %% [markdown]
# # Bleaching analysis
# ## Background correction
# This is specific for every detector & optical set up. The calculated offset can be applied to all images taken with the same filter and laser settings. Background images were taken by moving away from any detectable sample, and taking an image with the smallest laser power possible.

# %%
data_dir = "../data/msg_data/2024"

# %%
# Read & plot background image
bg_files = BioImage(f"{data_dir}/1022/mNG-mSc3-background.czi", reader=bioio_czi.Reader).get_image_data("CYX")

plt.imshow(bg_files[0, :, :])
plt.title('Background Image Frame green')
plt.show()

plt.imshow(bg_files[1, :, :]) 
plt.title('Background Image Frame red')
plt.show()

# Calculate average background intensity and assign to specific imaging channel
bg_green = np.mean(bg_files[0, :, :])
bg_red = np.mean(bg_files[1, :, :])

print(bg_green, bg_red)


# %%
bleach_dir = f"{data_dir}/1101/bleach"
# Read in all data, in nested dictionary structure
bl_img = {}
for fluorophore_path in Path(bleach_dir).iterdir():
    if fluorophore_path.is_dir():
        fluorophore = fluorophore_path.name
        bl_img[fluorophore] = {}
        for power_path in fluorophore_path.iterdir():
            if power_path.is_dir():
                power = power_path.name
                bl_img[fluorophore][power] = {}
                print(power_path)
                power_path = power_path / '2024-11-01' # this subfolder is added by the data acquisition software
                bl_img[fluorophore][power]['msc3'] = {} # msc3 snapshots were taken to verify injection
                bl_img[fluorophore][power]['bleaching'] = {}
                for file in power_path.iterdir(): 
                    if 'msc3' in file.name.lower(): # finds msc3 images, regardless of case
                        bl_img[fluorophore][power]['msc3'][file.name[-5:-4]] = BioImage(file, reader=bioio_czi.Reader).get_image_data("YX")
                    elif file.name.endswith('.czi'):
                        bl_img[fluorophore][power]['bleaching'][file.name[-5:-4]] = BioImage(file, reader=bioio_czi.Reader).get_image_data("TYX")



# %%
#Double-check data download and data format
for fluorophore in bl_img:
    for power in bl_img[fluorophore]:
        for data_type in bl_img[fluorophore][power]:
            if isinstance(bl_img[fluorophore][power][data_type], dict):
                for rep in bl_img[fluorophore][power][data_type]:
                    print(f"{fluorophore} - {power} - {data_type} - {rep}: {bl_img[fluorophore][power][data_type][rep].shape}")

# %%
# Substract background noise from data (27 snapshots in 'msc3', 800 frames * 27 = 21600 frames in 'bleaching')
for fluorophore in bl_img.keys(): 
    for power in bl_img[fluorophore].keys():
        for rep in sorted(bl_img[fluorophore][power]['msc3'].keys()):
            # Correct green channel
            bl_img[fluorophore][power]['msc3'][rep] = bl_img[fluorophore][power]['msc3'][rep] - bg_red
            bl_img[fluorophore][power]['msc3'][rep][bl_img[fluorophore][power]['msc3'][rep] <= 0] = np.nan # set background values to nan
        
        for rep in sorted(bl_img[fluorophore][power]['bleaching'].keys()):
            # Correct green channel
            bl_img[fluorophore][power]['bleaching'][rep] = bl_img[fluorophore][power]['bleaching'][rep] - bg_green
            bl_img[fluorophore][power]['bleaching'][rep][bl_img[fluorophore][power]['bleaching'][rep] <= 0] = np.nan

# %%
# Calculate average intensity of bleaching experiments, ignorning the first and last 1% of frames
for fluorophore in bl_img.keys(): 
    for power in bl_img[fluorophore].keys():
        bl_img[fluorophore][power]['bleaching_avg'] = {}
        for rep in bl_img[fluorophore][power]['bleaching'].keys():
            # calculate number of frames
            n_frames = bl_img[fluorophore][power]['bleaching'][rep].shape[0]
            frames_to_trim = int(n_frames * 0.05)   # 5% of frames
            # trim data 
            trimmed_data = bl_img[fluorophore][power]['bleaching'][rep][frames_to_trim:-frames_to_trim]
            # calculate mean intensity
            bl_img[fluorophore][power]['bleaching_avg'][rep] = np.nanmean(trimmed_data, axis=(1,2))

# %%
# Plot all bleaching curves, normalized to 100 at t=0
fig1_bleach_corr_avg_norm_100 = plt.figure(figsize=(8,6))

## Define color scheme for fluorophores
fluorophore_colors = {'mSG(J)': 'blue', 'mSG(BJ)': 'green', 'mSG(B)': 'red', 'mNG': 'orange'}

## Define line styles for power levels
power_linestyles = {'3percent': '-', '6percent': '--', '12percent': '-.', '1percent': ':'}

for fluorophore in bl_img.keys():
    color = fluorophore_colors[fluorophore]
    for power in bl_img[fluorophore].keys():
        linestyle = power_linestyles[power]
        for rep in bl_img[fluorophore][power]['bleaching_avg'].keys():
            x = bl_img[fluorophore][power]['bleaching_avg'][rep]
            x = x / x[0] * 100
            plt.plot(x,
                     color=color,
                     linestyle=linestyle,
                     label=f'{fluorophore} {power} rep{rep}')

plt.xlabel('Frame')
plt.ylabel('Mean Intensity')
plt.title('Bleaching Curves (normalized to mean intensity 100 at t=0)')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()


# %%
# Add normalized bleaching curves (normalized to 100 at t=0) to data structure
for fluorophore in bl_img.keys():
    for power in bl_img[fluorophore].keys():
        bl_img[fluorophore][power]['bleaching_avg_norm'] = {}  # empty dictionary for normalized bleaching curves
        for rep in bl_img[fluorophore][power]['bleaching_avg'].keys():
            # Get bleaching curve and normalize to 100
            bleach_curve = bl_img[fluorophore][power]['bleaching_avg'][rep]
            bl_img[fluorophore][power]['bleaching_avg_norm'][rep] = bleach_curve / bleach_curve[0] * 100


# %% [markdown]
# # Averaged bleaching curves
# Here, I average over the three replicates per fluorescent protein.

# %%
# Calculate average bleaching curves for each condition
for fluorophore in bl_img.keys():
    for power in natsorted(bl_img[fluorophore].keys()):
        # Stack all replicates into a single array
        all_reps = np.stack([bl_img[fluorophore][power]['bleaching_avg_norm'][rep] 
                             for rep in sorted(bl_img[fluorophore][power]['bleaching_avg_norm'].keys())])
        
        # Calculate mean and standard deviation across replicates
        bl_img[fluorophore][power]['avg_bleaching_norm'] = np.nanmean(all_reps, axis=0)
        bl_img[fluorophore][power]['avg_bleaching_std_norm'] = np.nanstd(all_reps, axis=0)
        
        # Perform exponential decay fit on averaged data
        av_int = bl_img[fluorophore][power]['avg_bleaching_norm']
        t = np.arange(len(av_int))
    
            


# %%
# Plot average bleaching curves and fits
fig, axs = plt.subplots(1, 4, figsize=(15, 4))

fluorophores = ['mSG(B)', 'mSG(BJ)', 'mSG(J)', 'mNG']
powers = ['1percent', '3percent', '6percent', '12percent']
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#9467bd']  # Blue, orange, green, purple

for col, power in enumerate(powers):
    ax = axs[col]
    
    for fidx, fluorophore in enumerate(fluorophores):
        if fluorophore in bl_img and power in bl_img[fluorophore]:
            # Plot average bleaching curve
            avg_curve = bl_img[fluorophore][power]['avg_bleaching_norm']
            std_curve = bl_img[fluorophore][power]['avg_bleaching_std_norm']
            
            ax.plot(avg_curve, '-', label=fluorophore, color=colors[fidx], alpha=1, linewidth=2)
            # Plot standard deviation
            ax.fill_between(range(len(avg_curve)), 
                          avg_curve - std_curve,
                          avg_curve + std_curve,
                          color=colors[fidx],
                          alpha=0.2)
    
    # Plot half bleaching line to extract time-to-half-bleaching
    ax.axhline(y=50, color='grey', linestyle=':', alpha=0.5)
    
    # Add labels and titles
    ax.set_title(power)
    ax.set_xlabel('Frame')
    if col == 0:
        ax.set_ylabel('Normalized Intensity')
    
    # Set y-axis limits the same for all plots
    ax.set_ylim(0, 101)
    
    ax.legend()



plt.tight_layout()
plt.savefig('../results/bleaching_curves_averaged_individual_SD.pdf')
plt.show()

# %%
# Plot average bleaching curves and fits
fig, axs = plt.subplots(1, 4, figsize=(15, 4))

fluorophores = ['mSG(B)', 'mSG(BJ)', 'mSG(J)', 'mNG']
powers = ['1percent', '3percent', '6percent', '12percent']
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#9467bd']  # Blue, orange, green, purple

for col, power in enumerate(powers):
    ax = axs[col]
    
    for fidx, fluorophore in enumerate(fluorophores):
        if fluorophore in bl_img and power in bl_img[fluorophore]:
            # Plot average bleaching curve
            avg_curve = bl_img[fluorophore][power]['avg_bleaching_norm']
            std_curve = bl_img[fluorophore][power]['avg_bleaching_std_norm']
            
            ax.plot(avg_curve, '-', label=fluorophore, color=colors[fidx], alpha=1, linewidth=2)
            # Plot individual replicate curves
            for rep in sorted(bl_img[fluorophore][power]['bleaching_avg_norm'].keys()):
                rep_curve = bl_img[fluorophore][power]['bleaching_avg_norm'][rep]
                ax.plot(rep_curve, '-', color=colors[fidx], alpha=0.3, linewidth=1)
    
    # Plot half bleaching line to extract time-to-half-bleaching
    ax.axhline(y=50, color='grey', linestyle=':', alpha=0.5)
    
    # Add labels and titles
    ax.set_title(power)
    ax.set_xlabel('Frame')
    if col == 0:
        ax.set_ylabel('Normalized Intensity')
    
    # Set y-axis limits the same for all plots
    ax.set_ylim(0, 100)
    
    ax.legend()

plt.tight_layout()
plt.savefig('../results/bleaching_curves_averaged_individual.pdf')
plt.show()

# %% [markdown]
# # Exponential fit
# Here, I am fitting an exponential decay to the bleaching curves of each replicate. As for most conditions, the decrease in signal intensity did not reach half of the initial signal intensity within the time frame of the experiment, I will extrapolate the half-life based on the fit.
#

# %%
# # Example replicate
av_int = bl_img['mSG(J)']['3percent']['bleaching_avg_norm']['1']

t = np.arange(len(av_int))
# Exponential decay function
N_0 = max(av_int) # initial intensity (100 at t=0)

def exp_decay(t, k, c):
    return (N_0-c) * np.exp(-k * t) + c

# Fit exponential decay
popt, pcov = curve_fit(exp_decay, t, av_int,
                      p0=[0.1, 0],
                      bounds=([ 0, 0],
                             [np.inf, np.inf]))

# Fit prediction 
fit_curve = exp_decay(t, *popt)
half_life = np.log(2)/popt[0]
offset = popt[1]

# Plot
plt.figure(figsize=(8, 6))
plt.plot(t, av_int, 'b.', label='Data')
plt.plot(t, fit_curve, 'r-', label=f'Exponential fit')

# Plot half-life as dashed line
plt.axvline(x=half_life, color='g', linestyle='--', label=f'Half-life [frames]: {half_life:.2f}')
plt.axhline(y=offset, color='lightgrey', linestyle='--', label=f'Offset: {offset:.2f}')
plt.xlabel('Frame')
plt.ylabel('Mean Intensity')
plt.title('Exponential Decay Fit')
plt.legend()
plt.tight_layout()
plt.show()

# %%
# Perform exponential decay fitting for each replicate, using raw intensity data
for fluorophore in bl_img.keys():
    for power in natsorted(bl_img[fluorophore].keys()):
        bl_img[fluorophore][power]['fit_params'] = {}
        bl_img[fluorophore][power]['fit_curves'] = {}
        bl_img[fluorophore][power]['half_life'] = {}
        bl_img[fluorophore][power]['offset'] = {}
        
        for rep in sorted(bl_img[fluorophore][power]['bleaching_avg_norm'].keys()):
            av_int = bl_img[fluorophore][power]['bleaching_avg_norm'][rep]
            t = np.arange(len(av_int))
            N_0 = max(av_int)
            # Fit exponential decay
            popt, pcov = curve_fit(exp_decay, t, av_int,
                                    p0=[0.1, 0],
                                    bounds=([0, 0],
                                        [np.inf, np.inf]))
            # Store fit parameters and generate fitted curve
            bl_img[fluorophore][power]['fit_params'][rep] = popt
            fit_curve = exp_decay(t, *popt)
            bl_img[fluorophore][power]['fit_curves'][rep] = fit_curve
            
 
            # Calculate half-life & store offset
            half_life = np.log(2) / popt[0]
            bl_img[fluorophore][power]['half_life'][rep] = half_life
            bl_img[fluorophore][power]['offset'][rep] = popt[1]
            #print(fluorophore, power, rep)
            #print(f'k: {popt[0]}, c: {popt[1]}, N_0: {N_0}')





# %%
## Compare fit parameters between raw and normalized data
# Create DataFrame to store fit parameters
fit_params_df_og = {}
fit_params_df_og = pd.DataFrame(columns=['fluorophore', 'power', 'replicate', 'k', 'c', 'half_life'])

# Perform exponential decay fitting for each replicate
for fluorophore in bl_img.keys():
    for power in natsorted(bl_img[fluorophore].keys()):
        for rep in sorted(bl_img[fluorophore][power]['bleaching_avg'].keys()):
            av_int = bl_img[fluorophore][power]['bleaching_avg'][rep]
            t = np.arange(len(av_int))
            N_0 = max(av_int)
            # Fit exponential decay
            popt, pcov = curve_fit(exp_decay, t, av_int,
                                    p0=[0.1, min(av_int)],
                                    bounds=([0, 0],
                                        [np.inf, np.inf]))
            
            # Generate fitted curve 
            fit_curve = exp_decay(t, *popt)
            
            # Calculate half-life
            half_life = np.log(2) / popt[0] 
            
            # Add parameters to DataFrame
            new_row = pd.DataFrame({
                'fluorophore': [fluorophore],
                'power': [power],
                'replicate': [rep],
                'k': [popt[0]], 
                'c': [popt[1]],
                'half_life': [half_life]
            })
            fit_params_df_og = pd.concat([fit_params_df_og, new_row], ignore_index=True)
   
            # print(fluorophore, power, rep)
            # print(f'k: {popt[0]}, c: {popt[1]}, N_0: {N_0}')


# Create DataFrame to store fit parameters
fit_params_df_norm = {}
fit_params_df_norm = pd.DataFrame(columns=['fluorophore', 'power', 'replicate', 'k', 'c', 'half_life'])

# Perform exponential decay fitting for each replicate
for fluorophore in bl_img.keys():
    for power in natsorted(bl_img[fluorophore].keys()):
        for rep in sorted(bl_img[fluorophore][power]['bleaching_avg_norm'].keys()):
            av_int = bl_img[fluorophore][power]['bleaching_avg_norm'][rep]
            t = np.arange(len(av_int))
            N_0 = max(av_int)
            # Fit exponential decay
            popt, pcov = curve_fit(exp_decay, t, av_int,
                                    p0=[0.1, min(av_int)],
                                    bounds=([0, 0],
                                        [np.inf, np.inf]))
            
            # Generate fitted curve 
            fit_curve = exp_decay(t, *popt)
            
            # Calculate half-life
            half_life = np.log(2) / popt[0] 
            
            # Add parameters to DataFrame
            new_row = pd.DataFrame({
                'fluorophore': [fluorophore],
                'power': [power],
                'replicate': [rep],
                'k': [popt[0]], 
                'c': [popt[1]],
                'half_life': [half_life]
            })
            fit_params_df_norm = pd.concat([fit_params_df_norm, new_row], ignore_index=True)
   
            # print(fluorophore, power, rep)
            # print(f'k: {popt[0]}, c: {popt[1]}, N_0: {N_0}')
            


# %%
# Plot k and c values for different conditions

fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(10, 10))

# Plot k by fluorophore for normalized data
for power in fit_params_df_norm['fluorophore'].unique():
    mask = fit_params_df_norm['fluorophore'] == power
    ax1.scatter(fit_params_df_norm[mask]['power'], 
               fit_params_df_norm[mask]['k'],
               label=power,
               alpha=0.7)

ax1.set_xlabel('Power')
ax1.set_ylabel('k')
ax1.set_title('Normalized Data: time constant k vs Power')
ax1.legend()
ax1.tick_params(axis='x', rotation=45)

# Plot k by fluorophore for original data
for power in fit_params_df_og['fluorophore'].unique():
    mask = fit_params_df_og['fluorophore'] == power
    ax2.scatter(fit_params_df_og[mask]['power'], 
               fit_params_df_og[mask]['k'],
               label=power,
               alpha=0.7)

ax2.set_xlabel('Power')
ax2.set_ylabel('k')
ax2.set_title('Original Data: time constant k vs Power')
ax2.legend()
ax2.tick_params(axis='x', rotation=45)

# Plot c by fluorophore for normalized data
for power in fit_params_df_norm['fluorophore'].unique():
    mask = fit_params_df_norm['fluorophore'] == power
    ax3.scatter(fit_params_df_norm[mask]['power'], 
               fit_params_df_norm[mask]['c'],
               label=power,
               alpha=0.7)

ax3.set_xlabel('Power')
ax3.set_ylabel('c')
ax3.set_title('Normalized Data: offset c vs Power')
ax3.legend()
ax3.tick_params(axis='x', rotation=45)

# Plot c by fluorophore for original data
for power in fit_params_df_og['fluorophore'].unique():
    mask = fit_params_df_og['fluorophore'] == power
    ax4.scatter(fit_params_df_og[mask]['power'], 
               fit_params_df_og[mask]['c'],
               label=power,
               alpha=0.7)

ax4.set_xlabel('Power')
ax4.set_ylabel('c')
ax4.set_title('Original Data: offset c vs Power')
ax4.legend()
ax4.tick_params(axis='x', rotation=45)

plt.tight_layout()


# %%
# Plot bleaching curves per fluorophore and power level
fig, axs = plt.subplots(4, 4, figsize=(15, 15))

# Define the fixed order for fluorophores and powers
fluorophores = ['mSG(B)', 'mSG(BJ)', 'mSG(J)', 'mNG'] 
powers = ['1percent', '3percent', '6percent', '12percent']

# colors
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#9467bd', '#8c564b']  # Blue, orange, green, purple, brown

for row, fluorophore in enumerate(fluorophores):
    for col, power in enumerate(powers):
        ax = axs[row, col]
        
        # Only plot if data exists for this combination
        if fluorophore in bl_img and power in bl_img[fluorophore]:
            fit_lines = []
            data_lines = []
            half_life_lines = []
            offset_lines = []
            
            for idx, rep in enumerate(natsorted(bl_img[fluorophore][power]['bleaching_avg_norm'].keys())):
                color = colors[idx % len(colors)]  # Cycle through colors if more than 5 replicates
                data_line, = ax.plot(bl_img[fluorophore][power]['bleaching_avg_norm'][rep],
                        '.',
                        label=f'Rep {rep}',
                        color=color)
                data_lines.append(data_line)
                fit_line, = ax.plot(bl_img[fluorophore][power]['fit_curves'][rep], # Map to 0-100
                        color='black',
                        linewidth=1,
                        linestyle='--')
                fit_lines.append(fit_line)
                
                # Add half-life line
                half_life = bl_img[fluorophore][power]['half_life'][rep]
                exp_length = len(bl_img[fluorophore][power]['bleaching_avg_norm'][rep])
                
                if 0 < half_life <= exp_length:
                    half_life_line = ax.axvline(x=half_life, color=color, linestyle=':', alpha=0.7)
                    half_life_lines.append(half_life_line)
                    
                # Add offset (popt[1]) line
                offset = bl_img[fluorophore][power]['fit_params'][rep][1]
                offset_line = ax.axhline(y=offset, color=color, linestyle='-.', alpha=0.5)
                offset_lines.append(offset_line)
            
            # Sort legend
            handles = data_lines + [fit_lines[0]] if fit_lines else []
            if half_life_lines:
                handles += [half_life_lines[0]]
            if offset_lines:
                handles += [offset_lines[0]]
            labels = [f'Rep {i+1}' for i in range(len(data_lines))] + (['Fit'] if fit_lines else [])
            if half_life_lines:
                labels += ['Half-life']
            if offset_lines:
                labels += ['Offset']
            ax.legend(handles, labels)
        
        # Set titles and labels
        if row == 0:
            ax.set_title(power)
        if col == 0:
            ax.set_ylabel(f'{fluorophore}\nNormalized Intensity')
        if row == 3:
            ax.set_xlabel('Frame')
            
        # Set y-axis limits same for all plots
        ax.set_ylim(0, 110)

plt.tight_layout()
plt.savefig('../results/bleaching_curves_matrix.pdf')
plt.show()




# %%
