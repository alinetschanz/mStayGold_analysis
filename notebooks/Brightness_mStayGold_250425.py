# %%
import numpy as np

from bioio import BioImage
import bioio_czi

import skimage.io as io
from skimage.filters import threshold_otsu
from skimage.exposure import histogram, cumulative_distribution

from scipy.optimize import curve_fit
from scipy.optimize import brentq
from scipy.stats import kruskal

from matplotlib import pyplot as plt

from IPython.display import clear_output

import time


from scipy.optimize import curve_fit
from pathlib import Path

from natsort import natsorted

import seaborn as sns

import scipy.signal as signal

import os


import numpy as np

import itertools


from scikit_posthocs import posthoc_dunn
from scipy.stats import shapiro, levene
import scikit_posthocs as sp

# %% [markdown]
# # mStayGold photobleaching analysis
# ## Data description
# * Here we compare three mStayGold variants: `mSG-BJ` (mStayGold-BJ), `mSG-B` (mStayGold-B), and `mSG-J` (mStayGold-J). As a reference, we also include `mNG` (mNeonGreen), a standard in the field. These fluorescent proteins (FPs) were excited at a 488 nm wavelength.
# * mRNA encoding each variant was injected into zebrafish embryos (1-cell-stage). We co-injected `mSc3` (mScarlet3) with each injection to control for variation in injection volume.
# * Data:
#     * **Background images** (shot noise) in the green channel (488 nm) and red channel (561 nm). They will be used to calculate the offset for each imaging channel and are specific for the detector & optical set up.
#     * **Brightness**: For each variant, we acquired 3 dual-color images over 20 zebrafish embryos. mSG variants were imaged at 3% 488 nm, and mSc3 was imaged at 0.5% 561 nm. These images will be used to compare absolute brightness of the mSG variants, normalized by the signal intensity of mSc3.
#     * **Photostability**: 800 frames at variying laser power were acquired to generate a bleaching time series:
#         * 12% power.
#         * 6% power.
#         * 3% power.
#     * Total, per variant, we have 3 * 20 dual-color images (brightness) and 9 bleaching time series (photostability).
#
# ## Analysis
# * Correction:
#     * Determine the noise threshold: average intensity per background image.
#     * Subtract background value from the data (per pixel).
#     * Remove empty and saturated pixels (NaN).
# * Calculate brightness:
#     * Calculate average intensity per frame for corrected channels.
#     * Normalize each average intensity of mSG to the equivalent mSc3 frame.
# * Analysis
#     * Calculate average brightness per fish (mean of 3 replicates)
#     * Compare distributions across 4 conditions: 
#         * Test for normality
#         * ANOVA
#
#
#

# %% [markdown]
# # Brightness
# ## Loading data
# Load background images and acquisition data. Reformat data and assign correct channels

# %%
#Set data directory
data_dir = "/Users/alinetschanz/SwinburneLab Dropbox/Aline/Data"

# %%
#Read & plot background image
bg_files = BioImage(f"{data_dir}/2024/1022/mNG-mSc3-background.czi", reader=bioio_czi.Reader).get_image_data("CYX")

plt.imshow(bg_files[0, :, :])
plt.title('Background Image Frame green')
plt.show()

plt.imshow(bg_files[1, :, :]) 
plt.title('Background Image Frame red')
plt.show()

#Calculate average background intensity and assign to specific imaging channel
bg_green = np.mean(bg_files[0, :, :])
bg_red = np.mean(bg_files[1, :, :])

print(bg_green, bg_red)



# %%
#Load brightness data (3 replicates across 21 embryos per fluorescent protein)
stack_dir = f"{data_dir}/2025/0425"

stk_img = {}
for FP_path in Path(stack_dir).iterdir():
    if FP_path.is_dir():
        FP = FP_path.name
        stk_img[FP] = {}
        for file in FP_path.iterdir():
            if file.name.endswith('.czi'):
                fish_name = file.name.split('_')[0]
                rep = file.name.split('-')[-1].split('.')[0]
                if fish_name not in stk_img[FP]:
                    stk_img[FP][fish_name] = {}
                stk_img[FP][fish_name][rep] = BioImage(str(file), reader=bioio_czi.Reader).get_image_data("CYX")

#Sanity check: Plot example image
##Green channel
plt.imshow(stk_img['mSG(J)']['Fish11']['02'][1, :, :])
plt.title("488 nm channel - mSG")
plt.show()
##Red channel
plt.imshow(stk_img['mSG(J)']['Fish11']['02'][0, :, :])
plt.title("560 nm channel - Sc3")
plt.show()

# %%
#Sanity check for data format: 2 channels, 512 x 512 pixels; 3 replicates; 21 embryos per FP; 4 FP

print(stk_img['mSG(J)']['Fish01']['01'].shape)

for FP in stk_img.keys():
    for fish_name in sorted(stk_img[FP].keys()):
        for rep in sorted(stk_img[FP][fish_name].keys()):
            if isinstance(stk_img[FP][fish_name][rep], np.ndarray):
                shape = stk_img[FP][fish_name][rep].shape
                # print(f"{FP} {fish_name} {rep}: {shape}")
                if shape != (2, 512, 512):
                    print(f"WARNING: Unexpected array shape for {FP} {fish_name} {rep}. Expected (2, 512, 512), got {shape}")
            else:
                print(f"{FP} {fish_name} {rep}: Not an array")

##Print summary of dataset structure
print("Dataset Summary:")
for FP in stk_img.keys():
    num_fish = len(set(stk_img[FP].keys()))
    num_reps = set()
    for fish_name in stk_img[FP].keys():
        num_reps.update(stk_img[FP][fish_name].keys())
    
    print(f"{FP}:")
    print(f"  - Number of fish: {num_fish}")
    print(f"  - Replicates per fish: {len(num_reps)}")
    
    if len(num_reps) != 3:
        print(f"  - WARNING: {len(num_reps)} replicates instead of the expected 3!")
    print()




# %%
#Assing red and green channels per replicate
for FP in stk_img.keys():
    for fish_name in sorted(stk_img[FP].keys()):
        for rep in sorted(stk_img[FP][fish_name].keys()):
            if isinstance(stk_img[FP][fish_name][rep], np.ndarray):
                # Wrap in dictionary format
                stk_img[FP][fish_name][rep] = {
                    'data': stk_img[FP][fish_name][rep],
                    'corrected': {
                     'red': stk_img[FP][fish_name][rep][0],
                     'green': stk_img[FP][fish_name][rep][1]
                }
            }

##Number of saturated pixels
print(f"Total number of pixels: {stk_img['mNG']['Fish04']['02']['corrected']['green'].size}")
print(f"Number of saturated pixels: {np.sum(stk_img['mSG(J)']['Fish04']['02']['corrected']['red'] == 65535)}")

# %% [markdown]
# ## Background correction
# This is specific for every detector & optical set up. The calculated offset can be applied to all images taken with the same filter and laser settings. Background images were taken by moving away from any detectable sample, and taking an image with the smallest laser power possible.

# %%
#Background correction
##Remove saturated pixels
for FP in stk_img.keys():
    for fish_name in sorted(stk_img[FP].keys()):
        for rep in sorted(stk_img[FP][fish_name].keys()):
            # Perform the correction for the red channel
            stk_img[FP][fish_name][rep]['corrected']['red_corr'] = stk_img[FP][fish_name][rep]['corrected']['red'].astype(np.float64) # Convert to float64
            stk_img[FP][fish_name][rep]['corrected']['red_corr'][stk_img[FP][fish_name][rep]['corrected']['red_corr']== 65535] = np.nan # Set saturated pixels to NaN
            stk_img[FP][fish_name][rep]['corrected']['red_corr'] = stk_img[FP][fish_name][rep]['corrected']['red_corr'] - bg_red # Substract background
            stk_img[FP][fish_name][rep]['corrected']['red_corr'][stk_img[FP][fish_name][rep]['corrected']['red_corr'] <= 0] = np.nan  # Set background values to NaN

            # Perform the correction for the green channel
            stk_img[FP][fish_name][rep]['corrected']['green_corr'] = stk_img[FP][fish_name][rep]['corrected']['green'].astype(np.float64)
            stk_img[FP][fish_name][rep]['corrected']['green_corr'][stk_img[FP][fish_name][rep]['corrected']['green_corr'] == 65535] = np.nan
            stk_img[FP][fish_name][rep]['corrected']['green_corr'] = stk_img[FP][fish_name][rep]['corrected']['green_corr'] - bg_green
            stk_img[FP][fish_name][rep]['corrected']['green_corr'][stk_img[FP][fish_name][rep]['corrected']['green_corr'] <= 0] = np.nan 


#Sanity check: Plot example image
##Red channel
plt.imshow(stk_img['mSG(J)']['Fish04']['02']['corrected']['red_corr'][:,:])
plt.show()
##Green channel
plt.imshow(stk_img['mSG(J)']['Fish04']['02']['corrected']['green_corr'][:,:])
plt.show()





# %% [markdown]
# # Brightness
# Calculate average intensity per frame for both channels

# %%
#Calculate average intensity per frame
for FP in stk_img.keys():
    for fish_name in sorted(stk_img[FP].keys()):
        for rep in sorted(stk_img[FP][fish_name].keys()):
            stk_img[FP][fish_name][rep]['corrected']['red_avg'] = np.nanmean(stk_img[FP][fish_name][rep]['corrected']['red_corr'])
            stk_img[FP][fish_name][rep]['corrected']['green_avg'] = np.nanmean(stk_img[FP][fish_name][rep]['corrected']['green_corr'])

# %% [markdown]
# Plot green and red channel for each FP

# %%
FPs = ['mNG', 'mSG(BJ)', 'mSG(B)', 'mSG(J)']  # List of fluorescent proteins

for FP in FPs:
    fish_names = sorted(stk_img[FP].keys())
    green_data = []
    red_data = []

    for fish_name in fish_names:
        green_reps = [
            stk_img[FP][fish_name][rep]['corrected']['green_avg']
            for rep in sorted(stk_img[FP][fish_name].keys())
        ]
        red_reps = [
            stk_img[FP][fish_name][rep]['corrected']['red_avg']
            for rep in sorted(stk_img[FP][fish_name].keys())
        ]
        green_data.append(green_reps)
        red_data.append(red_reps)

    positions = np.arange(len(fish_names)) + 1
    offset = 0.2

    # Create plot
    plt.figure(figsize=(14, 6))

    # Scatter points
    for i in range(len(fish_names)):
        plt.scatter([positions[i] - offset] * len(green_data[i]), green_data[i], color='green', alpha=0.6, zorder=3)
        plt.scatter([positions[i] + offset] * len(red_data[i]), red_data[i], color='red', alpha=0.6, zorder=3)


    # Median lines
    green_medians = [np.median(g) for g in green_data]
    red_medians = [np.median(r) for r in red_data]

    plt.plot(positions - offset, green_medians, color='green', linestyle='-', linewidth=2, label='Green median')
    plt.plot(positions + offset, red_medians, color='red', linestyle='-', linewidth=2, label='Red median')

    # Customize plot
    plt.xticks(positions, fish_names, rotation=45, ha='right')
    plt.xlabel('Embryos')
    plt.ylabel('Average Intensity')
    plt.title(f'Fluorescent Protein: {FP}')
    plt.legend()
    plt.tight_layout()
    plt.show()

# %% [markdown]
# ## Normalization
# Normalize mSG intensity to mSc3 intensity

# %%
for FP in stk_img.keys(): 
    for fish_name in sorted(stk_img[FP].keys()):
        for rep in sorted(stk_img[FP][fish_name].keys()):
            stk_img[FP][fish_name][rep]['corrected']['green_avg_norm'] = {}
            
            img = stk_img[FP][fish_name][rep]['corrected']['green_avg'] # image
            ref = stk_img[FP][fish_name][rep]['corrected']['red_avg'] # reference
                
            # Calculate normalization factor for the current replicate
            factor = img / ref
                
            # Normalize data by factor
            stk_img[FP][fish_name][rep]['corrected']['green_avg_norm'] = stk_img[FP][fish_name][rep]['corrected']['green_avg'] * factor

# %%
#Access normalization factor for correction of represntative images
print(f"Normalization factor for mSG(J) Fish01 rep 2: {stk_img['mSG(B)']['Fish17']['02']['corrected']['green_avg'] / stk_img['mSG(B)']['Fish17']['02']['corrected']['red_avg']}")



# %% [markdown]
# # Plotting
# Plot normalized green fluorescence inentsity for each FP. Caluclate mean normalized intensity across each replicate and plot per FP.

# %%
#plot normalized intensity per FP
##Initialize dictionary to store data for box plot
data_dict = {FP: [] for FP in stk_img.keys()}

##Collect data for each fluorescent protein
for FP in stk_img.keys():
    for fish_name in stk_img[FP].keys():
        for rep in stk_img[FP][fish_name].keys():
            data_dict[FP].append(stk_img[FP][fish_name][rep]['corrected']['green_avg_norm'])

##Create box plot
plt.figure(figsize=(10, 6))
plt.boxplot([data_dict[FP] for FP in sorted(data_dict.keys())], 
            labels=sorted(data_dict.keys()))

##Add individual points for better visualization
for i, FP in enumerate(sorted(data_dict.keys()), 1):
    plt.scatter([i] * len(data_dict[FP]), data_dict[FP], alpha=0.5)

plt.ylabel('Normalized Fluorescence Intensity [AU]')
plt.xlabel('Fluorescent Protein')

plt.show()




# %%
#Calculate the mean intensity value across three replicates per embryo for each fluorophore
for FP in stk_img.keys():
    for fish_name in stk_img[FP].keys():
        # Collect values from all replicates for this fish
        rep_values = []
        for rep in sorted(stk_img[FP][fish_name].keys()):
            values = stk_img[FP][fish_name][rep]['corrected']['green_avg_norm']
            rep_values.append(values)
        
        # Calculate mean across all replicates
        mean_across_reps = np.mean(rep_values, axis=0)
        
        # Store the mean value back in each replicate
        for rep in sorted(stk_img[FP][fish_name].keys()):
            stk_img[FP][fish_name][rep]['corrected']['green_avg_norm_mean'] = mean_across_reps

# %%
#plot normalized intensity per FP

data_dict = {FP: [] for FP in stk_img.keys()}

##Collect mean values for each FP
for FP in stk_img.keys():
    for fish_name in stk_img[FP].keys():
        for rep in stk_img[FP][fish_name].keys():
            data_dict[FP].append(stk_img[FP][fish_name][rep]['corrected']['green_avg_norm_mean'])

##Create box plot
plt.figure(figsize=(10, 6))
plt.boxplot([data_dict[FP] for FP in sorted(data_dict.keys())], 
            labels=sorted(data_dict.keys()))

plt.ylabel('Normalized Mean Fluorescence Intensity [AU]')
plt.xlabel('Fluorescent Protein')

##Add scatter points
for i, FP in enumerate(sorted(data_dict.keys()), 1):
    plt.scatter([i] * len(data_dict[FP]), data_dict[FP], alpha=0.5)

plt.savefig('../Results/PDFs/brightness_boxplot_mSG_250425.pdf')
plt.show()




# %% [markdown]
# # Statistics
# To test for differences in brightness, I first assess normality and variance. 

# %%
# Plot density plots of log-transformed data for each FP


data_dict = {FP: [] for FP in stk_img.keys()}

# Collect mean values for each FP
for FP in stk_img.keys():
    for fish_name in stk_img[FP].keys():
        for rep in stk_img[FP][fish_name].keys():
            data_dict[FP].append(stk_img[FP][fish_name][rep]['corrected']['green_avg_norm_mean'])

# Log transform the data (add small constant to avoid log(0))
log_data_dict = {}
for FP, values in data_dict.items():
    arr = np.array(values)
    arr = arr[arr > 0]  # Remove non-positive values if any
    log_data_dict[FP] = np.log10(arr + 1e-6)

# Plot density plots
plt.figure(figsize=(10, 6))
for FP in sorted(log_data_dict.keys()):
    data = log_data_dict[FP]
    if len(data) > 0:
        sns.kdeplot(data, label=FP, fill=True, alpha=0.4)
plt.xlabel('log10(Normalized Mean Fluorescence Intensity [AU])')
plt.ylabel('Density')
plt.title('Density Plot of log10-Transformed Normalized Mean Fluorescence Intensity')
plt.legend(title='Fluorescent Protein')
plt.tight_layout()
plt.show()


# %%
# Test for normality of log-transformed data for each FP and test for equal variance


data_dict = {FP: [] for FP in stk_img.keys()}

# Collect mean values for each FP
for FP in stk_img.keys():
    for fish_name in stk_img[FP].keys():
        for rep in stk_img[FP][fish_name].keys():
            data_dict[FP].append(stk_img[FP][fish_name][rep]['corrected']['green_avg_norm_mean'])

# Log transform the data (add small constant to avoid log(0))
log_data_dict = {}
for FP, values in data_dict.items():
    arr = np.array(values)
    arr = arr[arr > 0]  # Remove non-positive values if any
    log_data_dict[FP] = np.log10(arr + 1e-6)

# Test for normality using Shapiro-Wilk test
print("Shapiro-Wilk normality test results (log10-transformed data):")
for FP in sorted(log_data_dict.keys()):
    data = log_data_dict[FP]
    if len(data) > 2:  # Shapiro-Wilk requires at least 3 data points
        stat, p = shapiro(data)
        print(f"{FP}: W={stat:.3f}, p={p:.4f} (n={len(data)})")
        if p > 0.05:
            print(f"  -> Data likely normal (fail to reject H0)")
        else:
            print(f"  -> Data not normal (reject H0)")
    elif len(data) == 2:
        print(f"{FP}: Only 2 data points, Shapiro-Wilk not applicable.")
    elif len(data) == 1:
        print(f"{FP}: Only 1 data point, normality test not applicable.")
    else:
        print(f"{FP}: No data available.")

# Test for equal variance (Levene's test) across FPs
# Only include FPs with at least 2 data points
log_data_for_levene = [log_data_dict[FP] for FP in sorted(log_data_dict.keys()) if len(log_data_dict[FP]) > 1]
FPs_for_levene = [FP for FP in sorted(log_data_dict.keys()) if len(log_data_dict[FP]) > 1]

if len(log_data_for_levene) > 1:
    stat, p = levene(*log_data_for_levene)
    print("\nLevene's test for equal variances (log10-transformed data):")
    print(f"FPs included: {FPs_for_levene}")
    print(f"Levene statistic={stat:.3f}, p={p:.4f}")
    if p > 0.05:
        print("  -> Variances likely equal (fail to reject H0)")
    else:
        print("  -> Variances not equal (reject H0)")
else:
    print("\nNot enough groups with >1 data point for Levene's test.")


# %% [markdown]
# Data is not normally distributed and of unequal varaince. I thus perform a Kruskal-Wallis test and a Dunn's post-hoc test with Holm correction

# %%
# Kruskal-Wallis test and Dunn's post-hoc test with Holm correction

# Prepare data: collect log10-transformed means for each FP
data_dict = {FP: [] for FP in stk_img.keys()}

for FP in stk_img.keys():
    for fish_name in stk_img[FP].keys():
        for rep in stk_img[FP][fish_name].keys():
            data_dict[FP].append(stk_img[FP][fish_name][rep]['corrected']['green_avg_norm_mean'])

# Log transform (add small constant to avoid log(0))
log_data_dict = {}
for FP, values in data_dict.items():
    arr = np.array(values)
    arr = arr[arr > 0]  # Remove non-positive values if any
    log_data_dict[FP] = np.log10(arr + 1e-6)

# Only include FPs with at least 2 data points for the tests
FPs_for_test = [FP for FP in sorted(log_data_dict.keys()) if len(log_data_dict[FP]) > 1]
log_data_for_test = [log_data_dict[FP] for FP in FPs_for_test]

# Kruskal-Wallis test

stat, p = kruskal(*log_data_for_test)
print("Kruskal-Wallis test (log10-transformed data):")
print(f"FPs included: {FPs_for_test}")
print(f"Kruskal-Wallis statistic={stat:.3f}, p={p:.4g}")
if p < 0.05:
    print("  -> Significant differences detected among groups (reject H0)")
else:
    print("  -> No significant differences among groups (fail to reject H0)")


# Dunn's post-hoc test with Holm correction
dunn_results = None

# Prepare data for posthoc_dunn: flat array and group labels
all_data = []
all_groups = []
for FP in FPs_for_test:
    all_data.extend(log_data_dict[FP])
    all_groups.extend([FP] * len(log_data_dict[FP]))

dunn_results = sp.posthoc_dunn([log_data_dict[FP] for FP in FPs_for_test], p_adjust='holm')
dunn_results.index = FPs_for_test
dunn_results.columns = FPs_for_test
print("\nDunn's post-hoc test with Holm correction (p-values):")
print(dunn_results)

# --- Plot boxplots and add stars for significant differences ---

fig, ax = plt.subplots(figsize=(10, 6))
box = ax.boxplot([log_data_dict[FP] for FP in FPs_for_test], labels=FPs_for_test, patch_artist=True)

ax.set_ylabel('log10(green_avg_norm_mean)')
ax.set_title('Boxplot of log10-transformed green_avg_norm_mean by FP')

# Color the boxes using default matplotlib colors (tab10)
default_colors = plt.get_cmap('tab10').colors
for patch, color in zip(box['boxes'], default_colors):
    patch.set_facecolor(color)

# Add significance stars for significant pairwise differences
def get_star(pval):
    if pval < 0.001:
        return '***'
    elif pval < 0.01:
        return '**'
    elif pval < 0.05:
        return '*'
    else:
        return None

# For each pair, if significant, draw a line and add a star
y_max = max([max(vals) if len(vals) > 0 else 0 for vals in log_data_for_test])
y_min = min([min(vals) if len(vals) > 0 else 0 for vals in log_data_for_test])
h = (y_max - y_min) * 0.05 if (y_max - y_min) > 0 else 0.1
num_pairs = 0
pairs_done = set()
for i, FP1 in enumerate(FPs_for_test):
    for j, FP2 in enumerate(FPs_for_test):
        if j <= i:
            continue
        pval = dunn_results.loc[FP1, FP2]
        star = get_star(pval)
        if star:
            # Draw a line between box i and box j
            x1, x2 = i+1, j+1
            y = y_max + h + num_pairs*h*1.5
            ax.plot([x1, x1, x2, x2], [y, y+h/2, y+h/2, y], lw=1.5, c='k')
            ax.text((x1+x2)/2, y+h/2, star, ha='center', va='bottom', color='k', fontsize=16)
            num_pairs += 1

plt.tight_layout()
plt.show()


