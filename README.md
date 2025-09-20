# mStayGold analysis

This is the code acompanying the micropublication "Systematic comparison of mStayGold and mGold2 variants for live imaging in zebrafish".

## Data description
* Here we compare three mStayGold variants: `mSG-BJ` (mStayGold-BJ), `mSG-B` (mStayGold-B), and `mSG-J` (mStayGold-J). As a reference, we also include `mNG` (mNeonGreen), a standard in the field. These fluorescent proteins (FPs) were excited at a 488 nm wavelength.
* mRNA encoding each variant was injected into zebrafish embryos (1-cell-stage). We co-injected `mSc3` (mScarlet3) with each injection to control for variation in injection volume.
* Data:
    * **Background images** (shot noise) in the green channel (488 nm) and red channel (561 nm). They will be used to calculate the offset for each imaging channel and are specific for the detector & optical set up.
    * **Brightness**: For each variant, we acquired 3 dual-color images over 20 zebrafish embryos. mSG variants were imaged at 3% 488 nm, and mSc3 was imaged at 0.5% 561 nm. These images will be used to compare absolute brightness of the mSG variants, normalized by the signal intensity of mSc3.
    * **Photostability**: 800 frames at variying laser power were acquired to generate a bleaching time series:
        * 12% power.
        * 6% power.
        * 3% power.
    * Total, per variant, we have 3 * 20 dual-color images (brightness) and 9 bleaching time series (photostability).

## Analysis
* Correction:
    * Determine the noise threshold: average intensity per background image.
    * Subtract background value from the data (per pixel).
    * Remove empty and saturated pixels (NaN).
* Calculate brightness:
    * Calculate average intensity per frame for corrected channels.
    * Normalize each average intensity of mSG to the equivalent mSc3 frame.
* Analysis
    * Calculate average brightness per fish (mean of 3 replicates)
    * Compare distributions across 4 conditions: 
        * Test for normality
        * ANOVA




### Setup
1. Install Conda (Miniconda or Anaconda).
2. Create the environment:
   ```bash
   conda env create --file environment.yml
   conda activate msg
   ```

### Working with notebooks
- Keep formats in sync (uses `jupytext.toml`):
  ```bash
  ./jupytext_sync        # one-time sync now
  ./jupytext_sync --watch # optional: continuous sync (built-in polling)
  ```

- Open Jupyter:
  ```bash
  jupyter lab
  ```
- Update environment if new dependencies are needed:
  Add the name of the required package to the list in `environment.yml`, then run:
  ```bash
  conda env update --file environment.yml --prune
  ```



