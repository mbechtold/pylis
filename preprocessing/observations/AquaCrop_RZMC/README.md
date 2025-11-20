# Soil Moisture Data Assimilation in AquaCrop (Based on Bechtold et al., in preparation)

## Overview
AquaCrop has limited vertical coupling, which restricts the propagation of assimilated surface soil moisture to deeper layers and the entire root zone.
To address this limitation, surface soil moisture retrievals (here: **SMAP L2 Soil Moisture Enhanced 9 km product**) are translated into deeper soil moisture using the **exponential filter** approach (Albergel et al., 2008).

This README provides a walkthrough for a test case located at:
`/staging/leuven/stg_00024/OUTPUT/michelb/LIS/testcases_michel/ac72_rzmc_da/`

---

## 1. Source Code for SM Data Assimilation in AquaCrop
Clone the repository and compile as usual:
```bash
git clone --branch working/ac72_SMAP_RZMC_DA --single-branch https://github.com/mbechtold/LISF.git
```

---

## 2. Run Open Loop (OL) Simulation with SMAP L2 Soil Moisture
Navigate to the OL directory:
```bash
cd OL_SMAP_L2
```
Adjust `LIS_OL_SMAP_L2.slurm` (email, paths to compiled code, other paths, etc.) and submit the job.

**Purpose of this run:**
- Align SMAP L2 retrievals with AquaCrop-modeled soil moisture.
- Apply QC flagging to SMAP L2 retrievals using LIS-implemented flags.

---

## 3. Optimize Exponential Filter and Generate RZMC Observation Files
Obtain the Python scripts for the exponential filter from:
[https://github.com/KUL-RSDA/pylis/preprocessing/observations/AquaCrop_RZMC/](https://github.com/KUL-RSDA/pylis/preprocessing/observations/AquaCrop_RZMC/)

Navigate to the EXP_FILTER directory:
```bash
cd EXP_FILTER
```
Adjust `expfilter_optimization.slurm` and submit.
Output files will be written to the folder specified in the command-line arguments.

**Optional:** Plot time series by adjusting and submitting `expfilter_plotting_timeseries.slurm`.

---

## 4. Run OL and Data Assimilation (DA) with Exp-Filtered (RZMC) Observations
Navigate to the respective directories and submit jobs:
```bash
cd OL_SMAP_RZMC
# Adjust LIS_OL_SMAP_RZMC.slurm and submit

cd DA_SMAP_RZMC
# Adjust LIS_DA_SMAP_RZMC.slurm and submit
```

---

### References
- Albergel, C. et al. (2008). *From near-surface to root-zone soil moisture using an exponential filter: an assessment of the method based on in-situ observations and model simulations. https://doi.org/10.5194/hess-12-1323-2008*

---

