SM DA in AquaCrop following Bechtold et al., in prep.

Rationale of approach:
Limited vertical coupling in AquaCrop prevents the model propogation of assimilated surface soil moisture to deeper layers and the whole roote zone
To overcome this limitation, the surface soil misoture retreivals (here: SMAP L2 Soil Moisture Enhanced 9 km product, url:) are first translated to a deeper soil miosture using
the exp filter (Albergel, 2015)

This file provides a walkthrough trough a test case that is provided in:
/staging/leuven/stg_00024/OUTPUT/michelb/LIS/testcases_michel/ac72_rzmc_da/

(1) Source code for SM DA in AquaCrop
git clone --branch working/ac72_SMAP_RZMC_DA --single-branch https://github.com/mbechtold/LISF.git
Compile code as usual

(2) Run open loop (OL) simulation with SMAP L2 SM
cd OL_SMAP_L2
Adjust (mail, paths, etc.) LIS_OL_SMAP_L2.slurm and submit
The purpose of this run is:
- Aligning of the SMAP L2 retrievals and AquaCrop modeled soil moisture
- QC flagging of SMAP L2 retrievals using the flags implemented in LIS

3) Optimization of exp filter and write out of rzmc obs files
Obtain expilter python scripts from 
https://github.com/KUL-RSDA/pylis/preprocessing/observations/AquaCrop_RZMC/
cd EXP_FILTER
Adjust expfilter_optimization.slurm and submit
Files will be written to a folder as defined in the command line arguments
Optionally plotting of time series:
Adjust expfilter_plotting_timeseries.slurm and submit

4) Run OL and data assimilation (DA) with the exp filtered (RZMC) observations
cd OL_SMAP_RZMC
Adjust LIS_OL_SMAP_RZMC.slurm and submit
cd DA_SMAP_RZMC
Adjust LIS_DA_SMAP_RZMC.slurm and submit

