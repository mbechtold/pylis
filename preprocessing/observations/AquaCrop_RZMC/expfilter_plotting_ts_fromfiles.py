#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Plot time series of (a) observational SMAP monthly CDF-matched data loaded
DIRECTLY from a folder of NetCDF files and (b) modeled root-zone soil moisture
aggregated over user-specified layers.

One PNG per randomly selected valid grid cell; file names include lat/lon
rounded to two digits.

Example filenames (per provided structure):
  SMAP_RZMC_AC72_YYYYMMDD.nc
Each file contains a single time slice (time=1) with variable "sm(time,lat,lon)".

Usage example:
python /data/leuven/317/vsc31786/python/zdenko/expfilter_plotting_ts_fromfiles.py \
  --exp_folder /staging/leuven/stg_00024/OUTPUT/michelb/LIS/testcases_michel/ac72_rzmc_da/OL_SMAP_L2 \
  --lis_input_file /staging/leuven/stg_00024/OUTPUT/michelb/LIS/testcases_michel/ac72_rzmc_da/LDT_FILE_GENERATION/lis_input.d01.nc \
  --obs_folder /staging/leuven/stg_00024/OUTPUT/michelb/LIS/testcases_michel/ac72_rzmc_da/EXP_FILTER/SMAP_RZMC_30CM_AC72_MONTHLY \
  --fig_folder /staging/leuven/stg_00024/OUTPUT/michelb/LIS/testcases_michel/ac72_rzmc_da/EXP_FILTER/figures \
  --nsamples 5 --seed 42 --rz_layers 1-6 \
  --tmin 2015-01-01 --tmax 2022-12-31
"""

# --- Standard libs
import os
import sys
import argparse

# --- Third-party
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt

# --- Project libs
sys.path.append("/data/leuven/317/vsc31786/python/zdenko/scripts")
from pylis_zdenko import readers

# ------------------------
# Arguments
# ------------------------
parser = argparse.ArgumentParser(
    description="Plot SMAP monthly CDF-matched obs (from folder) vs LIS RZMC layer-aggregated model"
)
parser.add_argument(
    "--exp_folder",
    required=True,
    help="Path to LIS/SHUI experiment root (contains 'output/' or 'OUTPUT/' subfolder)",
)
parser.add_argument(
    "--lis_input_file",
    required=True,
    help="Full path to LIS input file (defines grid & landmask).",
)
parser.add_argument("--obs_folder", required=True, help="Folder with monthly NetCDFs (time=1 per file)")
parser.add_argument("--pattern", default="*.nc", help="Glob pattern inside obs_folder (default: *.nc)")
parser.add_argument("--varname", default="sm", help="Variable name in NetCDF files (default: sm)")
parser.add_argument("--latname", default="lat", help="Latitude variable name (default: lat)")
parser.add_argument("--lonname", default="lon", help="Longitude variable name (default: lon)")
parser.add_argument("--timename", default="time", help="Time variable name (default: time)")
parser.add_argument("--nsamples", type=int, default=20, help="Number of random locations to plot (default: 20)")
parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
parser.add_argument(
    "--rz_layers",
    default="1-6",
    help=(
        "Root-zone layer selection by coordinate values. Examples: '1-10', '1,2,3,4,5', or 'all'. "
    ),
)
parser.add_argument("--min_pairs", type=int, default=100, help="Minimum overlapping time steps needed per pixel (default: 100)")
parser.add_argument("--fig_folder", type=str, required=True, default=None, help="folder for time series figures")
parser.add_argument("--tmin", type=str, default=None, help="Minimum date (inclusive), e.g. 2016-01-01")
parser.add_argument("--tmax", type=str, default=None, help="Maximum date (inclusive), e.g. 2020-12-31")

args = parser.parse_args()

nsamples = int(args.nsamples)
seed = int(args.seed)

# ------------------------

os.makedirs(args.fig_folder, exist_ok=True)

# ------------------------
# Helpers for layer selection
# ------------------------


def _parse_rz_layers(spec, layer_coords):
    lc = np.array(layer_coords, dtype=float)
    s = str(spec).strip().lower()
    if s == "all":
        return list(lc)
    if "-" in s:
        a, b = s.split("-", 1)
        a = float(a)
        b = float(b)
        if a > b:
            a, b = b, a
        want_vals = lc[(lc >= a) & (lc <= b)]
        return list(np.unique(np.sort(want_vals)))
    want_vals = np.array([float(x) for x in s.split(",") if x.strip() != ""], dtype=float)
    mask = np.isin(lc, want_vals)
    return list(np.unique(np.sort(lc[mask])))


# ------------------------
# Load observations from folder (stack along time)
# ------------------------

def load_obs_folder(folder, pattern, varname, latname, lonname, timename):
    """Load monthly files without dask by opening each file eagerly and concatenating.
    Assumes each file has a single time slice (time=1)."""
    import glob
    import os
    import numpy as np
    import xarray as xr

    files = sorted(glob.glob(os.path.join(folder, pattern)))
    if not files:
        raise FileNotFoundError(f"No files matching {pattern} in {folder}")

    print(f"[load_obs_folder] Found {len(files)} files in '{folder}' matching '{pattern}'")
    das = []

    for i, fp in enumerate(files, start=1):
        print(f"\n[load_obs_folder] ({i}/{len(files)}) Loading file: {fp}")
        ds = xr.open_dataset(fp, engine="netcdf4", decode_times=True)
        ds.load()  # bring everything into memory
        ds.close()

        # Basic dataset overview
        print(f"  Dataset coords: {list(ds.coords)}")
        print(f"  Dataset vars:   {list(ds.data_vars)}")
        print(f"  Dataset dims:   {dict(ds.dims)}")

        if varname not in ds.variables:
            raise RuntimeError(f"Variable '{varname}' not found in {fp}")
        for nm in (latname, lonname, timename):
            if nm not in ds.variables and nm not in ds.coords:
                raise RuntimeError(f"Coordinate '{nm}' not found in {fp}")

        da = ds[varname]

        # ensure dims order [time, lat, lon]
        want_dims = [timename, latname, lonname]
        if list(da.dims) != want_dims:
            print(f"  Transposing {varname} from dims {da.dims} to {want_dims}")
            da = da.transpose(*want_dims)

        # rename to canonical dims only if different names to avoid warnings
        rename_map = {}
        if timename != "time":
            rename_map[timename] = "time"
        if latname != "lat":
            rename_map[latname] = "lat"
        if lonname != "lon":
            rename_map[lonname] = "lon"
        if rename_map:
            print(f"  Renaming dims: {rename_map}")
            da = da.rename(rename_map)

        # enforce float32 and NaNs
        da = da.where(np.isfinite(da)).astype(np.float32)

        # sanity: each file should have time length 1
        if da.sizes["time"] != 1:
            raise RuntimeError(f"Expected single time step in file {fp}, found {da.sizes['time']}")

        # Per-file data overview
        time_values = da["time"].values
        try:
            t_str = np.array2string(time_values, max_line_width=80)
        except Exception:
            t_str = str(time_values)

        da_min = float(da.min().values)
        da_max = float(da.max().values)
        da_mean = float(da.mean().values)

        print(f"  Variable '{varname}':")
        print(f"    dims:   {da.dims}")
        print(f"    shape:  {da.shape}")
        print(f"    time:   {t_str}")
        print(f"    min:    {da_min:.4g}")
        print(f"    max:    {da_max:.4g}")
        print(f"    mean:   {da_mean:.4g}")
        print(f"    n_nans: {int(np.isnan(da.values).sum())}")

        das.append(da)

    # Concatenate along time and sort by time just in case
    out = xr.concat(das, dim="time")
    out = out.sortby("time")

    # Overview of concatenated result
    print("\n[load_obs_folder] Concatenated result:")
    print(f"  dims:   {out.dims}")
    print(f"  shape:  {out.shape}")
    print(f"  n_time: {out.sizes['time']}")
    print(f"  time range: {out['time'].min().values} -> {out['time'].max().values}")

    # Simple overview of time spacing (if > 1 timestep)
    if out.sizes["time"] > 1:
        time_vals = out["time"].values
        n_show = min(3, out.sizes["time"])
        print("  first time points:", time_vals[:n_show])
        print("  last time points: ", time_vals[-n_show:])
        deltas = np.diff(time_vals)
        print(f"  median delta between times: {np.median(deltas)}")

    # Basic stats over all data
    out_min = float(out.min().values)
    out_max = float(out.max().values)
    out_mean = float(out.mean().values)
    print("  data stats over all time:")
    print(f"    min:  {out_min:.4g}")
    print(f"    max:  {out_max:.4g}")
    print(f"    mean: {out_mean:.4g}")

    return out


# ------------------------
# Load model SM and aggregate to root-zone (from LIS via readers.lis_cube)
# ------------------------


def load_model_rz(exp_folder, lis_input_file, rz_layers, start_date, end_date,
                  lats_range, lons_range):
    """
    Load LIS soil moisture via readers.lis_cube and aggregate selected layers
    to root-zone, mimicking the behavior in expfilter_rzmc_gapopt_refactored.py.

    Parameters
    ----------
    exp_folder : str
        Root of LIS/SHUI experiment (contains 'output' subdir).
    lis_input_file : str
        Unused here, kept for API consistency.
    rz_layers : str
        Root-zone layer selection, e.g. '1-6', 'all', '1,2,3'.
    start_date, end_date : str
        'DD/MM/YYYY' strings.
    lats_range, lons_range : 1D arrays
        Latitude / longitude coordinates to pass to lis_cube
        (should match the obs grid).
    """
    print("  [MODEL] Calling readers.lis_cube (mimic pipeline)...")
    print(f"  [MODEL] lis_dir   = {exp_folder}/output or {exp_folder}/OUTPUT")
    print(f"  [MODEL] start/end = {start_date} -> {end_date}")
    print(f"  [MODEL] lats: {lats_range[0]}..{lats_range[-1]} (n={len(lats_range)})")
    print(f"  [MODEL] lons: {lons_range[0]}..{lons_range[-1]} (n={len(lons_range)})")

    # Primary and fallback output directories
    dir_primary = os.path.join(exp_folder, "output")
    dir_fallback = os.path.join(exp_folder, "OUTPUT")

    # Choose the directory that exists
    if os.path.isdir(dir_primary):
        output_dir = dir_primary
    elif os.path.isdir(dir_fallback):
        output_dir = dir_fallback
    else:
        raise FileNotFoundError(f"No output directory found in {exp_folder}")

    dc_ol = readers.lis_cube(
        output_dir,
        var="SoilMoist_tavg",
        n_layers=12,
        lats=lats_range,
        lons=lons_range,
        start=start_date,
        end=end_date,
        h=0,
    )

    sm_da = dc_ol  # expect dims: ('time','layer','lat','lon')
    print(f"  [MODEL] Loaded model data dims: {sm_da.dims}")
    print(f"  [MODEL] Shape: {sm_da.shape}")
    print(f"  [MODEL] Coords: {list(sm_da.coords)}")

    if "time" not in sm_da.dims:
        raise RuntimeError("Model DataArray has no 'time' dim – unexpected.")
    print(f"  [MODEL] Time range: {sm_da['time'].min().values} -> {sm_da['time'].max().values}")
    print(f"  [MODEL] n_time = {sm_da.sizes['time']}")

    if "layer" not in sm_da.dims:
        raise RuntimeError("Model soil moisture DataArray must contain a 'layer' dimension.")

    layer_coords = sm_da["layer"].values
    print(f"  [MODEL] Available layer coords: {layer_coords}")

    # Always include surface layer = 1 (like in the pipeline)
    surface_val = 1.0
    rz_vals = _parse_rz_layers(rz_layers, layer_coords)
    if surface_val not in rz_vals:
        rz_vals = sorted(list(np.unique(np.append(rz_vals, surface_val))))

    print(f"  [MODEL] RZ layer spec       : {rz_layers}")
    print(f"  [MODEL] Effective RZ layers : {rz_vals}")

    # Root-zone mean
    rz = sm_da.sel(layer=rz_vals).mean("layer")

    # Make sure dims are (time, lat, lon)
    for needed in ("time", "lat", "lon"):
        if needed not in rz.dims:
            raise RuntimeError(f"RZ DataArray missing required dim '{needed}'. Dims: {rz.dims}")

    if list(rz.dims) != ["time", "lat", "lon"]:
        print(f"  [MODEL] Transposing RZ data from dims {rz.dims} to ('time','lat','lon').")
        rz = rz.transpose("time", "lat", "lon")

    rz = rz.astype(np.float32)

    # Quick NaN stats
    n_tot = rz.size
    n_nan = int(np.isnan(rz.values).sum())
    frac_nan = n_nan / n_tot if n_tot > 0 else np.nan
    print(f"  [MODEL] RZ NaN fraction: {frac_nan*100:.1f}%")
    print(f"  [MODEL] RZ dims: {rz.dims}, shape: {rz.shape}")

    return rz, rz_vals


# ------------------------
# Load data
# ------------------------
print("Loading observations from:", args.obs_folder)
obs = load_obs_folder(
    args.obs_folder, args.pattern, args.varname, args.latname, args.lonname, args.timename
)  # DataArray [time, lat, lon]

# ------------------------
# Optional time filtering
# ------------------------
if args.tmin or args.tmax:
    print("Applying time filtering...")
    tmin = np.datetime64(args.tmin) if args.tmin else None
    tmax = np.datetime64(args.tmax) if args.tmax else None

    if tmin is not None:
        print(f"  tmin = {tmin}")
        obs = obs.sel(time=slice(tmin, None))
    if tmax is not None:
        print(f"  tmax = {tmax}")
        obs = obs.sel(time=slice(None, tmax))

    if obs.sizes.get("time", 0) == 0:
        raise RuntimeError("No obs data left after applying time filter; check --tmin/--tmax.")

print(f"Obs time range after optional filter: {obs.time.values[0]} -> {obs.time.values[-1]}")
print(f"Obs n_time = {obs.sizes['time']}")

# Determine time range for model reading (match obs period)
from datetime import datetime as _dt

tmin_obs_str = np.datetime_as_string(obs.time.values[0], unit="D")  # 'YYYY-MM-DD'
tmax_obs_str = np.datetime_as_string(obs.time.values[-1], unit="D")

tmin_dt = _dt.strptime(tmin_obs_str, "%Y-%m-%d")
tmax_dt = _dt.strptime(tmax_obs_str, "%Y-%m-%d")
start_date_model = tmin_dt.strftime("%d/%m/%Y")
end_date_model = tmax_dt.strftime("%d/%m/%Y")

print(f"Obs time range used for model: {tmin_obs_str} → {tmax_obs_str}")
print(f"Model start/end (DD/MM/YYYY): {start_date_model} → {end_date_model}")

# Use the obs grid as the target grid for the model (mimic pipeline behaviour)
lats_range = obs["lat"].values
lons_range = obs["lon"].values
print(f"Obs grid: lat[{lats_range[0]}..{lats_range[-1]}] (n={len(lats_range)}), "
      f"lon[{lons_range[0]}..{lons_range[-1]}] (n={len(lons_range)})")

print("Loading model soil moisture and aggregating layers from LIS output...")
rz, rz_vals = load_model_rz(
    args.exp_folder,
    args.lis_input_file,
    args.rz_layers,
    start_date_model,
    end_date_model,
    lats_range,
    lons_range,
)
print(f"Selected root-zone layers: {rz_vals}")

# Align model time to obs time (if needed)
obs, rz = xr.align(obs, rz, join="inner")
print(f"After xr.align(join='inner'):")
print(f"  Final obs time range:   {obs.time.values[0]} → {obs.time.values[-1]}")
print(f"  Final model time range: {rz.time.values[0]} → {rz.time.values[-1]}")
print(f"  Obs n_time = {obs.sizes['time']}, Model n_time = {rz.sizes['time']}")

# ------------------------
# Spatial alignment (auto-flip + rounding tolerance if needed)
# ------------------------


def _try_align_coords(obs_da, ref_da, tol_deg=1e-3):
    """Return a view of obs_da whose lat/lon ordering & values match ref_da within tol.
    Tries all flip combinations (none / flip-lat / flip-lon / flip-both) and then
    snaps coordinates to the model using nearest-neighbor reindex within tolerance.
    tol_deg is in degrees (default 1e-3 ≈ 100 m at mid-lats).
    """
    lat_r = ref_da["lat"]
    lon_r = ref_da["lon"]

    candidates = []
    base = obs_da
    candidates.append(base)
    candidates.append(base.isel(lat=slice(None, None, -1)))
    candidates.append(base.isel(lon=slice(None, None, -1)))
    candidates.append(base.isel(lat=slice(None, None, -1), lon=slice(None, None, -1)))

    tol = float(tol_deg)

    for cand in candidates:
        # quick size check
        if (cand.sizes["lat"] != ref_da.sizes["lat"]) or (cand.sizes["lon"] != ref_da.sizes["lon"]):
            continue
        # Snap to model coords with nearest
        try:
            cand2 = cand.reindex(lat=lat_r, method="nearest", tolerance=tol)
            cand2 = cand2.reindex(lon=lon_r, method="nearest", tolerance=tol)
        except Exception:
            continue
        # Verify final difference within tolerance
        lat_ok = np.allclose(cand2["lat"].values, lat_r.values, atol=tol, rtol=0)
        lon_ok = np.allclose(cand2["lon"].values, lon_r.values, atol=tol, rtol=0)
        if lat_ok and lon_ok:
            return cand2

    raise RuntimeError(
        "Could not align obs grid to model grid within tolerance. "
        "Consider increasing tolerance or performing explicit regridding."
    )


# Make sure sizes roughly match first
if (obs.sizes["lat"] != rz.sizes["lat"]) or (obs.sizes["lon"] != rz.sizes["lon"]):
    raise RuntimeError("Obs and model have different spatial sizes. Regridding step needed before plotting.")

print("Attempting coordinate alignment (flip + nearest reindex)...")
obs = _try_align_coords(obs, rz, tol_deg=1e-3)
print("Coordinate alignment successful.")

def _maybe_flip_to_match(obs_da, ref_da):
    """Ensure obs_da has same lat/lon order as ref_da.
    If lat or lon are reversed relative to ref, flip the corresponding dimension.
    Requires same coordinate *values* up to tolerance (or reversed)."""
    lat_o = obs_da["lat"].values
    lon_o = obs_da["lon"].values
    lat_r = ref_da["lat"].values
    lon_r = ref_da["lon"].values

    # Check direct match
    lat_match = np.allclose(lat_o, lat_r, rtol=0, atol=1e-6)
    lon_match = np.allclose(lon_o, lon_r, rtol=0, atol=1e-6)

    # Check reversed match
    lat_rev_match = np.allclose(lat_o[::-1], lat_r, rtol=0, atol=1e-6)
    lon_rev_match = np.allclose(lon_o[::-1], lon_r, rtol=0, atol=1e-6)

    if not (lat_match or lat_rev_match):
        raise RuntimeError("Latitude coordinates do not match model grid (even after reversal). Regridding needed.")
    if not (lon_match or lon_rev_match):
        raise RuntimeError("Longitude coordinates do not match model grid (even after reversal). Regridding needed.")

    out = obs_da
    if lat_rev_match and not lat_match:
        print("Obs latitude is reversed relative to model — flipping lat dimension.")
        out = out.isel(lat=slice(None, None, -1))
        out = out.assign_coords(lat=out["lat"].values)
    if lon_rev_match and not lon_match:
        print("Obs longitude is reversed relative to model — flipping lon dimension.")
        out = out.isel(lon=slice(None, None, -1))
        out = out.assign_coords(lon=out["lon"].values)
    return out


# Make sure sizes match first
if (obs.sizes["lat"] != rz.sizes["lat"]) or (obs.sizes["lon"] != rz.sizes["lon"]):
    raise RuntimeError("Obs and model have different spatial sizes. Regridding step needed before plotting.")

# Auto-flip if necessary to match model orientation
obs = _maybe_flip_to_match(obs, rz)
print("Final obs/model grid orientation aligned.")

# ------------------------
# Choose random pixels (randomization over all grid cells, then keep valid ones)
# ------------------------
ntime_o, nlat, nlon = obs.shape
ntime_m = rz.shape[0]
min_pairs = int(args.min_pairs)

print(f"\n[PAIR-COUNT DEBUG] ntime_obs = {ntime_o}, ntime_model = {ntime_m}")
print(f"[PAIR-COUNT DEBUG] min_pairs threshold = {min_pairs}")

rng = np.random.default_rng(seed)

# Build time indexers: map each obs time to nearest model time within tolerance (days)
obs_t = obs["time"].values.astype("datetime64[ns]")
mod_t = rz["time"].values.astype("datetime64[ns]")
mod_t_int = mod_t.astype("int64")  # ns
obs_t_int = obs_t.astype("int64")

# tolerance: 16 days to catch month-ends vs month-middles
tol_ns = np.int64(16) * np.int64(24 * 3600 * 1_000_000_000)
print(f"[PAIR-COUNT DEBUG] Time match tolerance = 16 days -> {tol_ns} ns")

import numpy as _np

order = _np.argsort(mod_t_int)
mod_sorted = mod_t_int[order]
idx_left = _np.searchsorted(mod_sorted, obs_t_int, side="left")
nearest_mod_idx_sorted = _np.clip(idx_left, 0, mod_sorted.size - 1)
left_minus = _np.maximum(nearest_mod_idx_sorted - 1, 0)
choose_left = _np.abs(obs_t_int - mod_sorted[left_minus]) <= _np.abs(
    obs_t_int - mod_sorted[nearest_mod_idx_sorted]
)
nearest_mod_idx_sorted = _np.where(choose_left, left_minus, nearest_mod_idx_sorted)
nearest_mod_time = mod_sorted[nearest_mod_idx_sorted]
within_tol = _np.abs(nearest_mod_time - obs_t_int) <= tol_ns

# Map back to original model indices
nearest_mod_idx = order[nearest_mod_idx_sorted]

print(f"[PAIR-COUNT DEBUG] obs timesteps with a model neighbour within tolerance: {within_tol.sum()} / {ntime_o}")

pairs_mask = _np.zeros((nlat, nlon), dtype=bool)
max_pairs_overall = 0  # DEBUG: track max overlapping pairs in any pixel

print("Scanning grid for valid pixels (nearest-time matching with 16-day tolerance)...")

# Pre-extract numpy arrays for speed
obs_np = obs.values  # [To, lat, lon]
rz_np = rz.values  # [Tm, lat, lon]

# For each pixel, count number of obs samples that have finite obs AND finite model at nearest time
for j in range(nlat):
    o_j = obs_np[:, j, :]  # [To, nlon]
    m_j = rz_np[:, j, :]  # [Tm, nlon]

    ok_obs = within_tol
    m_match = m_j[nearest_mod_idx, :]  # [To, nlon]

    counts = _np.sum(_np.isfinite(o_j) & _np.isfinite(m_match) & ok_obs[:, None], axis=0)

    # DEBUG: track max pairs for this latitude row
    row_max = counts.max()
    if row_max > max_pairs_overall:
        max_pairs_overall = row_max

    pairs_mask[j, :] = counts >= min_pairs

print(f"[PAIR-COUNT DEBUG] Max overlapping obs–model pairs in any pixel = {max_pairs_overall}")
print(f"[PAIR-COUNT DEBUG] (Reminder: current min_pairs = {min_pairs})")

# Count how many valid pixels exist at all
valid_js, valid_is = _np.where(pairs_mask)
n_valid = valid_js.size
print(f"[PAIR-COUNT DEBUG] Number of pixels meeting min_pairs criterion = {n_valid}")

if n_valid == 0:
    raise RuntimeError(
        "No pixels with sufficient overlapping data. "
        f"Max pairs in any pixel = {max_pairs_overall}, min_pairs = {min_pairs}. "
        "Try decreasing --min_pairs and/or checking the time tolerance & period."
    )

if n_valid < nsamples:
    raise RuntimeError(
        f"Requested nsamples={nsamples}, but only {n_valid} pixels meet the "
        f"min_pairs={min_pairs} criterion. Reduce --nsamples or --min_pairs."
    )

print(f"Found {n_valid} valid pixels (meeting min_pairs={min_pairs}).")

# ---- NEW PART: randomization over *all* grid cells, then select first nsamples valid ones ----
n_total = nlat * nlon
flat_order = rng.permutation(n_total)  # random order over *all* grid cells, independent of validity

sel_js = []
sel_is = []

for idx in flat_order:
    j = idx // nlon
    i = idx % nlon
    # Only keep if pixel is valid
    if pairs_mask[j, i]:
        sel_js.append(j)
        sel_is.append(i)
        if len(sel_js) == nsamples:
            break

# Sanity check (should always be true given n_valid >= nsamples)
if len(sel_js) < nsamples:
    raise RuntimeError(
        "Internal error: could not collect the requested number of valid samples "
        "even though enough valid pixels were detected."
    )

sel_js = _np.array(sel_js, dtype=int)
sel_is = _np.array(sel_is, dtype=int)
n_pick = len(sel_js)  # should equal nsamples

print(
    f"Selected {n_pick} valid pixels for plotting, "
    "using a random order defined over the full grid domain."
)

# ------------------------
# Output folder
# ------------------------
out_dir = os.path.join(args.fig_folder, f"timeseries_{args.rz_layers}")
os.makedirs(out_dir, exist_ok=True)
print("Output directory:", out_dir)

# ------------------------
# Plot
# ------------------------

time_obs = obs["time"].values
time_mod = rz["time"].values

for k, (j, i) in enumerate(zip(sel_js, sel_is), start=1):
    lat_pt = float(obs["lat"].values[j])
    lon_pt = float(obs["lon"].values[i])

    x = obs.values[:, j, i]  # SMAP monthly CDF-matched obs
    y = rz.values[:, j, i]  # LIS root-zone (layer-aggregated, often daily)

    fig, ax = plt.subplots(figsize=(10, 4.8), constrained_layout=True)
    ax.plot(time_mod, y, label="Model RZMC (layer-avg)", linewidth=1.2)
    ax.plot(
        time_obs,
        x,
        label="SMAP obs (monthly CDF-matched)",
        linewidth=1.4,
        linestyle="None",
        marker="o",
        markersize=3,
    )

    ax.set_title(
        f"RZMC model vs SMAP obs @ lat {lat_pt:.2f}, lon {lon_pt:.2f}  |  layers={rz_vals}"
    )
    ax.set_xlabel("Time")
    ax.set_ylabel("Soil moisture (m³ m⁻³)")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", frameon=False)

    fname = f"ts_lat{lat_pt:.2f}_lon{lon_pt:.2f}.png".replace("+", "")
    out_path = os.path.join(out_dir, fname)
    print(out_path)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)

    print(f"[{k:02d}/{n_pick:02d}] -> {out_path}")

print("Done.")

