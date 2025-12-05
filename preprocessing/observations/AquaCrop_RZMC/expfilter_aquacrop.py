#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
End-to-end pipeline to generate exponentially filtered root zone soil moisture from
satellite surface soil moisture in DAOBS and AquaCrop soil moisture in SURFACEMODEL

- Optimize tau per pixel 
- During optimization, construct the *final* filtered + CDF-matched time series
  for the optimal tau and store it in a 3D array in memory:
      rzmc_all(time, lat, lon)
- After tau optimization and saving R_max_and_tau_opt_*.nc, PASS1 writes
  out rzmc_all[t, :, :] per day into NetCDF files.

Gap / stabilization policy:

  - For gaps **shorter than tau** (in days, per pixel):
      * the filter uses the Wagner-form EMA and **holds y across missing inputs** 
        (i.e., the state is persisted during the gap).
      * no retroactive masking is applied; these “short gaps” behave like in the
        simpler “hold y on NaNs” approach.

  - For gaps **longer than tau**:
      * the entire gap (from its start) is retroactively forced to NaN.
      * when valid data resumes after such a long gap, we apply a stabilization
        window whose length depends on tau (as before).
      * this keeps the original “long gap” policy but with a threshold tied to
        tau instead of a fixed number of days.

  - After filtering, all days with missing SMAP input are forced to NaN, i.e.
    time resolution is not artificially increased by the filtering.

The EMA uses the Wagner et al. (1999) SWI formulation:

  w = exp(-Δt / τ)
  y_t = w * y_{t-1} + (1 - w) * x_t

Inputs:
  - --exp_folder        : root of the LIS/SHUI experiment (contains /output)
  - --lis_input_file    : LIS input file defining grid/landmask
  - --expfilter_folder  : folder where R_max_and_tau_opt_*.nc is written
  - --outdir_timeseries : folder for writing the resulting timeseris dataset
  - --start_date / --end_date (DD/MM/YYYY)
  - --rz_layers         : root-zone selection (e.g., 1-3, 1-6, 1-10, all)
  - --depth_cm          : target depth for metadata
  - --model             : tag in daily filenames
  - --moving_window     : moving-window length (days) for CDF matching (default 30)
  - --outdir_timeseries : subdirectory for daily NetCDF files

CDF policy:
  - climatological moving window:
    * window length = --moving_window (days, default 30)
    * window step   = 10 days in day-of-year space
    * windows are circular in day-of-year, so the beginning and end of the year
      are covered symmetrically.

Example:
python /data/leuven/317/vsc31786/python/zdenko/expfilter_aquacrop.py \
  --exp_folder /staging/leuven/stg_00024/OUTPUT/michelb/LIS/testcases_michel/ac72_rzmc_da/OL_SMAP_L2 \
  --expfilter_folder /staging/leuven/stg_00024/OUTPUT/michelb/LIS/testcases_michel/ac72_rzmc_da/EXP_FILTER \
  --lis_input_file /staging/leuven/stg_00024/OUTPUT/michelb/LIS/testcases_michel/ac72_rzmc_da/LDT_FILE_GENERATION/lis_input.d01.nc \
  --outdir_timeseries SMAP_RZMC_30CM_AC72_MONTHLY \
  --rz_layers 1-6 \
  --depth_cm 60 \
  --moving_window 30 \
  --start_date 01/04/2015 \
  --end_date 31/12/2022
"""

import os
import sys
import argparse
from datetime import datetime

import numpy as np
import xarray as xr
import pandas as pd
from joblib import Parallel, delayed
from tqdm import tqdm
# Project libs
sys.path.append("/data/leuven/317/vsc31786/python/zdenko/scripts")
from pylis_zdenko import readers


# Ensure immediate logging in batch
os.environ.setdefault("PYTHONUNBUFFERED", "1")
try:
    sys.stdout.reconfigure(line_buffering=True)
    sys.stderr.reconfigure(line_buffering=True)
except Exception:
    pass

# Keep threaded math libs from oversubscribing CPUs (can override in SLURM)
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")


# ====================================================
# Generic helpers
# ====================================================

def _sanitize_n_jobs(n_jobs: int) -> int:
    if n_jobs == 0:
        return 1
    return n_jobs


def parse_date(dstr):
    return datetime.strptime(dstr, "%d/%m/%Y")


def reindex_to_full_time(da, time_index):
    """Reindex a DataArray `da` to a full `time_index`, inserting NaNs where missing."""
    return da.reindex(time=time_index)


def pearsonr_nan(a, b):
    mask = np.isfinite(a) & np.isfinite(b)
    if mask.sum() < 3:
        return np.nan
    aa = a[mask]
    bb = b[mask]
    va = np.var(aa)
    vb = np.var(bb)
    if va == 0.0 or vb == 0.0:
        return np.nan
    aa = aa - aa.mean()
    bb = bb - bb.mean()
    return float(np.dot(aa, bb) / np.sqrt(np.dot(aa, aa) * np.dot(bb, bb)))


def bias_correct_linear(x, ref):
    """Linear mean/variance bias correction of x to match ref."""
    x = x.copy()
    mask = np.isfinite(x) & np.isfinite(ref)
    if mask.sum() < 3:
        return x
    mx, mr = np.mean(x[mask]), np.mean(ref[mask])
    sx, sr = np.std(x[mask]), np.std(ref[mask])
    if sx > 0 and sr > 0:
        x[~np.isnan(x)] = (x[~np.isnan(x)] - mx) * (sr / sx) + mr
    else:
        x[~np.isnan(x)] = (x[~np.isnan(x)] - mx) + mr
    return x


def _flatten_land_index(landmask):
    idx = -np.ones_like(landmask, dtype=np.int32)
    flat = np.where(landmask.flatten())[0]
    idx_flat = -np.ones(landmask.size, dtype=np.int32)
    idx_flat[flat] = np.arange(flat.size, dtype=np.int32)
    return idx_flat.reshape(landmask.shape), flat.size


def _interp_to_percentile(x, xr_src, cdf_src):
    ok = np.where(np.isfinite(xr_src) & np.isfinite(cdf_src))[0]
    if ok.size < 2:
        return np.nan
    return float(np.interp(x, xr_src[ok], cdf_src[ok], left=0.0, right=1.0))


def _inv_cdf(p, xr_ref, cdf_ref):
    if np.isnan(p):
        return np.nan
    ok = np.where(np.isfinite(cdf_ref) & np.isfinite(xr_ref))[0]
    if ok.size < 2:
        return np.nan
    return float(np.interp(p, cdf_ref[ok], xr_ref[ok],
                           left=xr_ref[ok][0], right=xr_ref[ok][-1]))


def _parse_rz_layers(spec, layer_coords):
    lc = np.array(layer_coords, dtype=float)
    if isinstance(spec, (list, tuple, np.ndarray)):
        want_vals = np.array(spec, dtype=float)
    else:
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
        want_vals = np.array(
            [float(x) for x in s.split(",") if x.strip() != ""], dtype=float
        )
    mask = np.isin(lc, want_vals)
    return list(np.unique(np.sort(lc[mask])))


# ====================================================
# Gap / stabilization policy + EMA
# ====================================================

def get_stabilize_days(tau_days, min_days=5, max_days=60):
    """
    Map tau (days) to a stabilization window length (days).

    - For scalar tau: returns an int.
    - For array tau: returns an int array of the same shape.
    """
    tau_arr = np.asarray(tau_days, dtype=float)
    stab = np.zeros_like(tau_arr, dtype=np.int32)
    valid = np.isfinite(tau_arr) & (tau_arr > 0)
    if np.any(valid):
        # here we simply tie the window to tau itself (1 * tau), with bounds
        stab_val = np.round(tau_arr[valid]).astype(int)
        stab_val = np.clip(stab_val, min_days, max_days)
        stab[valid] = stab_val
    return stab


def exp_filter_ema_step(prev_y, x_now, dt_days, tau_days):
    """
    Single-step EMA update using the Wagner SWI form:

        w = exp(-Δt / τ)
        y_t = w * y_{t-1} + (1 - w) * x_t
    """
    if not np.isfinite(tau_days) or tau_days <= 0:
        return prev_y
    dt = max(1.0, float(dt_days))
    if np.isnan(prev_y):
        return x_now if np.isfinite(x_now) else np.nan
    if not np.isfinite(x_now):
        return prev_y
    w = np.exp(-dt / max(tau_days, 1e-6))
    return w * prev_y + (1.0 - w) * x_now

def exp_filter_ema_with_gap_1d(x, time_vals, tau_days):
    """
    1D version of EMA + gap/stabilization policy, used during tau optimization
    and for constructing 1D filtered series with the same rules.

    EMA uses Wagner SWI form:
        w = exp(-Δt / τ), y_t = w*y_{t-1} + (1-w)*x_t

    Gap policy (tau-dependent):

      - For gaps **shorter than tau_days**:
          * the state is simply held (y_t = y_{t-1}) across missing x,
            i.e., "hold y for short gaps".

      - For gaps **longer than tau_days**:
          * the entire gap is retroactively set to NaN.
          * a tau-dependent stabilization window is applied after the gap.

    Stabilization window length (after first valid and after long gaps)
    is tied to tau of this series via get_stabilize_days(tau_days).
    """
    ntime = x.shape[0]
    y_store = np.full(ntime, np.nan, dtype=np.float32)

    y_prev = np.nan
    gap_run = 0
    gap_start_idx = -1
    stabilize_left = 0
    has_seen_valid = False

    # Tau-dependent gap threshold (in days, integer, at least 1)
    if not np.isfinite(tau_days) or tau_days <= 0:
        gap_threshold = 1
    else:
        gap_threshold = max(1, int(round(float(tau_days))))

    stab_len_base = int(get_stabilize_days(tau_days))  # scalar int

    for t in range(ntime):
        if t == 0:
            dt_days = 1.0
        else:
            dt_days = (
                np.datetime64(time_vals[t], "D") - np.datetime64(time_vals[t - 1], "D")
            ).astype(int)
            dt_days = float(dt_days) if dt_days >= 1 else 1.0

        x_now = x[t]
        is_valid_x = np.isfinite(x_now)
        is_missing_x = not is_valid_x

        # Missing input → update gap tracker
        if is_missing_x:
            gap_run += 1
            if gap_run == 1:
                gap_start_idx = t
        else:
            # If we were in a gap and it was long (> gap_threshold), start stabilization
            if gap_run > gap_threshold:
                stabilize_left = stab_len_base
            gap_run = 0
            gap_start_idx = -1

        # EMA update using Wagner form
        if np.isfinite(tau_days) and tau_days > 0 and is_valid_x:
            dt = max(1.0, float(dt_days))
            if np.isnan(y_prev):
                y_now = x_now
            else:
                w = np.exp(-dt / max(tau_days, 1e-6))
                y_now = w * y_prev + (1.0 - w) * x_now
        else:
            # no valid tau or missing input -> carry previous state ("hold y")
            y_now = y_prev

        # First-ever valid
        if is_valid_x and not has_seen_valid:
            has_seen_valid = True
            stabilize_left = stab_len_base

        # Provisional write
        y_store[t] = y_now

        # Long gap retroactive NaNs:
        # if we are still in the gap, and its length has exceeded gap_threshold,
        # wipe the entire gap back to its start.
        if gap_run >= (gap_threshold + 1) and gap_start_idx >= 0:
            y_store[gap_start_idx : (t + 1)] = np.nan

        # Stabilization NaNs
        if stabilize_left > 0:
            y_store[t] = np.nan
            stabilize_left -= 1

        y_prev = y_now

    # Enforce original SMAP missingness
    missing = ~np.isfinite(x)
    y_store[missing] = np.nan

    return y_store


# ====================================================
# Moving-window CDF helpers (unchanged)
# ====================================================

MOVING_WINDOW_STEP_DAYS = 10  # fixed step asked by user
YEAR_LEN_FOR_DOY = 366       # circular length for day-of-year


def _build_moving_window_cdfs_1d(
    data_1d,
    doys,
    centers,
    obs_thresh,
    nbins,
    exclude_mask=None,
):
    """
    (Unused placeholder; kept for completeness.)
    """
    data_1d = np.asarray(data_1d, dtype=float)
    doys = np.asarray(doys, dtype=int)
    ntime = data_1d.size
    nwin = centers.size

    xr = np.full((nwin, 1, nbins), np.nan, dtype=np.float32)
    cdf = np.full((nwin, 1, nbins), np.nan, dtype=np.float32)

    valid = np.isfinite(data_1d)
    if exclude_mask is not None:
        valid = valid & (~exclude_mask)

    if not np.any(valid):
        return {"xr": xr, "cdf": cdf}

    # real implementation handled in wrapper
    return {"xr": xr, "cdf": cdf}


def build_moving_window_cdfs_1d(
    data_1d,
    doys,
    centers,
    half_window_days,
    obs_thresh,
    nbins,
    exclude_mask=None,
):
    """
    Build *climatological* moving-window CDFs for a 1D series.
    """
    data_1d = np.asarray(data_1d, dtype=float)
    doys = np.asarray(doys, dtype=int)
    nwin = centers.size

    xr = np.full((nwin, 1, nbins), np.nan, dtype=np.float32)
    cdf = np.full((nwin, 1, nbins), np.nan, dtype=np.float32)

    valid = np.isfinite(data_1d)
    if exclude_mask is not None:
        valid = valid & (~exclude_mask)

    if not np.any(valid):
        return {"xr": xr, "cdf": cdf}

    q = np.linspace(0.0, 1.0, nbins, dtype=np.float64)

    for iw, c in enumerate(centers):
        c = int(c)
        diff = np.abs(doys - c)
        diff_circ = np.minimum(diff, YEAR_LEN_FOR_DOY - diff)
        sel = valid & (diff_circ <= half_window_days)

        vals = data_1d[sel]
        if vals.size < obs_thresh:
            continue

        vals_sorted = np.sort(vals.astype(np.float64))
        xr_i = np.quantile(vals_sorted, q)

        xr[iw, 0, :] = xr_i.astype(np.float32)
        cdf[iw, 0, :] = q.astype(np.float32)

    return {"xr": xr, "cdf": cdf}


def build_window_index_for_time(doys, centers):
    """
    For each time index, find the index of the closest window center in circular DOY space.
    """
    doys = np.asarray(doys, dtype=int)
    centers = np.asarray(centers, dtype=int)
    diff = np.abs(doys[:, None] - centers[None, :])
    diff_circ = np.minimum(diff, YEAR_LEN_FOR_DOY - diff)
    return np.argmin(diff_circ, axis=1).astype(np.int32)


def _apply_qmap_series_mw(x_series, win_idx_for_time, gp, src_cdfs, ref_cdfs):
    """
    Moving-window quantile mapping for a 1D series and per-pixel CDFs.
    """
    x_series = np.asarray(x_series, dtype=float)
    ntime = x_series.size
    y = np.full(ntime, np.nan, dtype=np.float32)

    for k, val in enumerate(x_series):
        if not np.isfinite(val):
            continue
        w = int(win_idx_for_time[k])
        xr_s = src_cdfs["xr"][w, gp, :]
        cf_s = src_cdfs["cdf"][w, gp, :]
        xr_r = ref_cdfs["xr"][w, gp, :]
        cf_r = ref_cdfs["cdf"][w, gp, :]
        p = _interp_to_percentile(val, xr_s, cf_s)
        y[k] = _inv_cdf(p, xr_r, cf_r)

    return y


# ====================================================
# Daily write-out helpers
# ====================================================

def make_out_name(base_dir, model, date64):
    dstr = np.datetime_as_string(date64, unit="D").replace("-", "")  # YYYYMMDD
    return os.path.join(base_dir, f"SMAP_RZMC_{model}_{dstr}.nc")


def write_daily_nc(path, lat1d, lon1d, time_val, sm2d, compress_level=4):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    ds = xr.Dataset(
        data_vars=dict(
            sm=(("time", "lat", "lon"), sm2d[np.newaxis, :, :].astype("float32")),
        ),
        coords=dict(
            time=[np.datetime64(time_val, "ns")],
            lat=("lat", lat1d.astype("float32")),
            lon=("lon", lon1d.astype("float32")),
        ),
    )
    ds["sm"].attrs.update(
        {
            "long_name": "Volumetric Soil Moisture",
            "units": "m3 m-3",
            "valid_range": np.array([0.0, 1.0], dtype=np.float32),
            "_CoordinateAxes": "time lat lon",
        }
    )
    ds["lat"].attrs.update(
        {
            "units": "degrees_north",
            "_CoordinateAxisType": "Lat",
            "standard_name": "latitude",
            "valid_range": np.array([-90.0, 90.0], dtype=np.float32),
        }
    )
    ds["lon"].attrs.update(
        {
            "units": "degrees_east",
            "_CoordinateAxisType": "Lon",
            "standard_name": "longitude",
            "valid_range": np.array([-180.0, 180.0], dtype=np.float32),
        }
    )
    ds["time"].encoding.update(
        {
            "units": "days since 1970-01-01 00:00:00 UTC",
            "calendar": "standard",
        }
    )
    for k in ("units", "calendar"):
        ds["time"].attrs.pop(k, None)

    enc = {
        "sm": dict(
            zlib=True,
            complevel=int(compress_level),
            dtype="float32",
            _FillValue=np.float32(-9999.0),
        )
    }
    ds.to_netcdf(path, format="NETCDF4", encoding=enc)


# ====================================================
# CLI
# ====================================================

def parse_args():
    p = argparse.ArgumentParser(
        description="Optimize tau (using tau-dependent gap/stab rules and Wagner EMA) and write daily CDF-matched SMAP RZMC NetCDFs (moving-window CDF)"
    )
    p.add_argument(
        "--exp_folder",
        required=True,
        help=(
            "Path to LIS/SHUI experiment folder (root), "
            "e.g. /staging/.../SHUI2/EU_SMAP_OL_cal_0.05_scaling"
        ),
    )
    p.add_argument(
        "--lis_input_file",
        required=True,
        help="Full path to LIS input file (defines grid/landmask).",
    )
    p.add_argument(
        "--expfilter_folder",
        default=None,
        help=(
            "Folder where R_max_and_tau_opt_*.nc will be written. "
            "If not provided, it is derived from lis_input_file as "
            ".../LIS/SHUI2/expfilter_files."
        ),
    )
    p.add_argument(
        "--rz_layers",
        default="1-6",
        help=(
            "Root-zone layer selection by coordinate values. Examples: '1-6', "
            "'1,2,3,4,5', or 'all'. Surface layer value=1 is always included. "
            "If the effective RZ selection is exactly [1], tau is forced to 1e-6 "
            "without optimization."
        ),
    )
    p.add_argument(
        "--depth_cm",
        type=int,
        default=60,
        help="Target depth for metadata (e.g., 60 -> 60CM).",
    )
    p.add_argument(
        "--model",
        default="AC72",
        help="Model tag to encode in daily file names (default: AC72)",
    )
    p.add_argument(
        "--outdir_timeseries",
        required=True,
        help="Output directory for daily NetCDF files",
    )
    p.add_argument(
        "--moving_window",
        type=int,
        default=30,
        help="Length of moving window in days for climatological CDF matching (default: 30).",
    )
    p.add_argument(
        "--compress",
        type=int,
        default=4,
        help="NetCDF compression level (0-9)",
    )
    p.add_argument(
        "--start_date",
        default="01/04/2015",
        help="Start date DD/MM/YYYY (aligns with LIS/SMAP)",
    )
    p.add_argument(
        "--end_date",
        default="31/12/2023",
        help="End date DD/MM/YYYY",
    )
    p.add_argument(
        "--nbins",
        type=int,
        default=100,
        help="Number of bins for CDF estimation",
    )
    p.add_argument(
        "--obs_thresh",
        type=int,
        default=100,
        help="Min #obs per cell for CDF estimation",
    )
    p.add_argument(
        "--n_jobs",
        type=int,
        default=-1,
        help="Parallel jobs for tau sweep & CDF ( -1 = all CPUs )",
    )
    p.add_argument(
        "--progress_every",
        type=int,
        default=50,
        help="Print progress every N rows / days",
    )
    return p.parse_args()


# ====================================================
# Main
# ====================================================

def main():
    
    import os
    
    args = parse_args()
    os.makedirs(args.expfilter_folder, exist_ok=True)

    exp_folder = args.exp_folder
    lis_input_file = args.lis_input_file

    # expfilter_folder now controlled via CLI (with fallback to old behaviour)
    expfilter_folder = args.expfilter_folder
    os.makedirs(expfilter_folder, exist_ok=True)
    os.makedirs(os.path.join(expfilter_folder, args.outdir_timeseries), exist_ok=True)

    print("====================================================")
    print("Tau optimization + daily RZMC writer (gap/stab-consistent, Wagner EMA)")
    print("====================================================")
    print(f"Experiment folder : {exp_folder}")
    print(f"LIS input file    : {lis_input_file}")
    print(f"expfilter_folder  : {expfilter_folder}")
    print(f"RZ layers spec    : {args.rz_layers}")
    print(f"Depth (cm)        : {args.depth_cm}")
    print(f"Moden name        : {args.model}")
    print(f"Moving window size: {args.moving_window}")
    print(f"Outdir timeseries : {args.outdir_timeseries}")
    print("----------------------------------------------------")

    # ------------------------
    # Static grid / landmask
    # ------------------------
    print("[GRID] Loading LIS grid & land mask...")
    lc09_ds = xr.open_dataset(lis_input_file)

    print("[GRID] Deriving land flag from LIS input...")
    land_flag_da = readers.landflag(lis_input_file)
    if isinstance(land_flag_da, xr.DataArray):
        landmask = land_flag_da.values.astype(bool)
    else:
        landmask = np.asarray(land_flag_da).astype(bool)

    # ----------------------------
    # Time axis
    # ----------------------------
    print("[TIME] Building daily time index...")
    date_list = pd.date_range(
        start=parse_date(args.start_date),
        end=parse_date(args.end_date),
        freq="1D",
    )
    time_index = pd.DatetimeIndex(date_list)

    # ----------------------------
    # Load SMAP obs
    # ----------------------------
    print("[LOAD] Loading SMAP obs cube from original data...")

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

    # Now call obs_cube with the chosen directory
    dc_obs = readers.obs_cube(
        output_dir,
        lis_input_file,
        start=args.start_date,
        end=args.end_date,
        rescaled=False,
        subfolder="DAOBS",
        a="01",
        d="01",
        freq="1D",
    )

    dc_obs = reindex_to_full_time(dc_obs, time_index)
    obs = dc_obs  # (time, lat, lon)

    lats_range = np.round(obs["lat"].values[:, 0], 3)
    lons_range = np.round(obs["lon"].values[0, :], 3)

    # ----------------------------
    # Load LIS soil moisture cube
    # ----------------------------
    print("[LOAD] Loading LIS soil moisture cube from original output...")

    dc_ol = readers.lis_cube(
        output_dir,
        var="SoilMoist_tavg",
        lats=lats_range,
        lons=lons_range,
        start=args.start_date,
        end=args.end_date,
        h=0,
    )
    sm_da = dc_ol  # expect dims ('time','layer','lat','lon')
    if "layer" not in sm_da.dims:
        raise RuntimeError("Model soil moisture DataArray must contain a 'layer' dimension.")

    layer_coords = sm_da["layer"].values

    # Surface layer = 1 (hard-coded)
    surface_val = 1.0
    print(f"[LAYERS] Surface layer coordinate value = {surface_val} (hard-coded)")

    # Parse root-zone layers and ensure surface is included
    rz_vals = _parse_rz_layers(args.rz_layers, layer_coords)
    if surface_val not in rz_vals:
        rz_vals = sorted(list(np.unique(np.append(rz_vals, surface_val))))

    print(f"[LAYERS] RZ layer spec        : {args.rz_layers}")
    print(f"[LAYERS] Effective RZ layers  : {rz_vals}")

    lis_surface = sm_da.sel(layer=surface_val)
    rz = sm_da.sel(layer=rz_vals).mean("layer")

    print("[ALIGN] Aligning obs, LIS surface, and RZ on common time/space...")
    obs, lis_surface, rz = xr.align(obs, lis_surface, rz, join="inner")

    # ------------------------
    # Prepare arrays
    # ------------------------
    default_taus = np.array(
        [2, 4, 6, 8, 10, 12, 15, 18, 22, 26, 30, 35, 40, 46, 54, 65],
        dtype=float,
    )

    ntime, nlat, nlon = obs.shape
    print(f"[DATA] Shapes: time={ntime}, lat={nlat}, lon={nlon}")

    obs_np_raw = obs.values.astype(np.float32)        # original
    lis_surf_np = lis_surface.values.astype(np.float32)
    rz_np = rz.values.astype(np.float32)
    time_vals = obs["time"].values

    valid_land = landmask if landmask.shape == (nlat, nlon) else np.ones((nlat, nlon), dtype=bool)
    land_gpi, n_land = _flatten_land_index(valid_land)
    print(f"[LAND] Number of land pixels: {n_land}")

    # ------------------------
    # Bias correction: SMAP(surface) -> LIS(surface)
    # ------------------------
    print("[BIAS] Applying linear pre-filter bias correction...")
    obs_bc_np = np.full_like(obs_np_raw, np.nan, dtype=np.float32)

    def _bias_correct_row(j):
        row_bc = np.full((ntime, nlon), np.nan, dtype=np.float32)
        for i in range(nlon):
            if not valid_land[j, i]:
                continue
            x = obs_np_raw[:, j, i]
            r = lis_surf_np[:, j, i]
            if np.isfinite(x).sum() < 3 or np.isfinite(r).sum() < 3:
                row_bc[:, i] = x
                continue
            row_bc[:, i] = bias_correct_linear(x, r)
        return j, row_bc

    bc_results = Parallel(
        n_jobs=_sanitize_n_jobs(args.n_jobs),
        backend="loky",
        verbose=10,
    )(delayed(_bias_correct_row)(j) for j in range(nlat))

    for j, row_bc in tqdm(bc_results, total=nlat, desc="[BIAS] Rows", leave=True):
        obs_bc_np[:, j, :] = row_bc

    print("[BIAS] Done.")

    # Time / DOY vectors for CDF mapping
    time_da = obs["time"]
    doys_all = time_da.dt.dayofyear.values.astype(int)

    # Moving-window configuration
    moving_window_days = int(args.moving_window)
    half_window_days = moving_window_days / 2.0
    centers = np.arange(1, YEAR_LEN_FOR_DOY + 1, MOVING_WINDOW_STEP_DAYS, dtype=int)
    nwin = centers.size
    print(f"[CDF] Moving-window CDF: window={moving_window_days} days (half={half_window_days}), "
          f"step={MOVING_WINDOW_STEP_DAYS} days, nwin={nwin}")

    # For each time index, which window index should be used?
    win_idx_for_time = build_window_index_for_time(doys_all, centers)

    # ------------------------
    # Tau behavior: normal optimization vs forced tiny tau
    # ------------------------
    no_opt_tau_mode = (len(rz_vals) == 1 and np.isclose(rz_vals[0], 1.0))
    print(f"[TAU] No-optimization mode (rz_layers==[1])? {no_opt_tau_mode}")

    Score_max = np.full((nlat, nlon), np.nan, dtype=np.float32)
    Tau_opt   = np.full((nlat, nlon), np.nan, dtype=np.float32)

    # Final filtered + CDF-matched series (time, lat, lon)
    rzmc_all = np.full((ntime, nlat, nlon), np.nan, dtype=np.float32)

    if no_opt_tau_mode:
        # SPECIAL CASE: tau forced to tiny value everywhere
        print("[TAU] Enforcing tau=1e-6 for all land pixels (no optimization).")
        Tau_opt[:, :] = np.nan
        Tau_opt[valid_land] = 1e-6
        taus_used = np.array([1e-6], dtype=float)

        # First, compute R_max as correlation of bias-corrected SMAP vs RZ
        def _corr_row(j):
            row_R = np.full(nlon, np.nan, dtype=np.float32)
            for i in range(nlon):
                if not valid_land[j, i]:
                    continue
                x = obs_bc_np[:, j, i]
                y = rz_np[:, j, i]
                if np.isfinite(x).sum() < 10 or np.isfinite(y).sum() < 10:
                    continue
                row_R[i] = pearsonr_nan(x, y)
            return j, row_R

        corr_results = Parallel(
            n_jobs=_sanitize_n_jobs(args.n_jobs),
            backend="loky",
            verbose=10,
        )(delayed(_corr_row)(j) for j in range(nlat))

        for j, row_R in tqdm(corr_results, total=nlat, desc="[TAU] R (no-opt) rows", leave=True):
            Score_max[j, :] = row_R

        print("[TAU] Building final filtered + CDF-matched series for no-opt mode (tau=1e-6)...")

        def _build_rzmc_row_noopt(j):
            rzmc_row = np.full((ntime, nlon), np.nan, dtype=np.float32)

            for i in range(nlon):
                if not valid_land[j, i]:
                    continue

                x = obs_bc_np[:, j, i]
                y = rz_np[:, j, i]

                if np.isfinite(x).sum() < 10 or np.isfinite(y).sum() < 10:
                    continue

                valid_x = np.isfinite(x)

                # Reference CDFs (pixel only, cross-masked on SMAP availability) - moving window
                ref_pix_cdfs = build_moving_window_cdfs_1d(
                    y,
                    doys_all,
                    centers,
                    half_window_days,
                    obs_thresh=args.obs_thresh,
                    nbins=args.nbins,
                    exclude_mask=~valid_x,
                )

                tau = 1e-6
                xf = exp_filter_ema_with_gap_1d(x, time_vals, tau)

                src_cdfs = build_moving_window_cdfs_1d(
                    xf,
                    doys_all,
                    centers,
                    half_window_days,
                    obs_thresh=args.obs_thresh,
                    nbins=args.nbins,
                    exclude_mask=None,
                )
                src_pix = {
                    "xr": src_cdfs["xr"][:, 0:1, :],
                    "cdf": src_cdfs["cdf"][:, 0:1, :],
                }
                ref_pix = {
                    "xr": ref_pix_cdfs["xr"][:, 0:1, :],
                    "cdf": ref_pix_cdfs["cdf"][:, 0:1, :],
                }

                xf_qm = _apply_qmap_series_mw(
                    xf,
                    win_idx_for_time,
                    gp=0,
                    src_cdfs=src_pix,
                    ref_cdfs=ref_pix,
                )

                # Enforce original SMAP missingness explicitly
                xf_qm[~valid_x] = np.nan

                rzmc_row[:, i] = xf_qm.astype(np.float32)

            return j, rzmc_row

        rzmc_results_noopt = Parallel(
            n_jobs=_sanitize_n_jobs(args.n_jobs),
            backend="loky",
            verbose=10,
        )(delayed(_build_rzmc_row_noopt)(j) for j in range(nlat))

        for j, rzmc_row in tqdm(rzmc_results_noopt, total=nlat, desc="[TAU] RZMC (no-opt) rows", leave=True):
            rzmc_all[:, j, :] = rzmc_row

    else:
        # NORMAL MODE: tau optimization with EMA + tau-dependent gap/stabilization and moving-window post-CDF
        taus = default_taus.copy()
        taus_used = taus
        print("[TAU] Sweeping tau values and optimizing for Pearson R with moving-window CDF (using Wagner EMA + tau-dependent gap/stab)...")

        def _process_row(j):
            score_row = np.full(nlon, np.nan, dtype=np.float32)
            tau_row   = np.full(nlon, np.nan, dtype=np.float32)
            rzmc_row  = np.full((ntime, nlon), np.nan, dtype=np.float32)

            for i in range(nlon):
                if not valid_land[j, i]:
                    continue

                x = obs_bc_np[:, j, i]
                y = rz_np[:, j, i]

                if np.isfinite(x).sum() < 10 or np.isfinite(y).sum() < 10:
                    continue

                valid_x = np.isfinite(x)

                # Reference CDFs (pixel only, cross-masked on SMAP availability) - moving window
                ref_pix_cdfs = build_moving_window_cdfs_1d(
                    y,
                    doys_all,
                    centers,
                    half_window_days,
                    obs_thresh=args.obs_thresh,
                    nbins=args.nbins,
                    exclude_mask=~valid_x,
                )

                best_score = -np.inf
                best_tau   = np.nan
                best_series_qm = None
                gp_dummy = 0

                for tau in taus:
                    # Filter with EMA + tau-dependent gap/stabilization 1D (Wagner EMA)
                    xf = exp_filter_ema_with_gap_1d(x, time_vals, tau)

                    # Post-CDF (moving window, per-pixel)
                    src_cdfs = build_moving_window_cdfs_1d(
                        xf,
                        doys_all,
                        centers,
                        half_window_days,
                        obs_thresh=args.obs_thresh,
                        nbins=args.nbins,
                        exclude_mask=None,
                    )
                    src_pix = {
                        "xr": src_cdfs["xr"][:, 0:1, :],
                        "cdf": src_cdfs["cdf"][:, 0:1, :],
                    }
                    ref_pix = {
                        "xr": ref_pix_cdfs["xr"][:, 0:1, :],
                        "cdf": ref_pix_cdfs["cdf"][:, 0:1, :],
                    }
                    xf_qm = _apply_qmap_series_mw(
                        xf,
                        win_idx_for_time,
                        gp=gp_dummy,
                        src_cdfs=src_pix,
                        ref_cdfs=ref_pix,
                    )

                    # Enforce original SMAP missingness explicitly
                    xf_qm[~valid_x] = np.nan

                    score = pearsonr_nan(xf_qm, y)
                    if np.isnan(score):
                        continue
                    if score > best_score:
                        best_score = score
                        best_tau = tau
                        best_series_qm = xf_qm

                score_row[i] = np.nan if best_score == -np.inf else best_score
                tau_row[i]   = best_tau

                if best_series_qm is not None:
                    rzmc_row[:, i] = best_series_qm.astype(np.float32)

            return j, score_row, tau_row, rzmc_row

        tau_results = Parallel(
            n_jobs=_sanitize_n_jobs(args.n_jobs),
            backend="loky",
            verbose=10,
        )(delayed(_process_row)(j) for j in range(nlat))

        for j, score_row, tau_row, rzmc_row in tqdm(tau_results, total=nlat, desc="[TAU] Rows", leave=True):
            Score_max[j, :]   = score_row
            Tau_opt[j, :]     = tau_row
            rzmc_all[:, j, :] = rzmc_row

        print("[TAU] Sweep complete.")

    # ------------------------
    # Save R_max & tau_opt NetCDF
    # ------------------------
    print("[TAU] Saving NetCDF of R_max and tau_opt_days...")
    lats_da = xr.DataArray(lats_range, dims=("lat",))
    lons_da = xr.DataArray(lons_range, dims=("lon",))
    lat2d, lon2d = xr.broadcast(lats_da, lons_da)
    rz_layer_vals = np.array(rz_vals, dtype=float)

    out_ds_tau = xr.Dataset(
        data_vars={
            "R_max": (("x", "y"), Score_max),
            "tau_opt_days": (("x", "y"), Tau_opt),
            "rz_layers": (("rz_layer",), rz_layer_vals),
        },
        coords={
            "x": np.arange(Score_max.shape[0]),
            "y": np.arange(Score_max.shape[1]),
            "lat": (("x", "y"), lat2d.values),
            "lon": (("x", "y"), lon2d.values),
            "rz_layer": np.arange(rz_layer_vals.size, dtype=int),
        },
        attrs={
            "title": "Max Pearson R and optimal exponential filter timescale",
            "description": (
                "SMAP(surface) linearly bias-corrected to LIS(surface), exponentially "
                "filtered with per-pixel tau using a tau-dependent gap/stabilization "
                "policy consistent with the daily product (short gaps < tau use hold-y; "
                "long gaps > tau are fully masked with a tau-dependent stabilization "
                "window), moving-window CDF-matched to LIS(RZ) using "
                f"a window of length {moving_window_days} days in day-of-year "
                f"space (step {MOVING_WINDOW_STEP_DAYS} days), then scored via Pearson "
                "R vs LIS RZ. Root-zone selection always includes surface layer=1. "
                "If RZ is exactly layer=1, tau is forced to 1e-6 without optimization."
            ),
            "time_range": f"{args.start_date} to {args.end_date}",
            "metric": "R",
            "bias_correction": "linear",
            "postfilter_cdf": "moving_window",
            "cdf_nbins": args.nbins,
            "cdf_obs_thresh": args.obs_thresh,
            "cdf_moving_window_days": moving_window_days,
            "cdf_moving_window_step_days": MOVING_WINDOW_STEP_DAYS,
            "taus_days": ",".join(map(str, taus_used.tolist())),
            "n_jobs": args.n_jobs,
            "surface_layer": "1",
            "rz_layers_note": "See variable 'rz_layers(rz_layer)' for numeric layer coords.",
            "depth_cm": args.depth_cm,
        },
    )

    nc_tau_path = os.path.join(
        expfilter_folder, f"R_max_and_tau_opt_{args.rz_layers}_REVISED_tau_gap_holdY.nc"
    )
    out_ds_tau.to_netcdf(nc_tau_path)
    print(f"[TAU] -> Saved: {nc_tau_path}")

    # ====================================================
    # PASS1: write out the in-memory rzmc_all, with SMAP cross-mask
    # ====================================================
    print("[DAILY][PASS1] Writing daily NetCDFs from in-memory filtered + CDF-matched array (SMAP-masked)...")

    def write_one_day(t):
        # Cross-mask with original SMAP: keep only timesteps where SMAP exists
        valid_x_2d = np.isfinite(obs_np_raw[t, :, :])
        if not valid_x_2d.any():
            # No SMAP anywhere on this day -> skip writing this file
            return t
        sm2d = rzmc_all[t, :, :].copy()
        sm2d[~valid_x_2d] = np.nan

        out_path = make_out_name(
            os.path.join(args.expfilter_folder, args.outdir_timeseries),
            args.model,
            time_vals[t],
        )
        write_daily_nc(out_path, lats_range, lons_range, time_vals[t], sm2d, compress_level=args.compress)
        return t

    _ = Parallel(
        n_jobs=_sanitize_n_jobs(args.n_jobs),
        backend="loky",
        verbose=10,
    )(delayed(write_one_day)(t) for t in range(ntime))

    print("[DONE] Tau optimization + daily RZMC NetCDFs written from in-memory 3D array (SMAP-masked, tau-dependent gap policy).")


if __name__ == "__main__":
    main()

