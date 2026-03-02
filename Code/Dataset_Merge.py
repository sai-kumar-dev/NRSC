#!/usr/bin/env python3
"""
ERA5 PINN Production Pipeline
----------------------------------------------------
Stable single-node pipeline
- Chronological month sorting
- Duplicate timestamp removal
- Flux conversion (J/m^2 -> W/m^2)
- Air density computation
- Structural validation
- Explicit rechunking
- Zarr v2 backend (stable)
- Resume-safe checkpoints
"""

import os
import glob
import numpy as np
import xarray as xr
import dask

# =============================
# DASK CONFIG (Single Node)
# =============================

dask.config.set(scheduler="threads")
dask.config.set({"array.slicing.split_large_chunks": True})

# =============================
# CONFIG
# =============================

BASE_PATH = "nrsc/jayanth_works/era5_inputfluxes"
OUTPUT_ZARR = "/data/era5_pinn_dataset.zarr"
CHECKPOINT_DIR = "/data/era5_checkpoints"

YEARS = [1990]  # <<< change to range(1990, 2026) after successful test

R_D = 287.05
SECONDS_PER_HOUR = 3600.0

MONTH_ORDER = {
    "january": 1,
    "february": 2,
    "march": 3,
    "april": 4,
    "may": 5,
    "june": 6,
    "july": 7,
    "august": 8,
    "september": 9,
    "october": 10,
    "november": 11,
    "december": 12,
}

# =============================
# UTILITIES
# =============================

def sort_by_month(files):
    def extract_month(f):
        folder = os.path.basename(os.path.dirname(f))
        month_name = folder.split("_")[0].lower()
        return MONTH_ORDER.get(month_name, 0)
    return sorted(files, key=extract_month)


def checkpoint_exists(year):
    return os.path.exists(f"{CHECKPOINT_DIR}/{year}.done")


def write_checkpoint(year):
    open(f"{CHECKPOINT_DIR}/{year}.done", "w").close()


def open_year_dataset(year, stream, variables):
    path = f"{BASE_PATH}/{stream}/{year}/*/data_stream-oper_stepType-*.nc"
    files = glob.glob(path)

    if not files:
        raise RuntimeError(f"No files found for {year} {stream}")

    files = sort_by_month(files)

    ds = xr.open_mfdataset(
        files,
        combine="nested",
        concat_dim="valid_time",
        parallel=False,
        chunks={"valid_time": 24}
    )

    # Handle expver safely
    if "expver" in ds.dims:
        ds = ds.sel(expver="0001")
    elif "expver" in ds.variables:
        ds = ds.drop_vars("expver")

    ds = ds.rename({"valid_time": "time"})

    return ds[variables]


def convert_flux(ds):
    ds["sshf"] = ds["sshf"] / SECONDS_PER_HOUR
    ds["slhf"] = ds["slhf"] / SECONDS_PER_HOUR
    ds["sshf"].attrs["units"] = "W m^-2"
    ds["slhf"].attrs["units"] = "W m^-2"
    return ds


def compute_density(sp, t2m):
    rho = sp / (R_D * t2m)
    rho = rho.rename("rho")
    rho.attrs["units"] = "kg m^-3"
    return rho


def structural_checks(ds):
    if not ds.time.to_index().is_monotonic_increasing:
        raise RuntimeError("Time is not monotonic increasing")

    if len(ds.time) != len(np.unique(ds.time)):
        raise RuntimeError("Duplicate timestamps detected")


# =============================
# MAIN PIPELINE
# =============================

def main():

    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    for year in YEARS:

        print("\n==============================")
        print(f"Processing year {year}")
        print("==============================")

        if checkpoint_exists(year):
            print("Checkpoint exists. Skipping.")
            continue

        print("Loading instant variables...")
        ds_oper = open_year_dataset(
            year,
            "1990-2025",
            ["u10", "v10", "t2m", "sst"]
        )

        print("Loading accumulated flux...")
        ds_accum = open_year_dataset(
            year,
            "1990-2025",
            ["sshf", "slhf"]
        )

        print("Loading additional parameters...")
        ds_param = open_year_dataset(
            year,
            "1990-2025_add_parameters",
            ["sp", "d2m"]
        )

        print("Converting flux to W/m^2...")
        ds_accum = convert_flux(ds_accum)

        print("Computing air density...")
        rho = compute_density(ds_param["sp"], ds_oper["t2m"])

        print("Merging datasets...")
        ds = xr.merge([ds_oper, ds_param, ds_accum, rho])

        # Ensure chronological order
        ds = ds.sortby("time")

        # Remove duplicate timestamps safely
        _, unique_index = np.unique(ds["time"], return_index=True)
        ds = ds.isel(time=np.sort(unique_index))

        print("Running structural checks...")
        structural_checks(ds)

        # Explicit rechunk for Zarr stability
        print("Rechunking dataset...")
        ds = ds.chunk({
            "time": 24,
            "latitude": 100,
            "longitude": 100
        })

        print("Writing to Zarr (v2)...")

        if not os.path.exists(OUTPUT_ZARR):
            ds.to_zarr(
                OUTPUT_ZARR,
                mode="w",
                consolidated=True,
                zarr_version=2
            )
        else:
            ds.to_zarr(
                OUTPUT_ZARR,
                mode="a",
                append_dim="time",
                consolidated=True,
                zarr_version=2
            )

        write_checkpoint(year)

        print(f"Year {year} complete.")

    print("\nAll requested years processed successfully.")


if __name__ == "__main__":
    main()
