"""
ERA5 PINN Dataset Builder
--------------------------

Purpose:
    Construct a clean, physically consistent, merged dataset for
    Physics-Informed Neural Network (PINN) training.

Inputs:
    - ERA5 oper instant (u10, v10, t2m, sst)
    - ERA5 oper accum (sshf, slhf)
    - ERA5 add_parameters instant (sp, d2m)

Outputs:
    - Single compressed NetCDF:
        Variables:
            u10, v10, t2m, sst, d2m, sp, rho,
            sshf (W/m^2), slhf (W/m^2)

Scientific Notes:
    - Accumulated flux (J/m^2) converted to W/m^2
    - Air density computed via ideal gas law
    - expver explicitly handled
    - No regridding performed (grid consistency preserved)

Author: [Your Name]
"""

import xarray as xr
import numpy as np
import os

# ==============================================================
# CONFIGURATION
# ==============================================================

BASE_PATH = "nrsc/jayanth_works/era5_inputfluxes"

OPER_INSTANT_PATH = f"{BASE_PATH}/1990_2025/*/*/data_stream-oper_stepType-instant.nc"
OPER_ACCUM_PATH   = f"{BASE_PATH}/1990_2025/*/*/data_stream-oper_stepType-accum.nc"
PARAM_INSTANT_PATH = f"{BASE_PATH}/1990-2025_add_parameters/*/*/data_stream-oper_stepType-instant.nc"

OUTPUT_FILE = "era5_pinn_dataset_1990_2025.nc"

# Physical constants
R_D = 287.05  # Gas constant for dry air [J kg^-1 K^-1]
SECONDS_PER_HOUR = 3600.0


# ==============================================================
# UTILITY FUNCTIONS
# ==============================================================

def load_era5_dataset(path, variables):
    """
    Load ERA5 multi-file dataset lazily using xarray + dask.

    Parameters
    ----------
    path : str
        Glob path to NetCDF files.
    variables : list
        Variables to retain from dataset.

    Returns
    -------
    xarray.Dataset
        Cleaned dataset with selected variables.
    """

    ds = xr.open_mfdataset(
        path,
        combine="by_coords",
        parallel=True,
        chunks={"valid_time": 100}
    )

    # Handle expver dimension explicitly
    if "expver" in ds:
        ds = ds.sel(expver="0001")

    # Rename valid_time to time (CF consistency)
    ds = ds.rename({"valid_time": "time"})

    # Select required variables only (memory optimization)
    ds = ds[variables]

    return ds


def convert_flux_to_wm2(ds):
    """
    Convert accumulated flux (J/m^2) to W/m^2.

    ERA5 accum flux = energy accumulated over previous hour.
    """

    ds["sshf"] = ds["sshf"] / SECONDS_PER_HOUR
    ds["slhf"] = ds["slhf"] / SECONDS_PER_HOUR

    ds["sshf"].attrs["units"] = "W m^-2"
    ds["slhf"].attrs["units"] = "W m^-2"

    return ds


def compute_air_density(sp, t2m):
    """
    Compute air density using ideal gas law:

        rho = p / (R_d * T)

    Parameters
    ----------
    sp : DataArray
        Surface pressure [Pa]
    t2m : DataArray
        Air temperature at 2m [K]

    Returns
    -------
    DataArray
        Air density [kg/m^3]
    """

    rho = sp / (R_D * t2m)
    rho = rho.rename("rho")
    rho.attrs["units"] = "kg m^-3"
    rho.attrs["description"] = "Air density computed via ideal gas law"

    return rho


def perform_sanity_checks(ds):
    """
    Perform essential dataset sanity checks.
    """

    print("\n========== SANITY CHECK ==========")
    print("Time range:",
          str(ds.time.min().values),
          "→",
          str(ds.time.max().values))

    print("Total timesteps:", len(ds.time))

    # Basic NaN check
    print("\nMissing values per variable:")
    for var in ds.data_vars:
        print(var, int(ds[var].isnull().sum()))

    # Flux magnitude sanity
    print("\nMean sshf (W/m^2):",
          float(ds["sshf"].mean().compute()))
    print("Mean slhf (W/m^2):",
          float(ds["slhf"].mean().compute()))

    print("==================================\n")


# ==============================================================
# MAIN PIPELINE
# ==============================================================

def main():

    print("Loading ERA5 oper instant...")
    ds_oper = load_era5_dataset(
        OPER_INSTANT_PATH,
        ["u10", "v10", "t2m", "sst"]
    )

    print("Loading ERA5 oper accum...")
    ds_accum = load_era5_dataset(
        OPER_ACCUM_PATH,
        ["sshf", "slhf"]
    )

    print("Loading ERA5 additional parameters...")
    ds_param = load_era5_dataset(
        PARAM_INSTANT_PATH,
        ["sp", "d2m"]
    )

    print("Converting accumulated flux to W/m^2...")
    ds_accum = convert_flux_to_wm2(ds_accum)

    print("Computing air density...")
    rho = compute_air_density(ds_param["sp"], ds_oper["t2m"])

    print("Merging datasets...")
    ds_final = xr.merge([ds_oper, ds_param, ds_accum, rho])

    ds_final = ds_final.sortby("time")

    perform_sanity_checks(ds_final)

    print("Saving compressed NetCDF...")
    encoding = {var: {"zlib": True, "complevel": 4}
                for var in ds_final.data_vars}

    ds_final.to_netcdf(OUTPUT_FILE, encoding=encoding)

    print("Dataset successfully created:", OUTPUT_FILE)


if __name__ == "__main__":
    main()
