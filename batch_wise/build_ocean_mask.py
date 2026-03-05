#!/usr/bin/env python3

import os
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from datetime import datetime

# ------------------------------------------------
# CONFIG
# ------------------------------------------------

BASE_PATH = "/data/jayanth_works/era5_inputfluxes/1990-2025"
OUTPUT_FILE = "ocean_mask.npz"

# sample a few representative years
SAMPLE_YEARS = [1990, 2000, 2010, 2020]

# number of time steps per file to check
TIME_SCAN = 12

FIG_DIR = "mask_figures"


# ------------------------------------------------
# LOGGING
# ------------------------------------------------

def log(msg):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


# ------------------------------------------------
# FIND SST FILES
# ------------------------------------------------

def find_sst_files():

    files = []

    for year in SAMPLE_YEARS:

        ypath = os.path.join(BASE_PATH, str(year))

        if not os.path.exists(ypath):
            continue

        for month in sorted(os.listdir(ypath)):

            mpath = os.path.join(ypath, month)

            for f in os.listdir(mpath):

                if "sst" in f and f.endswith(".nc"):
                    files.append(os.path.join(mpath, f))
                    break

    return files


# ------------------------------------------------
# BUILD MASK
# ------------------------------------------------

def build_mask(files):

    ocean = None
    lat = None
    lon = None

    for f in files:

        log(f"Scanning {f}")

        ds = xr.open_dataset(f)

        sst = ds["sst"]

        if lat is None:
            lat = ds["latitude"].values
            lon = ds["longitude"].values

        T = min(TIME_SCAN, sst.shape[0])

        valid_any = None

        for t in range(T):

            valid = ~np.isnan(sst.isel(time=t).values)

            if valid_any is None:
                valid_any = valid
            else:
                valid_any |= valid

        if ocean is None:
            ocean = valid_any
        else:
            ocean |= valid_any

        ds.close()

    ys, xs = np.where(ocean)

    lat_vals = lat[ys]
    lon_vals = lon[xs]

    return ys, xs, lat_vals, lon_vals, ocean, lat, lon


# ------------------------------------------------
# SAVE MASK
# ------------------------------------------------

def save_mask(ys, xs, lat_vals, lon_vals, shape):

    log("Saving mask")

    np.savez_compressed(
        OUTPUT_FILE,
        ys=ys.astype(np.int32),
        xs=xs.astype(np.int32),
        lat=lat_vals.astype(np.float32),
        lon=lon_vals.astype(np.float32),
        grid_shape=shape
    )

    log(f"Ocean cells: {len(ys)}")
    log(f"Saved → {OUTPUT_FILE}")


# ------------------------------------------------
# VISUALIZATION
# ------------------------------------------------

def visualize(ocean, lat, lon, lat_vals, lon_vals):

    os.makedirs(FIG_DIR, exist_ok=True)

    log("Generating mask visualization")

    plt.figure(figsize=(10,5))

    plt.imshow(
        ocean,
        origin="lower",
        extent=[lon.min(), lon.max(), lat.min(), lat.max()],
        cmap="Blues"
    )

    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.title("Ocean Mask (Derived from SST)")

    plt.colorbar(label="Ocean=1")

    plt.savefig(os.path.join(FIG_DIR,"mask_grid.png"))
    plt.close()


    log("Generating ocean point scatter")

    plt.figure(figsize=(10,5))

    plt.scatter(lon_vals, lat_vals, s=1)

    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.title("Ocean Sampling Points")

    plt.savefig(os.path.join(FIG_DIR,"mask_points.png"))
    plt.close()


# ------------------------------------------------
# MAIN
# ------------------------------------------------

def main():

    log("Ocean mask creation started")

    files = find_sst_files()

    if not files:
        raise RuntimeError("No SST files found")

    log(f"Using {len(files)} SST files")

    ys, xs, lat_vals, lon_vals, ocean, lat, lon = build_mask(files)

    save_mask(ys, xs, lat_vals, lon_vals, ocean.shape)

    visualize(ocean, lat, lon, lat_vals, lon_vals)

    log("Ocean mask creation complete")


if __name__ == "__main__":
    main()
