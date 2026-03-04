#(era5_pinn) [nrsc@localhost test_aft]$ cat sampler.py
#!/usr/bin/env python3

"""
ERA5 PINN Sampler Engine
Research-grade streaming sampler with EDA + training integration
"""

import os
import argparse
import time
import json
import random
import psutil
import numpy as np
import xarray as xr
import pandas as pd
import torch

from datetime import datetime
from eda_stream import StreamEDA


# ------------------------------------------------
# CONFIG
# ------------------------------------------------

BASE_PATH = "/data/jayanth_works/era5_inputfluxes/1990-2025"
MASK_FILE = "ocean_mask.npz"

R_D = 287.05
SECONDS_PER_HOUR = 3600

INPUT_VARS = ["u10","v10","t2m","d2m","sp","rho","sst"]
TARGET_VARS = ["sshf","slhf"]

INPUT_SCALE = np.array(
    [20,20,320,320,105000,1.5,310],
    dtype=np.float32
)

TARGET_SCALE = np.array(
    [500,500],
    dtype=np.float32
)

dataset_cache = {}


# ------------------------------------------------
# LOGGING
# ------------------------------------------------

def log(msg):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


def sysinfo(stage):
    mem = psutil.virtual_memory()
    cpu = psutil.cpu_percent()

    print(
        f"[{stage}] Memory {mem.used/1e9:.2f}GB / "
        f"{mem.total/1e9:.2f}GB | CPU {cpu:.1f}%"
    )


# ------------------------------------------------
# LOAD OCEAN MASK
# ------------------------------------------------

def load_ocean_mask():

    mask = np.load(MASK_FILE)

    ys = mask["ys"]
    xs = mask["xs"]
    lat = mask["lat"]
    lon = mask["lon"]

    log(f"[INIT] Loaded ocean mask with {len(ys)} ocean points")

    return ys, xs, lat, lon


# ------------------------------------------------
# DATASET INDEX
# ------------------------------------------------

def build_index(start_year, end_year):

    index = {}

    for year in range(start_year, end_year+1):

        year_path = os.path.join(BASE_PATH, str(year))

        if not os.path.exists(year_path):
            continue

        for month in os.listdir(year_path):

            path = os.path.join(year_path, month)

            inst = None
            accum = None
            param = None

            for f in os.listdir(path):

                if not f.endswith(".nc"):
                    continue

                if "u10" in f and "t2m" in f:
                    inst = os.path.join(path, f)

                elif "sshf" in f:
                    accum = os.path.join(path, f)

                elif "d2m" in f:
                    param = os.path.join(path, f)

            if inst and accum and param:

                index[(year,month)] = {
                    "instant": inst,
                    "accum": accum,
                    "param": param
                }

    log(f"Indexed {len(index)} months")

    return index


# ------------------------------------------------
# DATASET CACHE
# ------------------------------------------------

def load_dataset(path):

    if path not in dataset_cache:
        dataset_cache[path] = xr.open_dataset(path)

    return dataset_cache[path]


# ------------------------------------------------
# SAMPLING
# ------------------------------------------------

def sample_month(paths, n, rng, ys, xs):

    ds_i = load_dataset(paths["instant"])
    ds_a = load_dataset(paths["accum"])
    ds_p = load_dataset(paths["param"])

    tdim = "valid_time" if "valid_time" in ds_i.dims else "time"

    time_vals = ds_i[tdim].values
    lat_vals = ds_i["latitude"].values
    lon_vals = ds_i["longitude"].values

    T = ds_i.sizes[tdim]

    u10 = ds_i["u10"].values
    v10 = ds_i["v10"].values
    t2m = ds_i["t2m"].values
    sst = ds_i["sst"].values

    d2m = ds_p["d2m"].values
    sp  = ds_p["sp"].values

    sshf = ds_a["sshf"].values / SECONDS_PER_HOUR
    slhf = ds_a["slhf"].values / SECONDS_PER_HOUR

    rho = sp / (R_D * t2m)

    Xs=[]
    Ys=[]
    metas=[]

    while sum(len(x) for x in Xs) < n:

        k = 5000

        ti = rng.integers(0, T, k)
        idx = rng.integers(0, len(ys), k)

        yi = ys[idx]
        xi = xs[idx]

        X = np.stack([
            u10[ti,yi,xi],
            v10[ti,yi,xi],
            t2m[ti,yi,xi],
            d2m[ti,yi,xi],
            sp[ti,yi,xi],
            rho[ti,yi,xi],
            sst[ti,yi,xi]
        ], axis=1)

        Y = np.stack([
            sshf[ti,yi,xi],
            slhf[ti,yi,xi]
        ], axis=1)

        good = ~np.isnan(X).any(axis=1)
        good &= ~np.isnan(Y).any(axis=1)

        X = X[good]
        Y = Y[good]

        times = time_vals[ti][good]

        lat = lat_vals[yi][good]
        lon = lon_vals[xi][good]

        wind = np.sqrt(X[:,0]**2 + X[:,1]**2)
        flux_mag = np.abs(Y[:,0]) + np.abs(Y[:,1])

        meta = pd.DataFrame({
            "time":times,
            "lat":lat,
            "lon":lon,
            "month":pd.to_datetime(times).month,
            "year":pd.to_datetime(times).year,
            "wind":wind,
            "flux_mag":flux_mag
        })

        Xs.append(X)
        Ys.append(Y)
        metas.append(meta)

    X = np.concatenate(Xs)[:n]
    Y = np.concatenate(Ys)[:n]
    meta = pd.concat(metas).iloc[:n]

    return X, Y, meta


# ------------------------------------------------
# TRAIN PLACEHOLDER
# ------------------------------------------------

def train_step(batch):

    # placeholder
    pass


# ------------------------------------------------
# ENGINE
# ------------------------------------------------

def run(args):

    rng = np.random.default_rng(args.seed)

    ys, xs, lat_grid, lon_grid = load_ocean_mask()

    config = vars(args)

    eda = StreamEDA(config)

    log("Starting PINN Sampling Engine")
    sysinfo("startup")

    index = build_index(args.start_year, args.end_year)

    keys = list(index.keys())

    sysinfo("index built")

    for b in range(args.batches):

        log(f"----- Batch {b+1}/{args.batches} -----")

        t0 = time.time()

        Xs=[]
        Ys=[]
        metas=[]

        while sum(len(x) for x in Xs) < args.batch_size:

            key = random.choice(keys)

            X,Y,meta = sample_month(
                index[key],
                2000,
                rng,
                ys,
                xs
            )

            Xs.append(X)
            Ys.append(Y)
            metas.append(meta)

        X = np.concatenate(Xs)[:args.batch_size]
        Y = np.concatenate(Ys)[:args.batch_size]
        meta = pd.concat(metas).iloc[:args.batch_size]

        # normalization
        X = X.astype(np.float32) / INPUT_SCALE
        Y = Y.astype(np.float32) / TARGET_SCALE

        X = torch.from_numpy(X)
        Y = torch.from_numpy(Y)

        batch = {
            "X": X,
            "Y": Y,
            "meta": meta,
            "batch_id": b,
            "timestamp": datetime.now().isoformat()
        }

        log(f"Batch tensors {X.shape} {Y.shape}")
        log(f"Batch time {time.time()-t0:.2f}s")

        sysinfo("after batch")

        eda.observe(batch)

        train_step(batch)

        yield batch

    log("All batches processed")

    eda.finalize()

    log("Engine finished")


# ------------------------------------------------
# CLI
# ------------------------------------------------

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--sampler",
        choices=[
            "random",
            "seasonal",
            "yearly",
            "flux_stratified",
            "hybrid"
        ],
        default="random"
    )

    parser.add_argument("--batch_size", type=int, default=15000)
    parser.add_argument("--batches", type=int, default=10)
    parser.add_argument("--start_year", type=int, default=1990)
    parser.add_argument("--end_year", type=int, default=2025)
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    engine = run(args)

    for _ in engine:
        pass


if __name__ == "__main__":
    main()
