#!/usr/bin/env python3

import os
import json
import pickle
import argparse
from datetime import datetime
from collections import OrderedDict, defaultdict

import numpy as np
import xarray as xr
import torch
import psutil


# ============================================================
# PATHS
# ============================================================

BASE_PATH="/data/jayanth_works/era5_inputfluxes/1990-2025"

MASK_FILE="ocean_mask.npz"

INDEX_CACHE="era5_index_cache.pkl"
SEASON_CACHE="season_index_cache.pkl"
SPATIAL_CACHE="spatial_tiles_cache.pkl"

CHECKPOINT_FILE="sampler_checkpoint.json"


# ============================================================
# CONSTANTS
# ============================================================

R_D=287.05
SECONDS_PER_HOUR=3600

DATASET_CACHE_LIMIT=6
SPATIAL_TILE=10

INPUT_SCALE=np.array([20,20,320,320,105000,1.5,310],dtype=np.float32)
TARGET_SCALE=np.array([500,500],dtype=np.float32)

SEASONS={
"winter":[12,1,2],
"spring":[3,4,5],
"summer":[6,7,8],
"autumn":[9,10,11]
}

FLUX_BINS=np.array([0,50,150,300,1e9])


# ============================================================
# LOGGING
# ============================================================

def now():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def sysinfo():

    mem=psutil.virtual_memory()

    return {
        "cpu_percent":psutil.cpu_percent(),
        "ram_used_gb":round(mem.used/1e9,2),
        "ram_total_gb":round(mem.total/1e9,2)
    }


def log(msg):

    stats=sysinfo()

    line=f"[{now()}] {msg} | CPU {stats['cpu_percent']}% | RAM {stats['ram_used_gb']}/{stats['ram_total_gb']} GB"

    print(line,flush=True)


# ============================================================
# OCEAN MASK
# ============================================================

def load_ocean_mask():

    mask=np.load(MASK_FILE)

    ys=mask["ys"]
    xs=mask["xs"]
    lat=mask["lat"]
    lon=mask["lon"]

    log(f"Ocean grid cells: {len(ys)}")

    return ys,xs,lat,lon


# ============================================================
# SPATIAL TILE CACHE
# ============================================================

def load_or_build_spatial_tiles(lat,lon):

    if os.path.exists(SPATIAL_CACHE):

        log("Loading spatial tile cache")

        with open(SPATIAL_CACHE,"rb") as f:
            return pickle.load(f)

    log("Building spatial tiles")

    tiles=defaultdict(list)

    for i,(la,lo) in enumerate(zip(lat,lon)):

        lat_tile=int((la+90)//SPATIAL_TILE)
        lon_tile=int((lo+180)//SPATIAL_TILE)

        tiles[(lat_tile,lon_tile)].append(i)

    tiles=dict(tiles)

    with open(SPATIAL_CACHE,"wb") as f:
        pickle.dump(tiles,f)

    log(f"Built {len(tiles)} spatial tiles")

    return tiles


# ============================================================
# DATASET INDEX
# ============================================================

def build_index(start_year,end_year):

    if os.path.exists(INDEX_CACHE):

        log("Loading dataset index cache")

        with open(INDEX_CACHE,"rb") as f:
            return pickle.load(f)

    log("Building dataset index")

    index={}

    for year in range(start_year,end_year+1):

        ypath=os.path.join(BASE_PATH,str(year))

        if not os.path.exists(ypath):
            continue

        for month in os.listdir(ypath):

            mpath=os.path.join(ypath,month)

            inst=accum=param=None

            for f in os.listdir(mpath):

                fp=os.path.join(mpath,f)

                if "u10" in f and "t2m" in f:
                    inst=fp

                elif "sshf" in f:
                    accum=fp

                elif "d2m" in f:
                    param=fp

            if inst and accum and param:

                index[(year,int(month))]={
                    "instant":inst,
                    "accum":accum,
                    "param":param
                }

    with open(INDEX_CACHE,"wb") as f:
        pickle.dump(index,f)

    log(f"Indexed {len(index)} months")

    return index


# ============================================================
# SEASON INDEX
# ============================================================

def load_or_build_season_index(index):

    if os.path.exists(SEASON_CACHE):

        log("Loading season index cache")

        with open(SEASON_CACHE,"rb") as f:
            return pickle.load(f)

    log("Building season index")

    season_index={k:[] for k in SEASONS}

    for (year,month) in index:

        for s,months in SEASONS.items():

            if month in months:
                season_index[s].append((year,month))

    with open(SEASON_CACHE,"wb") as f:
        pickle.dump(season_index,f)

    return season_index


# ============================================================
# DATASET CACHE
# ============================================================

dataset_cache=OrderedDict()

def load_dataset(path):

    if path in dataset_cache:

        dataset_cache.move_to_end(path)
        return dataset_cache[path]

    ds=xr.open_dataset(path)

    dataset_cache[path]=ds

    if len(dataset_cache)>DATASET_CACHE_LIMIT:

        _,v=dataset_cache.popitem(last=False)
        v.close()

    return ds


# ============================================================
# SAMPLE EXTRACTION
# ============================================================

def sample_month(paths,ys,xs,grid_idx,time_idx):

    ds_i=load_dataset(paths["instant"])
    ds_a=load_dataset(paths["accum"])
    ds_p=load_dataset(paths["param"])

    tdim="valid_time" if "valid_time" in ds_i.dims else "time"

    yi=ys[grid_idx]
    xi=xs[grid_idx]

    time_vals=ds_i[tdim].values

    u10=ds_i["u10"].isel({tdim:time_idx}).values[:,yi,xi]
    v10=ds_i["v10"].isel({tdim:time_idx}).values[:,yi,xi]
    t2m=ds_i["t2m"].isel({tdim:time_idx}).values[:,yi,xi]
    sst=ds_i["sst"].isel({tdim:time_idx}).values[:,yi,xi]

    d2m=ds_p["d2m"].isel({tdim:time_idx}).values[:,yi,xi]
    sp=ds_p["sp"].isel({tdim:time_idx}).values[:,yi,xi]

    sshf=ds_a["sshf"].isel({tdim:time_idx}).values[:,yi,xi]/SECONDS_PER_HOUR
    slhf=ds_a["slhf"].isel({tdim:time_idx}).values[:,yi,xi]/SECONDS_PER_HOUR

    rho=sp/(R_D*t2m)

    X=np.stack([u10,v10,t2m,d2m,sp,rho,sst],axis=1)
    Y=np.stack([sshf,slhf],axis=1)

    flux=np.abs(Y[:,0])+np.abs(Y[:,1])

    times=time_vals[time_idx]

    return X,Y,flux,times


# ============================================================
# SAMPLER ENGINE
# ============================================================

def run(args):

    rng=np.random.default_rng(args.seed)

    log("Sampler started")

    ys,xs,lat,lon=load_ocean_mask()

    GRID_SIZE=len(ys)

    tiles=load_or_build_spatial_tiles(lat,lon)

    tile_list=list(tiles.keys())

    sizes=np.array([len(tiles[t]) for t in tile_list])
    tile_weights=sizes/sizes.sum()

    index=build_index(args.start_year,args.end_year)

    season_index=load_or_build_season_index(index)

    keys=list(index.keys())

    year_index=defaultdict(list)

    for (y,m) in keys:
        year_index[y].append((y,m))

    year_list=list(year_index.keys())

    start_batch=0

    if os.path.exists(CHECKPOINT_FILE):

        with open(CHECKPOINT_FILE) as f:
            start_batch=json.load(f)["last_batch"]+1

    for b in range(start_batch,args.batches):

        log(f"Batch {b+1}/{args.batches}")

        Xs=[]
        Ys=[]
        lat_all=[]
        lon_all=[]
        time_all=[]

        seen=set()

        generated=0
        accepted=0

        current=0

        while current < args.batch_size:

            # ------------------------------------------------
            # SAMPLER STRATEGY
            # ------------------------------------------------

            if args.sampler=="temporal":

                year=rng.choice(year_list)
                key=rng.choice(year_index[year])
                grid_pool=np.arange(GRID_SIZE)

            elif args.sampler=="seasonal":

                season=rng.choice(list(SEASONS.keys()))
                key=rng.choice(season_index[season])
                grid_pool=np.arange(GRID_SIZE)

            elif args.sampler=="spatial":

                tile=rng.choice(tile_list,p=tile_weights)
                grid_pool=tiles[tile]
                key=rng.choice(keys)

            elif args.sampler=="hybrid":

                season=rng.choice(list(SEASONS.keys()))
                key=rng.choice(season_index[season])

                tile=rng.choice(tile_list,p=tile_weights)
                grid_pool=tiles[tile]

            else:

                key=rng.choice(keys)
                grid_pool=np.arange(GRID_SIZE)

            remaining=args.batch_size-current

            k=max(500,int(remaining*1.3))

            grid_idx=rng.choice(grid_pool,k,replace=len(grid_pool)<k)

            ds_i=load_dataset(index[key]["instant"])

            tdim="valid_time" if "valid_time" in ds_i.dims else "time"

            T=ds_i.sizes[tdim]

            time_idx=rng.integers(0,T,k)

            # ------------------------------------------------
            # DUPLICATE FILTER
            # ------------------------------------------------

            pairs=set()

            unique_time=[]
            unique_grid=[]

            for t,g in zip(time_idx,grid_idx):

                h=t*GRID_SIZE+g

                if h not in seen:

                    seen.add(h)

                    unique_time.append(t)
                    unique_grid.append(g)

            if not unique_time:
                continue

            time_idx=np.array(unique_time)
            grid_idx=np.array(unique_grid)

            generated+=len(grid_idx)

            X,Y,flux,times=sample_month(index[key],ys,xs,grid_idx,time_idx)

            if args.sampler in ("flux","hybrid"):

                target=rng.integers(0,len(FLUX_BINS)-1)

                mask=(flux>=FLUX_BINS[target])&(flux<FLUX_BINS[target+1])

                X=X[mask]
                Y=Y[mask]
                grid_idx=grid_idx[mask]
                time_idx=time_idx[mask]
                times=times[mask]

            if len(X)==0:
                continue

            accepted+=len(X)

            Xs.append(X)
            Ys.append(Y)

            lat_all.append(lat[grid_idx])
            lon_all.append(lon[grid_idx])
            time_all.append(times)

            current+=len(X)

        # ------------------------------------------------
        # FINALIZE BATCH
        # ------------------------------------------------

        X=np.concatenate(Xs)[:args.batch_size]
        Y=np.concatenate(Ys)[:args.batch_size]

        lat_batch=np.concatenate(lat_all)[:args.batch_size]
        lon_batch=np.concatenate(lon_all)[:args.batch_size]
        time_batch=np.concatenate(time_all)[:args.batch_size]

        X=X.astype(np.float32)/INPUT_SCALE
        Y=Y.astype(np.float32)/TARGET_SCALE

        X=torch.from_numpy(X)
        Y=torch.from_numpy(Y)

        eff=accepted/max(generated,1)

        log(f"accepted={accepted} rejected={generated-accepted} efficiency={eff:.3f}")

        sys=sysinfo()

        log(f"Batch ready | CPU {sys['cpu_percent']}% RAM {sys['ram_used_gb']}/{sys['ram_total_gb']} GB")

        with open(CHECKPOINT_FILE,"w") as f:
            json.dump({"last_batch":b},f)

        yield {
            "X":X,
            "Y":Y,
            "lat":lat_batch,
            "lon":lon_batch,
            "time":time_batch,
            "batch":b,
            "sampler":args.sampler,
            "generated":generated,
            "accepted":accepted,
            "efficiency":eff
        }


# ============================================================
# CLI
# ============================================================

def main():

    parser=argparse.ArgumentParser()

    parser.add_argument("--sampler",
        choices=["random","temporal","seasonal","spatial","flux","hybrid"],
        default="random")

    parser.add_argument("--batch_size",type=int,default=15000)
    parser.add_argument("--batches",type=int,default=10)
    parser.add_argument("--start_year",type=int,default=1990)
    parser.add_argument("--end_year",type=int,default=2025)
    parser.add_argument("--seed",type=int,default=42)

    args=parser.parse_args()

    engine=run(args)

    for _ in engine:
        pass


if __name__=="__main__":
    main()
