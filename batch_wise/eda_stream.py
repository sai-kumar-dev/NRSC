#!/usr/bin/env python3

import os
import json
import psutil
import socket
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from datetime import datetime


# ============================================================
# CONFIG
# ============================================================

INPUT_SCALE=np.array([20,20,320,320,105000,1.5,310])
TARGET_SCALE=np.array([500,500])

VAR_NAMES=["u10","v10","t2m","d2m","sp","rho","sst"]
TARGET_NAMES=["sshf","slhf"]

EDA_RANDOM_SEED=42

LAT_BANDS={
"tropics":(0,20),
"midlat":(20,45),
"high_lat":(45,70),
"polar":(70,90)
}

plt.rcParams["figure.dpi"]=140
plt.rcParams["savefig.bbox"]="tight"


# ============================================================
# UTILITIES
# ============================================================

def now():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def system_stats():

    mem=psutil.virtual_memory()

    return {
        "cpu_percent":psutil.cpu_percent(),
        "ram_used_gb":round(mem.used/1e9,2),
        "ram_total_gb":round(mem.total/1e9,2)
    }


def ensure_dir(path):
    os.makedirs(path,exist_ok=True)


def create_run_dir(base="runs"):

    ensure_dir(base)

    ts=datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    pid=os.getpid()

    run_dir=os.path.join(base,f"run_{ts}_{pid}")

    ensure_dir(run_dir)
    ensure_dir(os.path.join(run_dir,"logs"))

    meta={
        "created":now(),
        "hostname":socket.gethostname(),
        "pid":pid
    }

    with open(os.path.join(run_dir,"run_metadata.json"),"w") as f:
        json.dump(meta,f,indent=2)

    return run_dir


def log(run_dir,msg):

    stats=system_stats()

    line=f"[{now()}] {msg} | CPU {stats['cpu_percent']}% | RAM {stats['ram_used_gb']}/{stats['ram_total_gb']} GB"

    print(line,flush=True)

    with open(os.path.join(run_dir,"logs","eda.log"),"a") as f:
        f.write(line+"\n")


def save_json(path,data):

    with open(path,"w") as f:
        json.dump(data,f,indent=2)


# ============================================================
# BATCH EDA
# ============================================================

class BatchEDA:


    def __init__(self,batch,run_dir):

        self.batch=batch
        self.run_dir=run_dir
        self.batch_id=batch.get("batch",-1)

        self.X=batch["X"].cpu().numpy()
        self.Y=batch["Y"].cpu().numpy()

        self.lat=batch.get("lat")
        self.lon=batch.get("lon")
        self.time=batch.get("time")

        self.X_raw=self.X*INPUT_SCALE
        self.Y_raw=self.Y*TARGET_SCALE

        self.wind=np.sqrt(self.X_raw[:,0]**2+self.X_raw[:,1]**2)
        self.wind_dir=np.arctan2(self.X_raw[:,1],self.X_raw[:,0])

        self.flux=np.abs(self.Y_raw[:,0])+np.abs(self.Y_raw[:,1])

        self.batch_dir=os.path.join(run_dir,f"batch_{self.batch_id:03d}")
        self.fig_dir=os.path.join(self.batch_dir,"figures")

        ensure_dir(self.batch_dir)
        ensure_dir(self.fig_dir)


    def save(self,name):

        plt.tight_layout()
        plt.savefig(os.path.join(self.fig_dir,name))
        plt.close()


    # --------------------------------------------------------
    # DATA CHECKS
    # --------------------------------------------------------

    def data_checks(self):

        nan_x=int(np.isnan(self.X_raw).sum())
        nan_y=int(np.isnan(self.Y_raw).sum())

        inf_x=int(np.isinf(self.X_raw).sum())
        inf_y=int(np.isinf(self.Y_raw).sum())

        return nan_x,nan_y,inf_x,inf_y


    def duplicate_check(self):

        if self.lat is None or self.lon is None or self.time is None:
            return 0

        arr=np.column_stack((self.lat,self.lon,self.time))

        return len(arr)-len(np.unique(arr,axis=0))


    # --------------------------------------------------------
    # LATITUDE BANDS
    # --------------------------------------------------------

    def latitude_bands(self):

        if self.lat is None:
            return

        lat_abs=np.abs(self.lat)

        counts=[]

        for lo,hi in LAT_BANDS.values():

            counts.append(np.sum((lat_abs>=lo)&(lat_abs<hi)))

        plt.figure()

        plt.bar(LAT_BANDS.keys(),counts)

        plt.ylabel("samples")

        self.save("latitude_bands.png")


    # --------------------------------------------------------
    # SPATIAL COVERAGE
    # --------------------------------------------------------

    def spatial_density(self):

        if self.lat is None:
            return

        plt.figure(figsize=(10,5))

        plt.hexbin(self.lon,self.lat,gridsize=60,
                   extent=[-180,180,-90,90],
                   cmap="viridis")

        plt.xlabel("Longitude")
        plt.ylabel("Latitude")
        plt.grid(alpha=0.3)

        plt.colorbar(label="samples")

        self.save("spatial_density.png")


    # --------------------------------------------------------
    # FLUX DISTRIBUTION
    # --------------------------------------------------------

    def flux_distribution(self):

        p99=np.percentile(self.flux,99)

        plt.figure()

        sns.histplot(self.flux,bins=80)

        plt.axvline(p99,color="red",label="p99")

        plt.yscale("log")

        plt.legend()

        plt.xlabel("Flux magnitude")

        self.save("flux_distribution.png")


    # --------------------------------------------------------
    # WIND DISTRIBUTIONS
    # --------------------------------------------------------

    def wind_distributions(self):

        plt.figure()

        sns.histplot(self.wind,bins=80)

        plt.xlabel("Wind speed")

        self.save("wind_speed.png")

        plt.figure()

        sns.histplot(self.wind_dir,bins=60)

        plt.xlabel("Wind direction")

        self.save("wind_direction.png")


    # --------------------------------------------------------
    # WIND FLUX RELATION
    # --------------------------------------------------------

    def wind_flux(self):

        rng=np.random.default_rng(EDA_RANDOM_SEED)

        idx=rng.choice(len(self.wind),min(5000,len(self.wind)),replace=False)

        plt.figure()

        plt.scatter(self.wind[idx],self.flux[idx],s=2,alpha=0.3)

        plt.xlabel("Wind speed")
        plt.ylabel("Flux magnitude")

        self.save("wind_flux.png")


    # --------------------------------------------------------
    # PHYSICS CHECK
    # --------------------------------------------------------

    def t2m_sst_check(self):

        diff=np.abs(self.X_raw[:,2]-self.X_raw[:,6])

        plt.figure()

        sns.histplot(diff,bins=60)

        plt.xlabel("|t2m - sst|")

        self.save("t2m_sst_difference.png")


    # --------------------------------------------------------
    # FLUX SIGN
    # --------------------------------------------------------

    def flux_sign(self):

        for i,name in enumerate(TARGET_NAMES):

            plt.figure()

            sns.histplot(self.Y_raw[:,i],bins=80)

            plt.xlabel(name)

            self.save(f"{name}_sign.png")


    # --------------------------------------------------------
    # STATS
    # --------------------------------------------------------

    def stats(self,nan_x,nan_y,inf_x,inf_y,dup):

        stats={}

        stats["batch"]=self.batch_id
        stats["samples"]=len(self.X)

        stats["mean_flux"]=float(np.mean(self.flux))
        stats["mean_wind"]=float(np.mean(self.wind))

        stats["nan_inputs"]=nan_x
        stats["nan_targets"]=nan_y
        stats["inf_inputs"]=inf_x
        stats["inf_targets"]=inf_y

        stats["duplicates"]=dup

        stats["system"]=system_stats()

        save_json(os.path.join(self.batch_dir,"stats.json"),stats)

        pd.DataFrame([stats]).to_csv(
            os.path.join(self.batch_dir,"stats.csv"),
            index=False
        )


    def run_all(self):

        log(self.run_dir,f"Batch {self.batch_id} EDA")

        nan_x,nan_y,inf_x,inf_y=self.data_checks()
        dup=self.duplicate_check()

        try:

            self.latitude_bands()
            self.spatial_density()
            self.flux_distribution()
            self.wind_distributions()
            self.wind_flux()
            self.flux_sign()
            self.t2m_sst_check()

        except Exception as e:

            log(self.run_dir,f"Plot error {e}")

        self.stats(nan_x,nan_y,inf_x,inf_y,dup)


# ============================================================
# GLOBAL EDA
# ============================================================

class GlobalEDA:


    def __init__(self,run_dir):

        self.run_dir=run_dir

        self.grid=np.zeros((180,360))

        self.flux=[]
        self.wind=[]
        self.lat=[]
        self.months=[]

        self.batch_flux_mean=[]

        self.samples=0


    def add_batch(self,lat,lon,flux,wind,time):

        self.samples+=len(flux)

        self.flux.extend(flux)
        self.wind.extend(wind)

        self.batch_flux_mean.append(np.mean(flux))

        if lat is not None:

            self.lat.extend(lat)

            lat_idx=np.clip((lat+90).astype(int),0,179)
            lon_idx=np.clip((lon+180).astype(int),0,359)

            for la,lo in zip(lat_idx,lon_idx):
                self.grid[la,lo]+=1

        if time is not None:

            self.months.extend(pd.to_datetime(time).month)


    def finalize(self):

        global_dir=os.path.join(self.run_dir,"global")
        fig_dir=os.path.join(global_dir,"figures")

        ensure_dir(global_dir)
        ensure_dir(fig_dir)

        flux=np.array(self.flux)
        wind=np.array(self.wind)
        lat=np.array(self.lat)

        # coverage

        plt.figure(figsize=(10,5))

        plt.imshow(self.grid,origin="lower",
                   extent=[-180,180,-90,90],
                   cmap="viridis")

        plt.colorbar(label="sample density")

        plt.xlabel("Longitude")
        plt.ylabel("Latitude")

        plt.grid(alpha=0.3)

        plt.savefig(os.path.join(fig_dir,"coverage.png"))

        plt.close()

        # latitude distribution

        plt.figure()

        sns.histplot(np.abs(lat),bins=80)

        plt.xlabel("|latitude|")

        plt.savefig(os.path.join(fig_dir,"latitude_distribution.png"))

        plt.close()

        # seasonal distribution

        plt.figure()

        sns.histplot(self.months,bins=np.arange(1,14))

        plt.xlabel("Month")

        plt.savefig(os.path.join(fig_dir,"season_distribution.png"))

        plt.close()

        # flux distribution

        sample=np.random.choice(flux,min(len(flux),200000),replace=False)

        plt.figure()

        sns.histplot(sample,bins=100)

        plt.yscale("log")

        plt.savefig(os.path.join(fig_dir,"flux_hist.png"))

        plt.close()

        # wind flux

        idx=np.random.choice(len(wind),min(len(wind),20000),replace=False)

        plt.figure()

        plt.scatter(wind[idx],flux[idx],s=1,alpha=0.2)

        plt.xlabel("Wind speed")
        plt.ylabel("Flux magnitude")

        plt.savefig(os.path.join(fig_dir,"wind_flux.png"))

        plt.close()

        # drift

        plt.figure()

        plt.plot(self.batch_flux_mean)

        plt.xlabel("Batch")
        plt.ylabel("Mean Flux")

        plt.savefig(os.path.join(fig_dir,"flux_drift.png"))

        plt.close()

        stats={
            "total_samples":self.samples,
            "mean_flux":float(np.mean(flux)),
            "mean_wind":float(np.mean(wind))
        }

        save_json(os.path.join(global_dir,"global_stats.json"),stats)

        pd.DataFrame([stats]).to_csv(
            os.path.join(global_dir,"global_stats.csv"),
            index=False
        )

        log(self.run_dir,"Global EDA completed")
