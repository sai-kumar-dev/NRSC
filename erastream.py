(era5_pinn) [nrsc@localhost test_aft]$ cat eda_stream.py
import os
import json
import logging
import psutil
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch

from datetime import datetime
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import mutual_info_regression

sns.set_theme(style="whitegrid")

INPUT_NAMES = ["u10","v10","t2m","d2m","sp","rho","sst"]
TARGET_NAMES = ["sshf","slhf"]
ALL_NAMES = INPUT_NAMES + TARGET_NAMES


class StreamEDA:

    def __init__(self,config):

        self.config = config

        self.X_batches=[]
        self.Y_batches=[]
        self.meta_batches=[]

        self.run_dir=self._create_run_dir()

        self.paths={
            "summary":os.path.join(self.run_dir,"summary"),
            "sampling":os.path.join(self.run_dir,"sampling"),
            "spatial":os.path.join(self.run_dir,"spatial"),
            "physics":os.path.join(self.run_dir,"physics"),
            "importance":os.path.join(self.run_dir,"importance")
        }

        for p in self.paths.values():
            os.makedirs(p,exist_ok=True)

        self._setup_logging()
        self._save_config()

        logging.info("EDA initialized")
        self._log_system("startup")


    def _create_run_dir(self):

        ts=datetime.now().strftime("%Y%m%d_%H%M%S")
        path=os.path.join("eda_runs",f"run_{ts}")
        os.makedirs(path,exist_ok=True)
        return path


    def _setup_logging(self):

        log_file=os.path.join(self.run_dir,"eda.log")

        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s | %(levelname)s | %(message)s",
            handlers=[logging.FileHandler(log_file),
                      logging.StreamHandler()]
        )


    def _save_config(self):

        with open(os.path.join(self.run_dir,"config.json"),"w") as f:
            json.dump(self.config,f,indent=4)


    def _log_system(self,stage):

        mem=psutil.virtual_memory()
        cpu=psutil.cpu_percent(interval=0.1)

        logging.info(
            f"[SYSTEM] {stage} | CPU {cpu:.1f}% | "
            f"Memory {mem.used/1e9:.2f}/{mem.total/1e9:.2f}GB"
        )


    def observe(self,batch):

        X=batch["X"]
        Y=batch["Y"]
        meta=batch["meta"]

        if isinstance(X,torch.Tensor):
            X=X.cpu().numpy()

        if isinstance(Y,torch.Tensor):
            Y=Y.cpu().numpy()

        self.X_batches.append(X)
        self.Y_batches.append(Y)
        self.meta_batches.append(meta.reset_index(drop=True))

        logging.info(f"Batch {batch['batch_id']} observed")


    def finalize(self):

        logging.info("Finalizing EDA")

        X=np.concatenate(self.X_batches)
        Y=np.concatenate(self.Y_batches)
        meta=pd.concat(self.meta_batches,ignore_index=True)

        df=pd.DataFrame(
            np.concatenate([X,Y],axis=1),
            columns=ALL_NAMES
        )

        df=pd.concat([df,meta],axis=1)

        self._summary(df)
        self._sampling_diagnostics(df)
        self._spatial_maps(df)
        self._physics_diagnostics(df)
        self._feature_importance(df)
        self._latitude_flux_profile(df)

        logging.info("EDA complete")


    def _summary(self,df):

        df.describe().to_csv(
            os.path.join(self.paths["summary"],"describe.csv")
        )

        df.isna().sum().to_csv(
            os.path.join(self.paths["summary"],"missing.csv")
        )


    def _sampling_diagnostics(self,df):

        flux=np.abs(df.sshf)+np.abs(df.slhf)

        bins=[0,100,300,700,2000]
        classes=pd.cut(flux,bins)

        classes.value_counts().sort_index().plot.bar()

        plt.title("Flux class distribution")

        plt.savefig(
            os.path.join(self.paths["sampling"],
                         "flux_classes.png"),
            dpi=300
        )

        plt.close()


    def _spatial_maps(self,df):

        grid=df.groupby(["lat","lon"]).mean().reset_index()

        fig,ax=plt.subplots(1,2,figsize=(14,6))

        sc1=ax[0].scatter(
            grid.lon,
            grid.lat,
            c=grid.sshf,
            cmap="coolwarm",
            s=10
        )

        sc2=ax[1].scatter(
            grid.lon,
            grid.lat,
            c=grid.slhf,
            cmap="coolwarm",
            s=10
        )

        ax[0].set_title("Sensible Heat Flux")
        ax[1].set_title("Latent Heat Flux")

        plt.colorbar(sc1)

        plt.savefig(
            os.path.join(self.paths["spatial"],
                         "flux_maps.png"),
            dpi=300
        )

        plt.close()


    def _physics_diagnostics(self,df):

        wind=np.sqrt(df.u10**2 + df.v10**2)

        plt.hexbin(wind,df.slhf,gridsize=60)

        plt.xlabel("Wind speed")
        plt.ylabel("Latent heat flux")

        plt.savefig(
            os.path.join(self.paths["physics"],
                         "wind_flux.png"),
            dpi=300
        )

        plt.close()


    def _feature_importance(self,df):

        X=df[INPUT_NAMES]

        for target in TARGET_NAMES:

            y=df[target]

            rf=RandomForestRegressor(n_estimators=100)

            rf.fit(X,y)

            imp=pd.Series(
                rf.feature_importances_,
                index=INPUT_NAMES
            )

            imp.sort_values().plot.barh()

            plt.title(f"Feature importance → {target}")

            plt.savefig(
                os.path.join(
                    self.paths["importance"],
                    f"rf_{target}.png"
                ),
                dpi=300
            )

            plt.close()


    def _latitude_flux_profile(self,df):

        lat=df.groupby("lat")[TARGET_NAMES].mean()

        lat.plot()

        plt.title("Latitude Heat Flux Profile")

        plt.savefig(
            os.path.join(self.paths["physics"],
                         "latitude_flux_profile.png"),
            dpi=300
        )

        plt.close()
(era5_pinn) [nrsc@localhost test_aft]$
