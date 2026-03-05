#!/usr/bin/env python3

import os
import argparse
from datetime import datetime

from sampler import run as sampler_run
from eda_stream import BatchEDA, GlobalEDA, create_run_dir, log


# ------------------------------------------------
# PATHS
# ------------------------------------------------

MASK_FILE = "ocean_mask.npz"


# ------------------------------------------------
# MASK CREATION
# ------------------------------------------------

def ensure_ocean_mask():

    if os.path.exists(MASK_FILE):

        print("Ocean mask exists")
        return

    print("Ocean mask not found → building")

    import build_ocean_mask

    build_ocean_mask.main()


# ------------------------------------------------
# MAIN PIPELINE
# ------------------------------------------------

def run_pipeline(args):

    print("\n=== ERA5 FLUX PIPELINE START ===\n")

    # ------------------------------------------------
    # STEP 1 — OCEAN MASK
    # ------------------------------------------------

    ensure_ocean_mask()

    # ------------------------------------------------
    # STEP 2 — RUN DIRECTORY
    # ------------------------------------------------

    run_dir = create_run_dir()

    log(run_dir,"Pipeline started")

    # ------------------------------------------------
    # STEP 3 — GLOBAL EDA
    # ------------------------------------------------

    global_eda = GlobalEDA(run_dir)

    # ------------------------------------------------
    # STEP 4 — SAMPLER
    # ------------------------------------------------

    sampler_args = argparse.Namespace(
        sampler=args.sampler,
        batch_size=args.batch_size,
        batches=args.batches,
        start_year=args.start_year,
        end_year=args.end_year,
        seed=args.seed
    )

    engine = sampler_run(sampler_args)

    # ------------------------------------------------
    # STEP 5 — PROCESS BATCHES
    # ------------------------------------------------

    for batch in engine:

        # --------------------------------------------
        # Batch EDA
        # --------------------------------------------

        batch_eda = BatchEDA(batch, run_dir)

        batch_eda.run_all()

        # --------------------------------------------
        # Global EDA
        # --------------------------------------------

        global_eda.add_batch(
            batch["lat"],
            batch["lon"],
            batch_eda.flux,
            batch_eda.wind,
            batch["time"]
        )

    # ------------------------------------------------
    # STEP 6 — FINAL GLOBAL EDA
    # ------------------------------------------------

    global_eda.finalize()

    log(run_dir,"Pipeline completed")


# ------------------------------------------------
# CLI
# ------------------------------------------------

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--sampler",
        choices=[
            "random",
            "temporal",
            "seasonal",
            "spatial",
            "flux",
            "hybrid"
        ],
        default="hybrid"
    )

    parser.add_argument("--batch_size",type=int,default=15000)

    parser.add_argument("--batches",type=int,default=20)

    parser.add_argument("--start_year",type=int,default=1990)

    parser.add_argument("--end_year",type=int,default=2025)

    parser.add_argument("--seed",type=int,default=42)

    args = parser.parse_args()

    run_pipeline(args)


# ------------------------------------------------

if __name__ == "__main__":
    main()
