# ERA5 Zarr Dataset Builder

High-performance CLI tool for constructing structured, compressed, and reproducible ERA5 climate datasets in Zarr format.

Designed for research-scale workflows, PINN training, and large-scale atmospheric modeling.

---

## Overview

This tool builds multi-year ERA5 datasets from structured monthly NetCDF files and outputs optimized Zarr archives with:

- Float32 precision (memory efficient)
- Zstd compression (configurable)
- Time-based chunking (configurable)
- Robust time alignment
- Logging (file + console)
- Embedded dataset metadata
- Seasonal and monthly slicing support

---

## Directory Structure Expected

Input data must follow this structure:


/data/jayanth_works/era5_inputfluxes/1990-2025/<YEAR>/<month_year>/


Example:


1990-2025/
└── 2000/
└── january_2000/
├── instant_u10_v10_t2m_msl_sst.nc
├── accum_slhf_ssr_sshf_ssrd_tsr.nc
└── instant_d2m_sp.nc


---

## Installation

Requires Python 3.10+ and:

- xarray
- dask
- zarr
- numcodecs
- numpy

Example conda environment:


conda create -n era5_builder python=3.10 xarray dask zarr numcodecs numpy
conda activate era5_builder


---

## Basic Usage

Build full dataset (default 1990–2025):


python cli_industry.py --output full.zarr


---

## Seasonal Builds


python cli_industry.py --season DJF --output djf_all.zarr
python cli_industry.py --season JJA --output jja_all.zarr


Supported seasons:
- DJF
- MAM
- JJA
- SON

---

## Specific Month Across All Years

January only:


python cli_industry.py --months 1 --output jan_all.zarr


Multiple months:


python cli_industry.py --months 6 7 8 --output jja_manual.zarr


---

## Custom Year Range


python cli_industry.py --start 2000 --end 2010 --output 2000_2010.zarr


---

## Performance Tuning

Adjust time chunk size:


--chunk 1440


Default: `1440` (≈ 2 months of hourly data)

Adjust compression level:


--compression 2


Lower = faster write  
Higher = smaller file  

Example:


python cli_industry.py
--months 1
--chunk 2880
--compression 1
--output jan_fast.zarr


---

## Output Structure

Outputs are written to:


./outputs/<dataset_name>.zarr


Logs are written to:


./logs/build_<timestamp>.log


Each dataset contains metadata:

- creation timestamp
- year range
- selected months
- builder version

---

## Variables Included

- u10
- v10
- t2m
- sst
- sp
- d2m
- sshf (converted to W/m²)
- slhf (converted to W/m²)
- rho (computed air density)

---

## Design Philosophy

This builder is designed to:

- Scale linearly with dataset size
- Avoid unnecessary reindexing
- Minimize memory overhead
- Preserve exact time alignment
- Remain reproducible
- Support scientific traceability

It is intended to be run infrequently to build stable data archives used downstream for:

- PINN training
- ML workflows
- Climate modeling
- Seasonal analysis
- Flux research

---

## Performance Notes

Typical workstation (8 cores + HDD):

- Single year: ~3 minutes
- January across 36 years: ~11–12 minutes
- Full 30-year archive: scales linearly

Performance depends primarily on:
- Disk write speed
- Compression level
- Chunk size

---

## Future Extensions (Optional)

- YAML configuration support
- Docker containerization
- Automated missing-month detection
- Dataset checksum validation
- Distributed cluster support

---

## License

Research use. Customize as needed.

---

## Author

Sai Kumar  
ERA5 PINN Research Infrastructure
