# ERA5 PINN Dataset Builder

High-performance CLI pipeline for constructing structured, compressed, reproducible ERA5 climate datasets in Zarr format.

Designed for:

* Physics-Informed Neural Networks (PINNs)
* Seasonal climate analysis
* Air–sea flux research
* Large-scale atmospheric modeling
* Long-term dataset archiving

This builder converts monthly ERA5 NetCDF files into optimized Zarr archives suitable for large-scale ML training and scientific workflows.

---

# 🚀 Core Features

* Float32 precision (memory efficient)
* Zstd compression (configurable)
* Time-based chunking (configurable)
* Robust time alignment
* Automatic duplicate timestamp removal
* CLI-driven year / month / seasonal slicing
* Linear scaling with dataset size
* Reproducible build environment

---

# 📦 Conda Environment

This project uses the following environment:

```yaml
name: era5_pinn
channels:
  - conda-forge
  - defaults

dependencies:
  - python=3.10
  - numpy=1.26
  - pandas
  - xarray=2023.12.0
  - dask=2023.12.0
  - zarr=2.16.1
  - numcodecs=0.12.1
  - netcdf4
  - bottleneck
  - scipy
  - pip
```

Create the environment:

```bash
conda env create -f environment.yml
conda activate era5_pinn
```

---

# 📂 Required Input Directory Structure

Your ERA5 files must be structured exactly like this:

```
/data/jayanth_works/era5_inputfluxes/1990-2025/<YEAR>/<month_year>/
```

Example:

```
1990-2025/
└── 2000/
    └── january_2000/
        ├── instant_u10_v10_t2m_msl_sst.nc
        ├── accum_slhf_ssr_sshf_ssrd_tsr.nc
        └── instant_d2m_sp.nc
```

Each monthly folder must contain:

* instant_u10_v10_t2m_msl_sst.nc
* accum_slhf_ssr_sshf_ssrd_tsr.nc
* instant_d2m_sp.nc

---

# ▶️ Basic Usage

Build full dataset (1990–2025):

```bash
python cli_industry.py --output full_1990_2025.zarr
```

---

# 📆 Build Specific Year Range

```bash
python cli_industry.py \
  --start 2000 \
  --end 2010 \
  --output era5_2000_2010.zarr
```

---

# 🌦 Seasonal Builds

Supported seasons:

* DJF
* MAM
* JJA
* SON

Example:

```bash
python cli_industry.py --season DJF --output djf_all_years.zarr
python cli_industry.py --season JJA --output jja_all_years.zarr
```

---

# 🗓 Build Specific Months Across All Years

January only:

```bash
python cli_industry.py --months 1 --output january_all_years.zarr
```

Multiple months:

```bash
python cli_industry.py --months 6 7 8 --output jja_manual.zarr
```

---

# ⚙️ Performance Tuning

## Time Chunk Size

```bash
--chunk 1440
```

Default: 1440 (~2 months of hourly data)

Larger chunk:

* Faster sequential reads
* Fewer chunks written
* Better for HDD

Smaller chunk:

* Better random access
* More flexible slicing

---

## Compression Level

```bash
--compression 2
```

Compression levels:

* 1 → fastest write
* 2 → balanced (recommended)
* 3+ → smaller file, slower write

Example fast build:

```bash
python cli_industry.py \
  --months 1 \
  --chunk 2880 \
  --compression 1 \
  --output january_fast.zarr
```

---

# 📦 Output Structure

Outputs are written to:

```
./outputs/<dataset_name>.zarr
```

Logs are written to:

```
./logs/build_<timestamp>.log
```

Each dataset contains metadata:

* created_at
* year_range
* selected_months
* builder_version

---

# 📊 Variables Included

The final dataset contains:

* u10
* v10
* t2m
* sst
* sp
* d2m
* sshf (converted to W/m²)
* slhf (converted to W/m²)
* rho (computed air density)

---

# 🔍 Validate Built Dataset

Open dataset:

```python
import xarray as xr

ds = xr.open_zarr("outputs/full_1990_2025.zarr", consolidated=True)

print(ds)
print(ds.attrs)
```

Inspect chunking:

```python
print(ds.chunks)
```

---

# 🎯 Sampling for PINN Training

Example: Random sampling across entire dataset

```python
import numpy as np
import xarray as xr

ds = xr.open_zarr("outputs/full_1990_2025.zarr", consolidated=True)

n_samples = 200_000

time_idx = np.random.randint(0, len(ds.time), n_samples)
lat_idx  = np.random.randint(0, len(ds.latitude), n_samples)
lon_idx  = np.random.randint(0, len(ds.longitude), n_samples)

samples = ds.isel(
    time=xr.DataArray(time_idx, dims="points"),
    latitude=xr.DataArray(lat_idx, dims="points"),
    longitude=xr.DataArray(lon_idx, dims="points")
)

print(samples)
```

For efficient training:

* Open dataset once
* Keep Dask lazy evaluation
* Avoid reopening inside training loop

---

# 📈 Performance Expectations

On 8-core workstation + HDD:

| Build Type           | Approx Time    |
| -------------------- | -------------- |
| Single year          | ~3 minutes     |
| January (36 years)   | ~11–12 minutes |
| Full 30-year dataset | Linear scaling |

Performance depends mainly on:

* Disk speed
* Compression level
* Chunk size

---

# 🧠 Design Philosophy

This builder is meant to:

* Be run infrequently
* Produce stable long-term Zarr archives
* Act as infrastructure, not experiment code
* Ensure reproducibility
* Preserve scientific integrity

Once built, the dataset becomes:

* A permanent research repository
* A PINN training backbone
* A seasonal analysis archive
* A reproducible climate baseline

---

# 🔮 Future Extensions

* YAML configuration support
* Docker container
* Missing-month detection
* Dataset integrity verification
* Distributed Dask cluster build mode

---

# 👤 Author

Sai Kumar
ERA5 PINN Research Infrastructure
