import os
import numpy as np
import xarray as xr
from scipy.interpolate import griddata
from core import logger
from memory_utils import monitor_memory_usage, clear_memory

# -----------------------------
# Interpolation of NaNs (2D only)
# -----------------------------
def interpolate_nan_values(data, ocean_mask=None, method='linear', min_valid_points=4):
    """
    Interpolate NaN values in a 2D field.
    
    Parameters:
    - data : 2D np.ndarray (lat, lon)
    - ocean_mask : 2D bool array, True for ocean points
    - method : interpolation method ('linear', 'nearest', 'cubic')
    - min_valid_points : minimum points required for interpolation
    """
    data = data.copy().astype(np.float32)
    
    if ocean_mask is not None:
        if ocean_mask.shape != data.shape:
            raise ValueError(f"Ocean mask shape {ocean_mask.shape} does not match data {data.shape}")
        data[~ocean_mask] = np.nan

    valid_mask = ~np.isnan(data)
    if valid_mask.sum() < min_valid_points:
        logger.warning("Too few valid points for interpolation")
        return data

    # Prepare grid
    y, x = np.indices(data.shape)
    valid_points = np.column_stack((x[valid_mask], y[valid_mask]))
    valid_values = data[valid_mask]
    all_points = np.column_stack((x.ravel(), y.ravel()))

    try:
        interp = griddata(valid_points, valid_values, all_points, method=method, fill_value=np.nan)
        interp = interp.reshape(data.shape)

        # Fill remaining NaNs with nearest-neighbor
        if np.any(np.isnan(interp)):
            interp_nn = griddata(valid_points, valid_values, all_points, method='nearest').reshape(data.shape)
            interp[np.isnan(interp)] = interp_nn[np.isnan(interp)]

        if ocean_mask is not None:
            interp[~ocean_mask] = np.nan

        return interp

    except Exception as e:
        logger.warning(f"Interpolation failed: {e}")
        return data

def create_ocean_mask(data_dict):
    """
    Static ocean mask:
    A grid cell is ocean if SST is valid at ANY time step.
    Designed for sparse SST (ERA5-like).
    """
    logger.info("Creating static ocean mask (ANY-valid-time)")

    sst = data_dict['sst']  # (time, lat, lon)

    # Physical ocean SST range (Kelvin)
    valid_sst = (~np.isnan(sst)) & (sst >= 271.0) & (sst <= 308.0)

    # Ocean if SST ever exists
    ocean_mask = np.any(valid_sst, axis=0)

    ocean_points = ocean_mask.sum()
    total_points = ocean_mask.size

    logger.info(
        f"Ocean mask: {ocean_points}/{total_points} "
        f"({100 * ocean_points / total_points:.2f}%)"
    )

    if ocean_points == 0:
        raise RuntimeError(
            "Ocean mask empty. SST may be fully NaN or range-filtered."
        )

    return ocean_mask


# -----------------------------
# Load and preprocess data
# -----------------------------
def load_data(file_path, user_var_mapping=None):
    """
    Load NetCDF data, interpolate NaNs over ocean, and attach broadcasted ocean mask.
    """
    logger.info(f"Loading data from {file_path}")

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    ds = xr.open_dataset(file_path)

    # Default mapping for your dataset
    default_var_mapping = {
        'u10': 'u10',
        'v10': 'v10',
        'd2m': 'td2m',
        't2m': 't2m',
        'sst': 'sst',
        'sp':  'sp',
        'rho': 'rho'
    }
    var_mapping = user_var_mapping or default_var_mapping

    data_dict = {}
    for key, name in var_mapping.items():
        if name not in ds:
            raise KeyError(f"Missing variable in dataset: {name}")
        data_dict[key] = ds[name].values.astype(np.float32)
        logger.info(f"{key}: {data_dict[key].shape}")

    # Load dimensions
    data_dict['time'] = ds['time'].values
    data_dict['latitude'] = ds['lat'].values
    data_dict['longitude'] = ds['lon'].values
    ds.close()
    sst = data_dict['sst']
    print(f"NaNs in SST: {np.isnan(sst).sum()}")
    print(f"Total SST points: {sst.size}")
    print(f"SST stats: min={np.nanmin(sst):.2f}, max={np.nanmax(sst):.2f}\nsst shape : {sst.shape}")
    # -----------------------------
    # Create robust static ocean mask
    # -----------------------------
    static_mask = create_ocean_mask(data_dict)
    
    # Interpolate NaNs for key variables only over ocean
    interpolate_vars = ['sst', 't2m', 'd2m']
    for var in interpolate_vars:
        field = data_dict[var]
        for t in range(field.shape[0]):
            if np.isnan(field[t][static_mask]).any():
                field[t] = interpolate_nan_values(field[t], ocean_mask=static_mask)
        data_dict[var] = field

    # Broadcast ocean mask to 3D for convenience
    data_dict['ocean_mask'] = np.broadcast_to(static_mask, (data_dict['time'].shape[0], *static_mask.shape))

    return data_dict

# -----------------------------
# Prepare ML data in chunks
# -----------------------------
def prepare_data_in_chunks(
    data_dict,
    chunk_size=1_000_000,
    max_samples=20_000_000,
    subsample_ratio=0.02,
    seed=42
):
    """
    Prepare training data chunks from the full dataset using ocean mask and subsampling.
    """
    rng = np.random.default_rng(seed)
    input_vars = ['u10', 'v10', 'd2m', 't2m', 'sst', 'sp', 'rho']

    for v in input_vars + ['ocean_mask']:
        if v not in data_dict:
            raise KeyError(f"Missing required variable: {v}")

    time_dim, lat_dim, lon_dim = data_dict['sst'].shape
    ocean_mask_3d = data_dict['ocean_mask'].reshape(time_dim, -1)
    reshaped_data = {v: data_dict[v].reshape(time_dim, -1) for v in input_vars}

    total_ocean = int(ocean_mask_3d.sum())
    effective_ratio = min(subsample_ratio, max_samples / total_ocean)
    logger.info(f"Effective subsample ratio: {effective_ratio:.5f}")

    # Physical constants
    cp, L = 1004.0, 2.5e6
    C_H, C_E = 1.3e-3, 1.5e-3

    def sat_vp(T):
        Tc = T - 273.15
        return 611.2 * np.exp(17.67 * Tc / (Tc + 243.5))

    X_chunks, y_chunks = [], []
    collected = 0

    for t in range(time_dim):
        if collected >= max_samples:
            break

        idx = ocean_mask_3d[t]
        if not idx.any():
            continue

        # Extract ocean points
        X_t = np.column_stack([reshaped_data[v][t][idx] for v in input_vars])
        X_t = X_t[~np.isnan(X_t).any(axis=1)]
        if X_t.size == 0:
            continue

        # Subsample
        keep = rng.random(len(X_t)) < effective_ratio
        X_t = X_t[keep]
        if len(X_t) > chunk_size:
            X_t = X_t[rng.choice(len(X_t), chunk_size, replace=False)]

        # Physics: SHF & LHF
        wind = np.maximum(np.sqrt(X_t[:,0]**2 + X_t[:,1]**2), 0.5)
        sp = np.maximum(X_t[:,5], 5e4)
        q_a = 0.622 * sat_vp(X_t[:,2]) / (sp - 0.378 * sat_vp(X_t[:,2]))
        q_s = 0.622 * sat_vp(X_t[:,4]) / (sp - 0.378 * sat_vp(X_t[:,4]))
        SHF = X_t[:,6] * cp * C_H * wind * (X_t[:,4] - X_t[:,3])
        LHF = X_t[:,6] * L  * C_E * wind * (q_s - q_a)

        X_chunks.append(X_t.astype(np.float32))
        y_chunks.append(np.column_stack([SHF, LHF]).astype(np.float32))
        collected += len(X_t)

        # Compact chunks early
        if len(X_chunks) >= 20:
            X_chunks = [np.vstack(X_chunks)]
            y_chunks = [np.vstack(y_chunks)]

    if len(X_chunks) == 0:
        raise ValueError("No valid data collected. Check your data and ocean mask.")

    X = np.vstack(X_chunks)
    y = np.vstack(y_chunks)

    logger.info(f"Final dataset shapes: X={X.shape}, y={y.shape}")
    return X, y




