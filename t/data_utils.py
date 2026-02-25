from core import logger
import numpy as np
import os
import xarray as xr
from memory_utils import monitor_memory_usage, clear_memory
from scipy.interpolate import griddata

# Function to load and preprocess data
def load_data(file_path):
    logger.info(f"Loading data from {file_path}")

    # Check if file exists
    if not os.path.exists(file_path):
        logger.error(f"File not found: {file_path}")
        raise FileNotFoundError(f"File not found: {file_path}")

    try:
        ds = xr.open_dataset(file_path)
        logger.info("Available variables in the dataset:")
        for var in ds.variables:
            logger.info(f" - {var}")

        # Map variable names based on your NetCDF file
        var_mapping = {
            'u10': 'u10',
            'v10': 'v10',
            'd2m': 'd2m',
            't2m': 't2m',
            'sst': 'sst',
            'sp': 'sp',
            'rho': 'rhoao'
        }

        # Extract data
        data_dict = {}

        # Input variables
        for var_key, var_name in var_mapping.items():
            if var_name in ds:
                data_dict[var_key] = ds[var_name].values
                logger.info(f"Extracted {var_key} data with shape {data_dict[var_key].shape}")
            else:
                logger.warning(f"Warning: {var_name} not found in dataset for {var_key}")
        # Get dimensions for reference
        if 'valid_time' in ds.dims:
            data_dict['time'] = ds.valid_time.values
        if 'latitude' in ds.dims:
            data_dict['latitude'] = ds.latitude.values
        if 'longitude' in ds.dims:
            data_dict['longitude'] = ds.longitude.values

        ds.close()

        # Create ocean mask
        ocean_mask = create_ocean_mask(data_dict)
        data_dict['ocean_mask'] = ocean_mask

        # Apply interpolation to handle NaN values in ocean areas
        logger.info("Applying interpolation to handle NaN values in ocean areas...")
        for var_key in var_mapping.keys():
            if var_key in data_dict and var_key != 'ocean_mask':
                original_data = data_dict[var_key]
                interpolated_data = np.zeros_like(original_data)

                # Interpolate each time step
                for t in range(original_data.shape[0]):
                    time_slice = original_data[t]

                    # Only interpolate in ocean areas
                    ocean_slice = ocean_mask[t] if len(ocean_mask.shape) == 3 else ocean_mask

                    if np.any(ocean_slice):
                        interpolated_slice = interpolate_nan_values(time_slice)
                        # Keep only ocean points
                        interpolated_slice[~ocean_slice] = np.nan
                        interpolated_data[t] = interpolated_slice
                    else:
                        interpolated_data[t] = time_slice

                data_dict[var_key] = interpolated_data
                logger.info(f"Applied interpolation to {var_key}")

        return data_dict

    except Exception as e:
        logger.error(f"Error loading data: {str(e)}")
        raise

def prepare_data_in_chunks(data_dict, chunk_size=1000000, max_samples=20000000, subsample_ratio=0.02):
    """Prepare data in chunks with strict memory management and subsampling"""
    logger.info(f"Preparing data with max_samples={max_samples}, subsample_ratio={subsample_ratio}")

    # Required input variables
    input_vars = ['u10', 'v10', 'd2m', 't2m', 'sst', 'sp', 'rho']

    # Check shapes
    shapes = {var: data_dict[var].shape for var in input_vars if var in data_dict}
    logger.info("Input data shapes: %s", shapes)
    
    # Get dimensions
    time_dim = data_dict['time'].shape[0]
    lat_dim = data_dict['latitude'].shape[0]
    lon_dim = data_dict['longitude'].shape[0]
    logger.info(f"Dimensions: time={time_dim}, lat={lat_dim}, lon={lon_dim}")

    # Get ocean mask
    ocean_mask = data_dict['ocean_mask']

    # Calculate total number of ocean points
    if len(ocean_mask.shape) == 3:
        total_ocean_points = np.sum(ocean_mask)
    else:
        total_ocean_points = np.sum(ocean_mask) * time_dim

    logger.info(f"Total ocean data points: {total_ocean_points}")
    
    # Apply subsampling if dataset is too large
    if total_ocean_points > max_samples:
        effective_subsample_ratio = max_samples / total_ocean_points
        logger.info(f"Dataset too large. Using subsample ratio: {effective_subsample_ratio:.4f}")
    else:
        effective_subsample_ratio = subsample_ratio
        logger.info(f"Using configured subsample ratio: {effective_subsample_ratio:.4f}")

    # Reshape all variables to 2D arrays (time, lat*lon)
    reshaped_data = {}
    for var in input_vars:
        if var in data_dict:
            reshaped_data[var] = data_dict[var].reshape(time_dim, -1)

    # Reshape ocean mask
    if len(ocean_mask.shape) == 3:
        ocean_mask_2d = ocean_mask.reshape(time_dim, -1)
    else:
        ocean_mask_2d = np.tile(ocean_mask.reshape(1, -1), (time_dim, 1))

    # Process data in chunks with subsampling
    X_chunks = []
    y_chunks = []
    samples_collected = 0

    # Constants for bulk formulas
    cp = 1004.0  # Specific heat capacity of air at constant pressure (J/kg/K)
    L = 2.5e6    # Latent heat of vaporization (J/kg)
    C_H = 1.3e-3 # Transfer coefficient for sensible heat
    C_E = 1.5e-3 # Transfer coefficient for latent heat

    # Calculate saturation vapor pressure function
    def calc_saturation_vapor_pressure(temp_k):
        temp_c = temp_k - 273.15
        return 6.112 * np.exp(17.67 * temp_c / (temp_c + 243.5)) * 100  # in Pa

    # Process each time step with subsampling
    for t in range(time_dim):
        if t % 500 == 0:  # Reduced logging frequency
            logger.info(f"Processing time step {t+1}/{time_dim}, collected {samples_collected} samples")
            monitor_memory_usage()

        # Early termination if we have enough samples
        if samples_collected >= max_samples:
            logger.info(f"Reached maximum samples ({max_samples}). Stopping data collection.")
            break

        # Subsampling: skip this time step with probability (1 - subsample_ratio)
        if np.random.random() > effective_subsample_ratio:
            continue

        # Get ocean mask for this time step
        ocean_mask_t = ocean_mask_2d[t]
        if not np.any(ocean_mask_t):
            continue  # Skip if no ocean points

        # Extract data for this time step (only ocean points)
        try:
            u10 = reshaped_data['u10'][t][ocean_mask_t]
            v10 = reshaped_data['v10'][t][ocean_mask_t]
            td2m = reshaped_data['d2m'][t][ocean_mask_t]
            t2m = reshaped_data['t2m'][t][ocean_mask_t]
            sst = reshaped_data['sst'][t][ocean_mask_t]
            sp = reshaped_data['sp'][t][ocean_mask_t]
            rho = reshaped_data['rho'][t][ocean_mask_t]
        except KeyError as e:
            logger.warning(f"Missing variable {e} at time step {t}")
            continue

        # Stack features
        X_t = np.column_stack([u10, v10, td2m, t2m, sst, sp, rho])

        # Remove remaining NaN values
        mask = ~np.isnan(X_t).any(axis=1)
        X_t = X_t[mask]

        if X_t.shape[0] == 0:
            continue  # Skip if no valid data

        # Further subsampling within the time step if needed
        if len(X_t) > chunk_size:
            indices = np.random.choice(len(X_t), chunk_size, replace=False)
            X_t = X_t[indices]

        # Calculate wind speed
        wind_speed = np.sqrt(X_t[:, 0]**2 + X_t[:, 1]**2)

        # Calculate specific humidity from dewpoint temperature
        e_s_td = calc_saturation_vapor_pressure(X_t[:, 2])
        q_a = 0.622 * e_s_td / (X_t[:, 5] - 0.378 * e_s_td)

        # Calculate specific humidity at sea surface
        e_s_sst = calc_saturation_vapor_pressure(X_t[:, 4])
        q_s = 0.622 * e_s_sst / (X_t[:, 5] - 0.378 * e_s_sst)

        # Calculate fluxes using bulk formulas
        sensible_heat_flux = X_t[:, 6] * cp * C_H * wind_speed * (X_t[:, 4] - X_t[:, 3])
        latent_heat_flux = X_t[:, 6] * L * C_E * wind_speed * (q_s - q_a)

        # Combine into target array
        y_t = np.column_stack([sensible_heat_flux, latent_heat_flux])

        # Add to chunks
        X_chunks.append(X_t.astype(np.float32))  # Convert to float32 to save memory
        y_chunks.append(y_t.astype(np.float32))
        samples_collected += len(X_t)

        # Clear intermediate variables
        del u10, v10, td2m, t2m, sst, sp, rho, X_t, y_t
        del wind_speed, e_s_td, q_a, e_s_sst, q_s, sensible_heat_flux, latent_heat_flux

        # Periodic memory cleanup
        if t % 1000 == 0:
            clear_memory()

    # Concatenate all chunks
    logger.info("Concatenating data chunks...")
    if len(X_chunks) == 0:
        raise ValueError("No valid data collected. Check your data file and ocean mask.")
    
    X = np.vstack(X_chunks)
    y = np.vstack(y_chunks)

    # Clear intermediate data
    del X_chunks, y_chunks
    clear_memory()

    logger.info(f"Final data shapes: X={X.shape}, y={y.shape}")

    # Check for any remaining NaN or infinite values
    nan_count_X = np.sum(np.isnan(X))
    inf_count_X = np.sum(np.isinf(X))
    nan_count_y = np.sum(np.isnan(y))
    inf_count_y = np.sum(np.isinf(y))

    logger.info(f"Data quality check - X: {nan_count_X} NaNs, {inf_count_X} infs")
    logger.info(f"Data quality check - y: {nan_count_y} NaNs, {inf_count_y} infs")

    # Remove any remaining problematic values
    valid_mask = ~(np.isnan(X).any(axis=1) | np.isinf(X).any(axis=1) |
                   np.isnan(y).any(axis=1) | np.isinf(y).any(axis=1))

    X = X[valid_mask]
    y = y[valid_mask]

    logger.info(f"After cleaning: X={X.shape}, y={y.shape}")

    return X, y

# Function to create ocean mask
def create_ocean_mask(data_dict):
    """Create ocean mask based on SST availability and typical ocean characteristics"""
    logger.info("Creating ocean mask...")

    # Use SST as primary indicator for ocean points
    sst = data_dict['sst']

    # Create mask where SST is valid (not NaN) and within reasonable ocean temperature range
    # Ocean SST typically ranges from -2°C to 35°C (271K to 308K)
    ocean_mask = (~np.isnan(sst)) & (sst >= 271.0) & (sst <= 308.0)

    # Additional check: ocean points typically have higher surface pressure variability
    # and different temperature characteristics
    if 'sp' in data_dict:
        sp = data_dict['sp']
        # Surface pressure over ocean typically ranges from 98000 to 104000 Pa
        pressure_mask = (~np.isnan(sp)) & (sp >= 95000) & (sp <= 105000)
        ocean_mask = ocean_mask & pressure_mask

    # Calculate percentage of ocean points
    total_points = ocean_mask.size
    ocean_points = np.sum(ocean_mask)
    ocean_percentage = (ocean_points / total_points) * 100

    logger.info(f"Ocean mask created: {ocean_points}/{total_points} points ({ocean_percentage:.2f}%) identified as ocean")
    return ocean_mask

# Function to interpolate NaN values
def interpolate_nan_values(data, method='linear'):
    """Interpolate NaN values in 2D arrays using scipy griddata"""
    if np.all(np.isnan(data)):
        logger.warning("All values are NaN, cannot interpolate")
        return data

    # Get valid (non-NaN) points
    valid_mask = ~np.isnan(data)
    if np.sum(valid_mask) < 4:  # Need at least 4 points for interpolation
        logger.warning("Too few valid points for interpolation")
        return data

    # Create coordinate grids
    rows, cols = np.mgrid[0:data.shape[0], 0:data.shape[1]]

    # Get coordinates and values of valid points
    valid_coords = np.column_stack((rows[valid_mask], cols[valid_mask]))
    valid_values = data[valid_mask]

    # Get coordinates of all points
    all_coords = np.column_stack((rows.ravel(), cols.ravel()))

    try:
        # Interpolate
        interpolated_values = griddata(valid_coords, valid_values, all_coords,
                                     method=method, fill_value=np.nan)
        interpolated_data = interpolated_values.reshape(data.shape)

        # Fill remaining NaNs with nearest neighbor interpolation
        if np.any(np.isnan(interpolated_data)):
            interpolated_values_nearest = griddata(valid_coords, valid_values, all_coords,
                                                 method='nearest')
            interpolated_data_nearest = interpolated_values_nearest.reshape(data.shape)
            nan_mask = np.isnan(interpolated_data)
            interpolated_data[nan_mask] = interpolated_data_nearest[nan_mask]

        return interpolated_data
    except Exception as e:
        logger.warning(f"Interpolation failed: {str(e)}, returning original data")
        return data
