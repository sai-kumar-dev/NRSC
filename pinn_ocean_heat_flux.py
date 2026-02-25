import os
import numpy as np
import xarray as xr
import torch
import gc
import torch.cuda
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import time
import pickle
import logging
import json
from datetime import datetime
from scipy.interpolate import griddata
from scipy.ndimage import gaussian_filter

# Set up logging
log_dir = "logs"
os.makedirs(log_dir, exist_ok=True)
log_file = f"{log_dir}/pinn_training_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger()

# Set random seeds for reproducibility
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

# Function to clear memory
def clear_memory():
    """Clear memory to avoid OOM errors"""
    gc.collect()
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

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

# Function to apply smoothing for visualization
def apply_smoothing(data, sigma=1.0):
    """Apply Gaussian smoothing for visualization"""
    if np.all(np.isnan(data)):
        return data
    
    # Create a mask for valid data
    valid_mask = ~np.isnan(data)
    if np.sum(valid_mask) == 0:
        return data
    
    # Apply smoothing only to valid data
    smoothed_data = np.full_like(data, np.nan)
    
    # Fill NaNs with interpolated values for smoothing
    temp_data = data.copy()
    if np.any(np.isnan(temp_data)):
        temp_data = interpolate_nan_values(temp_data)
    
    # Apply Gaussian filter
    if not np.all(np.isnan(temp_data)):
        smoothed_data = gaussian_filter(temp_data, sigma=sigma)
    
    # Restore NaNs where original data was NaN
    smoothed_data[~valid_mask] = np.nan
    
    return smoothed_data

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

# Function to prepare data in chunks with ocean masking
def prepare_data_in_chunks(data_dict, chunk_size=1000000):
    """Prepare data in chunks to avoid memory issues, applying ocean mask"""
    logger.info("Preparing data in chunks with ocean masking...")
    
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
    
    # Process data in chunks
    X_chunks = []
    y_chunks = []
    
    # Constants for bulk formulas
    cp = 1004.0  # Specific heat capacity of air at constant pressure (J/kg/K)
    L = 2.5e6    # Latent heat of vaporization (J/kg)
    C_H = 1.3e-3 # Transfer coefficient for sensible heat
    C_E = 1.5e-3 # Transfer coefficient for latent heat
    
    # Calculate saturation vapor pressure function
    def calc_saturation_vapor_pressure(temp_k):
        temp_c = temp_k - 273.15
        return 6.112 * np.exp(17.67 * temp_c / (temp_c + 243.5)) * 100  # in Pa
    
    # Process each time step
    for t in range(time_dim):
        if t % 100 == 0:
            logger.info(f"Processing time step {t+1}/{time_dim}")
        
        # Get ocean mask for this time step
        ocean_mask_t = ocean_mask_2d[t]
        if not np.any(ocean_mask_t):
            continue  # Skip if no ocean points
        
        # Extract data for this time step (only ocean points)
        u10 = reshaped_data['u10'][t][ocean_mask_t]
        v10 = reshaped_data['v10'][t][ocean_mask_t]
        td2m = reshaped_data['d2m'][t][ocean_mask_t]
        t2m = reshaped_data['t2m'][t][ocean_mask_t]
        sst = reshaped_data['sst'][t][ocean_mask_t]
        sp = reshaped_data['sp'][t][ocean_mask_t]
        rho = reshaped_data['rho'][t][ocean_mask_t]
        
        # Stack features
        X_t = np.column_stack([u10, v10, td2m, t2m, sst, sp, rho])
        
        # Remove remaining NaN values (after interpolation, there should be very few)
        mask = ~np.isnan(X_t).any(axis=1)
        X_t = X_t[mask]
        
        if X_t.shape[0] == 0:
            continue  # Skip if no valid data
        
        # Calculate wind speed
        wind_speed = np.sqrt(X_t[:, 0]**2 + X_t[:, 1]**2)
        
        # Calculate specific humidity from dewpoint temperature
        e_s_td = calc_saturation_vapor_pressure(X_t[:, 2])  # Vapor pressure at dewpoint
        q_a = 0.622 * e_s_td / (X_t[:, 5] - 0.378 * e_s_td)  # Specific humidity of air
        
        # Calculate specific humidity at sea surface (assumed saturation)
        e_s_sst = calc_saturation_vapor_pressure(X_t[:, 4])
        q_s = 0.622 * e_s_sst / (X_t[:, 5] - 0.378 * e_s_sst)  # Specific humidity at sea surface
        
        # Calculate fluxes using bulk formulas
        sensible_heat_flux = X_t[:, 6] * cp * C_H * wind_speed * (X_t[:, 4] - X_t[:, 3])
        latent_heat_flux = X_t[:, 6] * L * C_E * wind_speed * (q_s - q_a)
        
        # Combine into target array
        y_t = np.column_stack([sensible_heat_flux, latent_heat_flux])
        
        # Add to chunks
        X_chunks.append(X_t)
        y_chunks.append(y_t)
        
        # Free memory
        del u10, v10, td2m, t2m, sst, sp, rho, X_t, y_t
        del wind_speed, e_s_td, q_a, e_s_sst, q_s, sensible_heat_flux, latent_heat_flux
    
    # Concatenate all chunks
    logger.info("Concatenating data chunks...")
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

# PINN Model Definition
class OceanHeatFluxPINN(nn.Module):
    def __init__(self, input_dim=7, hidden_dim=128, output_dim=2, num_layers=4):
        super(OceanHeatFluxPINN, self).__init__()
        
        layers = []
        layers.append(nn.Linear(input_dim, hidden_dim))
        layers.append(nn.Tanh())
        
        for _ in range(num_layers - 2):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.Tanh())
        
        layers.append(nn.Linear(hidden_dim, output_dim))
        
        self.network = nn.Sequential(*layers)
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_normal_(module.weight)
            nn.init.zeros_(module.bias)
    
    def forward(self, x):
        return self.network(x)

# Physics-informed loss function
def physics_informed_loss(model, X, y_true, lambda_physics=0.1):
    """
    Calculate physics-informed loss combining data loss and physics constraints
    """
    # Data loss
    y_pred = model(X)
    data_loss = nn.MSELoss()(y_pred, y_true)
    
    # Physics constraints
    # Extract variables from input
    u10 = X[:, 0]
    v10 = X[:, 1]
    td2m = X[:, 2]
    t2m = X[:, 3]
    sst = X[:, 4]
    sp = X[:, 5]
    rho = X[:, 6]
    
    # Calculate wind speed
    wind_speed = torch.sqrt(u10**2 + v10**2)
    
    # Physics constraint: sensible heat flux should be proportional to temperature difference
    temp_diff = sst - t2m
    sensible_flux_pred = y_pred[:, 0]
    
    # Physics constraint: latent heat flux should be related to humidity difference
    latent_flux_pred = y_pred[:, 1]
    
    # Constraint 1: Sensible heat flux sign should match temperature difference sign
    sign_constraint = torch.mean((torch.sign(sensible_flux_pred) - torch.sign(temp_diff))**2)
    
    # Constraint 2: Flux magnitude should increase with wind speed
    wind_constraint = torch.mean((torch.abs(sensible_flux_pred) / (wind_speed + 1e-6) - 
                                 torch.abs(latent_flux_pred) / (wind_speed + 1e-6))**2)
    
    # Total physics loss
    physics_loss = sign_constraint + wind_constraint
    
    # Combined loss
    total_loss = data_loss + lambda_physics * physics_loss
    
    return total_loss, data_loss, physics_loss

# Training function
def train_pinn(model, train_loader, val_loader, num_epochs=100, learning_rate=0.001, 
               lambda_physics=0.1, device='cpu'):
    """Train the PINN model"""
    
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, 
                                                    patience=10, verbose=True)
    
    train_losses = []
    val_losses = []
    
    logger.info(f"Starting training for {num_epochs} epochs on {device}")
    
    for epoch in range(num_epochs):
        model.train()
        train_loss_epoch = 0.0
        train_data_loss_epoch = 0.0
        train_physics_loss_epoch = 0.0
        
        for batch_idx, (X_batch, y_batch) in enumerate(train_loader):
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            
            optimizer.zero_grad()
            
            total_loss, data_loss, physics_loss = physics_informed_loss(
                model, X_batch, y_batch, lambda_physics)
            
            total_loss.backward()
            optimizer.step()
            
            train_loss_epoch += total_loss.item()
            train_data_loss_epoch += data_loss.item()
            train_physics_loss_epoch += physics_loss.item()
        
        # Validation
        model.eval()
        val_loss_epoch = 0.0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                total_loss, _, _ = physics_informed_loss(model, X_batch, y_batch, lambda_physics)
                val_loss_epoch += total_loss.item()
        
        avg_train_loss = train_loss_epoch / len(train_loader)
        avg_val_loss = val_loss_epoch / len(val_loader)
        avg_data_loss = train_data_loss_epoch / len(train_loader)
        avg_physics_loss = train_physics_loss_epoch / len(train_loader)
        
        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)
        
        scheduler.step(avg_val_loss)
        
        if epoch % 10 == 0:
            logger.info(f"Epoch {epoch}/{num_epochs}")
            logger.info(f"  Train Loss: {avg_train_loss:.6f} (Data: {avg_data_loss:.6f}, Physics: {avg_physics_loss:.6f})")
            logger.info(f"  Val Loss: {avg_val_loss:.6f}")
        
        clear_memory()
    
    return train_losses, val_losses

# Main execution function
def main():
    try:
        # Configuration
        config = {
            'data_file': '/projects/nrsc01/air-sea_flux/data/merged_data.nc',  # Update with your file path
            'batch_size': 1024,
            'hidden_dim': 128,
            'num_layers': 4,
            'num_epochs': 200,
            'learning_rate': 0.001,
            'lambda_physics': 0.1,
            'train_split': 0.8,
            'val_split': 0.1
        }
        
        # Device setup
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {device}")
        
        # Load and prepare data
        data_dict = load_data(config['data_file'])
        X, y = prepare_data_in_chunks(data_dict)
        
        # Normalize features
        scaler_X = StandardScaler()
        scaler_y = StandardScaler()
        
        X_scaled = scaler_X.fit_transform(X)
        y_scaled = scaler_y.fit_transform(y)
        
        # Convert to tensors
        X_tensor = torch.FloatTensor(X_scaled)
        y_tensor = torch.FloatTensor(y_scaled)
        
        # Create dataset and split
        dataset = TensorDataset(X_tensor, y_tensor)
        
        train_size = int(config['train_split'] * len(dataset))
        val_size = int(config['val_split'] * len(dataset))
        test_size = len(dataset) - train_size - val_size
        
        train_dataset, val_dataset, test_dataset = random_split(
            dataset, [train_size, val_size, test_size])
        
        # Create data loaders
        train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], 
                                shuffle=True, num_workers=0)
        val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], 
                              shuffle=False, num_workers=0)
        test_loader = DataLoader(test_dataset, batch_size=config['batch_size'], 
                               shuffle=False, num_workers=0)
        
        logger.info(f"Dataset sizes - Train: {train_size}, Val: {val_size}, Test: {test_size}")
        
        # Initialize model
        model = OceanHeatFluxPINN(
            input_dim=X.shape[1],
            hidden_dim=config['hidden_dim'],
            output_dim=y.shape[1],
            num_layers=config['num_layers']
        ).to(device)
        
        logger.info(f"Model initialized with {sum(p.numel() for p in model.parameters())} parameters")
        
        # Train model
        train_losses, val_losses = train_pinn(
            model, train_loader, val_loader,
            num_epochs=config['num_epochs'],
            learning_rate=config['learning_rate'],
            lambda_physics=config['lambda_physics'],
            device=device
        )
        
        # Evaluate on test set
        model.eval()
        test_loss = 0.0
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            for X_batch, y_batch in test_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                y_pred = model(X_batch)
                
                # Convert back to original scale for evaluation
                y_pred_orig = scaler_y.inverse_transform(y_pred.cpu().numpy())
                y_true_orig = scaler_y.inverse_transform(y_batch.cpu().numpy())
                
                all_predictions.append(y_pred_orig)
                all_targets.append(y_true_orig)
                
                loss = nn.MSELoss()(y_pred, y_batch)
                test_loss += loss.item()
        
        avg_test_loss = test_loss / len(test_loader)
        logger.info(f"Test Loss: {avg_test_loss:.6f}")
        
        # Concatenate all predictions and targets
        all_predictions = np.vstack(all_predictions)
        all_targets = np.vstack(all_targets)
        
        # Calculate metrics
        from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
        
        # Sensible heat flux metrics
        mse_sensible = mean_squared_error(all_targets[:, 0], all_predictions[:, 0])
        mae_sensible = mean_absolute_error(all_targets[:, 0], all_predictions[:, 0])
        r2_sensible = r2_score(all_targets[:, 0], all_predictions[:, 0])
        
        # Latent heat flux metrics
        mse_latent = mean_squared_error(all_targets[:, 1], all_predictions[:, 1])
        mae_latent = mean_absolute_error(all_targets[:, 1], all_predictions[:, 1])
        r2_latent = r2_score(all_targets[:, 1], all_predictions[:, 1])
        
        logger.info("=== Model Performance ===")
        logger.info(f"Sensible Heat Flux - MSE: {mse_sensible:.2f}, MAE: {mae_sensible:.2f}, R²: {r2_sensible:.4f}")
        logger.info(f"Latent Heat Flux - MSE: {mse_latent:.2f}, MAE: {mae_latent:.2f}, R²: {r2_latent:.4f}")
        
        # Save model and results
        save_results(model, scaler_X, scaler_y, train_losses, val_losses, 
                    all_predictions, all_targets, config)
        
        # Create visualizations
        create_visualizations(train_losses, val_losses, all_predictions, all_targets)
        
        logger.info("Training completed successfully!")
        
    except Exception as e:
        logger.error(f"Error in main execution: {str(e)}")
        raise

def save_results(model, scaler_X, scaler_y, train_losses, val_losses, 
                predictions, targets, config):
    """Save model, scalers, and results"""
    
    # Create results directory
    results_dir = f"results_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    os.makedirs(results_dir, exist_ok=True)
    
    # Save model
    model_path = os.path.join(results_dir, 'pinn_model.pth')
    torch.save({
        'model_state_dict': model.state_dict(),
        'model_config': {
            'input_dim': model.network[0].in_features,
            'hidden_dim': config['hidden_dim'],
            'output_dim': model.network[-1].out_features,
            'num_layers': config['num_layers']
        }
    }, model_path)
    
    # Save scalers
    scaler_path = os.path.join(results_dir, 'scalers.pkl')
    with open(scaler_path, 'wb') as f:
        pickle.dump({'scaler_X': scaler_X, 'scaler_y': scaler_y}, f)
    
    # Save training history
    history_path = os.path.join(results_dir, 'training_history.pkl')
    with open(history_path, 'wb') as f:
        pickle.dump({
            'train_losses': train_losses,
            'val_losses': val_losses,
            'predictions': predictions,
            'targets': targets
        }, f)
    
    # Save configuration
    config_path = os.path.join(results_dir, 'config.json')
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2, default=str)
    
    logger.info(f"Results saved to {results_dir}")

def create_visualizations(train_losses, val_losses, predictions, targets):
    """Create and save visualization plots"""
    
    # Set up the plotting style
    plt.style.use('default')
    
    # Create figure with subplots
    fig = plt.figure(figsize=(20, 15))
    
    # 1. Training history
    ax1 = plt.subplot(3, 3, 1)
    plt.plot(train_losses, label='Training Loss', alpha=0.8)
    plt.plot(val_losses, label='Validation Loss', alpha=0.8)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training History')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.yscale('log')
    
    # 2. Sensible heat flux scatter plot
    ax2 = plt.subplot(3, 3, 2)
    plt.scatter(targets[:, 0], predictions[:, 0], alpha=0.5, s=1)
    min_val = min(targets[:, 0].min(), predictions[:, 0].min())
    max_val = max(targets[:, 0].max(), predictions[:, 0].max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.8)
    plt.xlabel('True Sensible Heat Flux (W/m²)')
    plt.ylabel('Predicted Sensible Heat Flux (W/m²)')
    plt.title('Sensible Heat Flux: Predicted vs True')
    plt.grid(True, alpha=0.3)
    
    # 3. Latent heat flux scatter plot
    ax3 = plt.subplot(3, 3, 3)
    plt.scatter(targets[:, 1], predictions[:, 1], alpha=0.5, s=1)
    min_val = min(targets[:, 1].min(), predictions[:, 1].min())
    max_val = max(targets[:, 1].max(), predictions[:, 1].max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.8)
    plt.xlabel('True Latent Heat Flux (W/m²)')
    plt.ylabel('Predicted Latent Heat Flux (W/m²)')
    plt.title('Latent Heat Flux: Predicted vs True')
    plt.grid(True, alpha=0.3)
    
    # 4. Residuals for sensible heat flux
    ax4 = plt.subplot(3, 3, 4)
    residuals_sensible = predictions[:, 0] - targets[:, 0]
    plt.scatter(targets[:, 0], residuals_sensible, alpha=0.5, s=1)
    plt.axhline(y=0, color='r', linestyle='--', alpha=0.8)
    plt.xlabel('True Sensible Heat Flux (W/m²)')
    plt.ylabel('Residuals (W/m²)')
    plt.title('Sensible Heat Flux Residuals')
    plt.grid(True, alpha=0.3)
    
    # 5. Residuals for latent heat flux
    ax5 = plt.subplot(3, 3, 5)
    residuals_latent = predictions[:, 1] - targets[:, 1]
    plt.scatter(targets[:, 1], residuals_latent, alpha=0.5, s=1)
    plt.axhline(y=0, color='r', linestyle='--', alpha=0.8)
    plt.xlabel('True Latent Heat Flux (W/m²)')
    plt.ylabel('Residuals (W/m²)')
    plt.title('Latent Heat Flux Residuals')
    plt.grid(True, alpha=0.3)
    
    # 6. Distribution of sensible heat flux
    ax6 = plt.subplot(3, 3, 6)
    plt.hist(targets[:, 0], bins=50, alpha=0.7, label='True', density=True)
    plt.hist(predictions[:, 0], bins=50, alpha=0.7, label='Predicted', density=True)
    plt.xlabel('Sensible Heat Flux (W/m²)')
    plt.ylabel('Density')
    plt.title('Sensible Heat Flux Distribution')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 7. Distribution of latent heat flux
    ax7 = plt.subplot(3, 3, 7)
    plt.hist(targets[:, 1], bins=50, alpha=0.7, label='True', density=True)
    plt.hist(predictions[:, 1], bins=50, alpha=0.7, label='Predicted', density=True)
    plt.xlabel('Latent Heat Flux (W/m²)')
    plt.ylabel('Density')
    plt.title('Latent Heat Flux Distribution')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 8. Error distribution for sensible heat flux
    ax8 = plt.subplot(3, 3, 8)
    plt.hist(residuals_sensible, bins=50, alpha=0.7, density=True)
    plt.axvline(x=0, color='r', linestyle='--', alpha=0.8)
    plt.xlabel('Residuals (W/m²)')
    plt.ylabel('Density')
    plt.title('Sensible Heat Flux Error Distribution')
    plt.grid(True, alpha=0.3)
    
    # 9. Error distribution for latent heat flux
    ax9 = plt.subplot(3, 3, 9)
    plt.hist(residuals_latent, bins=50, alpha=0.7, density=True)
    plt.axvline(x=0, color='r', linestyle='--', alpha=0.8)
    plt.xlabel('Residuals (W/m²)')
    plt.ylabel('Density')
    plt.title('Latent Heat Flux Error Distribution')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save the plot
    plot_path = f"pinn_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.show()
    
    logger.info(f"Visualizations saved to {plot_path}")

def load_trained_model(model_path, scaler_path):
    """Load a trained model and scalers for inference"""
    
    # Load model
    checkpoint = torch.load(model_path, map_location='cpu')
    model_config = checkpoint['model_config']
    
    model = OceanHeatFluxPINN(
        input_dim=model_config['input_dim'],
        hidden_dim=model_config['hidden_dim'],
        output_dim=model_config['output_dim'],
        num_layers=model_config['num_layers']
    )
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # Load scalers
    with open(scaler_path, 'rb') as f:
        scalers = pickle.load(f)
    
    return model, scalers['scaler_X'], scalers['scaler_y']

def predict_heat_fluxes(model, scaler_X, scaler_y, input_data, device='cpu'):
    """Make predictions using the trained model"""
    
    model.to(device)
    model.eval()
    
    # Normalize input data
    input_scaled = scaler_X.transform(input_data)
    input_tensor = torch.FloatTensor(input_scaled).to(device)
    
    with torch.no_grad():
        predictions_scaled = model(input_tensor)
        predictions = scaler_y.inverse_transform(predictions_scaled.cpu().numpy())
    
    return predictions

def create_spatial_visualization(data_dict, predictions, time_idx=0):
    """Create spatial visualization of heat flux predictions"""
    
    # Get spatial dimensions
    lat = data_dict['latitude']
    lon = data_dict['longitude']
    ocean_mask = data_dict['ocean_mask']
    
    # Create meshgrid
    lon_grid, lat_grid = np.meshgrid(lon, lat)
    
    # Initialize arrays for spatial data
    if len(ocean_mask.shape) == 3:
        mask_2d = ocean_mask[time_idx]
    else:
        mask_2d = ocean_mask
    
    sensible_flux_2d = np.full(mask_2d.shape, np.nan)
    latent_flux_2d = np.full(mask_2d.shape, np.nan)
    
    # Fill in predictions for ocean points
    ocean_indices = np.where(mask_2d)
    if len(ocean_indices[0]) > 0:
        sensible_flux_2d[ocean_indices] = predictions[:len(ocean_indices[0]), 0]
        latent_flux_2d[ocean_indices] = predictions[:len(ocean_indices[0]), 1]
    
    # Apply smoothing for better visualization
    sensible_flux_smooth = apply_smoothing(sensible_flux_2d, sigma=1.0)
    latent_flux_smooth = apply_smoothing(latent_flux_2d, sigma=1.0)
    
    # Create the spatial plot
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Sensible heat flux - raw
    im1 = axes[0, 0].contourf(lon_grid, lat_grid, sensible_flux_2d, 
                              levels=20, cmap='RdBu_r', extend='both')
    axes[0, 0].set_title('Sensible Heat Flux (Raw)')
    axes[0, 0].set_xlabel('Longitude')
    axes[0, 0].set_ylabel('Latitude')
    plt.colorbar(im1, ax=axes[0, 0], label='W/m²')
    
    # Sensible heat flux - smoothed
    im2 = axes[0, 1].contourf(lon_grid, lat_grid, sensible_flux_smooth, 
                              levels=20, cmap='RdBu_r', extend='both')
    axes[0, 1].set_title('Sensible Heat Flux (Smoothed)')
    axes[0, 1].set_xlabel('Longitude')
    axes[0, 1].set_ylabel('Latitude')
    plt.colorbar(im2, ax=axes[0, 1], label='W/m²')
    
    # Latent heat flux - raw
    im3 = axes[1, 0].contourf(lon_grid, lat_grid, latent_flux_2d, 
                              levels=20, cmap='viridis', extend='both')
    axes[1, 0].set_title('Latent Heat Flux (Raw)')
    axes[1, 0].set_xlabel('Longitude')
    axes[1, 0].set_ylabel('Latitude')
    plt.colorbar(im3, ax=axes[1, 0], label='W/m²')
    
    # Latent heat flux - smoothed
    im4 = axes[1, 1].contourf(lon_grid, lat_grid, latent_flux_smooth, 
                              levels=20, cmap='viridis', extend='both')
    axes[1, 1].set_title('Latent Heat Flux (Smoothed)')
    axes[1, 1].set_xlabel('Longitude')
    axes[1, 1].set_ylabel('Latitude')
    plt.colorbar(im4, ax=axes[1, 1], label='W/m²')
    
    plt.tight_layout()
    
    # Save the spatial plot
    spatial_plot_path = f"spatial_heat_flux_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
    plt.savefig(spatial_plot_path, dpi=300, bbox_inches='tight')
    plt.show()
    
    logger.info(f"Spatial visualization saved to {spatial_plot_path}")
    
    return sensible_flux_2d, latent_flux_2d

def analyze_model_sensitivity(model, scaler_X, scaler_y, sample_data, device='cpu'):
    """Analyze model sensitivity to input variables"""
    
    model.to(device)
    model.eval()
    
    # Variable names
    var_names = ['u10', 'v10', 'd2m', 't2m', 'sst', 'sp', 'rho']
    
    # Take a sample of data points
    n_samples = min(1000, len(sample_data))
    sample_indices = np.random.choice(len(sample_data), n_samples, replace=False)
    base_data = sample_data[sample_indices]
    
    # Calculate baseline predictions
    base_scaled = scaler_X.transform(base_data)
    base_tensor = torch.FloatTensor(base_scaled).to(device)
    
    with torch.no_grad():
        base_pred_scaled = model(base_tensor)
        base_pred = scaler_y.inverse_transform(base_pred_scaled.cpu().numpy())
    
    # Analyze sensitivity for each variable
    sensitivities = {}
    
    for i, var_name in enumerate(var_names):
        logger.info(f"Analyzing sensitivity for {var_name}...")
        
        # Create perturbations (±10% of the variable's range)
        var_range = np.ptp(base_data[:, i])
        perturbation = 0.1 * var_range
        
        # Positive perturbation
        perturbed_data_pos = base_data.copy()
        perturbed_data_pos[:, i] += perturbation
        
        # Negative perturbation
        perturbed_data_neg = base_data.copy()
        perturbed_data_neg[:, i] -= perturbation
        
        # Make predictions
        with torch.no_grad():
            # Positive perturbation
            pos_scaled = scaler_X.transform(perturbed_data_pos)
            pos_tensor = torch.FloatTensor(pos_scaled).to(device)
            pos_pred_scaled = model(pos_tensor)
            pos_pred = scaler_y.inverse_transform(pos_pred_scaled.cpu().numpy())
            
            # Negative perturbation
            neg_scaled = scaler_X.transform(perturbed_data_neg)
            neg_tensor = torch.FloatTensor(neg_scaled).to(device)
            neg_pred_scaled = model(neg_tensor)
            neg_pred = scaler_y.inverse_transform(neg_pred_scaled.cpu().numpy())
        
        # Calculate sensitivity (gradient approximation)
        sensitivity_sensible = (pos_pred[:, 0] - neg_pred[:, 0]) / (2 * perturbation)
        sensitivity_latent = (pos_pred[:, 1] - neg_pred[:, 1]) / (2 * perturbation)
        
        sensitivities[var_name] = {
            'sensible': {
                'mean': np.mean(sensitivity_sensible),
                'std': np.std(sensitivity_sensible),
                'median': np.median(sensitivity_sensible)
            },
            'latent': {
                'mean': np.mean(sensitivity_latent),
                'std': np.std(sensitivity_latent),
                'median': np.median(sensitivity_latent)
            }
        }
    
    # Create sensitivity visualization
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Sensible heat flux sensitivity
    sensible_means = [sensitivities[var]['sensible']['mean'] for var in var_names]
    sensible_stds = [sensitivities[var]['sensible']['std'] for var in var_names]
    
    ax1.bar(var_names, sensible_means, yerr=sensible_stds, capsize=5, alpha=0.7)
    ax1.set_title('Model Sensitivity - Sensible Heat Flux')
    ax1.set_ylabel('Sensitivity (W/m² per unit change)')
    ax1.tick_params(axis='x', rotation=45)
    ax1.grid(True, alpha=0.3)
    
    # Latent heat flux sensitivity
    latent_means = [sensitivities[var]['latent']['mean'] for var in var_names]
    latent_stds = [sensitivities[var]['latent']['std'] for var in var_names]
    
    ax2.bar(var_names, latent_means, yerr=latent_stds, capsize=5, alpha=0.7, color='orange')
    ax2.set_title('Model Sensitivity - Latent Heat Flux')
    ax2.set_ylabel('Sensitivity (W/m² per unit change)')
    ax2.tick_params(axis='x', rotation=45)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save sensitivity plot
    sensitivity_plot_path = f"model_sensitivity_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
    plt.savefig(sensitivity_plot_path, dpi=300, bbox_inches='tight')
    plt.show()
    
    # Log sensitivity results
    logger.info("=== Model Sensitivity Analysis ===")
    for var_name in var_names:
        logger.info(f"{var_name}:")
        logger.info(f"  Sensible Heat Flux: {sensitivities[var_name]['sensible']['mean']:.4f} ± {sensitivities[var_name]['sensible']['std']:.4f}")
        logger.info(f"  Latent Heat Flux: {sensitivities[var_name]['latent']['mean']:.4f} ± {sensitivities[var_name]['latent']['std']:.4f}")
    
    return sensitivities

def validate_physics_constraints(predictions, input_data):
    """Validate that predictions follow physical constraints"""
    
    logger.info("Validating physics constraints...")
    
    # Extract input variables
    u10 = input_data[:, 0]
    v10 = input_data[:, 1]
    td2m = input_data[:, 2]
    t2m = input_data[:, 3]
    sst = input_data[:, 4]
    
    # Extract predictions
    sensible_flux = predictions[:, 0]
    latent_flux = predictions[:, 1]
    
    # Calculate derived quantities
    wind_speed = np.sqrt(u10**2 + v10**2)
    temp_diff = sst - t2m
    
    # Constraint 1: Sensible heat flux should correlate with temperature difference
    correlation_temp = np.corrcoef(sensible_flux, temp_diff)[0, 1]
    
    # Constraint 2: Heat fluxes should correlate with wind speed
    correlation_wind_sensible = np.corrcoef(np.abs(sensible_flux), wind_speed)[0, 1]
    correlation_wind_latent = np.corrcoef(np.abs(latent_flux), wind_speed)[0, 1]
    
    # Constraint 3: Sign consistency for sensible heat flux
    sign_consistency = np.mean(np.sign(sensible_flux) == np.sign(temp_diff))
    
    # Constraint 4: Reasonable magnitude ranges
    sensible_range = (np.min(sensible_flux), np.max(sensible_flux))
    latent_range = (np.min(latent_flux), np.max(latent_flux))
    
    # Log results
    logger.info("=== Physics Constraint Validation ===")
    logger.info(f"Temperature-Sensible Heat Flux Correlation: {correlation_temp:.4f}")
    logger.info(f"Wind Speed-Sensible Heat Flux Correlation: {correlation_wind_sensible:.4f}")
    logger.info(f"Wind Speed-Latent Heat Flux Correlation: {correlation_wind_latent:.4f}")
    logger.info(f"Sign Consistency (Sensible Heat Flux): {sign_consistency:.4f}")
    logger.info(f"Sensible Heat Flux Range: {sensible_range[0]:.2f} to {sensible_range[1]:.2f} W/m²")
    logger.info(f"Latent Heat Flux Range: {latent_range[0]:.2f} to {latent_range[1]:.2f} W/m²")
    
    # Create validation plots
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Temperature difference vs sensible heat flux
    axes[0, 0].scatter(temp_diff, sensible_flux, alpha=0.5, s=1)
    axes[0, 0].set_xlabel('Temperature Difference (SST - T2M) [K]')
    axes[0, 0].set_ylabel('Sensible Heat Flux [W/m²]')
    axes[0, 0].set_title(f'Temp Diff vs Sensible Heat Flux (r={correlation_temp:.3f})')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Wind speed vs sensible heat flux magnitude
    axes[0, 1].scatter(wind_speed, np.abs(sensible_flux), alpha=0.5, s=1)
    axes[0, 1].set_xlabel('Wind Speed [m/s]')
    axes[0, 1].set_ylabel('|Sensible Heat Flux| [W/m²]')
    axes[0, 1].set_title(f'Wind Speed vs |Sensible Heat Flux| (r={correlation_wind_sensible:.3f})')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Wind speed vs latent heat flux magnitude
    axes[1, 0].scatter(wind_speed, np.abs(latent_flux), alpha=0.5, s=1)
    axes[1, 0].set_xlabel('Wind Speed [m/s]')
    axes[1, 0].set_ylabel('|Latent Heat Flux| [W/m²]')
    axes[1, 0].set_title(f'Wind Speed vs |Latent Heat Flux| (r={correlation_wind_latent:.3f})')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Sign consistency plot
    sign_match = (np.sign(sensible_flux) == np.sign(temp_diff)).astype(int)
    axes[1, 1].hist(sign_match, bins=2, alpha=0.7, edgecolor='black')
    axes[1, 1].set_xlabel('Sign Match (0=No, 1=Yes)')
    axes[1, 1].set_ylabel('Count')
    axes[1, 1].set_title(f'Sign Consistency: {sign_consistency:.3f}')
    axes[1, 1].set_xticks([0, 1])
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save validation plot
    validation_plot_path = f"physics_validation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
    plt.savefig(validation_plot_path, dpi=300, bbox_inches='tight')
    plt.show()
    
    return {
        'temp_correlation': correlation_temp,
        'wind_sensible_correlation': correlation_wind_sensible,
        'wind_latent_correlation': correlation_wind_latent,
        'sign_consistency': sign_consistency,
        'sensible_range': sensible_range,
        'latent_range': latent_range
    }

def run_inference_example():
    """Example of how to run inference with a trained model"""
    
    logger.info("Running inference example...")
    
    # Example input data (replace with your actual data)
    # Format: [u10, v10, d2m, t2m, sst, sp, rho]
    example_input = np.array([
        [5.0, 3.0, 285.0, 288.0, 290.0, 101325.0, 1.2],  # Example 1
        [10.0, -2.0, 280.0, 285.0, 295.0, 101000.0, 1.15], # Example 2
        [2.0, 1.0, 290.0, 292.0, 288.0, 101500.0, 1.25],   # Example 3
    ])
    
    try:
        # Load trained model (update paths as needed)
        model_path = 'results_latest/pinn_model.pth'
        scaler_path = 'results_latest/scalers.pkl'
        
        if os.path.exists(model_path) and os.path.exists(scaler_path):
            model, scaler_X, scaler_y = load_trained_model(model_path, scaler_path)
            
            # Make predictions
            predictions = predict_heat_fluxes(model, scaler_X, scaler_y, example_input)
            
            logger.info("=== Inference Results ===")
            for i, pred in enumerate(predictions):
                logger.info(f"Example {i+1}:")
                logger.info(f"  Input: u10={example_input[i,0]:.1f}, v10={example_input[i,1]:.1f}, "
                           f"d2m={example_input[i,2]:.1f}, t2m={example_input[i,3]:.1f}, "
                           f"sst={example_input[i,4]:.1f}, sp={example_input[i,5]:.0f}, rho={example_input[i,6]:.2f}")
                logger.info(f"  Predicted Sensible Heat Flux: {pred[0]:.2f} W/m²")
                logger.info(f"  Predicted Latent Heat Flux: {pred[1]:.2f} W/m²")
        else:
            logger.warning("Trained model not found. Please run training first.")
            
    except Exception as e:
        logger.error(f"Error in inference example: {str(e)}")

def create_time_series_analysis(data_dict, model, scaler_X, scaler_y, lat_idx=None, lon_idx=None):
    """Create time series analysis for a specific location"""
    
    if lat_idx is None:
        lat_idx = len(data_dict['latitude']) // 2
    if lon_idx is None:
        lon_idx = len(data_dict['longitude']) // 2
    
    logger.info(f"Creating time series analysis for lat_idx={lat_idx}, lon_idx={lon_idx}")
    
    # Extract time series data for the specific location
    input_vars = ['u10', 'v10', 'd2m', 't2m', 'sst', 'sp', 'rho']
    time_series_data = []
    
    for t in range(len(data_dict['time'])):
        row = []
        for var in input_vars:
            if var in data_dict:
                value = data_dict[var][t, lat_idx, lon_idx]
                row.append(value)
        time_series_data.append(row)
    
    time_series_data = np.array(time_series_data)
    
    # Remove NaN values
    valid_mask = ~np.isnan(time_series_data).any(axis=1)
    time_series_data = time_series_data[valid_mask]
    valid_times = data_dict['time'][valid_mask]
    
    if len(time_series_data) == 0:
        logger.warning("No valid data for time series analysis")
        return
    
    # Make predictions
    predictions = predict_heat_fluxes(model, scaler_X, scaler_y, time_series_data)
    
    # Create time series plot
    fig, axes = plt.subplots(3, 2, figsize=(15, 12))
    
    # Convert times to datetime for plotting
    if hasattr(valid_times[0], 'astype'):
        plot_times = pd.to_datetime(valid_times)
    else:
        plot_times = valid_times
    
    # Input variables
    var_names = ['Wind Speed (u10)', 'Wind Speed (v10)', 'Dewpoint (d2m)', 
                 'Temperature (t2m)', 'SST', 'Surface Pressure (sp)', 'Density (rho)']
    
    for i, (var_name, var_data) in enumerate(zip(var_names[:4], time_series_data.T[:4])):
        row, col = i // 2, i % 2
        axes[row, col].plot(plot_times, var_data)
        axes[row, col].set_title(var_name)
        axes[row, col].grid(True, alpha=0.3)
        axes[row, col].tick_params(axis='x', rotation=45)
    
    # Heat flux predictions
    axes[2, 0].plot(plot_times, predictions[:, 0], label='Sensible Heat Flux', color='red')
    axes[2, 0].set_title('Predicted Sensible Heat Flux')
    axes[2, 0].set_ylabel('W/m²')
    axes[2, 0].grid(True, alpha=0.3)
    axes[2, 0].tick_params(axis='x', rotation=45)
    
    axes[2, 1].plot(plot_times, predictions[:, 1], label='Latent Heat Flux', color='blue')
    axes[2, 1].set_title('Predicted Latent Heat Flux')
    axes[2, 1].set_ylabel('W/m²')
    axes[2, 1].grid(True, alpha=0.3)
    axes[2, 1].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    
    # Save time series plot
    time_series_plot_path = f"time_series_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
    plt.savefig(time_series_plot_path, dpi=300, bbox_inches='tight')
    plt.show()
    
    logger.info(f"Time series analysis saved to {time_series_plot_path}")
    
    return predictions, valid_times

def export_predictions_to_netcdf(data_dict, predictions, output_path):
    """Export predictions back to NetCDF format"""
    
    logger.info(f"Exporting predictions to {output_path}")
    
    # Get dimensions
    time_dim = len(data_dict['time'])
    lat_dim = len(data_dict['latitude'])
    lon_dim = len(data_dict['longitude'])
    
    # Initialize output arrays
    sensible_flux_3d = np.full((time_dim, lat_dim, lon_dim), np.nan)
    latent_flux_3d = np.full((time_dim, lat_dim, lon_dim), np.nan)
    
    # Get ocean mask
    ocean_mask = data_dict['ocean_mask']
    
    # Fill predictions into 3D arrays
    pred_idx = 0
    for t in range(time_dim):
        if len(ocean_mask.shape) == 3:
            mask_2d = ocean_mask[t]
        else:
            mask_2d = ocean_mask
        
        ocean_points = np.sum(mask_2d)
        if ocean_points > 0 and pred_idx < len(predictions):
            # Get indices of ocean points
            ocean_indices = np.where(mask_2d)
            
            # Fill in predictions
            end_idx = min(pred_idx + ocean_points, len(predictions))
            n_points = end_idx - pred_idx
            
            if n_points > 0:
                sensible_flux_3d[t][ocean_indices[0][:n_points], ocean_indices[1][:n_points]] = \
                    predictions[pred_idx:end_idx, 0]
                latent_flux_3d[t][ocean_indices[0][:n_points], ocean_indices[1][:n_points]] = \
                    predictions[pred_idx:end_idx, 1]
                
                pred_idx = end_idx
    
    # Create xarray dataset
    ds = xr.Dataset(
        {
            'sensible_heat_flux': (['time', 'latitude', 'longitude'], sensible_flux_3d),
            'latent_heat_flux': (['time', 'latitude', 'longitude'], latent_flux_3d),
            'ocean_mask': (['latitude', 'longitude'], ocean_mask[0] if len(ocean_mask.shape) == 3 else ocean_mask),
        },
        coords={
            'time': data_dict['time'],
            'latitude': data_dict['latitude'],
            'longitude': data_dict['longitude'],
        },
        attrs={
            'title': 'Ocean Heat Flux Predictions from PINN',
            'description': 'Sensible and latent heat flux predictions over ocean areas',
            'created': datetime.now().isoformat(),
            'units_sensible': 'W/m²',
            'units_latent': 'W/m²',
        }
    )
    
    # Add variable attributes
    ds['sensible_heat_flux'].attrs = {
        'long_name': 'Sensible Heat Flux',
        'units': 'W/m²',
        'description': 'Upward sensible heat flux from ocean to atmosphere'
    }
    
    ds['latent_heat_flux'].attrs = {
        'long_name': 'Latent Heat Flux',
        'units': 'W/m²',
        'description': 'Upward latent heat flux from ocean to atmosphere'
    }
    
    ds['ocean_mask'].attrs = {
        'long_name': 'Ocean Mask',
        'description': 'Boolean mask indicating ocean grid points'
    }
    
    # Save to NetCDF
    ds.to_netcdf(output_path)
    ds.close()
    
    logger.info(f"Predictions exported to {output_path}")

def run_comprehensive_analysis():
    """Run a comprehensive analysis including training and evaluation"""
    
    logger.info("Starting comprehensive PINN analysis...")
    
    try:
        # Update this path to your actual data file
        data_file = "/projects/nrsc01/air-sea_flux/data/merged_data.nc"  # Replace with actual path
        
        if not os.path.exists(data_file):
            logger.error(f"Data file not found: {data_file}")
            logger.info("Please update the data_file path in the run_comprehensive_analysis function")
            return
        
        # Configuration
        config = {
            'data_file': data_file,
            'batch_size': 1024,
            'hidden_dim': 128,
            'num_layers': 4,
            'num_epochs': 200,
            'learning_rate': 0.001,
            'lambda_physics': 0.1,
            'train_split': 0.8,
            'val_split': 0.1
        }
        
        # Run main training
        main()
        
        # Find the most recent results directory
        result_dirs = [d for d in os.listdir('.') if d.startswith('results_')]
        if result_dirs:
            latest_results = sorted(result_dirs)[-1]
            model_path = os.path.join(latest_results, 'pinn_model.pth')
            scaler_path = os.path.join(latest_results, 'scalers.pkl')
            
            # Load the trained model
            model, scaler_X, scaler_y = load_trained_model(model_path, scaler_path)
            
            # Load data for additional analysis
            data_dict = load_data(config['data_file'])
            X, y = prepare_data_in_chunks(data_dict)
            
            # Run sensitivity analysis
            analyze_model_sensitivity(model, scaler_X, scaler_y, X[:10000])  # Use subset for speed
            
            # Make predictions on test data
            predictions = predict_heat_fluxes(model, scaler_X, scaler_y, X[:10000])
            
            # Validate physics constraints
            validate_physics_constraints(predictions, X[:10000])
            
            # Create spatial visualization (for first time step)
            create_spatial_visualization(data_dict, predictions[:1000], time_idx=0)
            
            # Export predictions to NetCDF
            output_path = f"heat_flux_predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.nc"
            export_predictions_to_netcdf(data_dict, predictions, output_path)
            
            logger.info("Comprehensive analysis completed successfully!")
        
    except Exception as e:
        logger.error(f"Error in comprehensive analysis: {str(e)}")
        raise

# Command line interface
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Ocean Heat Flux PINN Training and Analysis')
    parser.add_argument('--mode', choices=['train', 'inference', 'analysis'], default='train',
                       help='Mode to run: train, inference, or analysis')
    parser.add_argument('--data_file', type=str, required=True,
                       help='Path to the NetCDF data file')
    parser.add_argument('--model_path', type=str,
                       help='Path to trained model (for inference mode)')
    parser.add_argument('--scaler_path', type=str,
                       help='Path to scalers (for inference mode)')
    parser.add_argument('--epochs', type=int, default=200,
                       help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=1024,
                       help='Batch size for training')
    parser.add_argument('--learning_rate', type=float, default=0.001,
                       help='Learning rate')
    parser.add_argument('--lambda_physics', type=float, default=0.1,
                       help='Physics loss weight')
    
    args = parser.parse_args()
    
    if args.mode == 'train':
        # Update config with command line arguments
        config = {
            'data_file': args.data_file,
            'batch_size': args.batch_size,
            'hidden_dim': 128,
            'num_layers': 4,
            'num_epochs': args.epochs,
            'learning_rate': args.learning_rate,
            'lambda_physics': args.lambda_physics,
            'train_split': 0.8,
            'val_split': 0.1
        }
        
        # Update the global config and run main
        globals()['config'] = config
        main()
        
    elif args.mode == 'inference':
        if not args.model_path or not args.scaler_path:
            logger.error("Model path and scaler path required for inference mode")
            exit(1)
        
        run_inference_example()
        
    elif args.mode == 'analysis':
        run_comprehensive_analysis()




