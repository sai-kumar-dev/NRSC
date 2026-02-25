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
        return data_dict
    
    except Exception as e:
        logger.error(f"Error loading data: {str(e)}")
        raise

# Function to prepare data in chunks
def prepare_data_in_chunks(data_dict, chunk_size=1000000):
    """Prepare data in chunks to avoid memory issues"""
    logger.info("Preparing data in chunks to avoid memory issues...")
    
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
    
    # Calculate total number of points
    total_points = time_dim * lat_dim * lon_dim
    logger.info(f"Total data points: {total_points}")
    
    # Reshape all variables to 2D arrays (time, lat*lon)
    reshaped_data = {}
    for var in input_vars:
        if var in data_dict:
            reshaped_data[var] = data_dict[var].reshape(time_dim, -1)
    
    # Process data in chunks
    X_chunks = []
    y_chunks = []
    
    # Constants for bulk formulas
    cp = 1004.0  # Specific heat capacity of air at constant pressure (J/kg/K)
    L = 2.5e6  # Latent heat of vaporization (J/kg)
    C_H = 1.3e-3  # Transfer coefficient for sensible heat
    C_E = 1.5e-3  # Transfer coefficient for latent heat
    
    # Calculate saturation vapor pressure function
    def calc_saturation_vapor_pressure(temp_k):
        temp_c = temp_k - 273.15
        return 6.112 * np.exp(17.67 * temp_c / (temp_c + 243.5)) * 100  # in Pa
    
    # Process each time step
    for t in range(time_dim):
        if t % 100 == 0:
            logger.info(f"Processing time step {t+1}/{time_dim}")
        
        # Extract data for this time step
        u10 = reshaped_data['u10'][t]
        v10 = reshaped_data['v10'][t]
        td2m = reshaped_data['d2m'][t]
        t2m = reshaped_data['t2m'][t]
        sst = reshaped_data['sst'][t]
        sp = reshaped_data['sp'][t]
        rho = reshaped_data['rho'][t]
        
        # Stack features
        X_t = np.column_stack([u10, v10, td2m, t2m, sst, sp, rho])
        
        # Remove NaN values
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
    
    logger.info(f"Final X shape: {X.shape}")
    logger.info(f"Final y shape: {y.shape}")
    
    # Log some statistics
    logger.info(f"Sensible heat flux - Mean: {np.mean(y[:, 0]):.2f}, Min: {np.min(y[:, 0]):.2f}, Max: {np.max(y[:, 0]):.2f}")
    logger.info(f"Latent heat flux - Mean: {np.mean(y[:, 1]):.2f}, Min: {np.min(y[:, 1]):.2f}, Max: {np.max(y[:, 1]):.2f}")
    
    # Free memory
    del X_chunks, y_chunks, reshaped_data
    
    return X, y

# Function to prepare data (original version - not used but kept for reference)
def prepare_data(data_dict):
    logger.info("Preparing data for training...")
    
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
    
    # Prepare input features
    # Reshape data to (time*lat*lon, features)
    X_list = []
    for var in input_vars:
        if var in data_dict:
            # Reshape to (time, lat*lon) then to (time*lat*lon, 1)
            reshaped = data_dict[var].reshape(time_dim, -1)
            X_list.append(reshaped)
    
    # Stack along time dimension
    X_stacked = np.stack(X_list, axis=-1)  # Shape: (time, lat*lon, features)
    
    # Reshape to (time*lat*lon, features)
    X = X_stacked.reshape(-1, len(input_vars))
    logger.info(f"X shape after reshaping: {X.shape}")
    
    # Remove NaN values
    mask = ~np.isnan(X).any(axis=1)
    X = X[mask]
    logger.info(f"X shape after removing NaNs: {X.shape}")
    
    # Calculate target variables (sensible and latent heat fluxes) using physics
    # We'll use these as "ground truth" for training
    logger.info("Calculating target variables using physics-based formulas...")
    u10 = X[:, 0]  # 10m u wind
    v10 = X[:, 1]  # 10m v wind
    td2m = X[:, 2]  # 2m dewpoint temperature
    t2m = X[:, 3]  # 2m temperature
    sst = X[:, 4]  # sea surface temperature
    sp = X[:, 5]  # surface pressure
    rho = X[:, 6]  # air density
    
    # Calculate wind speed
    wind_speed = np.sqrt(u10**2 + v10**2)
    
    # Constants for bulk formulas
    cp = 1004.0  # Specific heat capacity of air at constant pressure (J/kg/K)
    L = 2.5e6  # Latent heat of vaporization (J/kg)
    
    # Transfer coefficients (simplified)
    C_H = 1.3e-3  # Transfer coefficient for sensible heat
    C_E = 1.5e-3  # Transfer coefficient for latent heat
    
    # Calculate saturation vapor pressure using simplified formula
    def calc_saturation_vapor_pressure(temp_k):
        temp_c = temp_k - 273.15
        return 6.112 * np.exp(17.67 * temp_c / (temp_c + 243.5)) * 100  # in Pa
    
    # Calculate specific humidity from dewpoint temperature
    e_s_td = calc_saturation_vapor_pressure(td2m)  # Vapor pressure at dewpoint
    q_a = 0.622 * e_s_td / (sp - 0.378 * e_s_td)  # Specific humidity of air
    
    # Calculate specific humidity at sea surface (assumed saturation)
    e_s_sst = calc_saturation_vapor_pressure(sst)
    q_s = 0.622 * e_s_sst / (sp - 0.378 * e_s_sst)  # Specific humidity at sea surface
    
    # Calculate fluxes using bulk formulas
    sensible_heat_flux = rho * cp * C_H * wind_speed * (sst - t2m)
    latent_heat_flux = rho * L * C_E * wind_speed * (q_s - q_a)
    
    # Combine into target array
    y = np.column_stack([sensible_heat_flux, latent_heat_flux])
    logger.info(f"Y shape: {y.shape}")
    
    # Log some statistics
    logger.info(f"Sensible heat flux - Mean: {np.mean(sensible_heat_flux):.2f}, Min: {np.min(sensible_heat_flux):.2f}, Max: {np.max(sensible_heat_flux):.2f}")
    logger.info(f"Latent heat flux - Mean: {np.mean(latent_heat_flux):.2f}, Min: {np.min(latent_heat_flux):.2f}, Max: {np.max(latent_heat_flux):.2f}")
    
    return X, y

# PINN model definition
class PINN(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, output_dim=2):  # Reduced hidden_dim
        super(PINN, self).__init__()
        
        # Encoder - smaller network
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU()
        )
        
        # Decoder - smaller network
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, output_dim)
        )
    
    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

# Physics-based loss function
def compute_physics_loss(inputs, outputs, scaler_X, scaler_y=None):
    """Calculate physics-based loss using bulk formulas for air-sea fluxes"""
    # Convert to numpy for easier manipulation
    inputs_np = inputs.detach().cpu().numpy()
    outputs_np = outputs.detach().cpu().numpy()
    
    # Unscale the input features
    X = scaler_X.inverse_transform(inputs_np)
    
    # Extract relevant variables
    u10 = X[:, 0]  # 10m u wind
    v10 = X[:, 1]  # 10m v wind
    td2m = X[:, 2]  # 2m dewpoint temperature
    t2m = X[:, 3]  # 2m temperature
    sst = X[:, 4]  # sea surface temperature
    sp = X[:, 5]  # surface pressure
    rho = X[:, 6]  # air density
    
    # Calculate wind speed
    wind_speed = np.sqrt(u10**2 + v10**2)
    
    # Constants for bulk formulas
    cp = 1004.0  # Specific heat capacity of air at constant pressure (J/kg/K)
    L = 2.5e6  # Latent heat of vaporization (J/kg)
    
    # Transfer coefficients (simplified)
    C_H = 1.3e-3  # Transfer coefficient for sensible heat
    C_E = 1.5e-3  # Transfer coefficient for latent heat
    
    # Calculate saturation vapor pressure using simplified formula
    def calc_saturation_vapor_pressure(temp_k):
        temp_c = temp_k - 273.15
        return 6.112 * np.exp(17.67 * temp_c / (temp_c + 243.5)) * 100  # in Pa
    
    # Calculate specific humidity from dewpoint temperature
    e_s_td = calc_saturation_vapor_pressure(td2m)  # Vapor pressure at dewpoint
    q_a = 0.622 * e_s_td / (sp - 0.378 * e_s_td)  # Specific humidity of air
    
    # Calculate specific humidity at sea surface (assumed saturation)
    e_s_sst = calc_saturation_vapor_pressure(sst)
    q_s = 0.622 * e_s_sst / (sp - 0.378 * e_s_sst)  # Specific humidity at sea surface
    
    # Calculate fluxes using bulk formulas
    sensible_heat_flux_physics = rho * cp * C_H * wind_speed * (sst - t2m)
    latent_heat_flux_physics = rho * L * C_E * wind_speed * (q_s - q_a)
    
    # Combine into physics-based predictions
    physics_pred = np.column_stack([sensible_heat_flux_physics, latent_heat_flux_physics])
    
    # If scaler_y is provided, scale the physics predictions to match model output scale
    if scaler_y is not None:
        physics_pred = scaler_y.transform(physics_pred)
    
    # Convert back to torch tensor
    physics_pred_tensor = torch.tensor(physics_pred, dtype=torch.float32, device=outputs.device)
    
    # Calculate physics-based loss (MSE)
    physics_loss = torch.mean((outputs - physics_pred_tensor) ** 2)
    
    return physics_loss

# Combined loss function
def combined_loss(y_pred, y_true, inputs, scaler_X, scaler_y, physics_weight=0.3):
    """Combine data-driven loss and physics-based loss"""
    # Data-driven loss (MSE)
    data_loss = torch.mean((y_pred - y_true) ** 2)
    
    # Physics-based loss
    phys_loss = compute_physics_loss(inputs, y_pred, scaler_X, scaler_y)
    
    # Combined loss
    return (1 - physics_weight) * data_loss + physics_weight * phys_loss, data_loss, phys_loss

# Function to save checkpoint
def save_checkpoint(model, optimizer, epoch, train_losses, val_losses, data_losses, physics_losses,
                   best_val_loss, counter, output_dir, filename="checkpoint.pth"):
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'train_losses': train_losses,
        'val_losses': val_losses,
        'data_losses': data_losses,
        'physics_losses': physics_losses,
        'best_val_loss': best_val_loss,
        'counter': counter
    }
    torch.save(checkpoint, os.path.join(output_dir, filename))
    logger.info(f"Checkpoint saved at epoch {epoch+1}")

# Function to load checkpoint
def load_checkpoint(model, optimizer, output_dir, filename="checkpoint.pth"):
    checkpoint_path = os.path.join(output_dir, filename)
    if os.path.exists(checkpoint_path):
        logger.info(f"Loading checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch = checkpoint['epoch']
        train_losses = checkpoint['train_losses']
        val_losses = checkpoint['val_losses']
        data_losses = checkpoint['data_losses']
        physics_losses = checkpoint['physics_losses']
        best_val_loss = checkpoint['best_val_loss']
        counter = checkpoint['counter']
        return model, optimizer, epoch, train_losses, val_losses, data_losses, physics_losses, best_val_loss, counter, True
    else:
        logger.info("No checkpoint found, starting from scratch")
        return model, optimizer, 0, [], [], [], [], float('inf'), 0, False

# Training function
def train_model(model, train_loader, val_loader, scaler_X, scaler_y, device,
                lr=0.001, epochs=200, physics_weight=0.3, output_dir="pinn_results",
                checkpoint_freq=5, resume=True):
    """Train the PINN model with checkpointing and detailed logging"""
    # Initialize optimizer
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10, min_lr=1e-6)
    
    # Try to load checkpoint if resume is True
    if resume:
        model, optimizer, start_epoch, train_losses, val_losses, data_losses, physics_losses, best_val_loss, counter, loaded = load_checkpoint(model, optimizer, output_dir)
    else:
        start_epoch, train_losses, val_losses, data_losses, physics_losses = 0, [], [], [], []
        best_val_loss, counter, loaded = float('inf'), 0, False
    
    # If no checkpoint was loaded, initialize from scratch
    if not loaded:
        start_epoch = 0
        train_losses = []
        val_losses = []
        data_losses = []
        physics_losses = []
        best_val_loss = float('inf')
        counter = 0
    
    # Early stopping parameters
    patience = 20
    best_model_state = None
    
    # Create a progress tracking file
    progress_file = os.path.join(output_dir, "training_progress.json")
    
    # Training loop
    logger.info(f"Starting training from epoch {start_epoch+1} to {epochs}")
    for epoch in range(start_epoch, epochs):
        epoch_start_time = time.time()
        
        # Clear memory before each epoch
        clear_memory()
        
        # Training
        model.train()
        train_loss = 0.0
        train_data_loss = 0.0
        train_physics_loss = 0.0
        
        # Track batch progress
        batch_count = len(train_loader)
        for batch_idx, (inputs, targets) in enumerate(train_loader):
            if batch_idx % 50 == 0:
                logger.info(f"Epoch {epoch+1}/{epochs} - Batch {batch_idx+1}/{batch_count}")
            
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss, data_loss, phys_loss = combined_loss(outputs, targets, inputs, scaler_X, scaler_y, physics_weight)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            train_data_loss += data_loss.item()
            train_physics_loss += phys_loss.item()
        
        train_loss /= len(train_loader)
        train_data_loss /= len(train_loader)
        train_physics_loss /= len(train_loader)
        train_losses.append(train_loss)
        
        # Validation
        model.eval()
        val_loss = 0.0
        val_data_loss = 0.0
        val_physics_loss = 0.0
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss, data_loss, phys_loss = combined_loss(outputs, targets, inputs, scaler_X, scaler_y, physics_weight)
                val_loss += loss.item()
                val_data_loss += data_loss.item()
                val_physics_loss += phys_loss.item()
        
        val_loss /= len(val_loader)
        val_data_loss /= len(val_loader)
        val_physics_loss /= len(val_loader)
        val_losses.append(val_loss)
        data_losses.append(val_data_loss)
        physics_losses.append(val_physics_loss)
        
        # Calculate epoch time
        epoch_time = time.time() - epoch_start_time
        
        # Print progress
        logger.info(f"Epoch {epoch+1}/{epochs} - "
                   f"Train Loss: {train_loss:.6f} "
                   f"(Data: {train_data_loss:.6f}, Physics: {train_physics_loss:.6f}) - "
                   f"Val Loss: {val_loss:.6f} "
                   f"(Data: {val_data_loss:.6f}, Physics: {val_physics_loss:.6f}) - "
                   f"Time: {epoch_time:.2f}s")
        
        # Update progress file
        progress_data = {
            'current_epoch': epoch + 1,
            'total_epochs': epochs,
            'train_loss': train_loss,
            'val_loss': val_loss,
            'best_val_loss': min(best_val_loss, val_loss),
            'learning_rate': optimizer.param_groups[0]['lr'],
            'time_per_epoch': epoch_time,
            'estimated_remaining_time': epoch_time * (epochs - epoch - 1)
        }
        with open(progress_file, 'w') as f:
            json.dump(progress_data, f, indent=4)
        
        # Learning rate scheduler
        scheduler.step(val_loss)
        
        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = model.state_dict().copy()
            counter = 0
            # Save best model
            torch.save(best_model_state, os.path.join(output_dir, "best_model.pt"))
            logger.info(f"New best model saved with validation loss: {best_val_loss:.6f}")
        else:
            counter += 1
            if counter >= patience:
                logger.info(f"Early stopping at epoch {epoch+1}")
                break
        
        # Save checkpoint periodically
        if (epoch + 1) % checkpoint_freq == 0:
            save_checkpoint(model, optimizer, epoch, train_losses, val_losses,
                           data_losses, physics_losses, best_val_loss, counter,
                           output_dir)
    
    # Save final checkpoint
    save_checkpoint(model, optimizer, epoch, train_losses, val_losses,
                   data_losses, physics_losses, best_val_loss, counter,
                   output_dir, "final_checkpoint.pth")
    
    # Load best model
    model.load_state_dict(best_model_state)
    
    return model, train_losses, val_losses, data_losses, physics_losses

# Function to evaluate model
def evaluate_model(model, test_loader, scaler_X, scaler_y, device):
    """Evaluate the model on test data"""
    logger.info("Evaluating model on test data...")
    model.eval()
    test_loss = 0.0
    predictions = []
    targets = []
    
    with torch.no_grad():
        for batch_idx, (inputs, y_true) in enumerate(test_loader):
            if batch_idx % 20 == 0:
                logger.info(f"Evaluating batch {batch_idx+1}/{len(test_loader)}")
            
            inputs, y_true = inputs.to(device), y_true.to(device)
            y_pred = model(inputs)
            
            # Store predictions and targets
            predictions.append(y_pred.cpu().numpy())
            targets.append(y_true.cpu().numpy())
            
            # Calculate loss
            loss = torch.mean((y_pred - y_true) ** 2)
            test_loss += loss.item()
    
    test_loss /= len(test_loader)
    
    # Concatenate predictions and targets
    predictions = np.vstack(predictions)
    targets = np.vstack(targets)
    
    # Inverse transform to get actual values
    predictions_actual = scaler_y.inverse_transform(predictions)
    targets_actual = scaler_y.inverse_transform(targets)
    
    # Calculate metrics
    mse = np.mean((predictions_actual - targets_actual) ** 2, axis=0)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(predictions_actual - targets_actual), axis=0)
    
    logger.info(f"Test Loss: {test_loss:.6f}")
    logger.info(f"Sensible Heat Flux - RMSE: {rmse[0]:.2f} W/m², MAE: {mae[0]:.2f} W/m²")
    logger.info(f"Latent Heat Flux - RMSE: {rmse[1]:.2f} W/m², MAE: {mae[1]:.2f} W/m²")
    
    return predictions_actual, targets_actual, mse, rmse, mae

# Function to plot results
def plot_results(train_losses, val_losses, data_losses, physics_losses, predictions, targets, output_dir):
    """Plot training curves and prediction results"""
    logger.info("Plotting results...")
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Plot training curves
    plt.figure(figsize=(12, 8))
    plt.subplot(2, 1, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    
    plt.subplot(2, 1, 2)
    plt.plot(data_losses, label='Data Loss')
    plt.plot(physics_losses, label='Physics Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Data and Physics Loss Components')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/training_curves.png", dpi=300)
    
    # Plot scatter plots for predictions vs targets
    plt.figure(figsize=(15, 6))
    
    # Sensible heat flux
    plt.subplot(1, 2, 1)
    plt.scatter(targets[:, 0], predictions[:, 0], alpha=0.3)
    min_val = min(np.min(targets[:, 0]), np.min(predictions[:, 0]))
    max_val = max(np.max(targets[:, 0]), np.max(predictions[:, 0]))
    plt.plot([min_val, max_val], [min_val, max_val], 'r--')
    plt.xlabel('True Sensible Heat Flux (W/m²)')
    plt.ylabel('Predicted Sensible Heat Flux (W/m²)')
    plt.title('Sensible Heat Flux: Predicted vs True')
    
    # Latent heat flux
    plt.subplot(1, 2, 2)
    plt.scatter(targets[:, 1], predictions[:, 1], alpha=0.3)
    min_val = min(np.min(targets[:, 1]), np.min(predictions[:, 1]))
    max_val = max(np.max(targets[:, 1]), np.max(predictions[:, 1]))
    plt.plot([min_val, max_val], [min_val, max_val], 'r--')
    plt.xlabel('True Latent Heat Flux (W/m²)')
    plt.ylabel('Predicted Latent Heat Flux (W/m²)')
    plt.title('Latent Heat Flux: Predicted vs True')
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/prediction_scatter.png", dpi=300)
    
    # Plot histograms of errors
    plt.figure(figsize=(15, 6))
    
    # Sensible heat flux errors
    plt.subplot(1, 2, 1)
    errors = predictions[:, 0] - targets[:, 0]
    plt.hist(errors, bins=50, alpha=0.7)
    plt.xlabel('Error (W/m²)')
    plt.ylabel('Frequency')
    plt.title(f'Sensible Heat Flux Error Distribution\nMean: {np.mean(errors):.2f}, Std: {np.std(errors):.2f}')
    
    # Latent heat flux errors
    plt.subplot(1, 2, 2)
    errors = predictions[:, 1] - targets[:, 1]
    plt.hist(errors, bins=50, alpha=0.7)
    plt.xlabel('Error (W/m²)')
    plt.ylabel('Frequency')
    plt.title(f'Latent Heat Flux Error Distribution\nMean: {np.mean(errors):.2f}, Std: {np.std(errors):.2f}')
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/error_histograms.png", dpi=300)
    
    plt.close('all')
    logger.info("Plotting completed")

# Function to check progress
def check_progress(output_dir):
    """Check training progress from the progress file"""
    progress_file = os.path.join(output_dir, "training_progress.json")
    if os.path.exists(progress_file):
        with open(progress_file, 'r') as f:
            progress = json.load(f)
        
        current_epoch = progress['current_epoch']
        total_epochs = progress['total_epochs']
        train_loss = progress['train_loss']
        val_loss = progress['val_loss']
        best_val_loss = progress['best_val_loss']
        time_per_epoch = progress['time_per_epoch']
        remaining_time = progress['estimated_remaining_time']
        
        print(f"Training Progress: {current_epoch}/{total_epochs} epochs ({current_epoch/total_epochs*100:.1f}%)")
        print(f"Current train loss: {train_loss:.6f}")
        print(f"Current validation loss: {val_loss:.6f}")
        print(f"Best validation loss: {best_val_loss:.6f}")
        print(f"Time per epoch: {time_per_epoch:.2f} seconds")
        print(f"Estimated remaining time: {remaining_time/60:.2f} minutes ({remaining_time/3600:.2f} hours)")
        
        return True
    else:
        print("No progress file found. Training may not have started yet.")
        return False

# Function to monitor training progress
def monitor_progress(output_dir="pinn_results", interval=60):
    """Monitor training progress by checking the progress file periodically
    
    Args:
        output_dir: Directory where progress file is stored
        interval: Check interval in seconds
    """
    print(f"Monitoring training progress in {output_dir}...")
    print(f"Press Ctrl+C to stop monitoring")
    
    try:
        while True:
            found = check_progress(output_dir)
            
            if not found:
                print("Waiting for training to start...")
                
                # Check log file for recent activity
                log_dir = "logs"
                log_files = sorted([f for f in os.listdir(log_dir) if f.startswith("pinn_training_")], reverse=True)
                
                if log_files:
                    latest_log = os.path.join(log_dir, log_files[0])
                    print(f"\nRecent log entries from {latest_log}:")
                    try:
                        with open(latest_log, 'r') as f:
                            lines = f.readlines()
                            for line in lines[-10:]:  # Show last 10 lines
                                print(line.strip())
                    except Exception as e:
                        print(f"Error reading log file: {str(e)}")
            
            print("\nWaiting for next update...")
            time.sleep(interval)
    
    except KeyboardInterrupt:
        print("\nStopped monitoring")

# Main function
def main():
    start_time = time.time()
    
    # File path
    file_path = "/projects/nrsc01/air-sea_flux/data/merged_data.nc"
    
    # Create output directory
    output_dir = "pinn_results"
    os.makedirs(output_dir, exist_ok=True)
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # Load data
    try:
        data_dict = load_data(file_path)
    except Exception as e:
        logger.error(f"Failed to load data: {str(e)}")
        return
    
    # Prepare data
    try:
        X, y = prepare_data_in_chunks(data_dict)
    except Exception as e:
        logger.error(f"Failed to prepare data: {str(e)}")
        return
    
    # Scale data
    logger.info("Scaling data...")
    scaler_X = StandardScaler()
    X_scaled = scaler_X.fit_transform(X)
    scaler_y = StandardScaler()
    y_scaled = scaler_y.fit_transform(y)
    
    # Save scalers
    logger.info("Saving scalers...")
    with open(f"{output_dir}/scaler_X.pkl", 'wb') as f:
        pickle.dump(scaler_X, f)
    with open(f"{output_dir}/scaler_y.pkl", 'wb') as f:
        pickle.dump(scaler_y, f)
    
    # Convert to PyTorch tensors
    logger.info("Converting data to PyTorch tensors...")
    X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
    y_tensor = torch.tensor(y_scaled, dtype=torch.float32)
    
    # Create dataset
    dataset = TensorDataset(X_tensor, y_tensor)
    
    # Split into train, validation, and test sets
    logger.info("Splitting data into train, validation, and test sets...")
    train_size = int(0.7 * len(dataset))
    val_size = int(0.15 * len(dataset))
    test_size = len(dataset) - train_size - val_size
    train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])
    
    # Create data loaders
    batch_size = 512
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)
    
    logger.info(f"Train size: {len(train_dataset)}, Validation size: {len(val_dataset)}, Test size: {len(test_dataset)}")
    
    # Build model
    input_dim = X.shape[1]
    output_dim = y.shape[1]
    hidden_dim = 128
    
    # Check if we should resume from a checkpoint
    checkpoint_path = os.path.join(output_dir, "checkpoint.pth")
    if os.path.exists(checkpoint_path):
        logger.info("Found existing checkpoint. Will attempt to resume training.")
        resume = True
    else:
        logger.info("No checkpoint found. Starting training from scratch.")
        resume = False
    
    # Initialize model
    model = PINN(input_dim, hidden_dim, output_dim).to(device)
    logger.info(f"Model architecture:\n{model}")
    
    # Train model
    logger.info("Starting model training...")
    try:
        model, train_losses, val_losses, data_losses, physics_losses = train_model(
            model, train_loader, val_loader, scaler_X, scaler_y, device,
            lr=0.001, epochs=200, physics_weight=0.3, output_dir=output_dir,
            checkpoint_freq=5, resume=resume
        )
        
        # Save model
        logger.info("Saving final model...")
        torch.save(model.state_dict(), f"{output_dir}/pinn_model.pt")
        
        # Evaluate model
        logger.info("Evaluating model on test data...")
        predictions, targets, mse, rmse, mae = evaluate_model(model, test_loader, scaler_X, scaler_y, device)
        
        # Save evaluation metrics
        logger.info("Saving evaluation metrics...")
        np.savez(f"{output_dir}/evaluation_metrics.npz",
                mse=mse, rmse=rmse, mae=mae)
        
        # Plot results
        logger.info("Plotting results...")
        plot_results(train_losses, val_losses, data_losses, physics_losses,
                    predictions, targets, output_dir)
        
        # Calculate execution time
        end_time = time.time()
        execution_time = end_time - start_time
        logger.info(f"Execution time: {execution_time:.2f} seconds ({execution_time/60:.2f} minutes)")
    
    except KeyboardInterrupt:
        logger.info("Training interrupted by user. Saving current state...")
        # Save current model state
        torch.save(model.state_dict(), f"{output_dir}/interrupted_model.pt")
        logger.info("Model state saved. You can resume training later.")
    
    except Exception as e:
        logger.error(f"Error during training: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "monitor":
        # If called with "monitor" argument, just monitor progress
        output_dir = sys.argv[2] if len(sys.argv) > 2 else "pinn_results"
        interval = int(sys.argv[3]) if len(sys.argv) > 3 else 60
        monitor_progress(output_dir, interval)
    else:
        # Otherwise run the main training process
        main()


