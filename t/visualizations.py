import numpy as np
import matplotlib.pyplot as plt
from core import logger
from data_utils import interpolate_nan_values
from datetime import datetime
from scipy.ndimage import gaussian_filter

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
    total_ocean_points = len(ocean_indices[0])
    
    logger.info(f"Ocean points in spatial grid: {total_ocean_points}")
    logger.info(f"Predictions available: {len(predictions)}")
    
    if len(ocean_indices[0]) > 0 and len(predictions) > 0:
        # Use only the available predictions, don't exceed the array bounds
        n_predictions_to_use = min(len(predictions), total_ocean_points)
        
        if n_predictions_to_use > 0:
            # Fill only the first n_predictions_to_use ocean points
            sensible_flux_2d[ocean_indices[0][:n_predictions_to_use], 
                           ocean_indices[1][:n_predictions_to_use]] = predictions[:n_predictions_to_use, 0]
            latent_flux_2d[ocean_indices[0][:n_predictions_to_use], 
                         ocean_indices[1][:n_predictions_to_use]] = predictions[:n_predictions_to_use, 1]
            
            logger.info(f"Filled {n_predictions_to_use} ocean points with predictions")
        else:
            logger.warning("No predictions to fill in spatial visualization")
            return None, None
    else:
        logger.warning("No ocean points or predictions available for spatial visualization")
        return None, None

    # Apply smoothing for better visualization
    sensible_flux_smooth = apply_smoothing(sensible_flux_2d, sigma=1.0)
    latent_flux_smooth = apply_smoothing(latent_flux_2d, sigma=1.0)

    # Create the spatial plot
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # Check if we have valid data to plot
    if np.all(np.isnan(sensible_flux_2d)) or np.all(np.isnan(latent_flux_2d)):
        logger.warning("All values are NaN, cannot create meaningful spatial plot")
        plt.close(fig)
        return None, None

    # Sensible heat flux - raw
    try:
        im1 = axes[0, 0].contourf(lon_grid, lat_grid, sensible_flux_2d,
                                  levels=20, cmap='RdBu_r', extend='both')
        axes[0, 0].set_title('Sensible Heat Flux (Raw)')
        axes[0, 0].set_xlabel('Longitude')
        axes[0, 0].set_ylabel('Latitude')
        plt.colorbar(im1, ax=axes[0, 0], label='W/m²')
    except Exception as e:
        logger.warning(f"Could not create sensible heat flux raw plot: {e}")
        axes[0, 0].text(0.5, 0.5, 'No valid data', ha='center', va='center', transform=axes[0, 0].transAxes)
        axes[0, 0].set_title('Sensible Heat Flux (Raw) - No Data')

    # Sensible heat flux - smoothed
    try:
        im2 = axes[0, 1].contourf(lon_grid, lat_grid, sensible_flux_smooth,
                                  levels=20, cmap='RdBu_r', extend='both')
        axes[0, 1].set_title('Sensible Heat Flux (Smoothed)')
        axes[0, 1].set_xlabel('Longitude')
        axes[0, 1].set_ylabel('Latitude')
        plt.colorbar(im2, ax=axes[0, 1], label='W/m²')
    except Exception as e:
        logger.warning(f"Could not create sensible heat flux smoothed plot: {e}")
        axes[0, 1].text(0.5, 0.5, 'No valid data', ha='center', va='center', transform=axes[0, 1].transAxes)
        axes[0, 1].set_title('Sensible Heat Flux (Smoothed) - No Data')

    # Latent heat flux - raw
    try:
        im3 = axes[1, 0].contourf(lon_grid, lat_grid, latent_flux_2d,
                                  levels=20, cmap='viridis', extend='both')
        axes[1, 0].set_title('Latent Heat Flux (Raw)')
        axes[1, 0].set_xlabel('Longitude')
        axes[1, 0].set_ylabel('Latitude')
        plt.colorbar(im3, ax=axes[1, 0], label='W/m²')
    except Exception as e:
        logger.warning(f"Could not create latent heat flux raw plot: {e}")
        axes[1, 0].text(0.5, 0.5, 'No valid data', ha='center', va='center', transform=axes[1, 0].transAxes)
        axes[1, 0].set_title('Latent Heat Flux (Raw) - No Data')

    # Latent heat flux - smoothed
    try:
        im4 = axes[1, 1].contourf(lon_grid, lat_grid, latent_flux_smooth,
                                  levels=20, cmap='viridis', extend='both')
        axes[1, 1].set_title('Latent Heat Flux (Smoothed)')
        axes[1, 1].set_xlabel('Longitude')
        axes[1, 1].set_ylabel('Latitude')
        plt.colorbar(im4, ax=axes[1, 1], label='W/m²')
    except Exception as e:
        logger.warning(f"Could not create latent heat flux smoothed plot: {e}")
        axes[1, 1].text(0.5, 0.5, 'No valid data', ha='center', va='center', transform=axes[1, 1].transAxes)
        axes[1, 1].set_title('Latent Heat Flux (Smoothed) - No Data')

    plt.tight_layout()

    # Save the spatial plot
    spatial_plot_path = f"spatial_heat_flux_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
    plt.savefig(spatial_plot_path, dpi=300, bbox_inches='tight')
    plt.show()

    logger.info(f"Spatial visualization saved to {spatial_plot_path}")

    return sensible_flux_2d, latent_flux_2d

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
