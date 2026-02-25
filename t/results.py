from cli import RESULTS_DIR 
from datetime import datetime
import os 
import pickle
import json
import torch
from core import logger
import numpy as np
import xarray as xr
import pandas as pd

# Save results function
def save_results(model, scaler_X, scaler_y, train_losses, val_losses,
                predictions, targets, config):
    """Save model, scalers, and results"""
    global RESULTS_DIR
    # Create results directory
    RESULTS_DIR = f"results_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # Save model
    model_path = os.path.join(RESULTS_DIR, 'pinn_model.pth')
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
    scaler_path = os.path.join(RESULTS_DIR, 'scalers.pkl')
    with open(scaler_path, 'wb') as f:
        pickle.dump({'scaler_X': scaler_X, 'scaler_y': scaler_y}, f)

    # Save training history
    history_path = os.path.join(RESULTS_DIR, 'training_history.pkl')
    with open(history_path, 'wb') as f:
        pickle.dump({
            'train_losses': train_losses,
            'val_losses': val_losses,
            'predictions': predictions,
            'targets': targets
        }, f)

    # Save configuration
    config_path = os.path.join(RESULTS_DIR, 'config.json')
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2, default=str)

    logger.info(f"Results saved to {RESULTS_DIR}")
    return RESULTS_DIR

def create_performance_summary():
    """Create a comprehensive performance summary"""
    global RESULTS_DIR
    
    if RESULTS_DIR is None:
        logger.warning("No results directory available for summary")
        return
    
    # Load metrics
    metrics_path = os.path.join(RESULTS_DIR, 'comprehensive_metrics.json')
    if os.path.exists(metrics_path):
        with open(metrics_path, 'r') as f:
            metrics = json.load(f)
        
        # Create summary table - Fix: Convert to float first, then format
        summary_data = {
            'Metric': ['RMSE', 'MSE', 'MAE', 'R²', 'Accuracy', 'Precision', 'Recall', 'F1-Score'],
            'Sensible Heat Flux': [
                f"{float(metrics['sensible_heat_flux']['RMSE']):.4f}",
                f"{float(metrics['sensible_heat_flux']['MSE']):.4f}",
                f"{float(metrics['sensible_heat_flux']['MAE']):.4f}",
                f"{float(metrics['sensible_heat_flux']['R2']):.4f}",
                f"{float(metrics['sensible_heat_flux']['Accuracy']):.4f}",
                f"{float(metrics['sensible_heat_flux']['Precision']):.4f}",
                f"{float(metrics['sensible_heat_flux']['Recall']):.4f}",
                f"{float(metrics['sensible_heat_flux']['F1_Score']):.4f}"
            ],
            'Latent Heat Flux': [
                f"{float(metrics['latent_heat_flux']['RMSE']):.4f}",
                f"{float(metrics['latent_heat_flux']['MSE']):.4f}",
                f"{float(metrics['latent_heat_flux']['MAE']):.4f}",
                f"{float(metrics['latent_heat_flux']['R2']):.4f}",
                f"{float(metrics['latent_heat_flux']['Accuracy']):.4f}",
                f"{float(metrics['latent_heat_flux']['Precision']):.4f}",
                f"{float(metrics['latent_heat_flux']['Recall']):.4f}",
                f"{float(metrics['latent_heat_flux']['F1_Score']):.4f}"
            ]
        }
        
        # Create DataFrame and save
        df = pd.DataFrame(summary_data)
        summary_path = os.path.join(RESULTS_DIR, 'performance_summary.csv')
        df.to_csv(summary_path, index=False)
        
        logger.info("=== PERFORMANCE SUMMARY ===")
        logger.info(df.to_string(index=False))
        logger.info(f"Performance summary saved to {summary_path}")
        
        return df
    else:
        logger.warning("Metrics file not found")
        return None

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
