import os
import torch
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
from cli import RESULTS_DIR
from inference import load_trained_model, predict_heat_fluxes
from results import export_predictions_to_netcdf
from core import logger
from training_pipeline import train
from data_utils import load_data, prepare_data_in_chunks
from physics import validate_physics_constraints
from visualizations import create_spatial_visualization

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

def run_comprehensive_analysis(config):
    """Run a comprehensive analysis including training and evaluation"""
    global RESULTS_DIR
    
    logger.info("Starting comprehensive PINN analysis...")

    try:
        # Update this path to your actual data file
        #data_file = "/projects/nrsc01/air-sea_flux/data/era5_january_1995_2024_complete.nc"  # Replace with actual path

        if not os.path.exists(config['data_file']):
            logger.error(f"Data file not found: {config['data_file']}")
            logger.info("Please update the data_file path in the run_comprehensive_analysis function")
            return
        
        # Run main training
        train(config)

        # Use the results directory from main training
        if RESULTS_DIR and os.path.exists(RESULTS_DIR):
            model_path = os.path.join(RESULTS_DIR, 'pinn_model.pth')
            scaler_path = os.path.join(RESULTS_DIR, 'scalers.pkl')

            # Load the trained model
            model, scaler_X, scaler_y = load_trained_model(model_path, scaler_path)

            # Load data for additional analysis
            data_dict = load_data(config['data_file'])
            X, y = prepare_data_in_chunks(data_dict)

            # Run sensitivity analysis
            logger.info("Running sensitivity analysis...")
            analyze_model_sensitivity(model, scaler_X, scaler_y, X[:10000])  # Use subset for speed

            # Make predictions on test data
            logger.info("Making predictions for validation...")
            predictions = predict_heat_fluxes(model, scaler_X, scaler_y, X[:10000])

            # Validate physics constraints
            logger.info("Validating physics constraints...")
            validate_physics_constraints(predictions, X[:10000])

            # Create spatial visualization with proper data handling
            logger.info("Creating spatial visualization...")
            
            # Calculate how many ocean points we need for one time step
            ocean_mask = data_dict['ocean_mask']
            if len(ocean_mask.shape) == 3:
                ocean_points_per_timestep = np.sum(ocean_mask[0])
            else:
                ocean_points_per_timestep = np.sum(ocean_mask)
            
            logger.info(f"Ocean points per timestep: {ocean_points_per_timestep}")
            
            # Use enough predictions to fill at least one timestep
            n_predictions_needed = min(len(predictions), ocean_points_per_timestep)
            
            if n_predictions_needed > 0:
                create_spatial_visualization(data_dict, predictions[:n_predictions_needed], time_idx=0)
            else:
                logger.warning("Not enough predictions for spatial visualization")

            # Export predictions to NetCDF
            logger.info("Exporting predictions to NetCDF...")
            output_path = f"heat_flux_predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.nc"
            
            # Use a reasonable subset of predictions for export
            export_predictions = predictions[:min(len(predictions), 50000)]  # Limit to 50k predictions
            export_predictions_to_netcdf(data_dict, export_predictions, output_path)

            logger.info("Comprehensive analysis completed successfully!")

        else:
            logger.warning("No results directory found from training.")

    except Exception as e:
        logger.error(f"Error in comprehensive analysis: {str(e)}")
        raise
