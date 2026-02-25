import torch
import torch.nn as nn
import numpy as np
from core import logger
import matplotlib.pyplot as plt
from datetime import datetime

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
