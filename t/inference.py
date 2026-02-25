from core import logger
import numpy as np
import pickle
import torch
from models import OceanHeatFluxPINN
import os
from cli import RESULTS_DIR

def run_inference_example():
    """Example of how to run inference with a trained model"""
    global RESULTS_DIR
    
    logger.info("Running inference example...")

    # Example input data (replace with your actual data)
    # Format: [u10, v10, d2m, t2m, sst, sp, rho]
    example_input = np.array([
        [5.0, 3.0, 285.0, 288.0, 290.0, 101325.0, 1.2],  # Example 1
        [10.0, -2.0, 280.0, 285.0, 295.0, 101000.0, 1.15], # Example 2
        [2.0, 1.0, 290.0, 292.0, 288.0, 101500.0, 1.25],   # Example 3
    ])

    try:
        # Use the current results directory
        if RESULTS_DIR is None:
            # Try to find the most recent results directory
            result_dirs = [d for d in os.listdir('.') if d.startswith('results_')]
            if result_dirs:
                RESULTS_DIR = sorted(result_dirs)[-1]
            else:
                logger.warning("No results directory found. Please run training first.")
                return

        model_path = os.path.join(RESULTS_DIR, 'pinn_model.pth')
        scaler_path = os.path.join(RESULTS_DIR, 'scalers.pkl')

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
            logger.warning(f"Trained model not found at {model_path} or {scaler_path}")

    except Exception as e:
        logger.error(f"Error in inference example: {str(e)}")

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
