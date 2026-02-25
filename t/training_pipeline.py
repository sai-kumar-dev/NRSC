from memory_utils import set_memory_limits, monitor_memory_usage, clear_memory
from core import logger
from cli import RESULTS_DIR
import torch
import numpy as np
import os   
import json
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset, random_split
import torch.nn as nn
from models import OceanHeatFluxPINN
from training_utils import train_pinn
from data_utils import load_data, prepare_data_in_chunks
from inference import run_inference_example
from results import save_results
from visualizations import create_visualizations
from evaluation import calculate_comprehensive_metrics

def train(config):   
    global RESULTS_DIR
    try:
        # Set memory limits early
        set_memory_limits()
        # Device setup
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {device}")

        # Monitor initial memory
        monitor_memory_usage()

        # Load and prepare data with subsampling
        data_dict = load_data(config['data_file'])
        monitor_memory_usage()
        
        X, y = prepare_data_in_chunks(data_dict, 
                                    max_samples=config['max_samples'],
                                    subsample_ratio=config['subsample_ratio'])
        monitor_memory_usage()

        # Clear data_dict to free memory
        del data_dict
        clear_memory()

        # Convert to float32 to save memory
        X = X.astype(np.float32)
        y = y.astype(np.float32)

        logger.info(f"Using {len(X)} samples for training")


        # Normalize features
        scaler_X = StandardScaler()
        scaler_y = StandardScaler()

        X_scaled = scaler_X.fit_transform(X)
        y_scaled = scaler_y.fit_transform(y)

        # Convert to tensors
        X_tensor = torch.FloatTensor(X_scaled)
        y_tensor = torch.FloatTensor(y_scaled)

        # Create dataset and split (80% train, 20% test)
        dataset = TensorDataset(X_tensor, y_tensor)

        train_size = int(config['train_split'] * len(dataset))
        test_size = len(dataset) - train_size

        train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
        
        # Further split training data for validation (80% train, 10% val from original)
        val_size = int(0.125 * train_size)  # 10% of total data
        train_size_final = train_size - val_size
        
        train_dataset_final, val_dataset = random_split(train_dataset, [train_size_final, val_size])

        # Create data loaders
        train_loader = DataLoader(train_dataset_final, batch_size=config['batch_size'],
                                shuffle=True, num_workers=0)
        val_loader = DataLoader(val_dataset, batch_size=config['batch_size'],
                              shuffle=False, num_workers=0)
        test_loader = DataLoader(test_dataset, batch_size=config['batch_size'],
                               shuffle=False, num_workers=0)

        logger.info(f"Dataset sizes - Train: {train_size_final}, Val: {val_size}, Test: {test_size}")

        # Initialize model with LeakyReLU
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

        # Calculate comprehensive metrics for both outputs
        # In the main() function, replace the metrics calculation section:

        # Calculate comprehensive metrics for both outputs
        logger.info("=== Comprehensive Model Performance ===")

        # Sensible heat flux metrics (use sign-based classification)
        sensible_metrics = calculate_comprehensive_metrics(all_targets[:, 0], all_predictions[:, 0], 
                                                 flux_type='sensible')
        logger.info("Sensible Heat Flux Metrics:")
        for metric, value in sensible_metrics.items():
            logger.info(f"  {metric}: {value:.4f}")

        # Latent heat flux metrics (use magnitude-based classification)
        latent_metrics = calculate_comprehensive_metrics(all_targets[:, 1], all_predictions[:, 1], 
                                               flux_type='latent')
        logger.info("Latent Heat Flux Metrics:")
        for metric, value in latent_metrics.items():
            logger.info(f"  {metric}: {value:.4f}")


        # Overall metrics summary
        logger.info("=== Summary ===")
        logger.info(f"Sensible Heat Flux - RMSE: {sensible_metrics['RMSE']:.2f}, MAE: {sensible_metrics['MAE']:.2f}, R²: {sensible_metrics['R2']:.4f}")
        logger.info(f"Latent Heat Flux - RMSE: {latent_metrics['RMSE']:.2f}, MAE: {latent_metrics['MAE']:.2f}, R²: {latent_metrics['R2']:.4f}")
        logger.info(f"Sensible Heat Flux - Accuracy: {sensible_metrics['Accuracy']:.4f}, F1-Score: {sensible_metrics['F1_Score']:.4f}")
        logger.info(f"Latent Heat Flux - Accuracy: {latent_metrics['Accuracy']:.4f}, F1-Score: {latent_metrics['F1_Score']:.4f}")

        # Save model and results
        RESULTS_DIR = save_results(model, scaler_X, scaler_y, train_losses, val_losses,
                                 all_predictions, all_targets, config)

        # Save comprehensive metrics
        metrics_path = os.path.join(RESULTS_DIR, 'comprehensive_metrics.json')
        with open(metrics_path, 'w') as f:
            json.dump({
                'sensible_heat_flux': sensible_metrics,
                'latent_heat_flux': latent_metrics,
                'test_loss': avg_test_loss
            }, f, indent=2, default=str)

        # Create visualizations
        create_visualizations(train_losses, val_losses, all_predictions, all_targets)

        logger.info("Training completed successfully!")
        
        # Run inference example with the trained model
        run_inference_example()

    except MemoryError as e:
        logger.error(f"Memory error: {e}")
        logger.error("Try reducing max_samples or subsample_ratio in config")
        raise
    except Exception as e:
        logger.error(f"Error in main execution: {str(e)}")
        raise
    finally:
        clear_memory()
