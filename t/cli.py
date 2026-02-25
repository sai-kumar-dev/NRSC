from core import logger
import numpy as np
import torch
import os
from memory_utils import set_memory_limits
from training_pipeline import train
from inference import run_inference_example 
from analysis import run_comprehensive_analysis
from results import create_performance_summary

RESULTS_DIR = None

# Call this early
set_memory_limits()

# Set random seeds for reproducibility
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)


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
    parser.add_argument('--epochs', type=int, default=70,
                       help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=1024,
                       help='Batch size for training')
    parser.add_argument('--learning_rate', type=float, default=0.001,
                       help='Learning rate')
    parser.add_argument('--lambda_physics', type=float, default=0.1,
                       help='Physics loss weight')

    args = parser.parse_args()

    # Update config with command line arguments
    config = {
        'data_file': args.data_file,
        'batch_size': args.batch_size,
        'hidden_dim': 128,
        'num_layers': 4,
        'num_epochs': args.epochs,
        'learning_rate': args.learning_rate,
        'lambda_physics': args.lambda_physics,
        'train_split': 0.8, # Training sample percentage
        'test_split': 0.2,   # Testing sample percentage 
        'max_samples': 20000000,  # 20M samples max
        'subsample_ratio': 0.02   # Use 2% of data       
        }

    if args.mode == 'inference':
        if not args.model_path or not args.scaler_path:
            logger.error("Model path and scaler path required for inference mode")
            exit(1)

        # Set the paths for inference
        RESULTS_DIR = os.path.dirname(args.model_path)
        run_inference_example()

    elif args.mode == 'analysis':
        run_comprehensive_analysis(config)

    else:
        # Run main training
        train(config)
        
        # Create performance summary
        create_performance_summary()