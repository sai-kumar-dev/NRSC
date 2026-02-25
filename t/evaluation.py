from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import numpy as np

# Enhanced evaluation metrics function
def calculate_comprehensive_metrics(y_true, y_pred, flux_type='sensible', threshold_percentile=50):
    """Calculate comprehensive metrics including classification metrics"""
    
    # Regression metrics
    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)
    
    # Different classification strategies based on flux type
    if flux_type == 'sensible':
        # For sensible heat flux: Use sign-based classification (direction matters)
        # But filter out very small values near zero
        min_threshold = np.std(y_true) * 0.1  # 10% of standard deviation
        significant_mask = np.abs(y_true) > min_threshold
        
        if np.sum(significant_mask) > 0:
            y_true_filtered = y_true[significant_mask]
            y_pred_filtered = y_pred[significant_mask]
            
            # Sign-based classification
            y_true_binary = (y_true_filtered > 0).astype(int)
            y_pred_binary = (y_pred_filtered > 0).astype(int)
        else:
            # Fallback if no significant values
            y_true_binary = (y_true > 0).astype(int)
            y_pred_binary = (y_pred > 0).astype(int)
            
    elif flux_type == 'latent':
        # For latent heat flux: Use magnitude-based classification (high vs low)
        threshold_true = np.percentile(np.abs(y_true), threshold_percentile)
        
        # High flux = 1, Low flux = 0
        y_true_binary = (np.abs(y_true) > threshold_true).astype(int)
        y_pred_binary = (np.abs(y_pred) > threshold_true).astype(int)
    
    else:
        # Default: sign-based
        y_true_binary = (y_true > 0).astype(int)
        y_pred_binary = (y_pred > 0).astype(int)
    
    # Calculate classification metrics
    if len(np.unique(y_true_binary)) > 1:  # Check if we have both classes
        accuracy = accuracy_score(y_true_binary, y_pred_binary)
        precision = precision_score(y_true_binary, y_pred_binary, 
                                  average='weighted', zero_division=0)
        recall = recall_score(y_true_binary, y_pred_binary, 
                            average='weighted', zero_division=0)
        f1 = f1_score(y_true_binary, y_pred_binary, 
                     average='weighted', zero_division=0)
    else:
        # If only one class present
        unique_class = np.unique(y_true_binary)[0]
        pred_class = np.unique(y_pred_binary)[0] if len(np.unique(y_pred_binary)) == 1 else -1
        
        if unique_class == pred_class:
            accuracy = precision = recall = f1 = 1.0
        else:
            accuracy = precision = recall = f1 = 0.0
    
    return {
        'MSE': mse,
        'MAE': mae,
        'RMSE': rmse,
        'R2': r2,
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall': recall,
        'F1_Score': f1
    }
