# src/evaluation.py

import numpy as np

def smape(y_true, y_pred):
    """
    Calculates the Symmetric Mean Absolute Percentage Error (SMAPE).
    """
    numerator = np.abs(y_pred - y_true)
    denominator = (np.abs(y_true) + np.abs(y_pred)) / 2
    
    # Handle the case where both true and pred are zero
    # to avoid division by zero
    ratio = np.where(denominator == 0, 0, numerator / denominator)
    
    return np.mean(ratio) * 100