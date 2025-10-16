# src/predict.py

import pandas as pd
import os
import torch

from src import config
from src.data_preprocessing import engineer_features
from src.roberta_classifier import predict_with_roberta
from src.ensemble_model import predict_with_ensemble_model

def predict():
    """
    Loads trained models to make predictions on the test set.
    """
    print("Starting prediction pipeline...")
    print(f"Using device: {config.DEVICE.upper()}")

    # --- 1. Check if Models Exist ---
    roberta_path = os.path.join(config.ROBERTA_MODEL_PATH, "config.json")
    ensemble_path = config.ENSEMBLE_MODEL_PATH

    if not os.path.exists(roberta_path) or not os.path.exists(ensemble_path):
        print("\n!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        print("!!! ERROR: Trained models not found.                   !!!")
        print("!!! Please run 'python -m src.main' to train models first. !!!")
        print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        return

    # --- 2. Load Test Data ---
    print(f"\nStep 1: Loading test data from {config.TEST_FILE}...")
    if not os.path.exists(config.TEST_FILE):
        print(f"!!! ERROR: Test file not found at {config.TEST_FILE} !!!")
        return
        
    test_df = pd.read_csv(config.TEST_FILE)
    print(f"Loaded {len(test_df)} rows for prediction.")

    # --- 3. Feature Generation Pipeline ---
    # Generate predictions from RoBERTa to use as a feature
    print("\nStep 2: Generating price class predictions from RoBERTa...")
    test_df['roberta_pred_class'] = predict_with_roberta(test_df, config.TEXT_COLUMN)

    # Apply the same feature engineering steps used during training
    print("\nStep 3: Engineering features from catalog_content...")
    test_df_featured = engineer_features(test_df)
    print("Feature engineering complete.")

    # --- 4. Make Final Predictions ---
    print("\nStep 4: Making final price predictions with the ensemble model...")
    predictions = predict_with_ensemble_model(test_df_featured)
    test_df['predicted_price'] = predictions
    print("Prediction complete.")

    # --- 5. Save the Output ---
    output_df = test_df[['sample_id', 'predicted_price']]
    output_df.to_csv(config.TEST_OUTPUT_FILE, index=False)
    print(f"\n✅ Successfully saved predictions to {config.TEST_OUTPUT_FILE}")


if __name__ == "__main__":
    predict()