# # src/main.py

# import pandas as pd
# import os
# import torch

# from src import config
# from src.data_preprocessing import PriceProcessor, engineer_features
# from src.roberta_classifier import train_roberta, predict_with_roberta
# from src.ensemble_model import train_ensemble_model, predict_with_ensemble_model
# from src.evaluation import smape

# def main():
#     # --- 0. CRITICAL: Verify GPU is available ---
#     if not torch.cuda.is_available():
#         print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
#         print("!!! ERROR: No CUDA-enabled GPU found !!!")
#         print("!!!       Aborting training.       !!!")
#         print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
#         return
#     print(f"PyTorch confirmed CUDA is available. Using device: {config.DEVICE.upper()}")
    
#     # --- 1. Load Data ---
#     print("\nStep 1: Loading data...")
#     train_df = pd.read_csv(config.TRAIN_FILE)
#     test_df = pd.read_csv(config.TEST_FILE)
#     print(f"Train shape: {train_df.shape}, Test shape: {test_df.shape}")

#     # --- 2. Preprocess Price Column for Training ---
#     print("\nStep 2: Preprocessing price column for RoBERTa training...")
#     price_processor = PriceProcessor()
#     train_df[config.TARGET_CLASS_COLUMN] = price_processor.fit_transform(train_df[config.PRICE_COLUMN])
#     print(f"Price column transformed and binned into {train_df[config.TARGET_CLASS_COLUMN].nunique()} classes.")

#     # --- 3. Fine-Tune RoBERTa Classifier ---
#     print("\nStep 3: Fine-tuning RoBERTa for price classification...")
#     model_config_path = os.path.join(config.ROBERTA_MODEL_PATH, "config.json")
#     if not os.path.exists(model_config_path):
#         print("RoBERTa model not found. Starting training...")
#         train_roberta(train_df, config.TEXT_COLUMN, config.TARGET_CLASS_COLUMN)
#     else:
#         print("Found existing RoBERTa model. Skipping training.")

#     # --- 4. Get RoBERTa Predictions as Features ---
#     print("\nStep 4: Generating price class predictions from RoBERTa...")
#     train_df['roberta_pred_class'] = predict_with_roberta(train_df, config.TEXT_COLUMN)
#     test_df['roberta_pred_class'] = predict_with_roberta(test_df, config.TEXT_COLUMN)
#     print("RoBERTa predictions added as a new feature.")

#     # --- 5. Engineer Additional Features ---
#     print("\nStep 5: Engineering features from catalog_content...")
#     train_df_featured = engineer_features(train_df)
#     test_df_featured = engineer_features(test_df)
#     print("Feature engineering complete.")

#     # --- 6. Train Ensemble Model ---
#     print("\nStep 6: Training the final LightGBM ensemble model...")
#     train_ensemble_model(train_df_featured, config.PRICE_COLUMN)

#     # --- 7. Make Final Price Predictions ---
#     print("\nStep 7: Making final price predictions on the test set...")
#     test_predictions = predict_with_ensemble_model(test_df_featured)
#     test_df['predicted_price'] = test_predictions

#     # --- 8. Save Results ---
#     # The evaluation step has been removed as test.csv does not have a 'price' column.
#     print("\nStep 8: Saving submission file...")
#     submission_df = test_df[['sample_id', 'predicted_price']]
#     submission_df.to_csv(config.SUBMISSION_FILE, index=False)
#     print(f"✅ Submission file saved to {config.SUBMISSION_FILE}")

# if __name__ == "__main__":
#     main()

# src/main.py

import pandas as pd
import os
import torch

from src import config
from src.data_preprocessing import PriceProcessor, engineer_features
from src.roberta_classifier import train_roberta, predict_with_roberta
from src.ensemble_model import train_ensemble_model, predict_with_ensemble_model
from src.evaluation import smape

def main():
    # --- 0. CRITICAL: Verify GPU is available ---
    if not torch.cuda.is_available():
        print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        print("!!! ERROR: No CUDA-enabled GPU found !!!")
        print("!!!       Aborting training.       !!!")
        print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        return # Exit the script
    print(f"PyTorch confirmed CUDA is available. Using device: {config.DEVICE.upper()}")
    
    # --- 1. Load Data ---
    print("\nStep 1: Loading data...")
    train_df = pd.read_csv(config.TRAIN_FILE)
    test_df = pd.read_csv(config.TEST_FILE)
    print(f"Train shape: {train_df.shape}, Test shape: {test_df.shape}")

    # --- 2. Preprocess Price Column for Training ---
    print("\nStep 2: Preprocessing price column for RoBERTa training...")
    price_processor = PriceProcessor()
    train_df[config.TARGET_CLASS_COLUMN] = price_processor.fit_transform(train_df[config.PRICE_COLUMN])
    print(f"Price column transformed and binned into {train_df[config.TARGET_CLASS_COLUMN].nunique()} classes.")

    # --- 3. Fine-Tune RoBERTa Classifier ---
    print("\nStep 3: Fine-tuning RoBERTa for price classification...")
    
    # --- THIS IS THE IMPROVED LOGIC ---
    # Check for a specific file that indicates a successful save
    model_config_path = os.path.join(config.ROBERTA_MODEL_PATH, "config.json")
    if not os.path.exists(model_config_path):
        print("RoBERTa model not found. Starting training...")
        train_roberta(train_df, config.TEXT_COLUMN, config.TARGET_CLASS_COLUMN)
    else:
        print("Found existing RoBERTa model. Skipping training.")
    # --- END OF IMPROVED LOGIC ---

    # --- 4. Get RoBERTa Predictions as Features ---
    print("\nStep 4: Generating price class predictions from RoBERTa...")
    train_df['roberta_pred_class'] = predict_with_roberta(train_df, config.TEXT_COLUMN)
    test_df['roberta_pred_class'] = predict_with_roberta(test_df, config.TEXT_COLUMN)
    print("RoBERTa predictions added as a new feature.")

    # --- 5. Engineer Additional Features ---
    print("\nStep 5: Engineering features from catalog_content...")
    train_df_featured = engineer_features(train_df)
    test_df_featured = engineer_features(test_df)
    print("Feature engineering complete.")

# --- 6. Train Ensemble Model ---
    print("\nStep 6: Training the final Voting Regressor ensemble model (SVR + DT + AdaBoost)...")
    train_ensemble_model(train_df_featured, config.PRICE_COLUMN)

    # --- 7. Make Final Price Predictions ---
    print("\nStep 7: Making final price predictions on the test set...")
    test_predictions = predict_with_ensemble_model(test_df_featured)
    test_df['predicted_price'] = test_predictions

    # --- 8. Evaluate and Save Results ---
    print("\nStep 8: Evaluating the model and saving submission...")
    score = smape(test_df[config.PRICE_COLUMN], test_df['predicted_price'])
    print(f"SMAPE on Test Set: {score:.4f}")

    submission_df = test_df[['sample_id', 'predicted_price']]
    submission_df.to_csv(config.SUBMISSION_FILE, index=False)
    print(f"Submission file saved to {config.SUBMISSION_FILE}")

if __name__ == "__main__":
    main()