# # src/config.py

# import torch

# # --- File Paths ---
# DATA_DIR = "/kaggle/input/programfiles/data/"
# TRAIN_FILE = f"{DATA_DIR}train.csv"
# TEST_FILE = f"{DATA_DIR}test.csv"
# MODEL_DIR = "output/models/"
# SUBMISSION_FILE = "output/submission.csv"
# TEST_OUTPUT_FILE = "output/test_out.csv"

# # --- Preprocessing ---
# PRICE_COLUMN = "price"
# TEXT_COLUMN = "catalog_content"
# TARGET_CLASS_COLUMN = "price_class"
# NUM_PRICE_CLASSES = 14

# # --- RoBERTa Model ---
# ROBERTA_MODEL_NAME = "distilroberta-base"
# ROBERTA_MODEL_PATH = f"{MODEL_DIR}roberta_classifier"
# ROBERTA_BATCH_SIZE = 64   # Increased for speed
# ROBERTA_EPOCHS = 3           # Reduced, often sufficient
# ROBERTA_LEARNING_RATE = 1e-5

# # --- Ensemble Model (LightGBM) ---
# ENSEMBLE_MODEL_PATH = f"{MODEL_DIR}lgbm_regressor.pkl"
# LGBM_PARAMS = {
#     'objective': 'regression_l1', 'metric': 'mae', 'n_estimators': 2000,
#     'learning_rate': 0.01, 'feature_fraction': 0.8, 'bagging_fraction': 0.8,
#     'bagging_freq': 1, 'lambda_l1': 0.1, 'lambda_l2': 0.1,
#     'num_leaves': 31, 'verbose': -1, 'n_jobs': -1,
# }

# # --- General ---
# RANDOM_STATE = 42
# DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# src/config.py

import torch

# --- File Paths ---
DATA_DIR = "data/"
TRAIN_FILE = f"{DATA_DIR}train.csv"
TEST_FILE = f"{DATA_DIR}test.csv"
MODEL_DIR = "models/"
SUBMISSION_FILE = "submission.csv"

# --- Preprocessing ---
PRICE_COLUMN = "price"
TEXT_COLUMN = "catalog_content"
TARGET_CLASS_COLUMN = "price_class"
NUM_PRICE_CLASSES = 14

# --- RoBERTa Model ---
ROBERTA_MODEL_NAME = "distilroberta-base"
ROBERTA_MODEL_PATH = f"{MODEL_DIR}roberta_classifier"
ROBERTA_BATCH_SIZE = 64      # Increased for speed
ROBERTA_EPOCHS = 4          # Reduced, often sufficient
ROBERTA_LEARNING_RATE = 1e-5

# --- Ensemble Model (Voting Regressor: SVR + DecisionTree + AdaBoost) ---
ENSEMBLE_MODEL_PATH = f"{MODEL_DIR}voting_regressor.pkl"

SVM_PARAMS = {
    'kernel': 'rbf',
    'C': 10.0,
    'epsilon': 0.1,
    'gamma': 'scale',
}

DT_PARAMS = {
    'max_depth': None,
    'min_samples_split': 2,
    'min_samples_leaf': 1,
}

ADABOOST_PARAMS = {
    'n_estimators': 300,
    'learning_rate': 0.05,
    'loss': 'linear',
}

VOTING_WEIGHTS = [1.0, 1.0, 1.0]

# --- General ---
RANDOM_STATE = 42
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"