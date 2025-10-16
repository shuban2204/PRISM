# # src/ensemble_model.py

# import lightgbm as lgb
# import joblib
# import os
# import pandas as pd

# from src.config import LGBM_PARAMS, ENSEMBLE_MODEL_PATH, RANDOM_STATE

# def get_feature_columns(df):
#     """Identifies the feature columns for the ensemble model."""
#     # Exclude identifiers, raw text, and the original target
#     exclude_cols = [
#         'sample_id', 'catalog_content', 'image_links', 'image_link', 'price',
#         'price_class', 'value', 'unit', 'standardised_units'
#     ]
#     feature_cols = [col for col in df.columns if col not in exclude_cols]
#     return feature_cols

# def train_ensemble_model(df, target_col):
#     """Trains and saves the LightGBM regression model."""
#     feature_cols = get_feature_columns(df)
#     X_train = df[feature_cols]
#     y_train = df[target_col]

#     model = lgb.LGBMRegressor(**LGBM_PARAMS, random_state=RANDOM_STATE)
#     model.fit(X_train, y_train)

#     # Save the model
#     os.makedirs(os.path.dirname(ENSEMBLE_MODEL_PATH), exist_ok=True)
#     joblib.dump(model, ENSEMBLE_MODEL_PATH)
#     print(f"Ensemble model saved to {ENSEMBLE_MODEL_PATH}")
#     return model

# def predict_with_ensemble_model(df):
#     """Loads and predicts with the saved LightGBM model."""
#     feature_cols = get_feature_columns(df)
#     X_test = df[feature_cols]

#     model = joblib.load(ENSEMBLE_MODEL_PATH)
#     predictions = model.predict(X_test)
#     return predictions

# src/ensemble_model.py

from sklearn.ensemble import VotingRegressor, AdaBoostRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import joblib
import os
import pandas as pd

from src.config import (
    SVM_PARAMS,
    DT_PARAMS,
    ADABOOST_PARAMS,
    VOTING_WEIGHTS,
    ENSEMBLE_MODEL_PATH,
    RANDOM_STATE,
)

def get_feature_columns(df):
    """Identifies the feature columns for the ensemble model."""
    # Exclude identifiers, raw text, and the original target
    exclude_cols = [
        'sample_id', 'catalog_content', 'image_links', 'image_link', 'price',
        'price_class', 'value', 'unit', 'standardised_units'
    ]
    feature_cols = [col for col in df.columns if col not in exclude_cols]
    return feature_cols

def train_ensemble_model(df, target_col):
    """Trains and saves the Voting Regressor model (SVR + DecisionTree + AdaBoost)."""
    feature_cols = get_feature_columns(df)
    X_train = df[feature_cols]
    y_train = df[target_col]

    # Base learners
    svr = make_pipeline(StandardScaler(), SVR(**SVM_PARAMS))
    dtr = DecisionTreeRegressor(random_state=RANDOM_STATE, **DT_PARAMS)
    ada = AdaBoostRegressor(random_state=RANDOM_STATE, **ADABOOST_PARAMS)

    # Voting ensemble
    model = VotingRegressor(
        estimators=[
            ("svr", svr),
            ("dt", dtr),
            ("ada", ada),
        ],
        weights=VOTING_WEIGHTS,
        n_jobs=None,  # VotingRegressor in sklearn doesn't support n_jobs (only in some versions for estimators); keep None
    )

    model.fit(X_train, y_train)

    # Save the model
    os.makedirs(os.path.dirname(ENSEMBLE_MODEL_PATH), exist_ok=True)
    joblib.dump(model, ENSEMBLE_MODEL_PATH)
    print(f"Ensemble model saved to {ENSEMBLE_MODEL_PATH}")
    return model

def predict_with_ensemble_model(df):
    """Loads and predicts with the saved Voting Regressor model."""
    feature_cols = get_feature_columns(df)
    X_test = df[feature_cols]

    model = joblib.load(ENSEMBLE_MODEL_PATH)
    predictions = model.predict(X_test)
    return predictions