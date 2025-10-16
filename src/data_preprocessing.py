# src/data_preprocessing.py

import re
import pandas as pd
import numpy as np
from scipy import stats
from scipy.stats.mstats import winsorize
import joblib
import os

from src.config import NUM_PRICE_CLASSES, MODEL_DIR

# ----------------- Feature Engineering (from your provided code) -----------------
weight_units = {'g','gram','grams','gramm','gr','kg','k','lb','pound','pounds','oz','ounce','ounces'}
volume_units = {'ml','millilitre','milliliter','mililitro','ltr','liter','liters','l','fl oz','fluid ounce','fluid ounces','fluid ounces'}
count_units  = {'count','ct','each','unit','units','pack','packs','piece','pieces','box','bottle','bottles','bag','bags','case','carton','capsule','jar','pouch','bucket','tea bags','paper cupcake liners'}
length_units = {'cm','mm','m','meter','meters','inch','inches','in','ft','foot','feet'}

def extract_value_unit(text):
    lines = [l.strip() for l in str(text).split("\n") if l.strip()]
    if len(lines) < 2:
        return pd.Series([None, None])
    value_line = lines[-2]
    unit_line = lines[-1]
    match_val = re.search(r"[\d\.]+", value_line)
    value = float(match_val.group()) if match_val else None
    unit = re.sub(r'Unit[s]?:\s*', '', unit_line, flags=re.IGNORECASE).strip()
    return pd.Series([value, unit])

def extract_ipq(text):
    patterns = [r'pack of (\d+)', r'(\d+)\s*count', r'box of (\d+)', r'case of (\d+)', r'(\d+)\s*pieces', r'(\d+)\s*units', r'(\d+)\s*ct', r'(\d+)\s*each']
    text_lower = str(text).lower()
    for pat in patterns:
        match = re.search(pat, text_lower)
        if match:
            return int(match.group(1))
    return 1

def map_unit_class(unit):
    if unit is None: return 'others'
    u = unit.strip().lower()
    if u in weight_units: return 'weight'
    if u in volume_units: return 'volume'
    if u in count_units: return 'count'
    if u in length_units: return 'length'
    return 'others'

def convert_to_standard(value, unit):
    if value is None or unit is None: return value
    u = unit.strip().lower()
    if u in {'oz', 'ounce', 'ounces'}: return value * 28.3495
    if u in {'lb', 'pound', 'pounds'}: return value * 453.592
    if u in {'kg', 'k'}: return value * 1000
    if u in {'g', 'gram', 'grams', 'gramm', 'gr'}: return value
    if u in {'ltr', 'liter', 'liters', 'l'}: return value * 1000
    if u in {'fl oz', 'fluid ounce', 'fluid ounces'}: return value * 29.5735
    if u in {'ml', 'millilitre', 'milliliter', 'mililitro'}: return value
    if u in {'cm'}: return value / 100
    if u in {'mm'}: return value / 1000
    if u in {'inch', 'inches', 'in'}: return value * 0.0254
    if u in {'ft', 'foot', 'feet'}: return value * 0.3048
    if u in {'m', 'meter', 'meters'}: return value
    if u in count_units: return value
    return value

def engineer_features(df):
    """Applies all feature engineering steps."""
    temp_df = df.copy()
    temp_df[['value', 'unit']] = temp_df['catalog_content'].apply(extract_value_unit)
    temp_df['IPQ'] = temp_df['catalog_content'].apply(extract_ipq)
    temp_df['standardised_units'] = temp_df['unit'].apply(map_unit_class)
    temp_df['updated_values'] = temp_df.apply(lambda row: convert_to_standard(row['value'], row['unit']), axis=1)
    
    # Fill NaNs created during feature engineering
    temp_df['updated_values'] = temp_df['updated_values'].fillna(temp_df['updated_values'].median())
    temp_df['IPQ'] = temp_df['IPQ'].fillna(1)
    
    # One-hot encode and ensure all possible columns are present
    unit_dummies = pd.get_dummies(temp_df['standardised_units'], prefix='unit')
    for col in ['unit_weight', 'unit_volume', 'unit_count', 'unit_length', 'unit_others']:
        if col not in unit_dummies.columns:
            unit_dummies[col] = 0
    temp_df = pd.concat([temp_df, unit_dummies], axis=1)
    
    return temp_df

# ----------------- Price Transformation Pipeline -----------------

class PriceProcessor:
    """Handles Box-Cox, Winsorization, and Binning for the price column."""
    def __init__(self):
        self.lambda_ = None
        self.winsor_limits = (0.01, 0.01) # Winsorize 1% from both tails
        self.bin_edges = None
        self.path = os.path.join(MODEL_DIR, "price_processor.joblib")

    def fit_transform(self, price_series):
        # 1. Box-Cox Transform (add 1 to handle prices <= 0)
        transformed_price, self.lambda_ = stats.boxcox(price_series + 1)
        
        # 2. Winsorization
        winsorized_price = winsorize(transformed_price, limits=self.winsor_limits)
        
        # 3. Binning into classes
        binned_price, self.bin_edges = pd.qcut(
            winsorized_price, 
            q=NUM_PRICE_CLASSES, 
            labels=False, 
            retbins=True, 
            duplicates='drop'
        )
        self.save()
        return binned_price

    def save(self):
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        joblib.dump(self, self.path)

    @classmethod
    def load(cls):
        return joblib.load(os.path.join(MODEL_DIR, "price_processor.joblib"))