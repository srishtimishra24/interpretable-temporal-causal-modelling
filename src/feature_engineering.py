import pandas as pd
from config import LAG_COUNT

def add_lags(df, lags=LAG_COUNT):
    lagged_list = []
    for lag in range(1, lags + 1):
        shifted = df.shift(lag).copy()
        shifted.columns = [f"{col}_lag{lag}" for col in df.columns]
        lagged_list.append(shifted)
    return pd.concat(lagged_list, axis=1).fillna(0)


# ============================================
# STEP 5 — BASE VARIABLE EXTRACTION
# Longer suffixes checked first — prevents
# _small_increase being parsed as _increase.
# ============================================

EVENT_SUFFIXES = ["_small_increase", "_increase", "_decrease", "_spike"]

def extract_base_variable(feature_name):
    for lag in range(1, LAG_COUNT + 1):
        lag_suffix = f"_lag{lag}"
        if feature_name.endswith(lag_suffix):
            feature_name = feature_name[: -len(lag_suffix)]
            break
    for suffix in EVENT_SUFFIXES:
        if feature_name.endswith(suffix):
            feature_name = feature_name[: -len(suffix)]
            break
    return feature_name

