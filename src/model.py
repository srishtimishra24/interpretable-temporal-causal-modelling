"""
Core Tsetlin Machine implementation for temporal causal discovery.

This module contains the model training pipeline, clause activation
analysis, and stability selection used to identify causal relationships
in multivariate time-series data.
"""

from collections import defaultdict

import numpy as np
import pandas as pd
from pyTsetlinMachine.tm import MultiClassTsetlinMachine
from sklearn.feature_selection import mutual_info_classif

from config import (
    TM_CLAUSES,
    TM_THRESHOLD,
    TM_S,
    TM_EPOCHS,
    MI_TOP_K,
    N_RUNS,
    STABILITY_THRESHOLD,
)

from feature_engineering import extract_base_variable


# ============================================
# TSETLIN MACHINE PARAMETERS
# ============================================

TOP_K_PER_RUN = 3


# ============================================
# SINGLE-RUN CLAUSE ACTIVATION SCORING
# ============================================

def get_implicated_vars(tm, X, feature_names):
    """
    Identify the most influential variables for a trained
    Tsetlin Machine using clause activation analysis.
    """

    clause_outputs = tm.transform(X)
    n_clauses_total = clause_outputs.shape[1]
    n_features = len(feature_names)

    var_scores = defaultdict(float)

    for clause_idx in range(n_clauses_total):

        clause_fires = clause_outputs[:, clause_idx]

        fire_rate = clause_fires.mean()

        if fire_rate < 0.01 or fire_rate > 0.99:
            continue

        for feat_idx in range(n_features):

            feature_active = X[:, feat_idx]

            on_mask = feature_active == 1
            off_mask = feature_active == 0

            if on_mask.sum() == 0 or off_mask.sum() == 0:
                continue

            rate_on = clause_fires[on_mask].mean()
            rate_off = clause_fires[off_mask].mean()

            diff = rate_on - rate_off

            if diff > 0:

                base_var = extract_base_variable(feature_names[feat_idx])
                feat_name = feature_names[feat_idx]

                if "_lag1" in feat_name:
                    weight = 3
                elif "_lag2" in feat_name:
                    weight = 2
                else:
                    weight = 1

                var_scores[base_var] += weight * diff

    if not var_scores:
        return set()

    top_k = sorted(
        var_scores.items(),
        key=lambda x: -x[1]
    )[:TOP_K_PER_RUN]

    return {var for var, _ in top_k}


# ============================================
# STABILITY SELECTION
# ============================================

def stability_selection(X_selected, y, feature_names):
    """
    Train the Tsetlin Machine multiple times and retain
    variables that consistently appear across runs.
    """

    vote_counts = defaultdict(int)
    min_votes = int(np.ceil(STABILITY_THRESHOLD * N_RUNS))

    for _ in range(N_RUNS):

        tm = MultiClassTsetlinMachine(
            TM_CLAUSES,
            TM_THRESHOLD,
            TM_S,
        )

        tm.fit(
            X_selected,
            y,
            epochs=TM_EPOCHS,
        )

        implicated = get_implicated_vars(
            tm,
            X_selected,
            feature_names,
        )

        for var in implicated:
            vote_counts[var] += 1

    stable_vars = {
        var: count
        for var, count in vote_counts.items()
        if count >= min_votes
    }

    return stable_vars, dict(vote_counts)


# ============================================
# RUN PIPELINE FOR A SINGLE TARGET VARIABLE
# ============================================

def run_tm_for_target(target, event_df, lagged_df):
    """
    Execute the complete Tsetlin Machine pipeline for a
    single target variable.
    """

    print(f"  -> Target: {target}")

    label_col = f"{target}_increase"

    if label_col not in event_df.columns:
        print(f"     [SKIP] {label_col} not found.")
        return [], {}

    y = event_df[label_col].values.astype(np.uint32)

    n_pos = int(y.sum())
    n_neg = len(y) - n_pos

    if n_pos < 20 or n_neg < 20:
        print(
            f"     [SKIP] Too few examples "
            f"(pos={n_pos}, neg={n_neg})."
        )
        return [], {}

    X_cols = [
        col
        for col in lagged_df.columns
        if target not in col
    ]

    X = lagged_df[X_cols]

    mi = mutual_info_classif(
        X,
        y,
        random_state=42,
    )

    mi_series = pd.Series(
        mi,
        index=X.columns,
    )

    n_select = min(MI_TOP_K, len(mi_series))

    top_features = (
        mi_series
        .sort_values(ascending=False)
        .head(n_select)
        .index
        .tolist()
    )

    X_selected = X[top_features].values.astype(np.uint32)

    stable_vars, vote_counts = stability_selection(
        X_selected,
        y,
        top_features,
    )

    min_votes = int(np.ceil(STABILITY_THRESHOLD * N_RUNS))

    edges = [
        (var, target, vote_counts[var])
        for var in stable_vars
        if var != target
    ]

    print(
        f"     Edges found: {len(edges)} "
        f"(>= {min_votes}/{N_RUNS} runs)"
    )

    return edges, vote_counts