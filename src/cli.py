#!/usr/bin/env python
# coding: utf-8

"""
Tsetlin Machine Causal Discovery — Command Line Interface

Usage
-----
Minimal (no evaluation):
    python tm_causal_cli.py --input data.csv --output graph.txt

With ground truth evaluation:
    python tm_causal_cli.py --input data.csv --output graph.txt \
                            --ground-truth structure.txt

All options:
    python tm_causal_cli.py --input data.csv --output graph.txt \
                            --ground-truth structure.txt \
                            --tau 3 \
                            --runs 10 \
                            --stability 0.7 \
                            --top-k 3 \
                            --clauses 500 \
                            --epochs 100

Output format
-------------
graph.txt: one directed edge per line
    source_variable destination_variable

Example
-------
    python tm_causal_cli.py --input data/Web_Activity/preprocessed_1.csv \
                            --output discovered_graph.txt \
                            --ground-truth data/Web_Activity/structure.txt
"""

import argparse
import time
import sys
import os

import numpy as np
import pandas as pd
from collections import defaultdict
from sklearn.feature_selection import mutual_info_classif
from pyTsetlinMachine.tm import MultiClassTsetlinMachine


# ============================================
# ARGUMENT PARSING
# ============================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Tsetlin Machine Causal Discovery for multivariate time series",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    # Required
    parser.add_argument(
        "--input", required=True,
        help="Path to input CSV file (multivariate time series)"
    )
    parser.add_argument(
        "--output", required=True,
        help="Path to output edge list file"
    )

    # Optional evaluation
    parser.add_argument(
        "--ground-truth", default=None,
        help="Path to ground truth edge list file for evaluation (optional)"
    )

    # Optional column to drop
    parser.add_argument(
        "--drop-col", default=None,
        help="Column to drop from input (e.g. timestamp, Unnamed: 0)"
    )

    # TM hyperparameters
    parser.add_argument(
        "--tau", type=int, default=3,
        help="Maximum temporal lag (default: 3)"
    )
    parser.add_argument(
        "--runs", type=int, default=10,
        help="Number of TM runs for stability selection (default: 10)"
    )
    parser.add_argument(
        "--stability", type=float, default=0.7,
        help="Stability threshold — fraction of runs a variable must "
             "appear in to be kept as a causal edge (default: 0.7)"
    )
    parser.add_argument(
        "--top-k", type=int, default=3,
        help="Top-K candidates nominated per run (default: 3)"
    )
    parser.add_argument(
        "--clauses", type=int, default=500,
        help="Number of TM clauses (default: 500)"
    )
    parser.add_argument(
        "--epochs", type=int, default=100,
        help="TM training epochs per run (default: 100)"
    )
    parser.add_argument(
        "--mi-top-k", type=int, default=15,
        help="Top-K features selected by mutual information (default: 15)"
    )

    return parser.parse_args()


# ============================================
# PIPELINE FUNCTIONS
# ============================================

EVENT_SUFFIXES = ["_small_increase", "_increase", "_decrease", "_spike"]

def extract_base_variable(feature_name, lag_count):
    for lag in range(1, lag_count + 1):
        lag_suffix = f"_lag{lag}"
        if feature_name.endswith(lag_suffix):
            feature_name = feature_name[: -len(lag_suffix)]
            break
    for suffix in EVENT_SUFFIXES:
        if feature_name.endswith(suffix):
            feature_name = feature_name[: -len(suffix)]
            break
    return feature_name


def load_data(input_path, drop_col):
    df = pd.read_csv(input_path)
    if drop_col and drop_col in df.columns:
        df = df.drop(columns=[drop_col])
    # Auto-detect and drop index-like columns
    for col in df.columns:
        if col.lower() in ["unnamed: 0", "index", "timestamp"]:
            df = df.drop(columns=[col])
    df = df.select_dtypes(include=[float, int])
    return df


def load_ground_truth(structure_path):
    true_edges = set()
    with open(structure_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) == 2:
                true_edges.add((parts[0], parts[1]))
    return true_edges


def create_events(df):
    event_cols = {}
    for col in df.columns:
        delta = df[col].diff().fillna(0)
        if delta.abs().max() == 0:
            continue
        pos_deltas = delta[delta > 0]
        neg_deltas = delta[delta < 0]

        p75  = pos_deltas.quantile(0.75) if len(pos_deltas) > 0 else float("inf")
        p90  = pos_deltas.quantile(0.90) if len(pos_deltas) > 0 else float("inf")
        p60  = pos_deltas.quantile(0.60) if len(pos_deltas) > 0 else float("inf")
        p25n = neg_deltas.quantile(0.25) if len(neg_deltas) > 0 else float("-inf")

        p75  = p75  if p75  > 0 else float("inf")
        p90  = p90  if p90  > 0 else float("inf")
        p60  = p60  if p60  > 0 else float("inf")
        p25n = p25n if p25n < 0 else float("-inf")

        event_cols[col + "_increase"]       = (delta > p75).astype(int)
        event_cols[col + "_decrease"]       = (delta < p25n).astype(int)
        event_cols[col + "_spike"]          = (delta > p90).astype(int)
        event_cols[col + "_small_increase"] = (delta > p60).astype(int)

    return pd.DataFrame(event_cols).fillna(0)


def add_lags(df, lags):
    lagged_list = []
    for lag in range(1, lags + 1):
        shifted = df.shift(lag).copy()
        shifted.columns = [f"{col}_lag{lag}" for col in df.columns]
        lagged_list.append(shifted)
    return pd.concat(lagged_list, axis=1).fillna(0)


def get_implicated_vars(tm, X, feature_names, top_k, lag_count):
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
            on_mask  = feature_active == 1
            off_mask = feature_active == 0
            if on_mask.sum() == 0 or off_mask.sum() == 0:
                continue

            diff = clause_fires[on_mask].mean() - clause_fires[off_mask].mean()
            if diff > 0:
                base_var  = extract_base_variable(feature_names[feat_idx], lag_count)
                feat_name = feature_names[feat_idx]
                weight = 3 if "_lag1" in feat_name else (2 if "_lag2" in feat_name else 1)
                var_scores[base_var] += weight * diff

    if not var_scores:
        return set()

    top = sorted(var_scores.items(), key=lambda x: -x[1])[:top_k]
    return set(var for var, _ in top)


def run_target(target, event_df, lagged_df, args):
    label_col = target + "_increase"
    if label_col not in event_df.columns:
        return []

    y     = event_df[label_col].values.astype(np.uint32)
    n_pos = int(y.sum())
    n_neg = int(len(y) - n_pos)
    if n_pos < 20 or n_neg < 20:
        return []

    X_cols = [c for c in lagged_df.columns if target not in c]
    X      = lagged_df[X_cols]

    mi        = mutual_info_classif(X, y, random_state=42)
    mi_series = pd.Series(mi, index=X.columns)
    n_select  = min(args.mi_top_k, len(mi_series))
    top_features = mi_series.sort_values(ascending=False).head(n_select).index.tolist()

    X_selected    = X[top_features].values.astype(np.uint32)
    feature_names = top_features

    vote_counts = defaultdict(int)
    for _ in range(args.runs):
        tm = MultiClassTsetlinMachine(args.clauses, 20, 5.0)
        tm.fit(X_selected, y, epochs=args.epochs)
        implicated = get_implicated_vars(
            tm, X_selected, feature_names, args.top_k, args.tau
        )
        for var in implicated:
            vote_counts[var] += 1

    min_votes = int(np.ceil(args.stability * args.runs))
    return [
        (var, target)
        for var, count in vote_counts.items()
        if var != target and count >= min_votes
    ]


def evaluate(predicted_edges, true_edges):
    pred = set(predicted_edges)
    true = true_edges

    tp = pred & true
    fp = pred - true
    fn = true - pred

    precision = len(tp) / (len(tp) + len(fp) + 1e-9)
    recall    = len(tp) / (len(tp) + len(fn) + 1e-9)
    f1        = 2 * precision * recall / (precision + recall + 1e-9)

    pred_undir = set(tuple(sorted(e)) for e in pred)
    true_undir = set(tuple(sorted(e)) for e in true)
    tp_adj = pred_undir & true_undir
    fp_adj = pred_undir - true_undir
    fn_adj = true_undir - pred_undir
    p_adj  = len(tp_adj) / (len(tp_adj) + len(fp_adj) + 1e-9)
    r_adj  = len(tp_adj) / (len(tp_adj) + len(fn_adj) + 1e-9)
    f1_adj = 2 * p_adj * r_adj / (p_adj + r_adj + 1e-9)

    return {
        "precision":      round(precision, 3),
        "recall":         round(recall, 3),
        "f1_orientation": round(f1, 3),
        "f1_adjacency":   round(f1_adj, 3),
        "tp": len(tp), "fp": len(fp), "fn": len(fn),
        "predicted": len(pred), "true": len(true),
    }


# ============================================
# MAIN
# ============================================

def main():
    args = parse_args()
    t_start = time.time()

    print("=" * 60)
    print("Tsetlin Machine Causal Discovery")
    print("=" * 60)
    print(f"Input:       {args.input}")
    print(f"Output:      {args.output}")
    print(f"Ground truth:{args.ground_truth or 'None (no evaluation)'}")
    print(f"Tau max:     {args.tau}")
    print(f"Runs:        {args.runs}")
    print(f"Stability:   {args.stability} "
          f"({int(np.ceil(args.stability * args.runs))}/{args.runs} runs)")
    print(f"Top-K/run:   {args.top_k}")
    print(f"Clauses:     {args.clauses}")
    print(f"Epochs:      {args.epochs}")
    print("=" * 60)

    # Load
    print("\nLoading data...")
    df = load_data(args.input, args.drop_col)
    print(f"  Shape: {df.shape}")
    print(f"  Variables: {df.columns.tolist()}")

    # Transform
    print("\nTransforming to events...")
    event_df  = create_events(df)
    lagged_df = add_lags(event_df, args.tau)

    # Run pipeline on all variables
    targets = sorted(df.columns.tolist())
    print(f"\nRunning TM on {len(targets)} targets...")

    all_edges = []
    for target in targets:
        edges = run_target(target, event_df, lagged_df, args)
        if edges:
            print(f"  {target}: {len(edges)} edges found")
        all_edges.extend(edges)

    # Deduplicate
    all_edges = list(set(all_edges))

    # Write output
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w") as f:
        for src, dst in sorted(all_edges):
            f.write(f"{src} {dst}\n")

    t_elapsed = time.time() - t_start
    print(f"\nDiscovered {len(all_edges)} edges")
    print(f"Runtime: {t_elapsed:.1f}s")
    print(f"Graph written to: {args.output}")

    # Evaluate if ground truth provided
    if args.ground_truth:
        print("\n" + "=" * 60)
        print("EVALUATION")
        print("=" * 60)
        true_edges = load_ground_truth(args.ground_truth)
        metrics    = evaluate(set(all_edges), true_edges)
        print(f"  True edges:      {metrics['true']}")
        print(f"  Predicted edges: {metrics['predicted']}")
        print(f"  TP: {metrics['tp']}  FP: {metrics['fp']}  FN: {metrics['fn']}")
        print(f"  Precision:       {metrics['precision']}")
        print(f"  Recall:          {metrics['recall']}")
        print(f"  F1 (orientation):{metrics['f1_orientation']}")
        print(f"  F1 (adjacency):  {metrics['f1_adjacency']}")

    print("\nDone.")


if __name__ == "__main__":
    main()