#!/usr/bin/env python
# coding: utf-8

# ============================================
# MASTER TSETLIN MACHINE CAUSAL DISCOVERY
# Fully automatic pipeline - no manual steps
#
# Edge selection: stability selection only
#   - Run TM N_RUNS times per target
#   - Count how many runs each variable appears in
#   - Keep variables appearing in >= STABILITY_THRESHOLD runs
#   - No hand-tuned per-dataset thresholds
#
# Interpretability: vote counts drive the exhibit
#   - Highest vote count = most stable edge
#   - Display shows exact run-by-run stability
# ============================================

import numpy as np
import pandas as pd
import random
import networkx as nx
import matplotlib.pyplot as plt
import time

from collections import defaultdict
from sklearn.feature_selection import mutual_info_classif
from pyTsetlinMachine.tm import MultiClassTsetlinMachine


# ============================================
# REPRODUCIBILITY
# ============================================

np.random.seed(42)
random.seed(42)


# ============================================
# DATASET CONFIGURATIONS
# ============================================

DATASET_CONFIGS = {
    "Antivirus": {
        "files": [
            "data/Antivirus_Activity/preprocessed_1.csv",
            "data/Antivirus_Activity/preprocessed_2.csv",
        ],
        "structure": "data/Antivirus_Activity/structure.txt",
        "drop_col":  "timestamp",
        "manual_targets": [
            "Default_Transaction",
            "Chargement_portail",
            "Chargement_IE",
        ],
    },
    "Web_Activity": {
        "files": [
            "data/Web_Activity/preprocessed_1.csv",
            "data/Web_Activity/preprocessed_2.csv",
        ],
        "structure": "data/Web_Activity/structure.txt",
        "drop_col":  "timestamp",
        "manual_targets": [
            "Cpu_global",
            "Net_Out_Global",
            "Cpu_http",
            "Cpu_php",
            "Nb_connection_mysql",
        ],
    },
    "Middleware": {
        "files": [
            "data/Middleware_oriented_message_Activity/monitoring_metrics_1.csv",
            "data/Middleware_oriented_message_Activity/monitoring_metrics_2.csv",
        ],
        "structure": "data/Middleware_oriented_message_Activity/structure.txt",
        "drop_col":  "timestamp",
        "manual_targets": [
            "messages_causality_1",
            "cpu_global_prct",
            "ram_global_prct",
            "disk_io_write_mega_byte",
            "disk_io_read_mega_byte",
        ],
    },
    "Storm": {
        "files": [
            "data/Storm_Ingestion_Activity/storm_data_normal.csv",
        ],
        "structure": "data/Storm_Ingestion_Activity/storm_structure.txt",
        "drop_col":  "Unnamed: 0",
        "manual_targets": [
            "message_dispatcher_bolt",
            "check_message_bolt",
            "metric_bolt",
            "capacity_last_metric_bolt",
            "Real_time_merger_bolt",
            "group_status_information_bolt",
            "capacity_elastic_search_bolt",
        ],
    },
}


# ============================================
# TM HYPERPARAMETERS
# ============================================

TM_CLAUSES            = 200
TM_THRESHOLD          = 20
TM_S                  = 5.0
TM_EPOCHS             = 100
MI_TOP_K              = 15
LAG_COUNT             = 3
N_RUNS                = 10
STABILITY_THRESHOLD   = 0.7   # variable must appear in 70%+ of runs
CLAUSE_CORR_THRESHOLD = 0.3   # per-run activation threshold


# ============================================
# TARGET MODE
# "automatic" — sweeps ALL variables (default)
# ============================================

TARGET_MODE = "automatic"


# ============================================
# STEP 1 — LOAD DATA
# ============================================

def load_dataset(config):
    frames = [pd.read_csv(f) for f in config["files"]]
    df = pd.concat(frames, ignore_index=True)
    drop_col = config["drop_col"]
    if drop_col in df.columns:
        df = df.drop(columns=[drop_col])
    return df


# ============================================
# STEP 2 — LOAD GROUND TRUTH
# ============================================

def load_ground_truth(structure_path):
    true_edges = set()
    with open(structure_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) == 2:
                src, dst = parts
                true_edges.add((src, dst))
    return true_edges


# ============================================
# STEP 3 — EVENT TRANSFORMATION
# Percentile-based thresholds adapt to each
# column's distribution automatically.
# ============================================

def create_events(df):
    event_cols = {}
    for col in df.columns:
        delta = df[col].diff().fillna(0)

        if delta.abs().max() == 0:
            continue

        pos_deltas = delta[delta > 0]
        neg_deltas = delta[delta < 0]

        p75  = pos_deltas.quantile(0.75) if len(pos_deltas) > 0 else 0
        p90  = pos_deltas.quantile(0.90) if len(pos_deltas) > 0 else 0
        p60  = pos_deltas.quantile(0.60) if len(pos_deltas) > 0 else 0
        p25n = neg_deltas.quantile(0.25) if len(neg_deltas) > 0 else 0

        p75  = p75  if p75  > 0 else float("inf")
        p90  = p90  if p90  > 0 else float("inf")
        p60  = p60  if p60  > 0 else float("inf")
        p25n = p25n if p25n < 0 else float("-inf")

        event_cols[col + "_increase"]       = (delta > p75).astype(int)
        event_cols[col + "_decrease"]       = (delta < p25n).astype(int)
        event_cols[col + "_spike"]          = (delta > p90).astype(int)
        event_cols[col + "_small_increase"] = (delta > p60).astype(int)

    return pd.DataFrame(event_cols).fillna(0)


# ============================================
# STEP 4 — TEMPORAL LAGS
# ============================================

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


# ============================================
# STEP 6 — SINGLE-RUN CLAUSE ACTIVATION SCORING
#
# For each run, scores every candidate variable
# by summing clause firing rate differentials
# (rate_on - rate_off) across all informative clauses,
# weighted by lag distance.
#
# Then returns only the TOP_K_PER_RUN highest-scoring
# variables as the nominated causes for this run.
#
# This forces selectivity: each run nominates at most
# TOP_K_PER_RUN causes. Stability selection then counts
# how many runs nominate each variable. Only variables
# nominated in >= STABILITY_THRESHOLD of runs survive.
#
# Without this, every variable passes in every run
# and stability selection cannot distinguish signal
# from noise.
# ============================================

TOP_K_PER_RUN = 3   # each run nominates its top-K causal candidates

def get_implicated_vars(tm, X, feature_names):
    clause_outputs  = tm.transform(X)
    n_clauses_total = clause_outputs.shape[1]
    n_features      = len(feature_names)
    var_scores      = defaultdict(float)

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

            rate_on  = clause_fires[on_mask].mean()
            rate_off = clause_fires[off_mask].mean()
            diff     = rate_on - rate_off

            if diff > 0:
                base_var  = extract_base_variable(feature_names[feat_idx])
                feat_name = feature_names[feat_idx]
                # Lag-weighted score accumulation
                if "_lag1" in feat_name:
                    weight = 3
                elif "_lag2" in feat_name:
                    weight = 2
                else:
                    weight = 1
                var_scores[base_var] += weight * diff

    if not var_scores:
        return set()

    # Return only top-K variables by score — forces selectivity
    top_k = sorted(var_scores.items(), key=lambda x: -x[1])[:TOP_K_PER_RUN]
    return set(var for var, _ in top_k)


# ============================================
# STEP 7 — STABILITY SELECTION
#
# Run TM N_RUNS times. Count how many runs each
# variable is implicated in. Keep only variables
# appearing in >= STABILITY_THRESHOLD * N_RUNS runs.
#
# vote_counts: {var: int} — exact run counts
# stable_vars: {var: int} — filtered to stable only
# ============================================

def stability_selection(X_selected, y, feature_names):
    vote_counts = defaultdict(int)
    min_votes   = int(np.ceil(STABILITY_THRESHOLD * N_RUNS))

    for run_idx in range(N_RUNS):
        tm = MultiClassTsetlinMachine(TM_CLAUSES, TM_THRESHOLD, TM_S)
        tm.fit(X_selected, y, epochs=TM_EPOCHS)
        implicated = get_implicated_vars(tm, X_selected, feature_names)
        for var in implicated:
            vote_counts[var] += 1

    stable_vars = {
        var: count
        for var, count in vote_counts.items()
        if count >= min_votes
    }

    return stable_vars, dict(vote_counts)


# ============================================
# STEP 8 — FULL TM PIPELINE FOR ONE TARGET
# ============================================

def run_tm_for_target(target, event_df, lagged_df):
    print(f"  -> Target: {target}")

    label_col = target + "_increase"
    if label_col not in event_df.columns:
        print(f"     [SKIP] {label_col} not found.")
        return [], {}

    y     = event_df[label_col].values.astype(np.uint32)
    n_pos = int(y.sum())
    n_neg = int(len(y) - n_pos)

    if n_pos < 20 or n_neg < 20:
        print(f"     [SKIP] Too few examples "
              f"(pos={n_pos}, neg={n_neg}).")
        return [], {}

    X_cols = [c for c in lagged_df.columns if target not in c]
    X      = lagged_df[X_cols]

    mi        = mutual_info_classif(X, y, random_state=42)
    mi_series = pd.Series(mi, index=X.columns)
    n_select  = min(MI_TOP_K, len(mi_series))
    top_features = (
        mi_series
        .sort_values(ascending=False)
        .head(n_select)
        .index.tolist()
    )

    X_selected    = X[top_features].values.astype(np.uint32)
    feature_names = top_features

    stable_vars, vote_counts = stability_selection(
        X_selected, y, feature_names
    )

    min_votes = int(np.ceil(STABILITY_THRESHOLD * N_RUNS))

    edges = [
        (var, target, vote_counts[var])
        for var in stable_vars
        if var != target
    ]

    print(f"     Edges found: {len(edges)}  "
          f"(>= {min_votes}/{N_RUNS} runs)")

    return edges, vote_counts


# ============================================
# STEP 9 — TARGET SELECTION
# ============================================

def get_targets(mode, config, df_columns, true_edges):
    if mode == "automatic":
        return sorted(df_columns.tolist())
    elif mode == "gt_auto":
        return sorted(set(dst for _, dst in true_edges))
    elif mode == "manual":
        return config["manual_targets"]
    else:
        raise ValueError(f"Unknown TARGET_MODE: {mode}")


# ============================================
# STEP 10 — EVALUATE
# ============================================

def evaluate(predicted_edges, true_edges):
    pred = set((src, dst) for src, dst, _ in predicted_edges)
    true = true_edges

    tp = pred & true
    fp = pred - true
    fn = true - pred

    precision = len(tp) / (len(tp) + len(fp) + 1e-9)
    recall    = len(tp) / (len(tp) + len(fn) + 1e-9)
    f1        = (2 * precision * recall) / (precision + recall + 1e-9)

    pred_undir = set(tuple(sorted(e)) for e in pred)
    true_undir = set(tuple(sorted(e)) for e in true)

    tp_adj = pred_undir & true_undir
    fp_adj = pred_undir - true_undir
    fn_adj = true_undir - pred_undir

    p_adj  = len(tp_adj) / (len(tp_adj) + len(fp_adj) + 1e-9)
    r_adj  = len(tp_adj) / (len(tp_adj) + len(fn_adj) + 1e-9)
    f1_adj = (2 * p_adj * r_adj) / (p_adj + r_adj + 1e-9)

    return {
        "precision":      round(precision, 3),
        "recall":         round(recall, 3),
        "f1":             round(f1, 3),
        "f1_adjacency":   round(f1_adj, 3),
        "f1_orientation": round(f1, 3),
        "tp":             len(tp),
        "fp":             len(fp),
        "fn":             len(fn),
        "predicted":      len(pred),
        "true":           len(true),
    }


# ============================================
# STEP 11 — VISUALIZE GRAPH
# ============================================

def visualize_graph(edges, dataset_name, mode):
    G = nx.DiGraph()
    for src, dst, _ in edges:
        G.add_edge(src, dst)

    plt.figure(figsize=(12, 7))
    pos = nx.kamada_kawai_layout(G)
    nx.draw(
        G, pos,
        with_labels=True,
        node_color="lightblue",
        node_size=2500,
        font_size=8,
        arrows=True
    )
    plt.title(f"Causal Graph - {dataset_name} ({mode})")
    fname = f"graph_{dataset_name}_{mode}.png"
    plt.savefig(fname, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Graph saved: {fname}")


# ============================================
# STEP 12 — INTERPRETABILITY EXHIBIT
# Shows the most stable discovered edge and the
# full vote breakdown for that target variable.
# vote_counts values are integer run counts (0-N_RUNS).
# ============================================

def interpretability_exhibit(all_edges, all_vote_counts):
    if not all_edges:
        return

    # Best edge = highest integer vote count
    best      = max(all_edges, key=lambda e: e[2])
    src, dst, votes = best
    min_votes = int(np.ceil(STABILITY_THRESHOLD * N_RUNS))

    print(f"\n  [INTERPRETABILITY] Most stable edge: {src} -> {dst}")
    print(f"  Appeared in {votes}/{N_RUNS} runs "
          f"({100*votes//N_RUNS}% stability)")

    if dst in all_vote_counts:
        counts = all_vote_counts[dst]
        print(f"\n  Stability of all candidate causes for '{dst}':")
        print(f"  (threshold: {min_votes}/{N_RUNS} runs to be kept)\n")
        for var, cnt in sorted(counts.items(), key=lambda x: -x[1]):
            bar    = "█" * cnt + "░" * (N_RUNS - cnt)
            kept   = "✓ KEPT  " if cnt >= min_votes else "✗ PRUNED"
            print(f"    {kept}  {var:<42} [{bar}] {cnt}/{N_RUNS}")


# ============================================
# RUN ONE DATASET
# ============================================

def run_dataset(dataset_name, config, mode):
    print(f"\n{'='*60}")
    print(f"DATASET: {dataset_name}  |  MODE: {mode}")
    print(f"{'='*60}")

    t_start    = time.time()
    df         = load_dataset(config)
    true_edges = load_ground_truth(config["structure"])
    event_df   = create_events(df)
    lagged_df  = add_lags(event_df)
    targets    = get_targets(mode, config, df.columns, true_edges)

    if mode == "manual":
        targets = [t for t in targets
                   if t + "_increase" in event_df.columns]

    min_votes = int(np.ceil(STABILITY_THRESHOLD * N_RUNS))
    print(f"Targets ({len(targets)}): {targets}")
    print(f"Stability threshold: {STABILITY_THRESHOLD} "
          f"({min_votes}/{N_RUNS} runs)\n")

    all_edges       = []
    all_vote_counts = {}

    for target in targets:
        edges, vote_counts = run_tm_for_target(
            target, event_df, lagged_df
        )
        all_edges.extend(edges)
        if vote_counts:
            all_vote_counts[target] = vote_counts

    # Deduplicate: keep highest vote count per src->dst pair
    edge_map = {}
    for src, dst, votes in all_edges:
        key = (src, dst)
        if key not in edge_map or votes > edge_map[key]:
            edge_map[key] = votes
    all_edges = [(src, dst, v) for (src, dst), v in edge_map.items()]

    t_elapsed = time.time() - t_start
    print(f"\nTotal edges after deduplication: {len(all_edges)}")
    print(f"Runtime: {t_elapsed:.1f}s")

    metrics            = evaluate(all_edges, true_edges)
    metrics["runtime_s"] = round(t_elapsed, 1)

    interpretability_exhibit(all_edges, all_vote_counts)
    visualize_graph(all_edges, dataset_name, mode)

    return metrics


# ============================================
# MAIN
# ============================================

def main():
    all_results = []

    for dataset_name, config in DATASET_CONFIGS.items():
        metrics            = run_dataset(dataset_name, config, TARGET_MODE)
        metrics["dataset"] = dataset_name
        metrics["mode"]    = TARGET_MODE
        all_results.append(metrics)

    print(f"\n{'='*60}")
    print("SUMMARY RESULTS")
    print(f"{'='*60}")

    results_df = pd.DataFrame(all_results)[[
        "dataset", "mode",
        "precision", "recall", "f1",
        "f1_adjacency", "f1_orientation",
        "tp", "fp", "fn", "predicted", "true",
        "runtime_s"
    ]]

    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 120)
    print(results_df.to_string(index=False))

    output_file = f"results_TM_{TARGET_MODE}.csv"
    results_df.to_csv(output_file, index=False)
    print(f"\nResults saved to: {output_file}")


if __name__ == "__main__":
    main()