"""
Main entry point for the Interpretable Temporal Causal Modelling framework.
"""

import time
import pandas as pd

from config import DATASET_CONFIGS, TARGET_MODE
from data_loader import load_dataset, load_ground_truth
from preprocessing import create_events
from feature_engineering import add_lags
from model import run_tm_for_target
from evaluation import evaluate
from visualization import visualize_graph
from interpretability import interpretability_exhibit


def get_targets(mode, config, df_columns, true_edges):
    """Return the list of target variables based on the selected mode."""

    if mode == "automatic":
        return sorted(df_columns.tolist())

    elif mode == "gt_auto":
        return sorted(set(dst for _, dst in true_edges))

    elif mode == "manual":
        return config["manual_targets"]

    else:
        raise ValueError(f"Unknown mode: {mode}")


def run_dataset(dataset_name, config, mode):
    """Run the complete causal discovery pipeline for one dataset."""

    print("=" * 60)
    print(f"Dataset : {dataset_name}")
    print(f"Mode    : {mode}")
    print("=" * 60)

    start = time.time()

    # Load data
    df = load_dataset(config)
    true_edges = load_ground_truth(config["structure"])

    # Feature engineering
    event_df = create_events(df)
    lagged_df = add_lags(event_df)

    # Choose targets
    targets = get_targets(mode, config, df.columns, true_edges)

    if mode == "manual":
        targets = [
            t for t in targets
            if f"{t}_increase" in event_df.columns
        ]

    all_edges = []
    all_vote_counts = {}

    for target in targets:

        edges, vote_counts = run_tm_for_target(
            target,
            event_df,
            lagged_df,
        )

        all_edges.extend(edges)

        if vote_counts:
            all_vote_counts[target] = vote_counts

    # Remove duplicate edges
    edge_map = {}

    for src, dst, votes in all_edges:

        key = (src, dst)

        if key not in edge_map or votes > edge_map[key]:
            edge_map[key] = votes

    all_edges = [
        (src, dst, votes)
        for (src, dst), votes in edge_map.items()
    ]

    runtime = time.time() - start

    metrics = evaluate(all_edges, true_edges)
    metrics["runtime_s"] = round(runtime, 2)

    interpretability_exhibit(
        all_edges,
        all_vote_counts,
    )

    visualize_graph(
        all_edges,
        dataset_name,
        mode,
    )

    return metrics


def main():

    results = []

    for dataset_name, config in DATASET_CONFIGS.items():

        metrics = run_dataset(
            dataset_name,
            config,
            TARGET_MODE,
        )

        metrics["dataset"] = dataset_name
        metrics["mode"] = TARGET_MODE

        results.append(metrics)

    results_df = pd.DataFrame(results)

    print("\nSummary Results\n")
    print(results_df)

    results_df.to_csv(
        f"results/results_TM_{TARGET_MODE}.csv",
        index=False,
    )


if __name__ == "__main__":
    main()