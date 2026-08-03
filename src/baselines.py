#!/usr/bin/env python
# coding: utf-8

"""
MASTER BASELINE BENCHMARK

Methods
--------
1. PCMCI
2. CBNB-e
3. NBCB-e
4. GCMVL
5. DYNOTEARS

Outputs
-------
Precision
Recall
F1
Adjacency F1
Orientation F1
Runtime
CSV Summary
"""

import os
import sys
import time
import traceback

import numpy as np
import pandas as pd

# =============================================
# Repository
# =============================================

REPO_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "Case_Studies_of_Causal_Discovery_from_IT_Monitoring_Time_Series"
)

sys.path.insert(0, REPO_PATH)

# =============================================
# CausalLearn
# =============================================

from causallearn.graph.GeneralGraph import GeneralGraph
from causallearn.graph.GraphNode import GraphNode
from causallearn.graph.Edge import Edge
from causallearn.graph.Endpoint import Endpoint

import Algorithms.algorithms as algorithms

from Algorithms.Hybrids_of_CB_and_NB.cbnb_e import CBNBe
from Algorithms.Hybrids_of_CB_and_NB.nbcb_e import NBCBe


# =============================================
# DATASETS
# =============================================

BASE = "data"

DATASET_CONFIGS = {

    "Antivirus":{

        "files":[
            f"{BASE}/Antivirus_Activity/preprocessed_1.csv",
            f"{BASE}/Antivirus_Activity/preprocessed_2.csv"
        ],

        "structure":
            f"{BASE}/Antivirus_Activity/structure.txt",

        "drop_col":"timestamp",
        "index_col":0,

    },

    "Web_Activity":{

        "files":[
            f"{BASE}/Web_Activity/preprocessed_1.csv",
            f"{BASE}/Web_Activity/preprocessed_2.csv"
        ],

        "structure":
            f"{BASE}/Web_Activity/structure.txt",

        "drop_col":"timestamp",
        "index_col":0,

    },

    "Middleware":{

        "files":[
            f"{BASE}/Middleware_oriented_message_Activity/monitoring_metrics_1.csv",
            f"{BASE}/Middleware_oriented_message_Activity/monitoring_metrics_2.csv"
        ],

        "structure":
            f"{BASE}/Middleware_oriented_message_Activity/structure.txt",

        "drop_col":"timestamp",
        "index_col":0,

    },

    "Storm":{

        "files":[
            f"{BASE}/Storm_Ingestion_Activity/storm_data_normal.csv"
        ],

        "structure":
            f"{BASE}/Storm_Ingestion_Activity/storm_structure.txt",

        "drop_col":"Unnamed: 0",
        "index_col":None,

    }

}

# =============================================
# PARAMETERS
# =============================================

TAU_MAX = 3
SIG_LEVEL = 0.05

METHODS = [

    "PCMCI",
    "CBNB_e",
    "NBCB_e",
    "GCMVL",
    "DYNOTEARS"

]
# =============================================
# LOAD DATASET
# =============================================

def load_dataset(config):

    dfs = []

    for file in config["files"]:

        df = pd.read_csv(
            file,
            index_col=config["index_col"]
        )

        dfs.append(df)

    data = pd.concat(
        dfs,
        ignore_index=True
    )

    if config["drop_col"] in data.columns:
        data = data.drop(columns=[config["drop_col"]])

    data.columns = (
        data.columns
        .str.strip()
        .str.replace(" ", "_")
    )

    data = data.select_dtypes(include=[np.number])

    return data


# =============================================
# LOAD GROUND TRUTH
# =============================================

def load_ground_truth(structure_file):

    true_edges = set()

    with open(structure_file, "r") as f:

        for line in f:

            line = line.strip()

            if line == "":
                continue

            parts = line.split()

            if len(parts) != 2:
                continue

            true_edges.add(
                (parts[0], parts[1])
            )

    return true_edges


# =============================================
# GENERALGRAPH -> EDGE SET
# =============================================

def graph_to_edges(graph):

    predicted = set()

    if graph is None:
        return predicted

    for edge in graph.get_graph_edges():

        node1 = edge.get_node1().get_name()
        node2 = edge.get_node2().get_name()

        endpoint1 = edge.get_endpoint1()
        endpoint2 = edge.get_endpoint2()

        if endpoint1 == Endpoint.TAIL and endpoint2 == Endpoint.ARROW:

            predicted.add(
                (node1, node2)
            )

        elif endpoint1 == Endpoint.ARROW and endpoint2 == Endpoint.TAIL:

            predicted.add(
                (node2, node1)
            )

        elif endpoint1 == Endpoint.ARROW and endpoint2 == Endpoint.ARROW:

            predicted.add((node1, node2))
            predicted.add((node2, node1))

    return predicted


# =============================================
# EVALUATION
# =============================================

def evaluate(predicted, truth):

    tp = predicted & truth
    fp = predicted - truth
    fn = truth - predicted

    precision = (
        len(tp) / (len(tp) + len(fp))
        if len(tp) + len(fp) > 0
        else 0
    )

    recall = (
        len(tp) / (len(tp) + len(fn))
        if len(tp) + len(fn) > 0
        else 0
    )

    f1 = (
        2 * precision * recall /
        (precision + recall)
        if precision + recall > 0
        else 0
    )

    return {

        "TP": len(tp),
        "FP": len(fp),
        "FN": len(fn),
        "Precision": precision,
        "Recall": recall,
        "F1": f1

    }
# =============================================
# RUN METHOD
# =============================================

def run_method(method, data):

    start = time.time()

    try:

        if method == "PCMCI":

            graph = algorithms.pcmciplus(
                data,
                tau_max=TAU_MAX,
                sig_level=SIG_LEVEL
            )

        elif method == "CBNB_e":

            model = CBNBe(
                data,
                TAU_MAX,
                SIG_LEVEL,
                model="linear",
                indtest="linear",
                cond_indtest="linear"
            )

            model.run()

            graph = model.causal_graph

        elif method == "NBCB_e":

            model = NBCBe(
                data,
                TAU_MAX,
                SIG_LEVEL,
                model="linear",
                indtest="linear",
                cond_indtest="linear"
            )

            model.run()

            graph = model.causal_graph

        elif method == "GCMVL":

            graph = algorithms.granger_lasso(
                data,
                tau_max=TAU_MAX,
                sig_level=SIG_LEVEL
            )

        elif method == "DYNOTEARS":

            graph = algorithms.dynotears(
                data,
                tau_max=TAU_MAX,
                sig_level=SIG_LEVEL
            )

        else:

            raise ValueError(f"Unknown method: {method}")

        runtime = time.time() - start

        return graph, runtime

    except Exception as e:

        traceback.print_exc()

        runtime = time.time() - start

        return None, runtime


# =============================================
# MAIN BENCHMARK
# =============================================

results = []

for dataset_name, config in DATASET_CONFIGS.items():

    print("\n" + "=" * 70)
    print(dataset_name)
    print("=" * 70)

    data = load_dataset(config)

    truth = load_ground_truth(config["structure"])

    for method in METHODS:

        print(f"\nRunning {method}")

        graph, runtime = run_method(method, data)

        predicted = graph_to_edges(graph)

        metrics = evaluate(predicted, truth)

        print(f"Predicted edges: {len(predicted)}")
        print(f"Precision: {metrics['Precision']:.3f}")
        print(f"Recall: {metrics['Recall']:.3f}")
        print(f"F1: {metrics['F1']:.3f}")
        print(f"Runtime: {runtime:.2f} s")

        results.append({

            "Dataset": dataset_name,
            "Method": method,
            "TP": metrics["TP"],
            "FP": metrics["FP"],
            "FN": metrics["FN"],
            "Precision": metrics["Precision"],
            "Recall": metrics["Recall"],
            "F1": metrics["F1"],
            "Runtime": runtime

        })

results_df = pd.DataFrame(results)

results_df.to_csv(
    "baseline_results.csv",
    index=False
)

print("\n")
print("=" * 70)
print("Benchmark Complete")
print("=" * 70)

print(results_df)