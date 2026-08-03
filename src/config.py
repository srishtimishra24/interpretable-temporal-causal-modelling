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
import random


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