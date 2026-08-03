import pandas as pd
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
