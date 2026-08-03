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
