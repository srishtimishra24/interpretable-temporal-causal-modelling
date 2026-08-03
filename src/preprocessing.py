import pandas as pd
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
