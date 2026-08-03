import numpy as np
from config import STABILITY_THRESHOLD, N_RUNS
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
            kept   = "KEPT  " if cnt >= min_votes else "PRUNED"
            print(f"    {kept}  {var:<42} [{bar}] {cnt}/{N_RUNS}")

