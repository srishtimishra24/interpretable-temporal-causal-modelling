import matplotlib.pyplot as plt
import networkx as nx
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
    fname = f"images/graph_{dataset_name}_{mode}.png"
    plt.savefig(fname, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Graph saved: {fname}")
