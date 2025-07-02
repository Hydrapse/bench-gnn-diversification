import networkx as nx
import torch
import os.path as osp

from torch_geometric.utils import to_networkx


def calculate_adj_exponent(data, root: str, a=0.5, b=0.5, c=0.5):
    G = to_networkx(data, to_undirected=True)

    try:
        centrality1 = torch.load(osp.join(root, "degree_centrality"))
        centrality2 = torch.load(osp.join(root, "clustering_coefficients"))
        centrality3 = torch.load(osp.join(root, "eigenvector_centrality"))
    except Exception as e:
        print("Calculating centralities")
        centrality1 = nx.degree_centrality(G)
        centrality1 = torch.tensor(
            [centrality1[node] for node in range(data.num_nodes)]
        )
        torch.save(centrality1, osp.join(root, "degree_centrality"))

        centrality2 = nx.clustering(G)
        centrality2 = torch.tensor(
            [centrality2[node] for node in range(data.num_nodes)]
        )
        torch.save(centrality2, osp.join(root, "clustering_coefficients"))

        centrality3 = nx.eigenvector_centrality(G)
        centrality3 = torch.tensor(
            [centrality3[node] for node in range(data.num_nodes)]
        )
        torch.save(centrality3, osp.join(root, "eigenvector_centrality"))

    for i, centrality in enumerate([centrality1, centrality2, centrality3]):
        num_outliers = int(0.05 * centrality.size()[0])
        arr = centrality

        _, indices = torch.topk(arr.abs(), num_outliers)
        arr[indices] = 1.0  # outliers
        min_value = arr.min()
        max_value = arr[arr != 1.0].max()

        centrality = (arr - min_value) / (max_value - min_value)
        centrality[arr == 1.0] = 1.0
        if i == 0:
            centrality1 = centrality
        if i == 1:
            centrality2 = centrality
        if i == 2:
            centrality3 = centrality

    centrality = a * centrality1 + b * centrality2 + c * centrality3
    min_val = torch.min(centrality)
    max_val = torch.max(centrality)

    centrality = (centrality - min_val) / (max_val - min_val)
    return centrality
