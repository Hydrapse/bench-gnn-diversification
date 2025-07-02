"""
Methods for node disparity scores calculation
"""
import os
import os.path as osp
import math

import networkx as nx
import nx_cugraph as nxcg
import scipy.sparse as sparse
import numpy as np
import torch
from torch_geometric.data import Data
from torch_geometric.utils import scatter, to_networkx, k_hop_subgraph, to_edge_index, to_undirected
import torch.nn.functional as F
from torch_geometric.utils import spmm
from torch_sparse import SparseTensor
from tqdm import tqdm, trange
from sklearn.preprocessing import StandardScaler

from utils.utils import adj_norm, KNN
import faiss


def get_pagerank(adj, alpha_=0.15, epsilon_=1e-6, max_iter=100):
    adj = adj_norm(adj, norm='rw', add_self_loop=False)
    adj = adj.t()

    num_nodes = adj.size(dim=0)
    s = torch.full((num_nodes,), 1.0 / num_nodes, device=adj.device()).view(-1, 1)
    x = s.clone()

    for i in range(max_iter):
        x_last = x
        x = alpha_ * s + (1 - alpha_) * (adj @ x)
        # check convergence, l1 norm
        if (abs(x - x_last)).sum() < num_nodes * epsilon_:
            # print(f'power-iter      Iterations: {i}, NNZ: {(x.view(-1) > 0).sum()}')
            return x.view(-1)

    return x.view(-1)


def get_degree(adj, undirected=True):
    deg_in = adj.sum(dim=1).to(torch.long)
    deg_out = adj.sum(dim=0).to(torch.long)
    if undirected and not deg_in.equal(deg_out):
        return deg_in + deg_out
    else:
        return deg_in


def get_node_homophily(adj, y, hop=1, path=None):
    if path and os.path.exists(path):
        out = torch.load(path)
        return out.to(y.device)

    y = y.squeeze(-1) if y.dim() > 1 else y
    y = y.argmax(-1) if y.dim() > 1 else y

    adj_list = [adj]
    for h in range(1, hop):
        adj_list.append(adj @ adj_list[-1])
    row, col, _ = adj_list[-1].coo()

    out = torch.zeros(row.size(0), device=row.device)
    out[y[row] == y[col]] = 1.
    out = scatter(out, col, 0, dim_size=y.size(0), reduce='mean').cpu()

    if path is not None:
        torch.save(out, path)

    return out.to(y.device)


def get_clustering_coef(data: Data):
    """
    Support Cuda; Convert to undirected graph
    :param data:
    :return: clustering coefficient
    """
    G = to_networkx(data, to_undirected=True)
    G = nxcg.from_networkx(G)

    cluster_coef = nx.clustering(G)
    cluster_coef = torch.tensor([cluster_coef[node] for node in range(data.num_nodes)])
    cluster_coef = cluster_coef.to(data.y.device)

    return cluster_coef


def get_neighborhood_confusion(_data, _dataset, threshold=0.7, hop=1, path=None,
                               return_nc=False, undirected=False):
    if undirected:
        file_path = osp.join(path, f'subg_{_dataset}_hop{hop}_undir.pt') if path else None
    else:
        file_path = osp.join(path, f'subg_{_dataset}_hop{hop}.pt') if path else None

    # get node-wise subgraphs
    if file_path and os.path.exists(file_path):
        subg_dict = torch.load(file_path)
        subg_dict = {k: v.to(_data.x.device) for k, v in subg_dict.items()}
    else:
        if hasattr(_data, 'edge_index') and _data.edge_index:
            edge_index = _data.edge_index
        else:
            edge_index = to_edge_index(_data.adj_t)[0]
        if undirected:
            edge_index = to_undirected(edge_index)
        subg_dict = {}
        for nid in trange(_data.num_nodes):
            subg_dict[nid] = k_hop_subgraph(nid, hop, edge_index)[0]
        if file_path:
            torch.save({k: v.cpu() for k, v in subg_dict.items()}, file_path)

    nc = torch.empty(_data.num_nodes)
    if len(_data.y.shape) == 1:
        if -1 in _data.y:
            labels = F.one_hot(_data.y + 1)
        else:
            labels = F.one_hot(_data.y)
    else:
        labels = _data.y
    for i, subg in subg_dict.items():
        neigh_labels = torch.index_select(labels, 0, subg)
        if len(neigh_labels):
            nc[i] = len(neigh_labels) / torch.max(torch.sum(neigh_labels, dim=0)).item()
        else:
            nc[i] = 1.0
    nc = nc.to(_data.x.device)

    if return_nc:
        return nc

    # low_nc: 1 ; high_nc: 0
    threshold = 2 ** (threshold * math.log2(labels.shape[1]))
    return (nc <= threshold).int()


def get_directional_label_distance(_data):
    if len(_data.y.shape) == 1:
        y = F.one_hot(_data.y).to(torch.float)
    else:
        y = _data.y

    # _norm = 'rw'
    # neigh_y = adj_norm(_data.adj_t, norm=_norm) @ y
    # neigh_y_rev = adj_norm(_data.adj_t.t(), norm=_norm) @ y

    neigh_y = spmm(_data.adj_t, y, reduce='mean')
    neigh_y_rev = spmm(_data.adj_t.t(), y, reduce='mean')
    dist = torch.norm(neigh_y - neigh_y_rev, p=2, dim=1) / (y.shape[1] ** 0.5)
    return dist


def get_directional_label_similarity(_data):
    if len(_data.y.shape) == 1:
        y = F.one_hot(_data.y).to(torch.float)
    else:
        y = _data.y

    neigh_y = spmm(_data.adj_t, y, reduce='mean')
    neigh_y_rev = spmm(_data.adj_t.t(), y, reduce='mean')
    dist = (neigh_y * neigh_y_rev).sum(dim=1)
    return dist


def get_directional_label_cos_similarity(_data):
    if len(_data.y.shape) == 1:
        y = F.one_hot(_data.y).to(torch.float)
    else:
        y = _data.y

    neigh_y = spmm(_data.adj_t, y, reduce='mean')
    neigh_y_rev = spmm(_data.adj_t.t(), y, reduce='mean')
    dist = F.cosine_similarity(neigh_y, neigh_y_rev, dim=1)
    return dist


def get_direction_informativeness(_data):
    assert not _data.adj_t.is_symmetric()

    in_degree = _data.adj_t.sum(dim=1)
    out_degree = _data.adj_t.t().sum(dim=1)
    degree = in_degree + out_degree
    directionality = (in_degree - out_degree) / (in_degree + out_degree + 1e-6)

    dir_corr = abs(np.corrcoef(directionality, _data.y)[0, 1].item())
    deg_corr = abs(np.corrcoef(degree, _data.y)[0, 1].item())

    return dir_corr * math.log(dir_corr / deg_corr)


def get_directionality(_data):
    assert not _data.adj_t.is_symmetric()

    in_degree = _data.adj_t.sum(dim=1)
    out_degree = _data.adj_t.t().sum(dim=1)
    return (in_degree - out_degree) / (in_degree + out_degree + 1e-6)


def get_intra_class_directionality(_data):
    y = _data.y
    y = y.squeeze(-1) if y.dim() > 1 else y
    y = y.argmax(-1) if y.dim() > 1 else y

    row, col, _ = _data.adj_t.coo()
    homo_mk = torch.zeros(row.size(0), device=row.device)
    homo_mk[y[row] == y[col]] = 1.

    in_degree = scatter(homo_mk, col, 0, dim_size=y.size(0), reduce='sum')
    out_degree = scatter(homo_mk, row, 0, dim_size=y.size(0), reduce='sum')


    return (in_degree - out_degree) / (in_degree + out_degree + 1e-6)


def get_intra_class_neighborhood_label_similarity(data, cos=False, return_all_classes=False):
    y = data.y
    y = y.squeeze(-1) if y.dim() > 1 else y
    y = y.argmax(-1) if y.dim() > 1 else y

    if -1 in y: y += 1  # penn94

    _adj = adj_norm(data.adj_t, norm='rw', add_self_loop=False).to(torch.float)
    _y = F.one_hot(data.y, num_classes=int(data.y.max()+1)).to(torch.float)
    _agg_y = _adj @ _y

    row, col, _ = data.adj_t.coo()
    if cos:
        edge_y_sim = F.cosine_similarity(_agg_y[row], _agg_y[col], dim=1)
    else:
        edge_y_sim = (_agg_y[row] * _agg_y[col]).sum(dim=1)

    if return_all_classes:
        node_y_sim = scatter(edge_y_sim, col, 0, dim_size=data.num_nodes, reduce='mean')
        return node_y_sim
    else:
        # return intra-class similarity
        homo_mk = torch.zeros(row.size(0), device=row.device, dtype=torch.bool)
        homo_mk[y[row] == y[col]] = True
        homo_node_y_sim = scatter(edge_y_sim[homo_mk], col[homo_mk], 0, dim_size=data.num_nodes,
                                  reduce='mean')
        return homo_node_y_sim



def get_intra_class_degree(data):
    y = data.y
    y = y.squeeze(-1) if y.dim() > 1 else y
    y = y.argmax(-1) if y.dim() > 1 else y

    row, col, _ = data.adj_t.coo()

    homo_mk = torch.zeros(row.size(0), device=row.device)
    homo_mk[y[row] == y[col]] = 1.
    intra_class_degree = scatter(homo_mk, col, 0, dim_size=y.size(0), reduce='sum')
    return intra_class_degree


def get_intra_class_neighborhood_feature_distance(data, return_all_classes=False):
    y = data.y
    y = y.squeeze(-1) if y.dim() > 1 else y
    y = y.argmax(-1) if y.dim() > 1 else y

    feat = F.normalize(data.x, p=1)
    row, col, _ = data.adj_t.coo()
    edge_feat_dist = (feat[row] - feat[col]).norm(p=2, dim=1)

    if return_all_classes:
        return scatter(edge_feat_dist, col, 0, dim_size=data.num_nodes, reduce='mean')
    else:
        # return intra-class similarity
        homo_mk = torch.zeros(row.size(0), device=row.device, dtype=torch.bool)
        homo_mk[y[row] == y[col]] = True
        return scatter(edge_feat_dist[homo_mk], col[homo_mk], 0, dim_size=data.num_nodes, reduce='mean')


def get_intra_class_neighborhood_feature_similarity(data, cos=False, return_all_classes=False):
    y = data.y
    y = y.squeeze(-1) if y.dim() > 1 else y
    y = y.argmax(-1) if y.dim() > 1 else y

    feat = F.normalize(data.x, p=1)
    row, col, _ = data.adj_t.coo()

    if cos:
        edge_feat_sim = F.cosine_similarity(feat[row], feat[col], dim=1)
    else:
        edge_feat_sim = (feat[row] * feat[col]).sum(dim=1)

    if return_all_classes:
        return scatter(edge_feat_sim, col, 0, dim_size=data.num_nodes, reduce='mean')
    else:
        # return intra-class similarity
        homo_mk = torch.zeros(row.size(0), device=row.device, dtype=torch.bool)
        homo_mk[y[row] == y[col]] = True
        return scatter(edge_feat_sim[homo_mk], col[homo_mk], 0, dim_size=data.num_nodes, reduce='mean')

def get_max_neighbor_label_ratio(data):
    y = data.y
    y = y.squeeze(-1) if y.dim() > 1 else y
    y = y.argmax(-1) if y.dim() > 1 else y

    if -1 in y: y += 1  # penn94
    _adj = adj_norm(data.adj_t, norm='rw', add_self_loop=False).to(torch.float)
    _y = F.one_hot(y, num_classes=int(data.y.max()+1)).to(torch.float)
    _agg_y = _adj @ _y
    _agg_y = _agg_y / _agg_y.sum(dim=-1, keepdim=True)
    _max_ratio, _ = _agg_y.max(dim=-1)
    return _max_ratio

def get_neighbor_label_entropy(data):
    y = data.y
    y = y.squeeze(-1) if y.dim() > 1 else y
    y = y.argmax(-1) if y.dim() > 1 else y

    if -1 in y: y += 1  # penn94
    _adj = adj_norm(data.adj_t, norm='rw', add_self_loop=False).to(torch.float)
    _y = F.one_hot(y, num_classes=int(data.y.max()+1)).to(torch.float)
    _agg_y = _adj @ _y
    probs = _agg_y / _agg_y.sum(dim=-1, keepdim=True)
    entropy = - (probs * probs.clamp(min=1e-8).log()).sum(dim=-1)
    return entropy


def get_kmeans_clustering_mask(embeddings, num_clusters):
    faiss_kmeans = faiss.Kmeans(d=embeddings.shape[1], k=num_clusters, gpu=True)
    embed_np = embeddings
    faiss_kmeans.train(embed_np)
    _, cluster_results = faiss_kmeans.index.search(embed_np, 1)
    cluster_results = cluster_results.reshape(-1, )
    cluster_results = torch.from_numpy(cluster_results)
    masks = cluster_results.unsqueeze(0) == torch.arange(num_clusters).unsqueeze(1) # [k, N]
    return masks

def get_balanced_kmeans_clustering_mask(embeddings, num_clusters=2):
    assert num_clusters == 2, "This function only supports 2 equal-size clusters"
    faiss_kmeans = faiss.Kmeans(d=embeddings.shape[1], k=2, gpu=True)
    embed_np = embeddings
    faiss_kmeans.train(embed_np)

    D, I = faiss_kmeans.index.search(embed_np, 2)  # D: [N, 2], distances to both centroids

    idx = np.arange(embed_np.shape[0])
    sorted_idx = np.argsort(np.abs(D[:, 0] - D[:, 1]))[::-1]  # sort by distance to the closest centroid

    half = embed_np.shape[0] // 2
    cluster_labels = torch.zeros(embed_np.shape[0], dtype=torch.long, device=embeddings.device)
    cluster_0_count = 0
    cluster_1_count = 0

    for i in sorted_idx:
        if D[i, 0] < D[i, 1]:
            if cluster_0_count < half:
                cluster_labels[i] = 0
                cluster_0_count += 1
            else:
                cluster_labels[i] = 1
                cluster_1_count += 1
        else:
            if cluster_1_count < half:
                cluster_labels[i] = 1
                cluster_1_count += 1
            else:
                cluster_labels[i] = 0
                cluster_0_count += 1

    masks = cluster_labels.unsqueeze(0) == torch.arange(num_clusters).unsqueeze(1) # [k, N]
    return masks

def get_balanced_aggregated_kmeans_clustering_mask(data, num_clusters=2):
    assert num_clusters == 2, "This function only supports 2 equal-size clusters"
    embeddings = data.x
    adj = adj_norm(data.adj_t, norm='sym', add_self_loop=False).to(torch.float)
    embeddings = adj @ embeddings
    embeddings = adj @ embeddings  # two-hop aggregation
    faiss_kmeans = faiss.Kmeans(d=embeddings.shape[1], k=2, gpu=True)
    embed_np = embeddings
    faiss_kmeans.train(embed_np)

    D, I = faiss_kmeans.index.search(embed_np, 2)  # D: [N, 2], distances to both centroids

    idx = np.arange(embed_np.shape[0])
    sorted_idx = np.argsort(np.abs(D[:, 0] - D[:, 1]))[::-1]  # sort by distance to the closest centroid

    half = embed_np.shape[0] // 2
    cluster_labels = torch.zeros(embed_np.shape[0], dtype=torch.long, device=embeddings.device)
    cluster_0_count = 0
    cluster_1_count = 0

    for i in sorted_idx:
        if D[i, 0] < D[i, 1]:
            if cluster_0_count < half:
                cluster_labels[i] = 0
                cluster_0_count += 1
            else:
                cluster_labels[i] = 1
                cluster_1_count += 1
        else:
            if cluster_1_count < half:
                cluster_labels[i] = 1
                cluster_1_count += 1
            else:
                cluster_labels[i] = 0
                cluster_0_count += 1
    masks = cluster_labels.unsqueeze(0) == torch.arange(num_clusters).unsqueeze(1) # [k, N]
    return masks

# def get_LSI(data, max_hop=200, epsilon=0.03, ):
#     feat = F.normalize(data.x, p=1)
#     adj = adj_norm(data.adj_t, norm='rw')
#
#     deg = data.adj_t.sum(1)
#     adj_inf = (deg + 1) / (data.num_edges + data.num_nodes)
#     feat_inf = adj_inf.view(1, -1) @ feat
#
#     smoothed_feats = [feat]
#     for k in tqdm(range(1, max_hop+1)):
#         smoothed_feats.append(adj @ smoothed_feats[-1])
#
#     LSI = torch.zeros(data.num_nodes, device=feat_inf.device)
#     mask_before = torch.zeros(data.num_nodes, dtype=torch.bool, device=feat.device)
#
#     for k, feat_k in enumerate(smoothed_feats):
#         dist = (feat_k - feat_inf).norm(p=2, dim=1)
#         mask = (dist < epsilon).masked_fill_(mask_before, False)
#         mask_before.masked_fill_(mask, True)
#         LSI.masked_fill_(mask, k)
#
#     mask_final = torch.ones(data.num_nodes, dtype=torch.bool, device=feat.device)
#     mask_final.masked_fill_(mask_before, False)
#     LSI.masked_fill_(mask_final, max_hop)
#
#     del smoothed_feats
#     return LSI.cpu()


def get_LSI(data, max_hop=6):
    feat = F.normalize(data.x, p=1)
    # adj = adj_norm(data.adj_t, norm='rw')
    adj = adj_norm(data.adj_t, norm='sym')

    deg = data.adj_t.sum(1)
    adj_inf = (deg + 1) / (data.num_edges + data.num_nodes)
    feat_inf = adj_inf.view(1, -1) @ feat

    smoothed_feats = [feat]
    for k in range(1, max_hop + 1):
        smoothed_feats.append(adj @ smoothed_feats[-1])

    # calculate feature distance (to stationary point) for each node
    s_dists = []
    for k, feat_k in enumerate(smoothed_feats):
        dist = (feat_k - feat_inf).norm(p=2, dim=1)
        s_dists.append(dist)
    s_dists = torch.stack(s_dists, dim=1)

    del smoothed_feats
    return s_dists


def get_LSI2(data, max_hop=6):
    feat = F.normalize(data.x, p=1)
    adj = adj_norm(data.adj_t, norm='sym')

    smoothed_feats = [feat]
    for k in range(0, max_hop + 1):
        smoothed_feats.append(adj @ smoothed_feats[-1])

    # calculate feature distance (to its original feature) for each node
    s_dists = []
    for k, feat_k in enumerate(smoothed_feats):
        dist = (feat_k - feat).norm(p=2, dim=1)
        s_dists.append(dist)
    s_dists = torch.stack(s_dists[1:], dim=1)

    del smoothed_feats
    return s_dists


def get_local_assortativity(data=None, path=None):
    if os.path.exists(path):
        return torch.load(path).to(data.x.device)
    G, y = to_networkx(data.cpu()), data.y.cpu()
    assort_m, assort_t, z = _localAssortF(G, np.array(y))
    out = torch.from_numpy(assort_t).to(torch.float)
    if path is not None:
        torch.save(out, path)
    return out.to(data.x.device)


def get_feature_dist(adj, x, train_mask, test_mask, knn_batch_size=1024):
    adj = adj_norm(adj)
    x_2 = adj @ (adj @ x)

    knn = KNN(x_2[train_mask], k=1)
    v_val, v_idx = knn.predict(x_2[test_mask], knn_batch_size)
    return v_val.squeeze(), v_idx.squeeze()


def get_homophily_dist(homophily, v_train_idx, train_mask, test_mask):
    v_train_idx = v_train_idx.to(homophily.device)
    train_mask = train_mask.to(homophily.device)
    test_mask = test_mask.to(homophily.device)

    v_homo = homophily[train_mask][v_train_idx]
    u_homo = homophily[test_mask]
    homo_dist = torch.abs(u_homo - v_homo)
    return homo_dist



def _localAssortF(G, M, pr=np.arange(0., 1., 0.1), missingValue=-1):
    n = len(M)
    ncomp = (M != missingValue).sum()
    # m = len(E)
    m = G.number_of_edges()

    A = nx.to_scipy_sparse_array(G, weight=None)

    degree = np.array(A.sum(1)).flatten()

    D = sparse.diags(1. / degree, 0, format='csc')
    W = D.dot(A)
    c = len(np.unique(M))
    if ncomp < n:
        c -= 1

    # calculate node weights for how "complete" the
    # metadata is around the node
    Z = np.zeros(n)
    Z[M == missingValue] = 1.
    Z = W.dot(Z) / degree

    values = np.ones(ncomp)
    yi = (M != missingValue).nonzero()[0]
    yj = M[M != missingValue]
    Y = sparse.coo_matrix((values, (yi, yj)), shape=(n, c)).tocsc()

    assortM = np.empty((n, len(pr)))
    assortT = np.empty(n)

    eij_glob = np.array(Y.T.dot(A.dot(Y)).todense())
    eij_glob /= np.sum(eij_glob)
    ab_glob = np.sum(eij_glob.sum(1) * eij_glob.sum(0))

    WY = W.dot(Y).tocsc()

    for i in tqdm(range(n)):
        pis, ti, it = _calculateRWRrange(A, degree, i, pr, n)

        YPI = sparse.coo_matrix((ti[M != missingValue], (M[M != missingValue],
                                                         np.arange(n)[M != missingValue])),
                                shape=(c, n)).tocsr()
        e_gh = YPI.dot(WY).toarray()
        Z[i] = np.sum(e_gh)
        e_gh /= np.sum(e_gh)
        trace_e = np.trace(e_gh)
        assortT[i] = trace_e

    assortT -= ab_glob
    assortT /= (1. - ab_glob + 1e-200)

    return assortM, assortT, Z


def _calculateRWRrange(A, degree, i, prs, n, trans=True, maxIter=1000):
    pr = prs[-1]
    D = sparse.diags(1. / degree, 0, format='csc')
    W = D * A
    diff = 1
    it = 1

    F = np.zeros(n)
    Fall = np.zeros((n, len(prs)))
    F[i] = 1
    Fall[i, :] = 1
    Fold = F.copy()
    T = F.copy()

    if trans:
        W = W.T

    oneminuspr = 1 - pr

    while diff > 1e-9:
        F = pr * W.dot(F)
        F[i] += oneminuspr
        Fall += np.outer((F - Fold), (prs / pr) ** it)
        T += (F - Fold) / ((it + 1) * (pr ** it))

        diff = np.sum((F - Fold) ** 2)
        it += 1
        if it > maxIter:
            print(i, "max iterations exceeded")
            diff = 0
        Fold = F.copy()

    return Fall, T, it