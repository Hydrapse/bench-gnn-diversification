import faiss
import math
from typing import Tuple

import random
import numpy as np
import scipy.sparse as sp
from scipy.special import comb
import torch
import torch.nn.functional as F
import torch_sparse
from sklearn.cluster import KMeans
from torch import Tensor
from torch_geometric.typing import Adj
from torch_scatter import scatter
from torch_sparse import SparseTensor, matmul
from torch_geometric.utils import dropout_adj, to_edge_index, to_scipy_sparse_matrix, \
    from_scipy_sparse_matrix, to_torch_sparse_tensor
from tqdm import tqdm
import os.path as osp

def get_laplacian(adj, norm='sym', add_self_loop=False):
    if add_self_loop:
        adj = torch_sparse.fill_diag(adj, 1.0)
    deg = adj.sum(dim=1)
    deg_inv_sqrt = deg.pow_(-0.5)
    deg_inv_sqrt.masked_fill_(deg_inv_sqrt == float('inf'), 0)
    adj_sym = deg_inv_sqrt.view(-1, 1) * adj * deg_inv_sqrt.view(1, -1)
    identity = torch_sparse.SparseTensor.eye(adj.size(0), dtype=adj_sym.dtype(), device=adj_sym.device())
    laplacian = identity + adj_sym.mul_nnz(torch.tensor(-1.0, dtype=adj_sym.storage.value().dtype, device=adj_sym.device())) # subtraction
    row, col, edge_weight = laplacian.coo()
    edge_index = torch.stack([row, col], dim=0)
    return edge_index, edge_weight

class Dict(dict):

    def __getattr__(self, key):
        return self.get(key)

    def __setattr__(self, key, value):
        self[key] = value


def setup_seed(seed):
    if seed == -1:
        return
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def dropout_edge(edge_index: Adj, p: float, training: bool = True):
    if not training or p == 0.:
        return edge_index

    if isinstance(edge_index, SparseTensor):
        if edge_index.storage.value() is not None:
            value = F.dropout(edge_index.storage.value(), p=p)
            edge_index = edge_index.set_value(value, layout='coo')
        else:
            mask = torch.rand(edge_index.nnz(), device=edge_index.storage.row().device) > p
            edge_index = edge_index.masked_select_nnz(mask, layout='coo')
    else:
        edge_index, edge_attr = dropout_adj(edge_index, p=p)

    return edge_index


def adj_norm(adj, norm='rw', add_self_loop=True, exponent=-0.5):
    assert norm in ['sym', 'rw', 'col', 'dir', 'src', 'attn', 'none', 'high', 'second']

    if add_self_loop:
        adj = torch_sparse.fill_diag(adj, 1.0)
    deg = adj.sum(dim=1)

    if norm == 'none':
        return adj

    if norm == 'sym':
        deg_inv_sqrt = deg.pow_(-0.5)
        deg_inv_sqrt.masked_fill_(deg_inv_sqrt == float('inf'), 0)
        adj = deg_inv_sqrt.view(-1, 1) * adj * deg_inv_sqrt.view(1, -1)
    elif norm == 'rw':
        deg_inv = deg.pow_(-1)
        deg_inv.masked_fill_(deg_inv == float('inf'), 0)
        adj = deg_inv.view(-1, 1) * adj
    elif norm in ['src', 'col']:
        deg_inv = deg.pow_(-1)
        deg_inv.masked_fill_(deg_inv == float('inf'), 0)
        adj = adj * deg_inv.view(1, -1)
    elif norm == 'dir':
        """
            Applies the normalization for directed graphs:
            \mathbf{D}_{out}^{-1/2} \mathbf{A} \mathbf{D}_{in}^{-1/2}.
        """
        in_deg = torch_sparse.sum(adj, dim=0)
        in_deg_inv_sqrt = in_deg.pow_(exponent)
        in_deg_inv_sqrt.masked_fill_(in_deg_inv_sqrt == float("inf"), 0.0)

        out_deg = torch_sparse.sum(adj, dim=1)
        out_deg_inv_sqrt = out_deg.pow_(exponent)
        out_deg_inv_sqrt.masked_fill_(out_deg_inv_sqrt == float("inf"), 0.0)

        adj = out_deg_inv_sqrt.view(-1, 1) * adj * in_deg_inv_sqrt.view(1, -1)
    elif norm == 'attn':
        assert isinstance(exponent, Tensor)
        in_deg_inv = deg.pow(-exponent)
        in_deg_inv.masked_fill_(in_deg_inv == float('inf'), 0)

        out_deg_inv = deg.pow(exponent - 1)
        out_deg_inv.masked_fill_(out_deg_inv == float('inf'), 0)

        adj = out_deg_inv.view(-1, 1) * adj * in_deg_inv.view(1, -1)
    elif norm == 'high':
        """
            \mathbf{L} = \mathbf{I} - \mathbf{A}_{sym}
        """
        deg_inv_sqrt = deg.pow_(-0.5)
        deg_inv_sqrt.masked_fill_(deg_inv_sqrt == float('inf'), 0)
        adj_sym = deg_inv_sqrt.view(-1, 1) * adj * deg_inv_sqrt.view(1, -1)
        
        identity = torch_sparse.SparseTensor.eye(adj.size(0), dtype=adj_sym.dtype(), device=adj_sym.device())
        laplacian = torch_sparse.add(identity, adj_sym.mul_nnz(torch.tensor(-1.0, dtype=adj_sym.storage.value().dtype, device=adj_sym.device()))) # subtraction
        adj = laplacian
    elif norm == 'second': # set model.cached to true, otherwise will be slow
        """
            \mathbf{A}_{sym}^2
        """
        deg_inv_sqrt = deg.pow_(-0.5)
        deg_inv_sqrt.masked_fill_(deg_inv_sqrt == float('inf'), 0)
        adj_sym = deg_inv_sqrt.view(-1, 1) * adj * deg_inv_sqrt.view(1, -1)
        adj = adj_sym @ adj_sym

    else:
        raise NotImplementedError

    return adj

# def adj_norm_bern(adj, add_self_loop=True, bern_K=None, coe=None):
#     """
#     \sum_{k=0}^K \theta_k \frac{1}{2^K}\binom{K}{k}(2 \mathbf{I}-\mathbf{L})^{K-k} \mathbf{L}^k
#     """
#     # cannot use cache because the bern parameter is optimized
#     if add_self_loop:
#         adj = torch_sparse.fill_diag(adj, 1.0)
#     deg = adj.sum(dim=1)
    
#     assert bern_K is not None, "Bernstein K must be specified"
#     K = bern_K
#     deg_inv_sqrt = deg.pow_(-0.5)
#     deg_inv_sqrt.masked_fill_(deg_inv_sqrt == float('inf'), 0)
#     adj_sym = deg_inv_sqrt.view(-1, 1) * adj * deg_inv_sqrt.view(1, -1)
    
#     TEMP=F.relu(coe)
#     if not adj.has_value():
#         adj = adj.set_value(torch.ones(adj.nnz(), dtype=torch.float32, device=adj.device()), layout='coo')
#     # L
#     L = adj + adj_sym.mul_nnz(torch.tensor(-1.0, device=adj_sym.device()))  # subtraction
#     # 2I - L
#     L_comp = torch_sparse.fill_diag(adj, 2.0) + L.mul_nnz(torch.tensor(-1.0, device=adj_sym.device()))  # subtraction
    
#     output = torch.zeros(adj.size(0), adj.size(1), device=adj.device(), dtype=torch.float32)

#     def sparse_matrix_power(sparse_tensor, k):
#         if k == 0:
#             N = sparse_tensor.size(0)
#             return torch.eye(N, device=sparse_tensor.device())  # dense identity
#         result = sparse_tensor
#         for _ in range(1, k):
#             result = matmul(result, sparse_tensor)
#         return result

#     for k in range(K + 1):
#         L_k = sparse_matrix_power(L, k)
#         Lc_k = sparse_matrix_power(L_comp, K - k)
#         term = TEMP[k] * (1 / 2 ** K) * comb(K, k) * (Lc_k @ L_k)
#         output += term

#     return output


def adj_to_directed_ADPA_norm(adj, r=0.5, num_nodes=None):
    # if isinstance(adj, SparseTensor):
    #     adj = to_edge_index(adj)[0]
    # adj = to_scipy_sparse_matrix(adj.cpu()).tocoo()

    # adj_t = adj.T
    adj_t = adj.t()
    adj_aa = adj @ adj
    adj_aat = adj @ adj_t
    adj_ata = adj_t @ adj
    adj_atat = adj_t @ adj_t

    # adj = adj + sp.eye(adj.shape[0])
    # adj_t = adj_t + sp.eye(adj_t.shape[0])

    adj = torch_sparse.fill_diag(adj, 1.0)
    adj_t = torch_sparse.fill_diag(adj_t, 1.0)

    deg_a = adj.sum(dim=1)
    deg_at = adj_t.sum(dim=1)
    deg_aa = adj_aa.sum(dim=1)
    deg_aat = adj_aat.sum(dim=1)
    deg_ata = adj_ata.sum(dim=1)
    deg_atat = adj_atat.sum(dim=1)

    # deg_a = np.array(adj.sum(1))
    # deg_at = np.array(adj_t.sum(1))
    # deg_aa = np.array(adj_aa.sum(1))
    # deg_aat = np.array(adj_aat.sum(1))
    # deg_ata = np.array(adj_ata.sum(1))
    # deg_atat = np.array(adj_atat.sum(1))

    new_adj_list = []
    for _deg, _adj in zip([deg_a, deg_at, deg_aa, deg_aat, deg_ata, deg_atat],
                          [adj, adj_t, adj_aa, adj_aat, adj_ata, adj_atat]):
        in_deg_inv_sqrt = _deg.pow_(r - 1)
        in_deg_inv_sqrt.masked_fill_(in_deg_inv_sqrt == float("inf"), 0.0)

        out_deg_inv_sqrt = _deg.pow_(-r)
        out_deg_inv_sqrt.masked_fill_(out_deg_inv_sqrt == float("inf"), 0.0)

        new_adj_list.append(
            out_deg_inv_sqrt.view(-1, 1) * _adj * in_deg_inv_sqrt.view(1, -1))

    # r_inv_sqrt_left = np.power(deg_a, r - 1).flatten()
    # r_inv_sqrt_left[np.isinf(r_inv_sqrt_left)] = 0.
    # r_mat_inv_sqrt_left = sp.diags(r_inv_sqrt_left)
    # r_inv_sqrt_right = np.power(deg_a, -r).flatten()
    # r_inv_sqrt_right[np.isinf(r_inv_sqrt_right)] = 0.
    # r_mat_inv_sqrt_right = sp.diags(r_inv_sqrt_right)
    # adj_normalized = r_mat_inv_sqrt_left @ adj @ r_mat_inv_sqrt_right
    #
    # r_inv_sqrt_left = np.power(deg_at, r - 1).flatten()
    # r_inv_sqrt_left[np.isinf(r_inv_sqrt_left)] = 0.
    # r_mat_inv_sqrt_left = sp.diags(r_inv_sqrt_left)
    # r_inv_sqrt_right = np.power(deg_at, -r).flatten()
    # r_inv_sqrt_right[np.isinf(r_inv_sqrt_right)] = 0.
    # r_mat_inv_sqrt_right = sp.diags(r_inv_sqrt_right)
    # adj_t_normalized = r_mat_inv_sqrt_left @ adj_t @ r_mat_inv_sqrt_right
    #
    # r_inv_sqrt_left = np.power(deg_aa, r - 1).flatten()
    # r_inv_sqrt_left[np.isinf(r_inv_sqrt_left)] = 0.
    # r_mat_inv_sqrt_left = sp.diags(r_inv_sqrt_left)
    # r_inv_sqrt_right = np.power(deg_aa, -r).flatten()
    # r_inv_sqrt_right[np.isinf(r_inv_sqrt_right)] = 0.
    # r_mat_inv_sqrt_right = sp.diags(r_inv_sqrt_right)
    # adj_aa_normalized = r_mat_inv_sqrt_left @ adj_aa @ r_mat_inv_sqrt_right
    #
    # r_inv_sqrt_left = np.power(deg_aat, r - 1).flatten()
    # r_inv_sqrt_left[np.isinf(r_inv_sqrt_left)] = 0.
    # r_mat_inv_sqrt_left = sp.diags(r_inv_sqrt_left)
    # r_inv_sqrt_right = np.power(deg_aat, -r).flatten()
    # r_inv_sqrt_right[np.isinf(r_inv_sqrt_right)] = 0.
    # r_mat_inv_sqrt_right = sp.diags(r_inv_sqrt_right)
    # adj_aat_normalized = r_mat_inv_sqrt_left @ adj_aat @ r_mat_inv_sqrt_right
    #
    # r_inv_sqrt_left = np.power(deg_ata, r - 1).flatten()
    # r_inv_sqrt_left[np.isinf(r_inv_sqrt_left)] = 0.
    # r_mat_inv_sqrt_left = sp.diags(r_inv_sqrt_left)
    # r_inv_sqrt_right = np.power(deg_ata, -r).flatten()
    # r_inv_sqrt_right[np.isinf(r_inv_sqrt_right)] = 0.
    # r_mat_inv_sqrt_right = sp.diags(r_inv_sqrt_right)
    # adj_ata_normalized = r_mat_inv_sqrt_left @ adj_ata @ r_mat_inv_sqrt_right
    #
    # r_inv_sqrt_left = np.power(deg_atat, r - 1).flatten()
    # r_inv_sqrt_left[np.isinf(r_inv_sqrt_left)] = 0.
    # r_mat_inv_sqrt_left = sp.diags(r_inv_sqrt_left)
    # r_inv_sqrt_right = np.power(deg_atat, -r).flatten()
    # r_inv_sqrt_right[np.isinf(r_inv_sqrt_right)] = 0.
    # r_mat_inv_sqrt_right = sp.diags(r_inv_sqrt_right)
    # adj_atat_normalized = r_mat_inv_sqrt_left @ adj_atat @ r_mat_inv_sqrt_right
    #
    # adj_list = [adj_normalized.tocsr(), adj_t_normalized.tocsr(), adj_aa_normalized.tocsr(),
    #             adj_aat_normalized.tocsr(), adj_ata_normalized.tocsr(), adj_atat_normalized.tocsr()]
    # new_adj_list = []
    # for _adj in adj_list:
    #     _adj = from_scipy_sparse_matrix(_adj)[0]
    #     new_adj_list.append(to_torch_sparse_tensor(_adj, size=num_nodes))

    return new_adj_list
    # return adj_normalized, adj_t_normalized, adj_aa_normalized, adj_aat_normalized, adj_ata_normalized, adj_atat_normalized


def group_values(values: Tensor, num_groups=5, group_by='sample', log_scale=False):
    assert group_by in ['sample', 'domain']
    assert values.size(0) == values.view(-1).size(0)
    device = values.device
    if group_by == 'sample':
        values_, indices = values.sort()
        group_idx = torch.arange(num_groups).repeat_interleave(int(values.size(0) / num_groups))
        group_idx = torch.cat(
            [group_idx, torch.full((values.size(0) % num_groups,), num_groups - 1)])
        group_val = scatter(values_, group_idx.to(device), reduce='mean')
    elif group_by == 'domain':
        group_idx = torch.zeros_like(values, dtype=torch.long)
        if log_scale:
            values = torch.log10(values)
        min_val, max_val = values.min(), values.max()
        step = (max_val - min_val) / num_groups
        group_val = []
        for i in range(num_groups):
            mask = (values > min_val + i * step) & (values <= min_val + (i + 1) * step)
            if i == 0:
                mask = mask | (values == min_val)
            group_idx[mask] = i
            group_val.append((min_val + (2 * i + 1) / 2 * step))
        group_val = torch.tensor(group_val)
        if log_scale:
            group_val = torch.pow(10, group_val)
        indices = torch.arange(values.size(0))
    else:
        raise NotImplementedError
    return group_val.to(device), group_idx.to(device), indices.to(device)


def pred_fn(y_hat, y) -> Tuple[Tensor, Tensor]:
    if y_hat.shape[1] == 1:  # binary, auc_roc
        pred = y_hat
    elif y.dim() == 1:  # multi-class
        pred = y_hat.argmax(dim=-1)
    else:  # multi-label
        pred = (y_hat > 0).float()
    return pred, y


def loss_fn(y_hat, y) -> Tensor:
    if y_hat.shape[1] == 1:  # binary
        return F.binary_cross_entropy_with_logits(y_hat.squeeze(), y.float())
    elif y.dim() > 1:  # multi-label
        return F.binary_cross_entropy_with_logits(y_hat, y)
    else:  # multi-class
        loss = F.cross_entropy(y_hat, y)

        # y = F.cross_entropy(y_hat, y, reduction='none')
        # epsilon = 1 - math.log(2)
        # y = torch.log(epsilon + y) - math.log(epsilon)
        # loss = y.mean()

        return loss


def distance_matrix(x, y=None, p=2):
    # Returns a pair wise distance between all elements in matrix x and y
    y = x if y is None else y

    n = x.size(0)
    m = y.size(0)
    d = x.size(1)

    x = x.unsqueeze(1).expand(n, m, d)
    y = y.unsqueeze(0).expand(n, m, d)

    dist = torch.linalg.vector_norm(x - y, p, 2)
    return dist


class KNN:
    def __init__(self, train_pts, k=1, p=2):
        self.train_pts = train_pts
        self.k = k
        self.p = p

    def predict(self, test_pts, batch_size=1024):
        top_vals, top_idx = [], []
        # Ensure at least one batch is created
        num_batches = max(1, test_pts.size(0) // batch_size)

        # If there are fewer test points than batch_size, we just process the whole tensor in one batch
        for batch_test in tqdm(test_pts.tensor_split(num_batches)):
            dist = distance_matrix(batch_test, self.train_pts, self.p)
            vals, idx = dist.topk(self.k, largest=False)
            top_vals.append(vals)
            top_idx.append(idx)

        # Concatenate the results from all batches
        top_vals, top_idx = torch.cat(top_vals), torch.cat(top_idx)
        return top_vals.cpu(), top_idx.cpu()


def padding_test_to_full(test_pts, num_full_pts, test_mask):
    out = torch.zeros(num_full_pts, dtype=test_pts.dtype, device=test_pts.device)
    out[test_mask] = test_pts
    return out


def mask_to_index(mask):
    if mask.dtype == torch.bool:
        return mask.nonzero(as_tuple=False).view(-1)
    else:
        return mask


def index_to_mask(index: Tensor, size: int = None) -> Tensor:
    if index.dtype == torch.bool:
        return index
    else:
        index = index.view(-1)
        size = int(index.max()) + 1 if size is None else size
        mask = index.new_zeros(size, dtype=torch.bool)
        mask[index] = True
        return mask

def convert_pyg_mask(_data):
    _train_mask = index_to_mask(_data.train_mask, size=_data.num_nodes).t()
    _val_mask = index_to_mask(_data.val_mask, size=_data.num_nodes).t()
    _test_mask = index_to_mask(_data.test_mask, size=_data.num_nodes).t()
    if _train_mask.dim() == 1:
        _train_mask = _train_mask.unsqueeze(0)
        _val_mask = _val_mask.unsqueeze(0)
        _test_mask = _test_mask.unsqueeze(0)
    return _train_mask, _val_mask, _test_mask


def get_masked_adj(_data, selected_nodes):
    N = _data.num_nodes

    selected_diag = torch.sparse_coo_tensor(
        indices=torch.arange(N).repeat(2, 1),
        values=selected_nodes,
        size=(N, N),
        dtype=torch.float32,
        device=selected_nodes.device
    )
    _masked_adj = _data.adj_t.to_torch_sparse_coo_tensor()
    _masked_adj = torch.sparse.mm(selected_diag, torch.sparse.mm(_masked_adj, selected_diag))
    _masked_adj = SparseTensor.from_torch_sparse_coo_tensor(_masked_adj)
    _row, _col, _val = _masked_adj.coo()
    edge_masked = _val.to(torch.bool)

    return SparseTensor(row=_row[edge_masked], col=_col[edge_masked], sparse_sizes=(N, N))


def get_intra_class_masked_adj(_data, selected_nodes, get_inter_class=False):
    """Get Intra-class graph (Remove test set labels)"""
    N = _data.num_nodes

    _adj = get_masked_adj(_data, selected_nodes)
    _row, _col, _ = _adj.coo()
    _homo_mk = torch.zeros(_row.size(0), device=_row.device, dtype=torch.bool)
    _homo_mk[_data.y[_row] == _data.y[_col]] = 1.

    if get_inter_class: _homo_mk = ~ _homo_mk

    return SparseTensor(row=_row[_homo_mk], col=_col[_homo_mk], sparse_sizes=(N, N))


def shuffle_intra_nodes(_y, _selected_nodes):
    """intra-class shuffling"""
    assert _y.dim() == 1
    origin_idx = torch.arange(_y.shape[0], dtype=torch.long, device=_y.device)
    shuffled_idx = origin_idx.clone()

    for g in _y.unique():
        mask = (_y == g) & _selected_nodes
        idx = mask.nonzero(as_tuple=True)[0]
        perm = idx[torch.randperm(idx.size(0))]
        shuffled_idx[idx] = origin_idx[perm]

    return shuffled_idx


def kmeans_clustering(embed, num_clusters, cuda=True):
    embed_np = embed.cpu().numpy()
    if cuda:
        faiss_kmeans = faiss.Kmeans(d=embed.shape[1], k=num_clusters, gpu=True)
        faiss_kmeans.train(embed_np)
        _, cluster_results = faiss_kmeans.index.search(embed_np, 1)
        cluster_results = cluster_results.reshape(-1, )
        center_data = faiss_kmeans.centroids
    else:
        km = KMeans(n_clusters=num_clusters, n_init=20).fit(embed_np)
        cluster_results = km.labels_
        center_data = km.cluster_centers_

    cluster_results = torch.from_numpy(cluster_results).to(embed.device)
    center_data = torch.from_numpy(center_data).to(embed.device)
    return cluster_results, center_data


def trained_models_conf_to_paths(dataset, proc_dir, confs, postfix=''):
    """
    TODO: check if exist duplication
    Convert a config object to one or several string representation.
    domain (+ hop) + arch
    """
    all_file_paths = []
    all_file_paths_postfixed = []
    for conf in confs:
        assert conf.arch is not None
        domain = conf.get('domain', '')
        if domain:  # e.g., group-frac-numExperts_expertID
            expert_id = domain.split('_')[-1]
            domain_args = domain[: -len(expert_id) - 1].split('-')
            domain = f'-group-{domain_args[0]}-frac-{domain_args[1]}-eid-{expert_id}-{domain_args[2]}'
            domain_postfixed = f'-group-{domain_args[0]}{postfix}-frac-{domain_args[1]}-eid-{expert_id}-{domain_args[2]}'
        # parse hops
        depths = conf.get('depths', '2')
        if '-' in depths:
            depths = depths.split('-')
            depths = list(range(int(depths[0]), int(depths[1])+1))
        elif ',' in depths:
            depths = depths.split(',')
        else:
            depths = [depths]
    # load logits
        for depth in depths:
            file_path = f'{proc_dir}/logit/{dataset}{domain}_{conf.arch}-conv{depth}.pt'
            if not osp.exists(file_path):
                raise FileNotFoundError(f'Logit file {file_path} not found!')
            all_file_paths.append(file_path)
            file_path_postfixed = f'{proc_dir}/logit/{dataset}{domain_postfixed}_{conf.arch}-conv{depth}.pt'
            all_file_paths_postfixed.append(file_path_postfixed)
    return all_file_paths, all_file_paths_postfixed
