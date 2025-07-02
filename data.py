import os
import os.path as osp
from typing import Tuple

import numpy as np
import pandas as pd
import scipy
import torch
import torch_geometric.transforms as T
import torchmetrics
from torch_sparse import SparseTensor
from sklearn.preprocessing import StandardScaler
from torch_geometric.data import Data, Batch, download_url
from torch_geometric.datasets import (Planetoid, WikiCS, Coauthor, Amazon,
                                      GNNBenchmarkDataset, Yelp, Flickr,
                                      Reddit2, PPI, WebKB, WikipediaNetwork, Actor)
import gdown

import sklearn
from ogb.nodeproppred import PygNodePropPredDataset, NodePropPredDataset
from torch_geometric.transforms import BaseTransform

from utils.load_data import load_twitch_gamer, load_facebook100
from utils.training import random_planetoid_splits, random_splits, even_quantile_labels
from utils.utils import dropout_edge, get_masked_adj
from disparity import get_node_homophily, get_degree, get_pagerank, get_LSI, get_LSI2, get_neighborhood_confusion, get_clustering_coef, get_directional_label_distance, get_directional_label_similarity, get_directional_label_cos_similarity, get_direction_informativeness, get_directionality, get_intra_class_directionality, get_intra_class_neighborhood_label_similarity, get_intra_class_degree, get_intra_class_neighborhood_feature_distance, get_intra_class_neighborhood_feature_similarity, get_max_neighbor_label_ratio, get_neighbor_label_entropy, get_kmeans_clustering_mask, get_balanced_kmeans_clustering_mask, get_balanced_aggregated_kmeans_clustering_mask



SPLIT_RUNS = 10


class StandardizeFeatures(BaseTransform):
    def __call__(self, data: Data) -> Data:
        scaler = StandardScaler()
        scaler.fit(data.x)
        X = scaler.transform(data.x)
        data.x = torch.from_numpy(X).type(data.x.dtype)
        return data


def load_split(path: str, data, train_rate: float, val_rate: float,
               num_classes=-1, filtered=False):
    file = osp.join(path, f'split_{train_rate}_{val_rate}{"_filtered" if filtered else ""}.pt')

    if osp.exists(file):
        split_dict = torch.load(file, weights_only=True)
    else:
        train_masks, val_masks, test_masks = [], [], []
        for i in range(SPLIT_RUNS):
            if num_classes > 0:
                train_mask, val_mask, test_mask = random_planetoid_splits(
                    data, num_classes, train_rate, val_rate)
            else:
                train_mask, val_mask, test_mask = random_splits(
                    data, train_rate, val_rate)
            train_masks.append(train_mask)
            val_masks.append(val_mask)
            test_masks.append(test_mask)
        split_dict = {
            'train_mask': torch.stack(train_masks, dim=1),
            'val_mask': torch.stack(val_masks, dim=1),
            'test_mask': torch.stack(test_masks, dim=1)
        }
        torch.save(split_dict, file)

    data.train_mask = split_dict['train_mask']
    data.val_mask = split_dict['val_mask']
    data.test_mask = split_dict['test_mask']
    return data


def load_ptb_edges(path, edge_index, num_nodes, ptb_type, ptb_ratio, **kwargs):
    if ptb_ratio <= 0.:
        return edge_index

    file = osp.join(path, f'perturbed_{ptb_type}-{ptb_ratio}_edges.pt')
    if osp.exists(file):
        new_edge_index = torch.load(file)
        return new_edge_index

    num_ptb_edges = int(ptb_ratio * edge_index.size(1))
    if ptb_type == 'rand':
        ptb_edges = torch.stack([
            torch.randint(low=0, high=num_nodes, size=(num_ptb_edges,)),
            torch.randint(low=0, high=num_nodes, size=(num_ptb_edges,)),
        ], dim=0)
        new_edge_index = torch.cat([edge_index, ptb_edges], dim=1)
    elif ptb_type == 'drop':
        new_edge_index = dropout_edge(edge_index, p=ptb_ratio)
    else:
        raise NotImplementedError

    torch.save(new_edge_index, file)
    return new_edge_index


def get_ogbn(root: str, name: str, **kwargs):
    pre_transform = T.Compose([T.ToSparseTensor()])
    transform = T.Compose([StandardizeFeatures()])
    name = 'ogbn-' + name
    dataset = PygNodePropPredDataset(name, root, transform=transform, pre_transform=pre_transform)
    split = dataset.get_idx_split()
    n_data = Data(train_mask=split['train'], val_mask=split['valid'], test_mask=split['test'])
    data = dataset[0].update(n_data)
    if kwargs.pop('undirected', True):
        data.adj_t = data.adj_t.to_symmetric()  # very important
    data.y = data.y.squeeze()
    one_hot_train_mask = torch.zeros(data.x.size(0), dtype=torch.bool)
    one_hot_train_mask[data.train_mask] = True
    data.train_mask = one_hot_train_mask
    one_hot_val_mask = torch.zeros(data.x.size(0), dtype=torch.bool)
    one_hot_val_mask[data.val_mask] = True
    data.val_mask = one_hot_val_mask
    one_hot_test_mask = torch.zeros(data.x.size(0), dtype=torch.bool)
    one_hot_test_mask[data.test_mask] = True
    data.test_mask = one_hot_test_mask
    return data, dataset.num_features, dataset.num_classes, dataset.processed_dir


def get_arxiv_year(root, train_rate=0.5, val_rate=0.25, nclass=5, **kwargs):
    pre_transform = T.Compose([T.ToSparseTensor()])
    # transform = T.Compose([StandardizeFeatures()])
    transform = T.Compose([])
    dataset = PygNodePropPredDataset('ogbn-arxiv', root, transform=transform, pre_transform=pre_transform)
    split = dataset.get_idx_split()
    n_data = Data(train_mask=split['train'], val_mask=split['valid'], test_mask=split['test'])
    data = dataset[0].update(n_data)

    if kwargs.pop('rev_adj', False):
        data.adj_t = data.adj_t.t()

    # data.adj_t = data.adj_t.to_symmetric()  # very important for year to have direct edges?

    label = even_quantile_labels(
        data.node_year.numpy().flatten(), nclass, verbose=False)
    data.y = torch.as_tensor(label, dtype=torch.long)

    proc_dir = f'{root}/ogbn_arxiv/year'
    os.makedirs(proc_dir, exist_ok=True)
    data = load_split(proc_dir, data, train_rate, val_rate)

    return data, dataset.num_features, nclass, proc_dir


def get_planetoid(root: str, name: str, split: str = 'public', **kwargs) -> Tuple[Data, int, int, str]:
    transform = T.Compose([T.NormalizeFeatures()])
    pre_transform = T.Compose([T.ToSparseTensor()])
    dataset = Planetoid(f'{root}/Planetoid', name, split, transform=transform, pre_transform=pre_transform)

    data, num_classes = dataset[0], dataset.num_classes

    if kwargs.get('train_rate', False):
        data = load_split(f'{root}/Planetoid/{name}', data, kwargs['train_rate'], kwargs['val_rate'],
                          num_classes=num_classes)

    return data, dataset.num_features, num_classes, dataset.processed_dir


def get_actor(root: str, train_rate=0.6, val_rate=0.2, **kwargs):
    trans_list = [T.ToSparseTensor(), T.NormalizeFeatures()]
    if kwargs.pop('undirected', False):  # default directed graph
        trans_list.insert(0, T.ToUndirected())
    transform = T.Compose(trans_list)
    dataset = Actor(root=f'{root}/Actor', transform=transform)

    data, num_classes = dataset[0], dataset.num_classes

    if train_rate:
        data = load_split(f'{root}/Actor', data, train_rate, val_rate, num_classes)

    return data, dataset.num_features, num_classes, dataset.processed_dir


def get_webkb(root: str, name: str, **kwargs) -> Tuple[Data, int, int, str]:
    transform = T.Compose([T.NormalizeFeatures()])
    pre_transform = T.Compose([T.ToSparseTensor()])
    dataset = WebKB(f'{root}/WebKB', name, transform=transform, pre_transform=pre_transform)
    data = dataset[0]
    split = 0
    data.train_mask = data.train_mask[:, split]
    data.val_mask = data.val_mask[:, split]
    data.test_mask = data.test_mask[:, split]
    return data, dataset.num_features, dataset.num_classes, dataset.processed_dir


def get_amazon(root: str, name: str, train_rate=0.6, val_rate=0.2, **kwargs) -> Tuple[Data, int, int, str]:
    transform = T.Compose([T.NormalizeFeatures()])
    pre_transform = T.Compose([T.ToSparseTensor()])
    dataset = Amazon(f'{root}/Amazon', name, transform=transform, pre_transform=pre_transform)
    if name == 'computers':
        name = name.capitalize()
    data = load_split(f'{root}/Amazon/{name}', dataset[0], train_rate, val_rate,
                      num_classes=dataset.num_classes)
    return data, dataset.num_features, dataset.num_classes, dataset.processed_dir


def get_wikinet(root: str, name: str, train_rate=None, val_rate=None, filtered=False, **kwargs) -> Tuple[Data, int, int, str]:
    # follow GPRGNN's setting
    if filtered:
        url = f'https://github.com/yandex-research/heterophilous-graphs/raw/main/data/{name}_filtered.npz'
        raw_dir = f'{root}/WikipediaNetwork/{name}'
        download_url(url, raw_dir, log=False)

        ndata = np.load(osp.join(raw_dir, f'{name}_filtered.npz'))
        data = Data(
            x=torch.tensor(ndata['node_features']),
            y=torch.tensor(ndata['node_labels']),
            edge_index=torch.tensor(ndata['edges']).t(),
            train_mask=torch.tensor(ndata['train_masks']).t(),
            val_mask=torch.tensor(ndata['val_masks']).t(),
            test_mask=torch.tensor(ndata['test_masks']).t(),
        )
        # trans_list = [T.RemoveSelfLoops(), T.AddSelfLoops(), T.ToSparseTensor()]
        trans_list = [T.ToSparseTensor()]
        if kwargs.pop('undirected', False):  # default directed graph
            trans_list.insert(0, T.ToUndirected())
        transform = T.Compose(trans_list)
        data = transform(data)

        num_features, num_classes = data.x.shape[1], int(data.y.max()+1)
    else:
        transform = T.Compose([])
        pre_transform = T.Compose([T.ToSparseTensor()])

        preProcDs = WikipediaNetwork(f'{root}/WikipediaNetwork', name,
                                     geom_gcn_preprocess=False, transform=transform, pre_transform=pre_transform)
        dataset = WikipediaNetwork(f'{root}/WikipediaNetwork', name,
                                   geom_gcn_preprocess=True, transform=transform, pre_transform=pre_transform)
        data = dataset[0]
        data.adj_t = preProcDs[0].adj_t

        num_features, num_classes = dataset.num_features, dataset.num_classes

    if train_rate:
        data = load_split(f'{root}/WikipediaNetwork/{name}', data,
                          train_rate, val_rate, num_classes, filtered)

    return data, num_features, num_classes, f'{root}/WikipediaNetwork/{name}/processed'


def get_heterophilious(root: str, name: str, train_rate=None, val_rate=None, **kwargs):
    url = f'https://github.com/yandex-research/heterophilous-graphs/raw/main/data/{name}.npz'
    raw_dir = f'{root}/Heterophilious/{name}'
    download_url(url, raw_dir, log=False)

    ndata = np.load(osp.join(raw_dir, f'{name}.npz'))
    data = Data(
        x=torch.tensor(ndata['node_features']),
        y=torch.tensor(ndata['node_labels']),
        edge_index=torch.tensor(ndata['edges']).t(),
        train_mask=torch.tensor(ndata['train_masks']).t(),
        val_mask=torch.tensor(ndata['val_masks']).t(),
        test_mask=torch.tensor(ndata['test_masks']).t(),
    )

    # transform_list = [T.RemoveSelfLoops(), T.AddSelfLoops(), T.ToSparseTensor()]
    transform_list = [T.ToSparseTensor()]
    if kwargs.pop('undirected', True):
        transform_list.insert(0, T.ToUndirected())
    transform = T.Compose(transform_list)
    data = transform(data)

    num_features, num_classes = data.x.shape[1], len(data.y.unique())

    if train_rate:
        data = load_split(raw_dir, data, train_rate, val_rate,
                          num_classes=num_classes)

    if num_classes == 2:  # binary
        num_classes = 1

    return data, num_features, num_classes, f'{raw_dir}/processed'


def get_wikics(root: str, **kwargs) -> Tuple[Data, int, int, str]:
    transform = T.Compose([])
    pre_transform = T.Compose([T.ToSparseTensor()])
    dataset = WikiCS(f'{root}/WIKICS', transform=transform, pre_transform=pre_transform)
    data = dataset[0]
    data.adj_t = data.adj_t.to_symmetric()  # already symmetric
    data.val_mask = data.stopping_mask
    data.stopping_mask = None
    return data, dataset.num_features, dataset.num_classes, dataset.processed_dir


def get_yelp(root: str, **kwargs) -> Tuple[Data, int, int, str]:
    transform = T.Compose([])
    pre_transform = T.Compose([T.ToSparseTensor()])
    dataset = Yelp(f'{root}/YELP', transform=transform, pre_transform=pre_transform)
    data = dataset[0]
    data.x = (data.x - data.x.mean(dim=0)) / data.x.std(dim=0)
    return data, dataset.num_features, dataset.num_classes, dataset.processed_dir


def get_flickr(root: str, **kwargs) -> Tuple[Data, int, int, str]:
    transform = T.Compose([])
    pre_transform = T.Compose([T.ToSparseTensor()])
    dataset = Flickr(f'{root}/Flickr', transform=transform, pre_transform=pre_transform)
    return dataset[0], dataset.num_features, dataset.num_classes, dataset.processed_dir


def get_reddit(root: str, **kwargs) -> Tuple[Data, int, int, str]:
    transform = T.Compose([])
    pre_transform = T.Compose([T.ToSparseTensor()])
    dataset = Reddit2(f'{root}/Reddit2', transform=transform, pre_transform=pre_transform)
    data = dataset[0]
    data.x = (data.x - data.x.mean(dim=0)) / data.x.std(dim=0)
    return data, dataset.num_features, dataset.num_classes, dataset.processed_dir


def get_ppi(root: str, split: str = 'train', **kwargs):
    pre_transform = T.Compose([T.ToSparseTensor()])
    dataset = PPI(f'{root}/PPI', split=split, pre_transform=pre_transform)
    data = Batch.from_data_list(dataset)
    data.batch = None
    data.ptr = None
    data[f'{split}_mask'] = torch.ones(data.num_nodes, dtype=torch.bool)
    return data, dataset.num_features, dataset.num_classes, dataset.processed_dir


def get_sbm(root: str, name: str, **kwargs):
    pre_transform = T.Compose([T.ToSparseTensor()])
    dataset = GNNBenchmarkDataset(f'{root}/SBM', name, split='train',
                                  pre_transform=pre_transform)
    data = Batch.from_data_list(dataset)
    data.batch = None
    data.ptr = None
    return data, dataset.num_features, dataset.num_classes, dataset.processed_dir


def get_facebook100(root: str, name: str, train_rate=0.5, val_rate=0.25,
                    ptb_type=None, ptb_ratio=0., **kwargs):
    name = name.title()
    url = f'https://github.com/CUAI/Non-Homophily-Large-Scale/raw/master/data/facebook100/{name}.mat'
    dataset_dir = f'{root}/facebook100/{name}'
    download_url(url, dataset_dir, log=False)

    mat = scipy.io.loadmat(f'{dataset_dir}/{name}.mat')
    features, edge_index, label = load_facebook100(mat)

    # already undirected edge_index
    trans_list = [T.ToUndirected(), T.ToSparseTensor()]

    if ptb_type is not None and ptb_ratio > 0:
        edge_index = load_ptb_edges(dataset_dir, edge_index, features.shape[0], ptb_type, ptb_ratio)
        trans_list.insert(0, T.RemoveDuplicatedEdges())

    data = Data(
        x=torch.tensor(features, dtype=torch.float),
        y=torch.tensor(label, dtype=torch.long),
        edge_index=edge_index,
    )
    transform = T.Compose(trans_list)
    data = transform(data)

    data = load_split(dataset_dir, data, train_rate, val_rate)

    num_features, num_classes = data.x.shape[1], (data.y.max() + 1).item()

    return data, num_features, num_classes, dataset_dir


def get_pokec(root: str, train_rate=0.5, val_rate=0.25, **kwargs):
    dataset_url = '1dNs5E7BrWJbgcHeQ_zuy5Ozp2tRCWG0y'

    os.makedirs(f'{root}/pokec', exist_ok=True)
    path = f'{root}/pokec/pokec.mat'
    if not osp.exists(path):
        gdown.download(id=dataset_url, output=path, quiet=False)
    ndata = scipy.io.loadmat(path)

    data = Data(
        x=torch.tensor(ndata['node_feat']).float(),
        y=torch.tensor(ndata['label'].flatten(), dtype=torch.long),
        edge_index=torch.tensor(ndata['edge_index'], dtype=torch.long),
    )
    transform = T.Compose([T.ToUndirected(), T.ToSparseTensor()])
    data = transform(data)

    num_features, num_classes = data.x.shape[1], (data.y.max() + 1).item()

    data = load_split(f'{root}/pokec', data, train_rate, val_rate)

    return data, num_features, num_classes, f'{root}/pokec'


def get_snap_patents(root: str, train_rate=0.5, val_rate=0.25, nclass=5, **kwargs):
    dataset_url = '1ldh23TSY1PwXia6dU0MYcpyEgX-w3Hia'

    os.makedirs(f'{root}/snap-patents', exist_ok=True)
    path = f'{root}/snap-patents/snap-patents.mat'
    if not osp.exists(path):
        gdown.download(id=dataset_url, output=path, quiet=False)
    ndata = scipy.io.loadmat(path)

    label = even_quantile_labels(ndata['years'].flatten(), nclass, verbose=False)
    data = Data(
        x=torch.tensor(ndata['node_feat'].todense(), dtype=torch.float),
        y=torch.tensor(label, dtype=torch.long),
        edge_index=torch.tensor(ndata['edge_index'], dtype=torch.long),
    )
    transform = T.Compose([T.ToSparseTensor()])  # directed graph
    data = transform(data)

    data = load_split(f'{root}/snap-patents', data, train_rate, val_rate)

    return data, data.x.shape[1], nclass, f'{root}/snap-patents'


def get_genius(root: str, train_rate=0.5, val_rate=0.25, **kwargs):
    url = f'https://github.com/CUAI/Non-Homophily-Large-Scale/raw/master/data/genius.mat'
    dataset_dir = f'{root}/genius'
    download_url(url, dataset_dir, log=False)
    ndata = scipy.io.loadmat(f'{dataset_dir}/genius.mat')

    undirected = kwargs.pop('undirected', True)

    data = Data(
        x=torch.tensor(ndata['node_feat'], dtype=torch.float),
        y=torch.tensor(ndata['label'].flatten(), dtype=torch.long),
        edge_index=torch.tensor(ndata['edge_index'], dtype=torch.long),
    )
    trans_list = [T.ToSparseTensor()]
    if undirected:
        trans_list.insert(0, T.ToUndirected())
    transform = T.Compose(trans_list)
    data = transform(data)

    data = load_split(dataset_dir, data, train_rate, val_rate)

    num_features, num_classes = data.x.shape[1], 1  # calculate AUC_ROC

    return data, num_features, num_classes, dataset_dir


def get_twitch_gamer(root: str, train_rate=0.5, val_rate=0.25, task='mature', normalize=True, **kwargs):
    # TODO: Google Drive URL is not available
    url_feat = '1fA9VIIEI8N0L27MSQfcBzJgRQLvSbrvR'
    url_edge = '1XLETC6dG3lVl7kDmytEJ52hvDMVdxnZ0'
    dataset_dir = f'{root}/twitch-gamer'
    os.makedirs(dataset_dir, exist_ok=True)

    if not osp.exists(f'{dataset_dir}/twitch-gamer_feat.csv'):
        gdown.download(id=url_feat, output=f'{dataset_dir}/twitch-gamer_feat.csv', quiet=False)
    if not osp.exists(f'{dataset_dir}/twitch-gamer_edges.csv'):
        gdown.download(id=url_edge, output=f'{dataset_dir}/twitch-gamer_edges.csv', quiet=False)
    edges = pd.read_csv(f'{dataset_dir}/twitch-gamer_edges.csv')
    nodes = pd.read_csv(f'{dataset_dir}/twitch-gamer_feat.csv')

    label, features = load_twitch_gamer(nodes, task)
    node_feat = torch.tensor(features, dtype=torch.float)
    if normalize:
        node_feat = node_feat - node_feat.mean(dim=0, keepdim=True)
        node_feat = node_feat / node_feat.std(dim=0, keepdim=True)

    data = Data(
        x=torch.tensor(node_feat, dtype=torch.float),
        y=torch.tensor(label),
        edge_index=torch.tensor(edges.to_numpy(), dtype=torch.long).t(),
    )
    transform = T.Compose([T.ToUndirected(), T.ToSparseTensor()])
    data = transform(data)

    data = load_split(f'{root}/twitch-gamer', data, train_rate, val_rate)

    num_features, num_classes = data.x.shape[1], (data.y.max() + 1).item()

    return data, num_features, num_classes, dataset_dir


def get_metric(name, num_classes):
    if name.lower() in ['yelp', 'ppi']:
        return torchmetrics.F1Score(task="multilabel", average='micro', num_labels=num_classes)
    elif name.lower() in ['minesweeper', 'tolokers', 'questions', 'genius']:
        return torchmetrics.AUROC(task="binary")
    else:
        return torchmetrics.Accuracy(task="multiclass", num_classes=num_classes)


def get_data(name: str, root: str, **kwargs):
    """
    Returns: pyg Data, number of features, number of classes, processed directory
    """
    if name.lower() in ['cora', 'citeseer', 'pubmed']:
        return get_planetoid(root, name, **kwargs)
    elif name.lower() in ['computers', 'photo']:
        return get_amazon(root, name, **kwargs)
    elif name.lower() == 'wikics':
        return get_wikics(root, **kwargs)
    elif name.lower() == 'actor':
        return get_actor(root, **kwargs)
    elif name.lower() in ['chameleon', 'squirrel']:
        return get_wikinet(root, name, **kwargs)
    elif name.lower() in ['roman_empire', 'amazon_ratings', 'minesweeper', 'tolokers', 'questions']:
        return get_heterophilious(root, name, **kwargs)
    elif name.lower() in ['cornell', 'texas', 'wisconsin']:
        return get_webkb(root, name, **kwargs)
    elif name.lower() in ['cluster', 'pattern']:
        return get_sbm(root, name, **kwargs)
    elif name.lower() == 'reddit':
        return get_reddit(root, **kwargs)
    elif name.lower() == 'ppi':
        return get_ppi(root, **kwargs)
    elif name.lower() == 'flickr':
        return get_flickr(root, **kwargs)
    elif name.lower() == 'yelp':
        return get_yelp(root, **kwargs)
    elif name.lower() in ['arxiv', 'products']:
        return get_ogbn(root, name, **kwargs)
    elif name.lower() == 'pokec':
        return get_pokec(root, **kwargs)
    elif name.lower() == 'arxiv-year':
        return get_arxiv_year(root, **kwargs)
    elif name.lower() in ['penn94', 'reed98', 'cornell5', 'amherst41']:
        return get_facebook100(root, name, **kwargs)
    elif name.lower() == 'snap-patents':
        return get_snap_patents(root, **kwargs)
    elif name.lower() == 'genius':
        return get_genius(root, **kwargs)
    elif name.lower() == 'twitch-gamers':
        return get_twitch_gamer(root, **kwargs)
    else:
        raise NotImplementedError


def get_expert_masks(proc_dir, data, use_test=True,
                     fraction=0.8, group='homophily', num_experts=2, expert_id=None,
                     **kwargs):
    """
    use_test: whether to use test set for metric calculating and node sorting.
    """
    N = data.x.size(0)
    assert N == data.train_mask.shape[0]
    dataset = kwargs['name']
    train_mask = data.train_mask.view(N, -1)
    val_mask = data.val_mask.view(N, -1)
    test_mask = data.test_mask.view(N, -1)

    # if use learned homophily, then force use_test=False
    if kwargs.get('learned_homo', ''):
        use_test = False

    def get_sorted_indices(_use_test, _use_val=True, no_learn=False):
        """
        load sorted indices from file if exists
        """
        identifiers = {}
        if group == 'neighbor_confusion':
            identifiers['threshold'] = kwargs.get('nc_threshold', 0.7)
            identifiers['hop'] = kwargs.get('nc_hop', 1)
        elif group == 'homophily':
            identifiers['hop'] = kwargs.get('hop', 1)
        os.makedirs(proc_dir, exist_ok=True)
        _file_path = os.path.join(proc_dir, f'sorted-indices_{dataset}_{group}')
        for k, v in identifiers.items():
            _file_path += f'_{k}-{v}'
        if not _use_test:
            if kwargs.get('learned_homo', '') and not no_learn:
                _file_path += f'_learnedHomo-{kwargs["learned_homo"]}'
            _file_path += '_filter-test'
        if not _use_val:
            _file_path += '_filter-val'
        _file_path += '.pt'
        if osp.exists(_file_path):
            _sorted_indices = torch.load(_file_path, weights_only=True)
            if _use_test:
                assert _sorted_indices.shape == (N,)
            else:
                assert _sorted_indices.shape == (N, train_mask.shape[1])
            return _sorted_indices
        if kwargs.get('learned_homo', '') and not no_learn:
            raise FileNotFoundError('Learned Homophily is not available')

        def _get_sorted_indices(_masked_adj):
            if group == 'homophily':
                _hop = kwargs.get('hop', 1)
                _metric_val = get_node_homophily(_masked_adj, data.y, _hop)
            elif group == 'homophily_2':
                _metric_val = get_node_homophily(_masked_adj, data.y, hop=2)
            elif group == 'degree':
                _metric_val = get_degree(data.adj_t)
            elif group == 'LSI_inf':
                _metric_val = get_LSI(data, max_hop=1).t()[1]
            elif group == 'LSI_self':
                _metric_val = get_LSI2(data, max_hop=1).t()[1]
            elif group == 'pagerank':
                _metric_val = get_pagerank(data.adj_t.t())
            elif group == 'random':
                _metric_val = torch.randperm(N)
            elif group == 'neighbor_confusion':
                threshold = kwargs.get('nc_threshold', 0.7)
                _hop = kwargs.get('nc_hop', 1)
                _metric_val = get_neighborhood_confusion(data, dataset, threshold=threshold, hop=_hop, path=proc_dir)
            elif group == 'clustering_coef':
                _metric_val = get_clustering_coef(data)
            elif group == 'directional_label_distance':
                _metric_val = get_directional_label_distance(data)
            elif group == 'directional_label_similarity':
                _metric_val = get_directional_label_similarity(data)
            elif group == 'directional_label_cos_similarity':
                _metric_val = get_directional_label_cos_similarity(data)
            elif group == 'direction_informativeness':
                _metric_val = get_direction_informativeness(data)
            elif group == 'directionality':
                _metric_val = get_directionality(data)
            elif group == 'intra_class_directionality':
                _metric_val = get_intra_class_directionality(data)
            elif group == 'intra_class_neighborhood_label_similarity':
                _metric_val = get_intra_class_neighborhood_label_similarity(data)
            elif group == 'intra_class_neighborhood_label_similarity_norm':
                _metric_val = (get_intra_class_neighborhood_label_similarity(data) /
                    get_intra_class_neighborhood_label_similarity(data, return_all_classes=True))
            elif group == 'intra_class_degree':
                _metric_val = get_intra_class_degree(data)
            elif group == 'intra_class_neighborhood_feature_distance':
                _metric_val = get_intra_class_neighborhood_feature_distance(data)
            elif group == 'intra_class_neighborhood_feature_similarity':
                _metric_val = get_intra_class_neighborhood_feature_similarity(data)
            elif group == 'intra_class_neighborhood_feature_cos_similarity':
                _metric_val = get_intra_class_neighborhood_feature_similarity(data, cos=True)
            elif group == 'intra_class_neighborhood_feature_cos_similarity_norm':
                _metric_val = (get_intra_class_neighborhood_feature_similarity(data, cos=True) /
                    get_intra_class_neighborhood_feature_similarity(data, cos=True, return_all_classes=True))
            elif group == 'intra_class_neighborhood_feature_cos_similarity_deg':
                _metric_val = get_intra_class_neighborhood_feature_similarity(data, cos=True) / get_degree(data.adj_t)
            elif group == 'max_neighbor_label_ratio':
                _metric_val = get_max_neighbor_label_ratio(data)
            elif group == 'neighbor_label_entropy':
                _metric_val = get_neighbor_label_entropy(data)
            else:
                raise NotImplementedError(f'Group {group} not implemented')
            return torch.argsort(_metric_val)

        if not _use_test:
            _sorted_indices = []
            for _i in range(test_mask.shape[1]):
                _selected_nodes = train_mask[:, _i] | val_mask[:, _i] if _use_val else train_mask[:, _i]
                _masked_adj = get_masked_adj(data, _selected_nodes)
                _sorted_indices.append(_get_sorted_indices(_masked_adj))
            _sorted_indices = torch.stack(_sorted_indices, dim=1)
        else:
            assert _use_val is True
            _sorted_indices = _get_sorted_indices(data.adj_t)
        torch.save(_sorted_indices, _file_path)
        return _sorted_indices

    all_expert_masks = []
    if not use_test:
        assert group == 'homophily'  # TODO: Currently support homophily only

        def _get_expert_mask(_original_mask, _sorted_index):
            _sorted_index = _sorted_index.to(_original_mask.device)
            _mask_size = _original_mask.sum().item()
            _start = int((1 - fraction) / (num_experts - 1) * expert_id * _mask_size)
            _end = int(_start + fraction * _mask_size)

            _sorted_mask = _original_mask[_sorted_index]
            _masked_sorted_indices_i = _sorted_index[_sorted_mask]
            _group_mask = torch.zeros(N, dtype=torch.bool).to(data.x.device)
            _group_mask[_masked_sorted_indices_i[_start:_end]] = True
            return _group_mask

        # calculate sorted indices for all masks
        use_val = kwargs.get('use_val', True)
        all_sorted_indices = [
            get_sorted_indices(_use_test=False, _use_val=use_val),  # train
            get_sorted_indices(_use_test=False, _use_val=True, no_learn=False),    # val
            get_sorted_indices(_use_test=True, _use_val=True        # test
                               ).unsqueeze(1).tile((1, train_mask.shape[1]))
        ]

        all_masks = [train_mask, val_mask, test_mask]
        for original_mask, sorted_indices in zip(all_masks, all_sorted_indices):
            expert_masks = [
                _get_expert_mask(original_mask[:, i], sorted_indices[:, i])
                for i in range(original_mask.shape[1])
            ]
            all_expert_masks.append(torch.stack(expert_masks, dim=1))
    else:
        if 'kmeans' in group:
            os.makedirs(proc_dir, exist_ok=True)
            _file_path = os.path.join(proc_dir, f'sorted-indices_{dataset}_{group}')
            identifiers = {'k': num_experts}
            for k, v in identifiers.items():
                _file_path += f'_{k}-{v}'
            _file_path += '.pt'
            if osp.exists(_file_path):
                group_masks = torch.load(_file_path, weights_only=True)
            else:
                if group == 'kmeans':
                    group_masks = get_kmeans_clustering_mask(data.x, num_experts)
                elif group == 'kmeans-balanced':
                    group_masks = get_balanced_kmeans_clustering_mask(data.x, num_experts)
                elif group == 'aggregated-kmeans-balanced':
                    group_masks = get_balanced_aggregated_kmeans_clustering_mask(data, num_experts)
                torch.save(group_masks, _file_path)
            group_mask = group_masks[expert_id, :]
        else:
            sorted_indices = get_sorted_indices(_use_test=True)
            group_mask = torch.zeros(N, dtype=torch.bool).to(data.x.device)
            if fraction == 1:
                group_mask.fill_(True)
            else:
                start = int((1 - fraction) / (num_experts - 1) * expert_id * N)
                end = int(start + fraction * N)
                group_mask[sorted_indices[start:end]] = True

        for original_mask in [train_mask, val_mask, test_mask]:
            expert_masks = []
            expert_mask = original_mask.clone()
            for i in range(expert_mask.shape[1]):
                expert_mask[:, i] = expert_mask[:, i] & group_mask
                expert_masks.append(expert_mask[:, i].contiguous())
            all_expert_masks.append(torch.stack(expert_masks, dim=1).squeeze().contiguous())

    return all_expert_masks


def get_expert_data(name: str, root: str, **kwargs):
    assert 'expert_id' in kwargs and kwargs['expert_id'] is not None

    data, num_features, num_classes, dataset_dir = get_data(name, root, **kwargs)

    kwargs['name'] = name
    use_test = kwargs.pop('use_test', True)  # whether to filter out test labels during metric calculation
    data.expert_train_masks, data.expert_val_masks, data.expert_test_masks = get_expert_masks(
        dataset_dir, data, use_test, **kwargs)

    return data, num_features, num_classes, dataset_dir
