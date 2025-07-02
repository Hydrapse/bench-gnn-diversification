import copy
import random

import dgl
import faiss
import numpy as np
import torch

import torch.nn as nn
import dgl.function as fn
import torch.nn.functional as F
from dgl.nn.pytorch import GraphConv, SAGEConv

# from libKMCUDA import kmeans_cuda
from sklearn.cluster import KMeans

from components.backbone import get_model
from components.layer import BaseMLP
from utils.utils import Dict, shuffle_intra_nodes, adj_norm, kmeans_clustering


def aug_feature_dropout(input_feat, drop_rate=0.2):
    """
    dropout features for augmentation.
    args:
        input_feat: input features
        drop_rate: dropout rate
    returns:
        aug_input_feat: augmented features
    """
    aug_input_feat = copy.deepcopy(input_feat).squeeze(0)
    drop_feat_num = int(aug_input_feat.shape[1] * drop_rate)
    drop_idx = random.sample([i for i in range(aug_input_feat.shape[1])], drop_feat_num)
    aug_input_feat[:, drop_idx] = 0
    aug_input_feat = aug_input_feat.unsqueeze(0)

    return aug_input_feat


def aug_feature_shuffle(input_feat):
    """
    shuffle the features for fake samples.
    args:
        input_feat: input features
    returns:
        aug_input_feat: augmented features
    """
    fake_input_feat = input_feat[:, np.random.permutation(input_feat.shape[1]), :]
    return fake_input_feat


# ------------------------from scratch------------------------
class GCN(nn.Module):
    def __init__(self, in_ft, out_ft, act):
        super(GCN, self).__init__()
        self.fc = nn.Linear(in_ft, out_ft, bias=False)
        self.act = nn.PReLU() if act == 'prelu' else act
        self.bias = nn.Parameter(torch.FloatTensor(out_ft))

        # init parameters
        torch.nn.init.xavier_uniform_(self.fc.weight.data)
        self.bias.data.fill_(0.0)

    def forward(self, feat, adj, sparse=False):
        h = self.fc(feat)
        if sparse:
            out = torch.unsqueeze(torch.spmm(adj, torch.squeeze(h, 0)), 0)
        else:
            out = torch.bmm(adj, h)
        out += self.bias
        return self.act(out)


class DinkNet(nn.Module):
    def __init__(self, n_in, n_h, n_cluster, tradeoff=1e-10, activation="prelu"):
        super(DinkNet, self).__init__()
        self.n_cluster = n_cluster
        self.cluster_center = torch.nn.Parameter(torch.Tensor(n_cluster, n_h))
        self.gcn = GCN(n_in, n_h, activation)
        self.lin = nn.Linear(n_h, n_h)
        self.discrimination_loss = nn.BCEWithLogitsLoss()
        self.tradeoff = tradeoff

    def forward(self, x_1, x_2, adj, sparse):
        h_1 = self.gcn(x_1, adj, sparse)
        h_2 = self.gcn(x_2, adj, sparse)
        z_1 = ((self.lin(h_1.squeeze(0))).sum(1))
        z_2 = ((self.lin(h_2.squeeze(0))).sum(1))
        logit = torch.cat((z_1, z_2), 0)
        return logit

    def embed(self, x, adj, power=5, sparse=True):
        local_h = self.gcn(x, adj, sparse)
        global_h = local_h.clone().squeeze(0)
        for i in range(power):
            global_h = adj @ global_h
        global_h = global_h.unsqueeze(0)
        local_h, global_h = map(lambda tmp: tmp.detach(), [local_h, global_h])
        h = local_h + global_h
        h = h.squeeze(0)
        h = F.normalize(h, p=2, dim=-1)
        return h

    @staticmethod
    def dis_fun(x, c):
        xx = (x * x).sum(-1).reshape(-1, 1).repeat(1, c.shape[0])
        cc = (c * c).sum(-1).reshape(1, -1).repeat(x.shape[0], 1)
        xx_cc = xx + cc
        xc = x @ c.T
        distance = xx_cc - 2 * xc
        return distance

    @staticmethod
    def no_diag(x, n):
        x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()

    def cal_loss(self, x, adj, finetune=True):
        # augmentations
        x_aug = aug_feature_dropout(x)
        x_shuffle = aug_feature_shuffle(x_aug)

        # discrimination loss
        logit = self.forward(x_aug, x_shuffle, adj, sparse=True)
        n = logit.shape[0] // 2
        disc_y = torch.cat((torch.ones(n), torch.zeros(n)), 0).to(logit.device)
        loss_disc = self.discrimination_loss(logit, disc_y)

        if finetune:
            # clustering loss
            h = self.embed(x, adj, power=5, sparse=True)
            sample_center_distance = self.dis_fun(h, self.cluster_center)
            center_distance = self.dis_fun(self.cluster_center, self.cluster_center)
            self.no_diag(center_distance, self.cluster_center.shape[0])
            clustering_loss = sample_center_distance.mean() - center_distance.mean()

            # tradeoff
            loss = clustering_loss + self.tradeoff * loss_disc

        else:
            loss = loss_disc
            sample_center_distance = None

        return loss, sample_center_distance

    def clustering(self, x, adj, finetune=True, cuda=True):
        h = self.embed(x, adj, sparse=True)
        if finetune:
            sample_center_distance = self.dis_fun(h, self.cluster_center)
            cluster_results = torch.argmin(sample_center_distance, dim=-1).cpu().detach().numpy()
        else:
            if cuda:
                # center, cluster_results = kmeans_cuda(h.cpu().detach().numpy(), self.n_cluster, verbosity=0, seed=3, device=h.get_device())
                # self.cluster_center.data = torch.tensor(center).to(h.device)
                raise Exception("cuda not supported")
            else:
                km = KMeans(n_clusters=self.n_cluster, n_init=20).fit(h.cpu().detach().numpy())
                cluster_results = km.labels_
                self.cluster_center.data = torch.tensor(km.cluster_centers_).to(h.device)

        return cluster_results


# ------------------------from dgl------------------------
class Encoder(nn.Module):
    def __init__(self, in_feats, n_hidden, n_layers, activation, gnn_encoder='gcn'):
        super(Encoder, self).__init__()
        self.gnn_encoder = gnn_encoder
        activation = nn.PReLU(n_hidden) if activation == 'prelu' else activation
        if gnn_encoder == 'gcn':
            self.conv = GCN_dgl(in_feats, n_hidden, n_layers, activation)
        # elif gnn_encoder == 'sgc':
        #     self.conv = SGConv(in_feats, n_hidden, k=power, cached=True)

    def forward(self, features, g, corrupt=False, batch_train=False):
        if corrupt:
            perm = torch.randperm(features.shape[0])
            features = features[perm]
        if self.gnn_encoder == 'gcn':
            features = self.conv(g, features, batch_train)
        elif self.gnn_encoder == 'sgc':
            features = self.conv(g, features, batch_train)
        return features


class GCN_dgl(nn.Module):
    def __init__(self, n_in, n_h, n_layers, activation, bias=True, weight=True):
        super(GCN_dgl, self).__init__()
        self.layers = nn.ModuleList()

        # input layer
        self.layers.append(GraphConv(n_in, n_h, weight=weight, bias=bias, activation=activation))
        # self.layers.append(SAGEConv(n_in, n_h, bias=bias, activation=activation, aggregator_type='mean'))

        # hidden layers
        for i in range(n_layers - 1):
            self.layers.append(GraphConv(n_h, n_h, weight=weight, bias=bias, activation=activation))
            # self.layers.append(SAGEConv(n_h, n_h, bias=bias, activation=activation, aggregator_type='mean'))

    def forward(self, g, feat, batch_train=False):
        h = feat.squeeze(0)
        if batch_train:
            for i, layer in enumerate(self.layers):
                h = layer(g[i], h)
        else:
            for i, layer in enumerate(self.layers):
                h = layer(g, h)
        return h


class DinkNet_dgl(nn.Module):
    def __init__(self, g_global, n_in, n_h, n_cluster, encoder_layers,
                 tradeoff=1e-10, activation='prelu', projector_layers=1,
                 dropout_rate=0.2, gnn_encoder='gcn', n_hop=10):
        super(DinkNet_dgl, self).__init__()
        self.g_global = g_global
        self.n_cluster = n_cluster
        self.cluster_center = torch.nn.Parameter(torch.Tensor(n_cluster, n_h))
        self.encoder = Encoder(n_in, n_h, encoder_layers, activation, gnn_encoder)
        self.mlp = torch.nn.ModuleList()
        for i in range(projector_layers):
            self.mlp.append(nn.Linear(n_h, n_h))
        self.discrimination_loss = nn.BCEWithLogitsLoss()
        self.tradeoff = tradeoff
        self.dropout_rate = dropout_rate

    def forward(self, x, g, batch_train):
        z_1 = self.encoder(x, g, corrupt=False, batch_train=batch_train)
        z_2 = self.encoder(x, g, corrupt=True, batch_train=batch_train)

        for i, lin in enumerate(self.mlp):
            z_1 = lin(z_1)
            z_2 = lin(z_2)

        logit = torch.cat((z_1.sum(1), z_2.sum(1)), 0)
        return logit

    def embed(self, x, g, power=10, batch_train=False):
        local_h = self.encoder(x, g, corrupt=False, batch_train=batch_train)

        feat = local_h.clone().squeeze(0)

        if batch_train:
            g = dgl.node_subgraph(self.g_global, g[-1].dstdata["_ID"].to(self.g_global.device)).to(feat.device)

        norm = torch.pow(g.in_degrees().float().clamp(min=1), -0.5).unsqueeze(1).to(local_h.device)
        for i in range(power):
            feat = feat * norm
            g.ndata['h2'] = feat
            g.update_all(fn.copy_u('h2', 'm'), fn.sum('m', 'h2'))
            feat = g.ndata.pop('h2')
            feat = feat * norm

        global_h = feat.unsqueeze(0)
        local_h, global_h = map(lambda tmp: tmp.detach(), [local_h, global_h])

        h = local_h + global_h
        h = h.squeeze(0)
        h = F.normalize(h, p=2, dim=-1)

        return h

    @staticmethod
    def dis_fun(x, c):
        xx = (x * x).sum(-1).reshape(-1, 1).repeat(1, c.shape[0])
        cc = (c * c).sum(-1).reshape(1, -1).repeat(x.shape[0], 1)
        xx_cc = xx + cc
        xc = x @ c.T
        distance = xx_cc - 2 * xc
        return distance

    @staticmethod
    def no_diag(x, n):
        x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()

    def cal_loss(self, x, g, batch_train=False, finetune=True):
        # augmentations
        x_aug = aug_feature_dropout(x, drop_rate=self.dropout_rate).squeeze(0)

        logit = self.forward(x_aug, g, batch_train=batch_train)

        # label of discriminative task
        n = logit.shape[0] // 2
        disc_y = torch.cat((torch.ones(n), torch.zeros(n)), 0).to(logit.device)

        # discrimination loss
        loss_disc = self.discrimination_loss(logit, disc_y)

        if finetune:
            # clustering loss
            h = self.embed(x, g, power=10, batch_train=batch_train)
            sample_center_distance = self.dis_fun(h, self.cluster_center)
            center_distance = self.dis_fun(self.cluster_center, self.cluster_center)
            self.no_diag(center_distance, self.cluster_center.shape[0])
            clustering_loss = sample_center_distance.mean() - center_distance.mean()

            # tradeoff
            loss = clustering_loss + self.tradeoff * loss_disc

        else:
            loss = loss_disc
            sample_center_distance = None

        return loss, sample_center_distance

    def clustering(self, x, adj, batch_train=False, finetune=True, cuda=True):
        h = self.embed(x, adj, power=10, batch_train=batch_train)
        # h = self.embed(x, adj, power=0, batch_train=batch_train)
        if finetune:
            sample_center_distance = self.dis_fun(h, self.cluster_center)
            cluster_results = torch.argmin(sample_center_distance, dim=-1).cpu().detach().numpy()
        else:
            if cuda:
                faiss_kmeans = faiss.Kmeans(d=h.shape[1], k=self.n_cluster, gpu=True)
                h_np = h.cpu().detach().numpy()
                faiss_kmeans.train(h_np)
                _, cluster_results = faiss_kmeans.index.search(h_np, 1)
                cluster_results = cluster_results.reshape(-1, )
                self.cluster_center.data = torch.from_numpy(faiss_kmeans.centroids).to(h.device)
                # center, cluster_results = kmeans_cuda(h.cpu().detach().numpy(), self.n_cluster, verbosity=0, seed=3, device=h.get_device())
                # self.cluster_center.data = torch.tensor(center).to(h.device)
            else:
                km = KMeans(n_clusters=self.n_cluster, n_init=20).fit(h.cpu().detach().numpy())
                cluster_results = km.labels_
                self.cluster_center.data = torch.tensor(km.cluster_centers_).to(h.device)
        return cluster_results, h


class IntraClassEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, layers, activation='prelu',
                 dropout=0., norm='layer', mixer='cat'):
        super(IntraClassEncoder, self).__init__()

        jk = 'null'
        intra_enc_conf = Dict({
            'name': 'GCN',
            'hidden_dim': hidden_dim,
            'init_layers': 0,
            'conv_layers': layers,
            'dropout': dropout,
            'norm': norm,
            'jk': jk,
            'activation': activation,
            # 'residual': 'cat',
            # 'adj_norm': 'rw',
        })
        masked_enc_conf = Dict({
            'name': 'GCN',
            'hidden_dim': hidden_dim,
            'init_layers': 0,
            'conv_layers': layers,
            'dropout': dropout,
            'norm': norm,
            'jk': jk,
            'activation': activation,
            # 'residual': 'cat',
            # 'adj_norm': 'rw',
        })
        full_enc_conf = Dict({
            'name': 'GCN',
            'hidden_dim': hidden_dim,
            'init_layers': 0,
            'conv_layers': layers,
            'dropout': dropout,
            'norm': norm,
            'jk': jk,
            'activation': activation,
            # 'residual': 'cat',
            # 'adj_norm': 'rw',
        })
        self.intra_enc = get_model(intra_enc_conf, input_dim, hidden_dim)
        self.masked_enc = get_model(masked_enc_conf, input_dim, hidden_dim)
        self.full_enc = get_model(full_enc_conf, input_dim, hidden_dim)

        self.mixer_type = mixer
        self.W = nn.Linear(3 * hidden_dim, hidden_dim)

    def encode(self, x, adj_intra, adj_masked, adj, enc_type='all'):
        xs = []
        if enc_type in ['intra', 'all']:
            xs.append(self.intra_enc(x, adj_intra))
        if enc_type in ['masked', 'all']:
            xs.append(self.masked_enc(x, adj_masked))
        if enc_type in ['full', 'all']:
            xs.append(self.full_enc(x, adj))
        return xs

    def mixer(self, xs):
        if self.mixer_type == 'cat':
            xs_arr, xs_sum = [], 0
            for _x in xs:
                xs_arr.append(_x)
                xs_sum += _x
            out = F.relu(self.W(torch.cat(xs_arr, dim=-1)) + xs_sum)
        elif self.mixer_type == 'sum':
            out = xs[0]
            for _x in xs[1:]:
                out += _x
        else:
            raise ValueError('Unknown mixer type: {}'.format(self.mixer_type))
        return out

    def forward(self, x, adj_intra, adj_masked, adj):
        x_intra, x_masked, x_full = self.encode(x, adj_intra, adj_masked, adj)
        out = self.mixer([x_intra, x_masked, x_full])
        # out = self.mixer([x_masked, x_full])
        # out = self.mixer([x_full])
        # out = self.mixer([x_intra])
        return out


class IntraClassRouter(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_clusters, encoder_conf,
                 tradeoff=1e-10, projector_layers=1,
                 feat_dropout=0.2, pipeline='trainSep', **kwargs):

        super(IntraClassRouter, self).__init__()

        self.n_cluster = num_clusters
        self.cluster_center = torch.nn.Parameter(torch.Tensor(num_clusters, hidden_dim))
        self.discrimination_loss = nn.BCEWithLogitsLoss()
        self.tradeoff = tradeoff
        self.feat_dropout = feat_dropout
        self.pipeline = pipeline  # trainSep/trainTog, corIntra, combSum

        self.encoder = IntraClassEncoder(input_dim, hidden_dim, **encoder_conf)

        mlp_conf = {
            'in_channels': hidden_dim,
            'hidden_channels': hidden_dim,
            'out_channels': hidden_dim,
            'num_layers': projector_layers,
            'dropout': 0,
            'norm': None,
            'keep_last_act': False,
        }
        self.projector = BaseMLP(**mlp_conf)

    def forward_together(self, x, adj_intra, adj_masked, adj, y, labeled_nodes,
                         trained_on_labeled_nodes=False):
        x_corrupt = self.corrupt(x, y, labeled_nodes, cor_type='rand_full')

        z = self.encoder(x, adj_intra, adj_masked, adj)
        z_sim = self.encoder(x_corrupt, adj_intra, adj_masked, adj)

        z = self.projector(z)
        z_sim = self.projector(z_sim)

        selected_nodes = labeled_nodes if trained_on_labeled_nodes else torch.ones_like(labeled_nodes)

        logit = torch.cat((z.sum(1)[selected_nodes], z_sim.sum(1)[selected_nodes]), 0)
        return logit

    def forward_separate(self, x, adj_intra, adj_masked, adj, y, labeled_nodes,
                         trained_on_labeled_nodes=False):
        x_corrupt = self.corrupt(x, y, labeled_nodes, cor_type='rand_full')
        x_cor_intra = self.corrupt(x, y, labeled_nodes,
                                   cor_type='rand_intra') if "corIntra" in self.pipeline else x_corrupt

        z_intra, z_masked, z_full = self.encoder.encode(x, adj_intra, adj_masked, adj)
        z_intra_sim = self.encoder.encode(x_cor_intra, adj_intra, adj_masked, adj, 'intra')[0]
        z_masked_sim = self.encoder.encode(x_corrupt, adj_intra, adj_masked, adj, 'masked')[0]
        z_full_sim = self.encoder.encode(x_corrupt, adj_intra, adj_masked, adj, 'full')[0]

        z = self.encoder.mixer([z_intra, z_masked, z_full])

        z_intra, z_intra_sim, z_masked, z_masked_sim, z_full, z_full_sim = map(
            lambda _z: self.projector(self.combine(z, _z)),
            [z_intra, z_intra_sim, z_masked, z_masked_sim, z_full, z_full_sim]
        )

        selected_nodes = labeled_nodes if trained_on_labeled_nodes else torch.ones_like(labeled_nodes)

        logit = torch.cat([
            z_intra.sum(1)[selected_nodes],
            z_masked.sum(1)[selected_nodes],
            z_full.sum(1)[selected_nodes],
            z_intra_sim.sum(1)[selected_nodes],
            z_masked_sim.sum(1)[selected_nodes],
            z_full_sim.sum(1)[selected_nodes],
        ], 0)
        return logit

    def combine(self, _a, _b):
        # TODO:
        if 'combSum' in self.pipeline:
            out = _a + _b
        else:
            out = torch.cat([_a, _b], dim=-1)
        return out

    def embed(self, x, adj_intra, adj_masked, adj, power=10, norm='sym'):
        local_h = self.encoder(x, adj_intra, adj_masked, adj)

        if power <= 0:
            h = local_h.detach()
        else:
            global_h = local_h.clone().squeeze(0)
            _adj = adj_norm(adj, norm)
            for i in range(power):
                global_h = _adj @ global_h
            global_h = global_h.unsqueeze(0)

            h = local_h.detach() + global_h.detach()

        h = h.squeeze(0)
        h = F.normalize(h, p=2, dim=-1)
        return h

    @staticmethod
    def corrupt(x, y, labeled_nodes, cor_type='rand_full'):
        if cor_type == 'rand_full':
            perm = torch.randperm(x.shape[0])
        elif cor_type == 'rand_intra':
            perm = shuffle_intra_nodes(y, labeled_nodes)
        else:
            raise ValueError
        x_corrupt = x[perm]
        return x_corrupt

    @staticmethod
    def dis_fun(x, c):
        xx = (x * x).sum(-1).reshape(-1, 1).repeat(1, c.shape[0])
        cc = (c * c).sum(-1).reshape(1, -1).repeat(x.shape[0], 1)
        xx_cc = xx + cc
        xc = x @ c.T
        distance = xx_cc - 2 * xc
        return distance

    @staticmethod
    def no_diag(x, n):
        x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()

    def cal_loss(self, x, adj_intra, adj_masked, adj, y, labeled_nodes, finetune=True):
        # augmentations
        x_aug = aug_feature_dropout(x, drop_rate=self.feat_dropout).squeeze(0)

        # forward TODO: only use labeled nodes to calculate loss
        if self.pipeline in ['trainTog']:
            forward_func = self.forward_together
        elif self.pipeline in ['trainSep', 'trainSep_corIntra']:
            forward_func = self.forward_separate
        else:
            raise ValueError
        logit = forward_func(x_aug, adj_intra, adj_masked, adj, y, labeled_nodes,
                             trained_on_labeled_nodes=False)

        # label of discriminative task
        n = logit.shape[0] // 2
        disc_y = torch.cat((torch.ones(n), torch.zeros(n)), 0).to(logit.device)

        # discrimination loss
        loss_disc = self.discrimination_loss(logit, disc_y)

        if finetune:
            # clustering loss
            h = self.embed(x, adj_intra, adj_masked, adj, power=10)
            sample_center_distance = self.dis_fun(h, self.cluster_center)
            center_distance = self.dis_fun(self.cluster_center, self.cluster_center)
            self.no_diag(center_distance, self.cluster_center.shape[0])
            clustering_loss = sample_center_distance.mean() - center_distance.mean()

            # tradeoff
            loss = clustering_loss + self.tradeoff * loss_disc
        else:
            loss = loss_disc
            sample_center_distance = None

        return loss, sample_center_distance

    def clustering(self, x, adj_intra, adj_masked, adj, finetune=False, power=10):
        h = self.embed(x, adj_intra, adj_masked, adj, power=power)
        if finetune:
            sample_center_distance = self.dis_fun(h, self.cluster_center)
            cluster_results = torch.argmin(sample_center_distance, dim=-1).detach()
        else:
            cluster_results, self.cluster_center.data = kmeans_clustering(h, self.n_cluster)
        return cluster_results, h
