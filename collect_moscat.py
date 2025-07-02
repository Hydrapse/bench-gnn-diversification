import logging
import os
import time
import warnings
from typing import List
import re
import json 

from torch_sparse import SparseTensor
import os.path as osp
from collections import defaultdict

import numpy as np
import torch
from torch import nn

from data import get_data, get_metric
from components.backbone import get_model
from logger import Logger
from utils.training import adjust_learning_rate
from utils.utils import adj_norm, loss_fn, pred_fn
from utils.utils import mask_to_index, index_to_mask
from components.layer import BaseMLP, AttentionChannelMixing
from utils.utils import Dict
from utils.utils import setup_seed

import torch.nn.functional as F
import torchmetrics
from torch_geometric.utils import scatter

import hydra
from omegaconf import OmegaConf, DictConfig
import warnings
import itertools
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)
warnings.filterwarnings("ignore")

log = logging.getLogger(__name__)
L_MAX = 6


def to_numpy(arr):
    if isinstance(arr, list):
        return [to_numpy(a) for a in arr]
    if isinstance(arr, torch.Tensor):
        return arr.detach().cpu().numpy()
    return np.asarray(arr)

def accuracy(pred, gold, mask):
    return np.mean((pred == gold)[mask])

def eval_weights(weights, logits, labels, val_masks, test_masks):
    """返回 (val_mean, val_std, test_mean, test_std)."""
    val_acc, test_acc = [], []
    num_masks = val_masks.shape[0]
    for r in range(logits.shape[0]):
        logits_w = (logits[r] * weights[:, None, None]).sum(axis=0)
        if logits_w.shape[-1] == 1:
            pred = (logits_w > 0).squeeze(-1)
        else:
            pred = logits_w.argmax(-1)
        val_acc.append(accuracy(pred, labels, val_masks[r % num_masks]))
        test_acc.append(accuracy(pred, labels, test_masks[r % num_masks]))
    val_acc, test_acc = np.array(val_acc), np.array(test_acc)
    return val_acc.mean(),  val_acc.std(),  test_acc.mean(),  test_acc.std()

def equal_grid_search(logits, labels, val_masks, test_masks, step=0.1):
    logits = to_numpy(logits)
    labels = to_numpy(labels)
    val_masks = to_numpy(val_masks)
    test_masks = to_numpy(test_masks)

    num_experts = logits.shape[1]
    grid = [round(i * step, 10) for i in range(int(1/step)+1)]  # e.g. 0.0 .. 1.0
    best_w = best_val = best_val_std = best_test = best_test_std = None

    for w_tuple in itertools.product(grid, repeat=num_experts):
        if abs(sum(w_tuple) - 1) > 1e-6:
            continue
        w = np.array(w_tuple)
        v_mean, v_std, t_mean, t_std = eval_weights(w, logits, labels, val_masks, test_masks)
        if best_val is None or v_mean > best_val:
            best_w, best_val, best_val_std = w, v_mean, v_std
            best_test, best_test_std = t_mean, t_std
    return best_w, best_val, best_val_std, best_test, best_test_std

def bayes_search(logits, labels, val_masks, test_masks,
                 n_trials=600, sampler=None, timeout=None):
    logits = to_numpy(logits)
    labels = to_numpy(labels)
    val_masks = to_numpy(val_masks)
    test_masks = to_numpy(test_masks)

    num_experts = logits.shape[1]

    def objective(trial):
        raw = np.array([
            trial.suggest_float(f"w{i}", 0.0, 1.0)
            for i in range(num_experts)
        ])
        weights = raw / raw.sum()
        val_mean, val_std, _, _ = eval_weights(
            weights, logits, labels, val_masks, test_masks
        )
        return val_mean

    study = optuna.create_study(direction="maximize",
                                sampler=sampler or optuna.samplers.TPESampler())
    study.optimize(objective, n_trials=n_trials,
                   timeout=timeout, show_progress_bar=False)

    best_raw = np.array([study.best_params[f"w{i}"] for i in range(num_experts)])
    best_w   = best_raw / best_raw.sum()
    best_val_mean, best_val_std, best_test_mean, best_test_std = eval_weights(
        best_w, logits, labels, val_masks, test_masks
    )
    return best_w, best_val_mean, best_val_std, best_test_mean, best_test_std


def get_best_expert_acc(_accs, _test_mask, _logits=None, _y=None, _log=False):
    r = _test_mask.size(0)

    best_acc, best_std = 0, 0
    for l in range(_accs.size(1)):
        _test_acc = []
        for run in range(_accs.size(0)):
            _mask = _test_mask[run % r]
            if _logits is not None and _logits.shape[-1] == 1:
                metric = torchmetrics.AUROC(task="binary")
                _acc = metric(_logits[run, l, _mask].cuda(), _y[_mask].cuda()).cpu()
            else:
                _acc = _accs[run, l, _mask].mean().item()
            _test_acc.append(_acc)

        if _log: print(f'{l}:  {np.mean(_test_acc):.4f} +- {np.std(_test_acc):.4f}')
        if np.mean(_test_acc) > best_acc:
            best_acc = np.mean(_test_acc)
            best_std = np.std(_test_acc)

    return best_acc, best_std


def count_peaks(vecs):
    """Ensure input vecs is a 2D tensor (batch_size, vector_length)"""
    l_shift = nn.functional.pad(vecs[:, 1:], (0, 1), "constant", float('-inf'))
    r_shift = nn.functional.pad(vecs[:, :-1], (1, 0), "constant", float('-inf'))
    peaks = (vecs > l_shift) & (vecs > r_shift)

    return peaks[:, :].sum(dim=1)


def check_dist(vecs, dim=0, eps=0.04):
    _all_correct = vecs.mean(dim=0) >= 1 - eps
    _all_wrong = vecs.mean(dim=0) <= eps

    _diffs = torch.diff(vecs, dim=dim)
    _mono_inc = torch.all(_diffs >= 0 - eps, dim=dim)
    _mono_dec = torch.all(_diffs <= 0 + eps, dim=dim)

    _cnt_peaks = count_peaks(vecs.t() if dim == 0 else vecs)
    _plateau_peak = _cnt_peaks == 0
    _single_peak = _cnt_peaks == 1 | _plateau_peak
    _double_peak = _cnt_peaks == 2
    _triple_peak = _cnt_peaks == 3

    return (
        _all_correct,
        _all_wrong,
        _mono_inc & ~_all_correct & ~_all_wrong,
        _mono_dec & ~_all_correct & ~_all_wrong,
        _single_peak & ~_all_correct & ~_all_wrong & ~_mono_inc & ~_mono_dec,
        _double_peak & ~_all_correct & ~_all_wrong & ~_mono_inc & ~_mono_dec,
        _triple_peak & ~_all_correct & ~_all_wrong & ~_mono_inc & ~_mono_dec,
    )


def _get_pagerank(adj, alpha_=0.15, epsilon_=1e-6, max_iter=100):
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


def _get_LSI(data, max_hop=6):
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


def _get_LSI2(data, max_hop=6):
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


def get_filtered_homophily(_data, node_mask):
    N = _data.x.size(0)
    train_mask_diag = torch.sparse_coo_tensor(
        indices=torch.arange(N, device=node_mask.device).repeat(2, 1),
        values=node_mask,
        size=(N, N),
        dtype=torch.float32
    )
    masked_adj = _data.adj_t.to_torch_sparse_coo_tensor()
    masked_adj = torch.sparse.mm(train_mask_diag, torch.sparse.mm(masked_adj, train_mask_diag))
    masked_adj = SparseTensor.from_torch_sparse_coo_tensor(masked_adj)

    row, col, val = masked_adj.coo()
    edge_mk = val.to(torch.bool)
    _row, _col = row[edge_mk], col[edge_mk]

    out = torch.zeros(_row.size(0), device=_row.device)
    out[_data.y[_row] == _data.y[_col]] = 1.
    out = scatter(out, _col, 0, dim_size=_data.y.size(0), reduce='mean')
    return out


def split_hop_dataset(in_train_mask, in_val_mask, in_test_mask,
                      all_wrong, all_correct,
                      num_runs, mask_train: List[str], val_ratio=0.1,
                      overfit_masks=None, mask_tr_het_ratio=-1):
    """Dataset Split"""

    num_masks = in_train_mask.size(0)
    num_nodes = in_train_mask.size(1)
    out_test_mask = in_test_mask

    # training on validation set has similar best_t but larger gap
    # between best_v and best_t. Works well on larger dataset
    if val_ratio <= 0:
        out_train_mask = in_val_mask
        out_val_mask = in_train_mask
    elif val_ratio >= 1:
        out_train_mask = in_train_mask
        out_val_mask = in_val_mask
    else:
        num_val = int(in_val_mask[0].sum() * val_ratio)
        out_train_mask = torch.stack(
            [index_to_mask(mask_to_index(in_val_mask[i])[num_val:], size=num_nodes) for i in
             range(num_masks)], dim=0)
        out_val_mask = in_val_mask & ~out_train_mask

        # mask heterophily region in GNN train
        if mask_tr_het_ratio == 0:
            out_train_mask = out_train_mask | in_train_mask
        elif 0 < mask_tr_het_ratio <= 1:
            out_train_mask = out_train_mask | (in_train_mask & ~overfit_masks)

    if num_masks == 1:
        out_train_mask = out_train_mask.tile(num_runs, 1)
        out_val_mask = out_val_mask.tile(num_runs, 1)
        out_test_mask = out_test_mask.tile(num_runs, 1)


    if mask_train is not None:
        if 'all_wrong' in mask_train:
            out_train_mask = out_train_mask & ~all_wrong
        if 'all_correct' in mask_train:
            out_train_mask = out_train_mask & ~all_correct

    return out_train_mask, out_val_mask, out_test_mask


class MoscatGating(nn.Module):
    def __init__(self, config, xs_dims, num_classes):
        super(MoscatGating, self).__init__()

        # encoder
        input_dim = config.decoder.hidden_dim
        mlp_conf = {
            'hidden_channels': input_dim,
            'out_channels': input_dim,
            'num_layers': 1,
            'dropout': 0,
            'norm': config.decoder.norm,
            'keep_last_act': True,
        }
        self.enc_residual = config.encoder.get('residual', True)
        self.encoders = nn.ModuleList()
        for x_dim in xs_dims:
            self.encoders.append(BaseMLP(x_dim, **mlp_conf))
        self.W = nn.Linear(len(self.encoders) * input_dim, input_dim)

        self.att_mixer = AttentionChannelMixing(xs_dims, input_dim, layer_norm=True)

        if config.decoder.norm == 'layer':
            self.feat_norm = nn.LayerNorm(input_dim, eps=1e-9)
        if config.decoder.norm == 'batch':
            self.feat_norm = nn.BatchNorm1d(input_dim)

        decoder_conf = Dict(OmegaConf.to_container(config.decoder))
        self.decoder = get_model(decoder_conf, input_dim, input_dim)
        self.classifier = nn.Linear(input_dim, num_classes)

    def forward(self, xs, edge_index=None):
        x, gating_weights = self.att_mixer(xs)

        # decoding
        if hasattr(self, 'feat_norm'):
            x = self.feat_norm(x)
        if edge_index is None:
            x = self.decoder(x)
        else:
            x = self.decoder(x, edge_index)

        # classification
        x = self.classifier(x)
        return x, gating_weights


def train_adaptive_mixer(sampler, optimizer, lr_scheduler,
                         xs, y, split_masks):
    sampler.train()
    optimizer.zero_grad()
    train_mask, val_mask, test_mask = split_masks

    prob_hop, _ = sampler(xs)

    loss = loss_fn(prob_hop[train_mask], y[train_mask])
    loss.backward()
    optimizer.step()
    if lr_scheduler is not None:
        lr_scheduler.step()


@torch.no_grad()
def test(sampler, xs, y, metric, masks):
    sampler.eval()
    out, gating_weights = sampler(xs)

    _accs, losses = [], []
    pred, y = pred_fn(out, y)
    for mask in masks:
        metric.reset()
        metric(pred[mask], y[mask])
        _accs.append(metric.compute())
        loss = loss_fn(out[mask], y[mask])
        losses.append(loss)

    return _accs, losses, out, gating_weights


def extract_conv_layers(filename):
    match = re.search(r'conv(\d+)', filename)
    if match:
        return int(match.group(1))
    else:
        return 2


@hydra.main(version_base=None, config_path='conf', config_name='config')
def main(conf):
    logger = Logger(conf=conf)
    data_conf = conf.dataset
    expert_confs = conf.expert
    train_conf = conf.train
    mixer_conf = conf.moscat

    conf_str = f'Moscat training config\n\n'
    for i, expert_path in enumerate(expert_confs):
        conf_str += (f'GNN Expert {i}:'
                     # f'\tarch: {expert_conf.arch}\n'
                     )
        conf_str += f'   {expert_path}\n'
        # if expert_conf.get('domain', ''):
        #     conf_str += f'\tdomain: {expert_conf.domain}\n'
        # if expert_conf.get('depths', ''):
        #     conf_str += f'\tdepths: {expert_conf.depths}\n'
    conf_str += (f'\nBasic Params: \n'
                 f'\tlog_epoch: {train_conf.log_epoch}\n'
                 f'\tearly_stopping: {train_conf.early_stopping}\n'
                 f'\tmixer_lr: {train_conf.mixer_lr}\n'
                 f'\thidden_dim: {mixer_conf.decoder.hidden_dim}\n'
                 f'\tnorm: {mixer_conf.decoder.norm}\n'
                 f'Moscat Params: \n'
                 f'\tval_ratio: {mixer_conf.val_ratio}\n'
                 f'\tmask_tr_het_ratio: {mixer_conf.get("mask_tr_het_ratio", -1)}\n'
                 f'\tmask_train: {mixer_conf.mask_train}\n')
    logger.critical(conf_str)

    proc_dir = f'{conf.proc_dir}/trial_{conf.trial_dir}'
    dataset_dir = osp.join(conf.data_dir, 'pyg')
    """
    curve_dir = osp.join(proc_dir, 'mixer_curve')
    logit_dir = osp.join(proc_dir, 'mixer_logit')
    attn_dir = osp.join(proc_dir, 'mixer_attn')
    os.makedirs(curve_dir, exist_ok=True)
    os.makedirs(logit_dir, exist_ok=True)
    os.makedirs(attn_dir, exist_ok=True)    
    """

    ## load data
    data, num_features, num_classes, dataset_dir = get_data(root=dataset_dir, **data_conf)
    metric = get_metric(data_conf.name, num_classes)
    dataset = data_conf.name

    ## load logits
    # expert_logits, expert_hops = [], []
    # for expert_conf in expert_confs:
    #     assert expert_conf.arch is not None
    #     # parse domain
    #     domain = expert_conf.get('domain', '')
    #     if domain:  # e.g., group-frac-numExperts_expertID
    #         expert_id = domain.split('_')[-1]
    #         domain_args = domain[: -len(expert_id) - 1].split('-')
    #         domain = f'-group-{domain_args[0]}-frac-{domain_args[1]}-eid-{expert_id}-{domain_args[2]}'
    #     # parse hops
    #     depths = expert_conf.get('depths', '2')
    #     if '-' in depths:
    #         depths = depths.split('-')
    #         depths = list(range(int(depths[0]), int(depths[1])+1))
    #     elif ',' in depths:
    #         depths = depths.split(',')
    #     else:
    #         depths = [depths]
    #     # load logits
    #     for depth in depths:
    #         if expert_conf.arch.split('-')[0] == 'MLP':
    #             expert_hops.append(0)
    #             depth = 3
    #         else:
    #             expert_hops.append(int(depth))
    #         expert_file = f'{proc_dir}/logit/{dataset}{domain}_{expert_conf.arch}-conv{depth}.pt'
    #         expert_logits.append(torch.load(expert_file, weights_only=True).detach().cpu())

    expert_logits, expert_hops = [], []
    for expert_path in expert_confs:
        expert_file = f'{proc_dir}/logit/{expert_path}'
        expert_logits.append(torch.load(expert_file, weights_only=True).detach().cpu())

        if 'MLP' in expert_path:
            expert_hops.append(0)
        else:
            expert_hops.append(extract_conv_layers(expert_path))

    expert_logits = torch.stack(expert_logits, dim=1)  # (runs, experts, nodes, logits)

    logger.critical(f'Runs: {expert_logits.shape[0]}, Num Experts: {expert_logits.shape[1]}\n')

    ## calculate accs
    if expert_logits.shape[-1] == 1:
        accs = (expert_logits > 0).squeeze(-1) == data.y
    else:
        accs = expert_logits.argmax(dim=-1) == data.y
    accs = accs.float()

    ## get split mask per run
    train_mask = index_to_mask(data.train_mask, size=data.num_nodes).t()
    val_mask = index_to_mask(data.val_mask, size=data.num_nodes).t()
    test_mask = index_to_mask(data.test_mask, size=data.num_nodes).t()
    if train_mask.dim() == 1:
        train_mask = train_mask.unsqueeze(0)
        val_mask = val_mask.unsqueeze(0)
        test_mask = test_mask.unsqueeze(0)
    NUM_RUNS = accs.shape[0]

    ## get train masking
    dist_masks = [check_dist(accs[i], dim=0, eps=0.04) for i in range(NUM_RUNS)]
    all_correct = torch.stack([masks[0] for masks in dist_masks], dim=0).to(conf.gpu)
    all_wrong = torch.stack([masks[1] for masks in dist_masks], dim=0).to(conf.gpu)

    ## structural encoding
    max_hop = 6
    disparity_dict = {
        'pagerank': _get_pagerank(data.adj_t.t()).view(-1, 1),
        'LSI': _get_LSI(data, max_hop=max_hop),
        'LSI2': _get_LSI2(data, max_hop=max_hop),
    }
    disparity_dict = {k: (v - v.mean(dim=0, keepdim=True)) / v.std(dim=0, keepdim=True)
                      for k, v in disparity_dict.items()}
    X = torch.cat([
        disparity_dict['pagerank'],
        disparity_dict['LSI'],
        disparity_dict['LSI2'],
    ], dim=1).to(conf.gpu)

    data.to(conf.gpu)
    disparity_dict = {k: v.to(conf.gpu) for k, v in disparity_dict.items()}
    X = X.to(conf.gpu)
    accs = accs.to(conf.gpu)
    expert_logits = expert_logits.to(conf.gpu)
    train_mask = train_mask.to(conf.gpu)
    val_mask = val_mask.to(conf.gpu)
    test_mask = test_mask.to(conf.gpu)
    metric.to(conf.gpu)

    # set up seed for preprocessing
    setup_seed(0)

    # By default, do not use experts' training set
    mask_tr_het_ratio, overfit_masks = mixer_conf.get('mask_tr_het_ratio', -1), []
    if 0 < mask_tr_het_ratio <= 1:
        for r in range(train_mask.shape[0]):
            selected_nodes = train_mask[r] | val_mask[r]
            homo = get_filtered_homophily(data, selected_nodes)

            train_homo = homo[train_mask[r]].mean()
            idx1 = mask_to_index(homo < train_homo)

            idx2 = idx1[torch.randperm(idx1.shape[0])[:int(idx1.shape[0] * mask_tr_het_ratio)]]
            overfit_masks.append(index_to_mask(idx2, homo.shape[0]))
        overfit_masks = torch.stack(overfit_masks, dim=0)

    ## training
    origin_val_mask = val_mask.clone()
    train_mask, val_mask, test_mask = split_hop_dataset(train_mask, val_mask, test_mask,
                                                        all_wrong, all_correct,
                                                        num_runs=NUM_RUNS,
                                                        mask_train=mixer_conf.mask_train,
                                                        val_ratio=mixer_conf.val_ratio,
                                                        overfit_masks=overfit_masks,
                                                        mask_tr_het_ratio=mask_tr_het_ratio
                                                        )

    total_train_time = 0.
    val_losses = []
    best_tests_cls, best_tests_hop = [], []
    total_logit, total_attn = [], []
    for i in range(NUM_RUNS):

        logger.info(f'------------------------Run {i}------------------------')
        setup_seed(i + 1)

        feat_type, xs = mixer_conf.encoder.get('feat_type', ['logit_2']), []
        if 'disparity' in feat_type:
            xs.append(X)
        if 'node_feat' in feat_type:
            xs.append(data.x)
        if 'logit_aug' in feat_type:  # Scope-aware Logit Augmentation
            lmax = mixer_conf.encoder.get('lmax', 6)
            deg_norm = mixer_conf.encoder.get('deg_norm', 'sym')
            adj_t = adj_norm(data.adj_t, norm=deg_norm, add_self_loop=False)

            for expert_idx, hop in enumerate(expert_hops):  # load expert logits
                _xs = [expert_logits[i][expert_idx]]

                # pseudo label embedding
                for _l in range(1, lmax + 1):
                    _xs.append(adj_t @ _xs[-1])

                # structural encoding
                if mixer_conf.encoder.get('pagerank', True):
                    _xs.append(disparity_dict['pagerank'])
                _xs.append(disparity_dict['LSI'][:, hop].unsqueeze(-1))
                _xs.append(disparity_dict['LSI2'][:, hop].unsqueeze(-1))

                xs.append(torch.cat(_xs, dim=1))

        xs_dims = [x.size(1) for x in xs]
        mixer = MoscatGating(mixer_conf, xs_dims, num_classes).to(conf.gpu)

        optimizer = torch.optim.Adam(mixer.parameters(), lr=train_conf.mixer_lr,
                                     weight_decay=train_conf.weight_decay)
        lr_scheduler = None

        best_s, best_ts = defaultdict(float), defaultdict(float)
        for epoch in range(1, train_conf.mixer_epoch + 1):
            tik = time.time()

            # adjust_learning_rate(optimizer, train_conf.lr, epoch)

            r = train_mask.size(0)
            train_adaptive_mixer(
                mixer, optimizer, lr_scheduler,
                xs, data.y,
                split_masks=[train_mask[i % r], val_mask[i % r], test_mask[i % r]]
               )

            tok = time.time()
            total_train_time += tok - tik

            (train_acc, val_acc, test_acc), (train_loss, val_loss, test_loss), logit, gating_weights = test(
                mixer, xs, data.y, metric, [train_mask[i % r], val_mask[i % r], test_mask[i % r]])
            val_losses.append(val_loss)
            logger.info(f'Epoch {epoch:03d}, Train: {train_acc: .4f}, '
                    f'Val: {val_acc: .4f}, Test: {test_acc: .4f}\n')

            if epoch > train_conf.get('log_epoch', 50):
                if val_acc > best_s['mixer_val_acc']:
                    best_s = {
                        'epoch': epoch,
                        'mixer_train_acc': train_acc,
                        'mixer_val_acc': val_acc,
                        'mixer_test_acc': test_acc,
                        'logit': logit,
                        'gating_weights': gating_weights
                    }

                if 0 < train_conf.early_stopping < epoch:
                    tmp = torch.tensor(val_losses[-(train_conf.early_stopping + 1): -1])
                    if val_loss > tmp.mean().item():
                        break

        logger.info(f"[Best Result] "
              f"Epoch: {int(best_s['epoch']):03d}, Train: {best_s['mixer_train_acc']:.4f}, "
              f"Val: {best_s['mixer_val_acc']:.4f}, Test: {best_s['mixer_test_acc']:.4f}")
        best_tests_cls.append(best_s['mixer_test_acc'])
        total_logit.append(best_s['logit'].cpu())
        total_attn.append(best_s['gating_weights'].cpu())

    best_tests_cls = torch.tensor(best_tests_cls)
    best_acc, best_std = get_best_expert_acc(accs, test_mask, expert_logits, data.y, _log=True)
    result_str = (f'Baseline Test: {best_acc*100:.2f} ±{best_std*100:.2f}\n'
                  f'Moscat Test: {best_tests_cls.mean()*100:.2f} ±{best_tests_cls.std()*100:.2f}')
    logger.critical(result_str)

    """Evaluate Ensemble and Upper Bound"""
    # # ensemble accuracy
    # best_ensemble_val_acc = 0.0
    # best_ensemble_test_acc = 0.0
    # best_alpha = -1
    # for alpha in np.arange(0., 1.1, 0.1):
    #     val_acc, test_acc = [], []
    #     for run in range(NUM_RUNS):
    #         logit = expert_logits[run][0] * alpha + expert_logits[run][1] * (1-alpha)
    #         y_hat, y = pred_fn(logit, data.y)
    #         all_acc = y_hat == y
    #
    #         val_acc.append(all_acc[val_mask[run%NUM_RUNS]].float().mean().item())
    #         test_acc.append(all_acc[test_mask[run%NUM_RUNS]].float().mean().item())
    #     if np.mean(val_acc) > best_ensemble_val_acc:
    #         best_ensemble_val_acc = np.mean(val_acc)
    #         best_enesmble_test_acc = np.mean(test_acc)
    #         acc_std = np.std(test_acc)
    #         best_alpha = alpha
    #
    # alpha = 0.5
    # mean_ensamble_val_acc = []
    # mean_ensamble_test_acc = []
    # for run in range(NUM_RUNS):
    #     logit = expert_logits[run][0] * alpha + expert_logits[run][1] * (1-alpha)
    #     y_hat, y = pred_fn(logit, data.y)
    #     all_acc = y_hat == y
    #
    #     mean_ensamble_val_acc.append(all_acc[val_mask[run%NUM_RUNS]].float().mean().item())
    #     mean_ensamble_test_acc.append(all_acc[test_mask[run%NUM_RUNS]].float().mean().item())
    #
    # best_enesmble_test_acc = max(np.mean(mean_ensamble_test_acc), best_enesmble_test_acc)

    # ensemble accuracy
    """
    w_best, v_mean, v_std, t_mean, t_std = equal_grid_search(
        expert_logits, data.y, origin_val_mask, test_mask, step=0.1
    )
    logger.critical(f"[Grid Search] best weights: {w_best.round(3)}")
    ens_result_str = f"Grid Ensemble Test: {t_mean * 100:.2f} ±{t_std * 100:.2f}"
    logger.critical(ens_result_str)
    """

    w_best, v_mean, v_std, t_mean, t_std = bayes_search(
        expert_logits, data.y, origin_val_mask, test_mask,
        n_trials=600, timeout=None
    )
    logger.critical(f"[Bayesian Opt] best weights: {w_best.round(3)}")
    ens_result_str = f"Bayesian Ensemble Test: {t_mean * 100:.2f} ±{t_std * 100:.2f}"
    logger.critical(ens_result_str)

    # upper bound accuracy
    test_acc = []
    for run in range(NUM_RUNS):
        all_acc = torch.zeros_like(data.y, dtype=torch.bool)
        for expert_id in range(expert_logits.shape[1]):
            logit = expert_logits[run][expert_id]
            y_hat, y = pred_fn(logit, data.y)
            all_acc |= (y_hat == y)
        test_acc.append(all_acc[test_mask[run % test_mask.shape[0]]].float().mean().item())
    upb_result_str = f"Upper Bound Test: {np.mean(test_acc)*100:.2f} ± {np.std(test_acc)*100:.2f}"
    logger.critical(upb_result_str)

    # compute error inconsistency
    if expert_logits.shape[-1] == 1:
        expert_accs = (expert_logits > 0).squeeze(-1) == data.y
    else:
        expert_accs = expert_logits.argmax(dim=-1) == data.y
    expert_accs = expert_accs.float()

    all_correct = expert_accs.mean(dim=1) == 1
    all_wrong = expert_accs.mean(dim=1) == 0
    err_inconsis_mask = ~(all_wrong | all_correct)

    all_eis = []
    for run in range(NUM_RUNS):
        err_inc = err_inconsis_mask[run][test_mask[run % test_mask.shape[0]]].float().mean().item()
        all_eis.append(err_inc)
    ei_result_str = f"Error Inconsistency: {np.mean(all_eis) * 100:.2f} ±{np.std(all_eis) * 100:.2f}"
    logger.critical(ei_result_str)

    all_result = {
        'ei': np.mean(all_eis),
        'best_acc': best_acc,
        'moscat_acc': best_tests_cls.mean().item(),
        'ens_acc': t_mean,
        'upb_acc': np.mean(test_acc),
    }

    # save 
    if conf.get('save_all_result', False):
        assert conf.get('save_all_result_path', None) is not None, \
            'Please specify the save_all_result_path in the config file.'
        assert conf.get('saving_prefix', None) != '', \
            'Please specify the saving_prefix in the config file.'

        result_path = conf.save_all_result_path

        # Load existing JSON if it exists
        if osp.exists(result_path):
            with open(result_path, 'r') as f:
                loaded_dict = json.load(f)
        else:
            logger.warning(f'No existing result file found at {result_path}. Creating a new one.')
            loaded_dict = {}

        # Add or update the key with the new result
        loaded_dict[conf.saving_prefix] = all_result

        # Save updated dictionary
        with open(result_path, 'w') as f:
            json.dump(loaded_dict, f, indent=2)

    if conf.get('log_time', False):
        logger.critical(f'Total Train Time: {total_train_time/NUM_RUNS:.2f}s')

    """Log"""
    logger.save_result(result_str)
    logger.save_config()

    """
    if conf.get('log_curve', False):
        filename = '{}_{}{}_mix{}'.format(conf.dataset.name, expert_conf.model, expert_conf.arch, mixer_conf.max_hop)
        filename += '.pt'
        torch.save(best_tests_cls, osp.join(curve_dir, filename))
    if conf.get('log_logit', False):
        filename = '{}_{}{}'.format(conf.dataset.name, expert_conf.model, expert_conf.arch)
        filename += '.pt'
        torch.save(torch.stack(total_logit, dim=0), osp.join(logit_dir, filename))
    if conf.get('log_attn', False):
        filename = '{}_{}{}_{}-{}'.format(conf.dataset.name, expert_conf.model, expert_conf.arch, mixer_conf.min_hop, mixer_conf.max_hop)
        filename += '.pt'
        torch.save(torch.stack(total_attn, dim=0), osp.join(attn_dir, filename))
    """


if __name__ == '__main__':
    main()
