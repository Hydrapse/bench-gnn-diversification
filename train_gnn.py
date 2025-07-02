import logging
import os
import time
import warnings
from collections import OrderedDict

import optuna
from optuna.trial import TrialState

from utils.atp_adj_norm import calculate_adj_exponent

warnings.filterwarnings("ignore", category=FutureWarning)

import numpy as np
import torch
from tqdm import tqdm

from data import get_data, get_expert_data, get_metric
from components.backbone import get_model
from disparity import get_degree
from utils.training import adjust_learning_rate, add_labels
from utils.utils import setup_seed, loss_fn, pred_fn, Dict
from components.loader import get_loader
from utils.utils import mask_to_index
from logger import Logger

import os.path as osp
from urllib.parse import urlparse

from omegaconf import OmegaConf
import hydra
from hydra.core.hydra_config import HydraConfig

warnings.filterwarnings("ignore")


def train(model, optimizer, data, train_mask, grad_norm=None, use_label=False, mask_rate=1.):
    model.train()
    idx = mask_to_index(train_mask)
    mask = torch.rand(idx.shape) < mask_rate
    train_idx = idx[mask]
    if use_label:
        x = add_labels(data.x, data.y, idx[~mask])
    else:
        x = data.x

    optimizer.zero_grad()
    out = model(x, data.adj_t)

    # Label Reuse
    # n_classes = (data.y.max() + 1).item()
    # unlabel_idx = torch.cat([train_idx, mask_to_index(data.val_mask), mask_to_index(data.test_mask)])
    # x[unlabel_idx, -n_classes:] = F.softmax(out[unlabel_idx], dim=-1).detach()
    # out = model(x, data.adj_t)

    loss = loss_fn(out[train_idx], data.y[train_idx])
    if hasattr(model, 'load_balance_loss'):
        loss += model.load_balance_loss
    loss.backward()
    if grad_norm is not None:
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_norm)
    optimizer.step()
    return loss


def mini_train(model, optimizer, loader, grad_norm=None):
    model.train()
    total_loss = 0
    for batch in tqdm(loader):
        optimizer.zero_grad()
        y_hat = model(batch.x, batch.adj_t)
        loss = loss_fn(y_hat, batch.y)
        loss.backward()
        if grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_norm)
        optimizer.step()
        total_loss += loss.item() / batch.batch_size
    return total_loss


@torch.no_grad()
def test(model, metric, data, masks, use_label=False):
    model.eval()

    if use_label:
        x = add_labels(data.x, data.y, data.train_mask)
    else:
        x = data.x

    out = model(x, data.adj_t)

    # out, hop_weights = model(x, data.adj_t)
    # torch.save(hop_weights, '/home/gangda/workspace/adapt-hop/processed/hop_weights/amazon_ratings_gamlp.pt')

    # Label Reuse
    # n_classes = (data.y.max() + 1).item()
    # unlabel_idx = torch.cat([mask_to_index(data.val_mask), mask_to_index(data.test_mask)])
    # x[unlabel_idx, -n_classes:] = F.softmax(out[unlabel_idx], dim=-1).detach()
    # out = model(x, data.adj_t)

    accs, losses = [], []
    pred, y = pred_fn(out, data.y)
    for mask in masks:
        metric.reset()
        metric(pred[mask], y[mask])
        accs.append(metric.compute())

        loss = loss_fn(out[mask], data.y[mask])
        losses.append(loss)

    return accs, losses, out


@hydra.main(version_base=None, config_path='conf', config_name='config')
def main(conf):
    ## init logger
    logger = Logger(conf=conf)
    logger.critical(OmegaConf.to_yaml(conf))
    sampler_conf = conf.sampler
    train_conf = conf.train
    model_conf = conf.model
    data_conf = conf.dataset

    # skip optuna repetitive trials
    hydra_cfg = HydraConfig.get()
    study_name = hydra_cfg.get("sweeper", {}).get("study_name", None)
    storage = hydra_cfg.get("sweeper", {}).get("storage", None)
    if study_name is not None and storage is not None:
        # study = optuna.load_study(study_name=study_name, storage=storage)
        if storage.startswith("sqlite:///"):
            db_path = urlparse(storage).path.lstrip('/')
            if not os.path.exists(db_path):
                logger.warning(f"Optuna SQLite file does not exist: {db_path}\nCreating a new study.")
        study = optuna.create_study(study_name=study_name, storage=storage, direction="maximize", load_if_exists=True)
        if len(study.trials) > 0:
            current_trial = study.trials[-1]
            for previous_trial in study.trials:
                if previous_trial.state == TrialState.COMPLETE and current_trial.params == previous_trial.params:
                    print(f"\nDuplicated trial: {current_trial.params}, return {previous_trial.value}\n")
                    return previous_trial.value

    
    ## working dir
    dataset_dir = osp.join(conf.data_dir, 'pyg')
    os.makedirs(dataset_dir, exist_ok=True)
    os.makedirs(conf.ckpt_dir, exist_ok=True)
    ckpt_path = osp.join(conf.ckpt_dir, '{}_{}_{}_{}.tar'.format(
        model_conf.name, model_conf.conv_layers, os.getpid(), int(time.time())))

    ## fixed dataset split
    seed = conf.get('seed', 0)
    setup_seed(seed)

    ## indicator for expert training
    train_expert = hasattr(data_conf, 'expert_id') and data_conf.expert_id >= 0

    ## indicator of group eval
    eval_by_group = data_conf.get('eval_by_group', False) or train_expert or data_conf.get('eval_group', [])

    ## validate params
    if train_expert or eval_by_group:
        assert ((hasattr(data_conf, 'group') or data_conf.get('eval_group', [])) and
                hasattr(data_conf, 'fraction') and
                hasattr(data_conf, 'num_experts') and
                data_conf.num_experts > 1)

    ## dataset
    if train_expert:
        logger.info('Start Expert Training!')
        data, num_features, num_classes, _ = get_expert_data(root=dataset_dir, **data_conf)
    else:
        data, num_features, num_classes, _ = get_data(root=dataset_dir, **data_conf)
    metric = get_metric(data_conf.name, num_classes)
    if train_conf.use_label:
        num_features = num_features + num_classes

    ## full-batch gpu training by default
    device = torch.device('cuda:{}'.format(conf.gpu) if torch.cuda.is_available() else 'cpu')

    ## update model conf
    model_conf = Dict(OmegaConf.to_container(model_conf))
    if model_conf.name.upper() == 'PNA':
        d = get_degree(data.adj_t, undirected=True)
        deg = torch.bincount(d)
        model_conf['deg'] = deg
    if model_conf.name.upper() == 'ACMGCN':
        model_conf['num_nodes'] = data.num_nodes
    if model_conf.name.upper() == 'POLYNORMER':
        if not hasattr(model_conf, 'global_dropout') or model_conf.global_dropout is None:
            model_conf['global_dropout'] = model_conf['dropout']
        if not hasattr(model_conf, 'init_dropout') or model_conf.init_dropout is None:
            model_conf['global_dropout'] = model_conf['dropout']
    if 'ATP' in model_conf.name.upper():
        heuristics_dir = osp.join(conf.proc_dir, 'heuristics')
        os.makedirs(heuristics_dir, exist_ok=True)
        r = calculate_adj_exponent(data, heuristics_dir,
                                   a=model_conf.a, b=model_conf.b, c=model_conf.c)
        model_conf['r'] = r.to(device)

    metric.to(device)
    data.to(device)

    ## log total train time (per run)
    total_train_time = 0.

    ## eval trained model on different subset
    expert_test_accs = OrderedDict()
    if eval_by_group:
        eval_groups = data_conf.get('eval_group', [])
        if len(eval_groups) == 0:
            eval_groups.append(data_conf.group)
        for group in eval_groups:
            expert_test_accs[group] = {i: [] for i in range(data_conf.num_experts)}

    ## plot convergence and generalization curve
    curve_dict = {}
    if conf.get('log_curve', False):
        for metric in ['train_loss', 'test_loss', 'train_acc', 'test_acc']:
            curve_dict[metric] = []
        conf.runs = 1  # only collect training curve results from 1 run

    best_val, best_test, runs_logit = [], [], []
    for i in range(1, conf.runs + 1):

        logger.info(f'------------------------Run {i}------------------------')
        seed = conf.get('seed', 0)
        setup_seed(seed + i)

        if train_expert:
            if data.expert_train_masks.dim() > 1 and data.expert_train_masks.size(1) > 1:
                train_mask = data.expert_train_masks[:,
                             (i - 1) % data.expert_train_masks.shape[-1]].squeeze()
            else:
                train_mask = data.expert_train_masks.squeeze()
            if data.expert_val_masks.dim() > 1 and data.expert_val_masks.size(1) > 1:
                val_mask = data.expert_val_masks[:, i - 1]
            else:
                val_mask = data.expert_val_masks.squeeze()
            if data.expert_test_masks.dim() > 1 and data.expert_test_masks.size(1) > 1:
                test_mask = data.expert_test_masks[:, i - 1]
            else:
                test_mask = data.expert_test_masks.squeeze()
        else:
            if len(data.train_mask.shape) > 1:
                train_mask = data.train_mask[:, i - 1]
            else:
                train_mask = data.train_mask
            if len(data.val_mask.shape) > 1:
                val_mask = data.val_mask[:, i - 1]
            else:
                val_mask = data.val_mask
            if len(data.test_mask.shape) > 1:
                test_mask = data.test_mask[:, i - 1]
            else:
                test_mask = data.test_mask

        is_mini = sampler_conf.name is not None
        train_loader = get_loader(sampler_conf, data, train_mask) if is_mini else None

        model = get_model(model_conf, num_features, num_classes).to(device)

        if model_conf.name.upper() in ['APPNP', 'GPRGNN']:
            optimizer = torch.optim.Adam([{
                'params': model.lin1.parameters(),
                'weight_decay': train_conf.weight_decay, 'lr': train_conf.lr
            }, {
                'params': model.lin2.parameters(),
                'weight_decay': train_conf.weight_decay, 'lr': train_conf.lr
            }, {
                'params': model.prop1.parameters(),
                'weight_decay': 0.0, 'lr': train_conf.lr
            }], lr=train_conf.lr)
        else:
            if train_conf.use_adamw:
                optimizer = torch.optim.AdamW(
                    model.parameters(), lr=train_conf.lr, weight_decay=train_conf.weight_decay
                )
            else:
                optimizer = torch.optim.Adam(
                    model.parameters(), lr=train_conf.lr, weight_decay=train_conf.weight_decay
                )

        ## for polynormer
        if hasattr(train_conf, 'local_epochs') and hasattr(train_conf, 'global_epochs'):
            train_conf.epoch = train_conf.local_epochs + train_conf.global_epochs

        best_val_acc, val_loss_history = 0, []
        for epoch in range(1, train_conf.epoch + 1):

            ## for polynormer
            if hasattr(train_conf, 'local_epochs') and epoch == train_conf.local_epochs + 1:
                print("start global attention!!!!!!")
                ckpt = torch.load(ckpt_path, weights_only=False)
                optimizer.load_state_dict(ckpt['optimizer_state_dict'])
                model.load_state_dict(ckpt['model_state_dict'])
                model._global = True

            ## increase lr as epoch grows, can improve convergence in most cases
            if model_conf.name.upper() not in ['POLYNORMER']:
                adjust_learning_rate(optimizer, train_conf.lr, epoch)

            tik = time.time()

            if is_mini:
                mini_train(model, optimizer, train_loader, train_conf.grad_norm)
            else:
                train(model, optimizer, data, train_mask,
                      grad_norm=train_conf.grad_norm,
                      use_label=train_conf.use_label,
                      mask_rate=train_conf.train_mask_rate)

            tok = time.time()
            total_train_time += tok - tik

            (train_acc, val_acc, test_acc), (train_loss, val_loss, test_loss), _ = test(
                model, metric, data, [train_mask, val_mask, test_mask],
                use_label=train_conf.use_label)

            if i == 1 and conf.get('log_curve', False):
                curve_dict['train_acc'].append(train_acc)
                curve_dict['test_acc'].append(test_acc)
                curve_dict['train_loss'].append(train_loss)
                curve_dict['test_loss'].append(test_loss)

            if hasattr(conf, 'use_true_epoch') and conf.use_true_epoch:
                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                if epoch == train_conf.epoch:
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'train_acc': train_acc,
                        'val_acc': val_acc,
                    }, ckpt_path)
            else:
                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'train_acc': train_acc,
                        'val_acc': val_acc,
                    }, ckpt_path)

            # logger.info(
            #     f'Epoch: {epoch:03d}, Train Loss: {train_loss: .4f}, Val Loss: {val_loss: .4f} '
            #     f'Train Acc: {train_acc:.4f}, Val Acc: {best_val_acc:.4f}')
            logger.info(
                f'Epoch: {epoch:03d}, Train Loss: {train_loss: .4f}, Best Val Acc: {best_val_acc: .4f} '
                f'Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}, Test Acc: {test_acc:.4f}')

            if epoch >= 0:
                val_loss_history.append(val_loss)
                if 0 < train_conf.early_stopping < epoch:
                    tmp = torch.tensor(val_loss_history[-(train_conf.early_stopping + 1): -1])
                    if val_loss > tmp.mean().item():
                        break
        ckpt = torch.load(ckpt_path, weights_only=False)
        model = get_model(model_conf, num_features, num_classes).to(device)
        model.load_state_dict(ckpt['model_state_dict'])
        (test_acc,), (test_loss,), logit = test(model, metric, data, [test_mask],
                                                use_label=train_conf.use_label)
        logger.info(f"[Best Model] Epoch: {ckpt['epoch']:02d}, Train: {ckpt['train_acc']:.4f}, "
                    f"Val: {ckpt['val_acc']:.4f}, Test: {test_acc:.4f}")
        best_val.append(float(best_val_acc))
        best_test.append(float(test_acc))
        runs_logit.append(logit.cpu())

        if eval_by_group:
            _data_conf = Dict(OmegaConf.to_container(data_conf))
            expert_test_masks = []
            for group in expert_test_accs.keys():
                _data_conf.group = group
                for k in range(data_conf.num_experts):
                    _data_conf.expert_id = k
                    expert_data, _, _, _ = get_expert_data(root=dataset_dir, **_data_conf)

                    if expert_data.expert_test_masks.dim() > 1 and expert_data.expert_test_masks.size(1) > 1:
                        expert_test_masks.append(expert_data.expert_test_masks[:, i - 1].to(device))
                    else:
                        expert_test_masks.append(expert_data.expert_test_masks.squeeze().to(device))

            test_accs, test_losses, _ = test(model, metric, data, expert_test_masks,
                    use_label=train_conf.use_label)

            for g_id, group in enumerate(expert_test_accs.keys()):
                for k in range(data_conf.num_experts):
                    test_acc = test_accs[g_id * data_conf.num_experts + k]
                    logger.info(f'{group} expert {k} --- Test Acc: {test_acc:.4f}')
                    expert_test_accs[group][k].append(test_acc.cpu().item())

    result_str = ''
    if train_expert:
        result_str += f' Expert: {data_conf.expert_id + 1}/{data_conf.num_experts}'
    if hasattr(conf, 'log_true_epoch') and conf.log_true_epoch:
        result_str += f' Epoch: {ckpt["epoch"]:02d}'
    result_str += (f'\nValid: {np.mean(best_val) * 100:.2f} ±{np.std(best_val) * 100:.2f}, '
                   f'Test: {np.mean(best_test) * 100:.2f} ±{np.std(best_test) * 100:.2f}\n')
    if eval_by_group:
        for group in expert_test_accs.keys():
            for k in range(data_conf.num_experts):
                group_test = np.array(expert_test_accs[group][k])
                result_str += (f'{group} expert {k} --- '
                               f'Test: {np.mean(group_test) * 100:.2f} ±{np.std(group_test) * 100:.2f}\n')

    logger.critical(result_str)
    if conf.get('log_time', False):
        logger.critical(f'Total Train Time: {total_train_time / conf.runs:.2f}s')

    """Log"""
    logger.save_result(result_str)
    logger.save_logit(torch.stack(runs_logit, dim=0))
    logger.save_curve(curve_dict)
    logger.save_config()

    if os.path.exists(ckpt_path):
        os.remove(ckpt_path)

    ## for optuna
    tuning_objective = np.mean(best_test)
    return float(tuning_objective)


if __name__ == '__main__':
    main()
