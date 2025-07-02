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

from omegaconf import OmegaConf
import hydra
from hydra.core.hydra_config import HydraConfig

warnings.filterwarnings("ignore")


def train(model, optimizer, data, train_mask, grad_norm=None):
    model.train()

    optimizer.zero_grad()
    out = model(data.x, data.adj_t)

    loss = loss_fn(out[train_mask], data.y[train_mask])
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
def test(model, metric, data, masks):
    model.eval()

    x = data.x
    out = model(x, data.adj_t)

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
    # skip optuna repetitive trials
    hydra_cfg = HydraConfig.get()
    study_name = hydra_cfg.get("sweeper", {}).get("study_name", None)
    storage = hydra_cfg.get("sweeper", {}).get("storage", None)
    if study_name is not None and storage is not None:
        study = optuna.load_study(study_name=study_name, storage=storage)
        current_trial = study.trials[-1]
        for previous_trial in study.trials:
            if previous_trial.state == TrialState.COMPLETE and current_trial.params == previous_trial.params:
                print(f"\nDuplicated trial: {current_trial.params}, return {previous_trial.value}\n")
                return previous_trial.value

    ## init logger
    logger = Logger(conf=conf)
    if hasattr(conf, 'skip_existing') and conf.skip_existing and osp.exists(osp.join(logger.logit_dir, logger.exp_name + '.pt')):
        logger.critical(f'Logit file {logger.exp_name}.pt already exists, skipping this experiment.')
        return 0.0
    logger.critical(OmegaConf.to_yaml(conf))
    train_conf = conf.train
    model_conf = conf.model
    data_conf = conf.dataset
    assert hasattr(train_conf, 'ft_lr')

    ## working dir
    dataset_dir = osp.join(conf.data_dir, 'pyg')
    os.makedirs(dataset_dir, exist_ok=True)
    os.makedirs(conf.ckpt_dir, exist_ok=True)
    ckpt_path = osp.join(conf.ckpt_dir, '{}_{}_{}_{}.tar'.format(
        model_conf.name, model_conf.conv_layers, os.getpid(), int(time.time())))

    ## fixed dataset split
    setup_seed(0)

    ## indicator for expert training
    train_expert = hasattr(data_conf, 'expert_id') and data_conf.expert_id >= 0
    assert train_expert
    assert ((hasattr(data_conf, 'group') or data_conf.get('eval_group', [])) and
            hasattr(data_conf, 'fraction') and
            hasattr(data_conf, 'num_experts') and
            data_conf.num_experts > 1)

    ## dataset
    data, num_features, num_classes, _ = get_data(root=dataset_dir, **data_conf)
    expert_data, _, _, _ = get_expert_data(root=dataset_dir, **data_conf)
    metric = get_metric(data_conf.name, num_classes)

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

    # load base GNN model if exists
    base_conf = OmegaConf.create(OmegaConf.to_container(conf, resolve=True, enum_to_str=True))
    if hasattr(base_conf.dataset, 'expert_id'):
        del base_conf.dataset.expert_id  # remove expert_id for base GNN
    base_gnn_logger = Logger(conf=base_conf)
    model_dir = osp.join(base_gnn_logger.log_dir, 'models')
    os.makedirs(model_dir, exist_ok=True)
    model_path = osp.join(model_dir, f'{base_gnn_logger.exp_name}.pt')
    if os.path.exists(model_path):
        logger.info(f'Loading base GNN model from {model_path}')
        all_models = torch.load(model_path, weights_only=False)
    else:
        logger.warning(f'Base GNN model not found at {model_path}, training from scratch.')
        all_models = {}
    metric.to(device)
    data.to(device)
    expert_data.to(device)

    ## log total train time (per run)
    total_train_time = 0.

    ## eval trained model on different subset
    expert_test_accs = OrderedDict()
    for group in [data_conf.group]:
        expert_test_accs[group] = {i: [] for i in range(data_conf.num_experts)}

    best_val, best_test, runs_logit = [], [], []
    for i in range(1, conf.runs + 1):

        logger.info(f'------------------------Run {i}------------------------')
        setup_seed(i)
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

        if os.path.exists(model_path):
            logger.info(f'Using base GNN model from {model_path}')
            model = get_model(model_conf, num_features, num_classes).to(device)
            model.load_state_dict(all_models[i]['model_state_dict'])
        else:
            logger.warning(f'Base GNN model not found at {model_path}, training from scratch.')
            
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

            best_val_acc, val_loss_history = 0, []
            for epoch in range(1, train_conf.epoch + 1):

                ## increase lr as epoch grows, can improve convergence in most cases
                adjust_learning_rate(optimizer, train_conf.lr, epoch)

                tik = time.time()

                train(model, optimizer, data, train_mask,
                    grad_norm=train_conf.grad_norm)

                tok = time.time()
                total_train_time += tok - tik

                (train_acc, val_acc, test_acc), (train_loss, val_loss, test_loss), _ = test(
                    model, metric, data, [train_mask, val_mask, test_mask])

                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'train_acc': train_acc,
                        'val_acc': val_acc,
                    }, ckpt_path)

                logger.info(
                    f'Epoch: {epoch:03d}, Train Loss: {train_loss: .4f}, Val Loss: {val_loss: .4f} '
                    f'Train Acc: {train_acc:.4f}, Val Acc: {best_val_acc:.4f}')
                # logger.info(
                #     f'Epoch: {epoch:03d}, Train Loss: {train_loss: .4f}, Best Val Acc: {best_val_acc: .4f} '
                #     f'Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}, Test Acc: {test_acc:.4f}')

                if epoch >= 0:
                    val_loss_history.append(val_loss)
                    if 0 < train_conf.early_stopping < epoch:
                        tmp = torch.tensor(val_loss_history[-(train_conf.early_stopping + 1): -1])
                        if val_loss > tmp.mean().item():
                            break

            ckpt = torch.load(ckpt_path, weights_only=False)
            if not os.path.exists(model_path):
                all_models[i] = {
                    'epoch': ckpt['epoch'],
                    'model_state_dict': ckpt['model_state_dict'],
                    'train_acc': ckpt['train_acc'],
                    'val_acc': ckpt['val_acc'],
                }
            model = get_model(model_conf, num_features, num_classes).to(device)
            model.load_state_dict(ckpt['model_state_dict'])
        (test_acc,), (test_loss,), logit = test(model, metric, data, [test_mask])
        logger.warning(f"[Best Model] Epoch: {all_models[i]['epoch']:02d}, Train: {all_models[i]['train_acc']:.4f}, "
                       f"Val: {all_models[i]['val_acc']:.4f}, Test: {test_acc:.4f}")
        # best_val.append(float(best_val_acc))
        # best_test.append(float(test_acc))
        # runs_logit.append(logit.cpu())


        ######  Finetune GNN on domain samples ######
        ft_epoch = train_conf.get('ft_epoch', 1000)
        ft_lr = train_conf.get('ft_lr', 0.001)

        # initialize optimizer
        if train_conf.use_adamw:
            optimizer = torch.optim.AdamW(model.parameters(), lr=ft_lr)
        else:
            optimizer = torch.optim.Adam(model.parameters(), lr=ft_lr)

        # train_expert masks
        if expert_data.expert_train_masks.dim() > 1 and expert_data.expert_train_masks.size(1) > 1:
            train_mask = expert_data.expert_train_masks[:,
                         (i - 1) % expert_data.expert_train_masks.shape[-1]].squeeze()
        else:
            train_mask = expert_data.expert_train_masks.squeeze()
        if expert_data.expert_val_masks.dim() > 1 and expert_data.expert_val_masks.size(1) > 1:
            val_mask = expert_data.expert_val_masks[:, i - 1]
        else:
            val_mask = expert_data.expert_val_masks.squeeze()
        if expert_data.expert_test_masks.dim() > 1 and expert_data.expert_test_masks.size(1) > 1:
            test_mask = expert_data.expert_test_masks[:, i - 1]
        else:
            test_mask = expert_data.expert_test_masks.squeeze()

        best_ft_val_acc = best_ft_test_acc = 0
        for epoch in range(1, ft_epoch + 1):

            # adjust_learning_rate(optimizer, ft_lr, epoch)

            train(model, optimizer, expert_data, train_mask, grad_norm=train_conf.grad_norm)

            (train_acc, val_acc, test_acc), (train_loss, val_loss, test_loss), _ = test(
                model, metric, expert_data, [train_mask, val_mask, test_mask])

            if val_acc > best_ft_val_acc:
                best_ft_val_acc = val_acc
                best_ft_test_acc = test_acc
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'train_acc': train_acc,
                    'val_acc': val_acc,
                }, ckpt_path)

            logger.info(f'FT Epoch: {epoch:03d}, Train Acc: {train_acc:.4f}, '
                        f'Val Acc: {val_acc:.4f}, Test Acc: {test_acc:.4f}, '
                        f'Best Val Acc: {best_ft_val_acc:.4f}, Best Test Acc: {best_ft_test_acc:.4f}')

        ckpt = torch.load(ckpt_path, weights_only=False)
        model = get_model(model_conf, num_features, num_classes).to(device)
        model.load_state_dict(ckpt['model_state_dict'])
        (test_acc,), (test_loss,), logit = test(model, metric, expert_data, [test_mask])
        logger.warning(f"[Best FT Model] Epoch: {ckpt['epoch']:02d}, Train: {ckpt['train_acc']:.4f}, "
                       f"Val: {ckpt['val_acc']:.4f}, Test: {test_acc:.4f}")
        best_val.append(float(best_ft_val_acc))
        best_test.append(float(test_acc))
        runs_logit.append(logit.cpu())

        ## eval_by_group
        _data_conf = Dict(OmegaConf.to_container(data_conf))
        expert_test_masks = []
        for group in expert_test_accs.keys():
            _data_conf.group = group
            for k in range(data_conf.num_experts):
                _data_conf.expert_id = k
                _expert_data, _, _, _ = get_expert_data(root=dataset_dir, **_data_conf)
                if _expert_data.expert_test_masks.dim() > 1 and _expert_data.expert_test_masks.size(1) > 1:
                    expert_test_masks.append(_expert_data.expert_test_masks[:, i - 1].to(device))
                else:
                    expert_test_masks.append(_expert_data.expert_test_masks.squeeze().to(device))

        test_accs, test_losses, _ = test(model, metric, expert_data, expert_test_masks)

        for g_id, group in enumerate(expert_test_accs.keys()):
            for k in range(data_conf.num_experts):
                test_acc = test_accs[g_id * data_conf.num_experts + k]
                logger.warning(f'{group} expert {k} --- Test Acc: {test_acc:.4f}')
                expert_test_accs[group][k].append(test_acc.cpu().item())

    result_str = ''
    if train_expert:
        result_str += f' Expert: {data_conf.expert_id + 1}/{data_conf.num_experts}'
    result_str += (f'\nValid: {np.mean(best_val) * 100:.2f} ±{np.std(best_val) * 100:.2f}, '
                   f'Test: {np.mean(best_test) * 100:.2f} ±{np.std(best_test) * 100:.2f}\n')

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
    logger.save_config()
    save_base_gnn = conf.get('save_base_gnn', False)
    if not os.path.exists(model_path) and save_base_gnn:
        torch.save(all_models, model_path)
        logger.info(f'Saved model to {model_path}')

    if os.path.exists(ckpt_path):
        os.remove(ckpt_path)

    ## for optuna
    tuning_objective = np.mean(best_test)
    return float(tuning_objective)


if __name__ == '__main__':
    main()
