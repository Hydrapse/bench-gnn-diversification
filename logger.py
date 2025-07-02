import os
import os.path as osp
import json
import time

import torch
import shutil
import logging
from datetime import datetime
from omegaconf import OmegaConf


# from torch.utils.tensorboard import SummaryWriter


class Logger:
    def __init__(self, conf=None):
        self.use_tensorboard = None
        self.writer = None

        self.conf = conf
        self.exp_name = self._init_experiment_name()
        self.exp_id = '{}_{}'.format(os.getpid(), int(time.time()))

        # set up logging directories
        self.log_dir: str = f'{conf.proc_dir}/trial_{conf.trial_dir}'
        self.config_dir = osp.join(self.log_dir, 'config')
        self.result_dir = osp.join(self.log_dir, 'result')
        self.curve_dir = osp.join(self.log_dir, 'curve')
        self.ckpt_dir = osp.join(self.log_dir, 'checkpoint')
        if conf.get('degradation', False):
            self.logit_dir = osp.join(self.log_dir, 'degradation')
        else:
            self.logit_dir = osp.join(self.log_dir, 'logit')

        os.makedirs(self.log_dir, exist_ok=True)
        os.makedirs(self.config_dir, exist_ok=True)
        os.makedirs(self.result_dir, exist_ok=True)
        os.makedirs(self.curve_dir, exist_ok=True)
        os.makedirs(self.logit_dir, exist_ok=True)
        os.makedirs(self.ckpt_dir, exist_ok=True)

        if self.conf.result_file != 'default':
            self.conf.log_result = True

        self.logger = logging.getLogger(__name__)  # init by hydra config

    def _init_experiment_name(self):
        """
        Initialize the experiment name. The structure of the experiment name is:
            {dataset_info}_{model_info}
        """
        data_conf = self.conf.dataset
        dataset_info = data_conf.name
        if hasattr(data_conf, 'undirected') and data_conf.undirected:
            dataset_info += '-undir'
        if hasattr(data_conf, 'rev_adj') and data_conf.rev_adj:
            dataset_info += '-revADJ'
        if data_conf.ptb_type is not None and data_conf.ptb_ratio > 0:
            dataset_info += '-ptb{}{}'.format(data_conf.ptb_type.upper(), data_conf.ptb_ratio)
        if hasattr(data_conf, 'expert_id'):
            if hasattr(data_conf, 'group'):
                dataset_info += '-group-{}'.format(data_conf.group)
                if data_conf.get('learned_homo', ''):
                    dataset_info += f'Learned{data_conf.learned_homo.upper()}'
                elif not data_conf.get('use_test', True):
                    dataset_info += 'FilterTest'
                    if not data_conf.get('use_val', True):
                        dataset_info += 'AndVal'
                if hasattr(self.conf.train, 'ft_lr'):
                    dataset_info += 'FineTune'
            if hasattr(data_conf, 'fraction'):
                dataset_info += '-frac-{}'.format(data_conf.fraction)
            dataset_info += '-eid-{}'.format(data_conf.expert_id)
            if hasattr(data_conf, 'num_experts'):
                dataset_info += '-{}'.format(data_conf.num_experts)
        
        


        if hasattr(self.conf, 'moscat'):
            mixer_conf = self.conf.moscat
            model_info = f'Moscat-lr{self.conf.train.mixer_lr}-val{mixer_conf.val_ratio}'
            model_info += f'-trHet{mixer_conf.get("mask_tr_het_ratio", -1)}'
        else:
            model_conf = self.conf.model
            model_info = model_conf.name.upper()
            if model_conf.jk is not None:
                model_info += '-jk{}'.format(model_conf.jk.upper())
            if model_conf.residual is not None:
                model_info += '-res{}'.format(model_conf.residual.upper())
            if model_conf.dropout > 0:
                model_info += '-dropout{}'.format(model_conf.dropout)
            if model_conf.dropedge > 0:
                model_info += '-dropedge{}'.format(model_conf.dropedge)
            if model_conf.get('deg_norm', None):
                model_info += '-{}'.format(model_conf.deg_norm)
            if model_conf.get('adj_norm', None) == 'high':
                model_info += '-laplacian'
            if model_conf.init_layers > 0:
                model_info += '-init{}'.format(model_conf.init_layers)
            model_info += '-conv{}'.format(model_conf.conv_layers)
        
        # if hasattr(self.conf, 'is_sweep') and self.conf.is_sweep:
        #     model_info = 'Sweep' + model_info
        if hasattr(self.conf, 'is_sweep') and self.conf.is_sweep:
            model_info += '-hidden{}'.format(model_conf.hidden_dim)
            model_info += '-epoch{}'.format(self.conf.train.epoch)
        if hasattr(self.conf, 'is_sweep_adj') and self.conf.is_sweep_adj:
            model_info += '-adj{}'.format(model_conf.adj_norm)
        if hasattr(self.conf, 'is_sweep_ft_lr') and self.conf.is_sweep_ft_lr:
            model_info += '-ftlr{}'.format(self.conf.train.ft_lr)
        if self.conf.model.name == 'BERNNET':
            model_info += '-K{}'.format(self.conf.model.K)


        exp_name = dataset_info + "_" + model_info

        if self.conf.get('seed', 0) != 0:
            exp_name += f'-seed{self.conf.seed}'

        return exp_name

    def info(self, message):
        self.logger.info(message)

    def warning(self, message):
        self.logger.warning(message)

    def critical(self, message):
        self.logger.critical(message)

    def error(self, message):
        self.logger.error(message)

    def log_scalar(self, tag, value, step):
        """
        Log a scalar value to TensorBoard.
        
        Args:
            tag (str): Name of the scalar.
            value (float): Value to log.
            step (int): Step number.
        """
        assert self.use_tensorboard, "TensorBoard is not enabled."
        self.writer.add_scalar(tag, value, step)

    def save_config(self):
        """
        Save specific experiment configuration/settings to a file.
        """
        file_path = os.path.join(self.config_dir, f'{self.exp_id}.yaml')
        if self.conf.get('log_result', False):
            selected_conf = {
                # 'sampler': self.conf.sampler,
                'dataset': self.conf.dataset,
                'train': self.conf.train,
                'model': self.conf.model,
            }
            if hasattr(self.conf, 'moscat'):
                selected_conf['moscat'] = self.conf.moscat
                selected_conf['expert'] = self.conf.expert
            selected_conf = OmegaConf.create(selected_conf)
            OmegaConf.save(selected_conf, file_path)
            self.logger.critical(f'Config saved in: {file_path}')

    def save_result(self, result_str):
        file_path = osp.join(self.result_dir, self.conf.result_file + '.txt')
        if self.conf.get('log_result', False):
            with open(file_path, 'a+', encoding='utf-8') as f:
                f.write(f'{self.exp_id}:{self.exp_name}\n')
                f.write(result_str)
                f.write('\n\n')
            self.logger.critical(f'Results saved in: {file_path}, ID: {self.exp_id}')

    def save_logit(self, total_logit):
        file_path = osp.join(self.logit_dir, self.exp_name + '.pt')
        if self.conf.get('log_logit', False):
            torch.save(total_logit, file_path)
            self.logger.critical(f'Logits saved in: {file_path}')

    def save_curve(self, curve_dict):
        file_path = osp.join(self.curve_dir, self.exp_name + '.pt')
        if self.conf.get('log_curve', False):
            curve_dict = {key: torch.tensor(value) for key, value in curve_dict.items()}
            torch.save(curve_dict, file_path)
            self.logger.critical(f'Curve saved in: {file_path}')

    @DeprecationWarning
    def save_model(self, model, filename=None):
        filename = 'model.pt' if filename is None else filename
        torch.save(model.state_dict(), osp.join(self.log_dir, filename))
        self.logger.critical(f'Model saved in: {osp.join(self.log_dir, filename)}')


class CheckpointPathManager:

    def __init__(self, conf, logger):
        self.conf = conf
        self.temp_dir = conf.ckpt_dir
        self.persis_dir = logger.ckpt_dir
        self.logit_dir = logger.logit_dir
        self.dataset = conf.dataset.name
        self.log_ckpt = conf.get('log_ckpt', True)

        os.makedirs(self.temp_dir, exist_ok=True)
        os.makedirs(self.persis_dir, exist_ok=True)

    def get_router_temp_path(self):
        return osp.join(self.temp_dir, '{}_{}_{}_{}.pt'.format(
            'router', self.conf.router.pipeline, os.getpid(), int(time.time())))

    def get_router_persis_path(self, run, finetune=False):
        file_name = self._router_config_to_filename(finetune)
        return osp.join(self.persis_dir, f'{self.dataset}_{file_name}_run{run}.pt')

    def get_expert_temp_path(self, expert_id):
        return osp.join(self.temp_dir, '{}_{}_{}_{}.pt'.format(
                 'expert', expert_id, os.getpid(), int(time.time())))

    def get_expert_persis_path(self, expert_id, run, finetune=False):
        file_name = self._expert_config_to_filename(expert_id, finetune)
        return osp.join(self.persis_dir, f'{self.dataset}_{file_name}_run{run}.pt')

    def get_expert_logit_path(self, expert_id):
        router_name = self._router_config_to_filename(finetune=True)
        expert_name = self._expert_config_to_filename(expert_id, finetune=True)
        return osp.join(self.logit_dir, f'{self.dataset}_{router_name}_{expert_name}.pt')

    def _router_config_to_filename(self, finetune=False) -> str:
        router = self.conf.router

        parts = [
            f"clusters{router.num_clusters}",
            f"ep{router.epoch}",
            f"lr{router.lr}",
            f"pat{router.patience}",
            router.pipeline,
            f"hd{router.hidden_dim}",
            f"proj{router.projector_layers}",
        ]

        # --- encoder sub-config ---
        enc = router.encoder
        parts.append(f"enc{enc.layers}{enc.activation}")
        if getattr(enc, "dropout", 0) > 0:
            parts.append(f"drop{enc.dropout}")
        if getattr(enc, "norm", None):
            parts.append(f"norm{enc.norm}")
        if getattr(enc, "mixer", None):
            parts.append(f"mixer{enc.mixer}")

        # --- augmentation ---
        if getattr(router, "feat_dropout", 0) > 0:
            parts.append(f"featdrop{router.feat_dropout}")

        # --- fine-tuning section ---
        if finetune:
            if getattr(router, "ft_epoch", 0) > 0:
                parts.extend([
                    f"fte{router.ft_epoch}",
                    f"ftlr{router.ft_lr}",
                    f"ftpat{router.ft_patience}",
                ])

            # --- always include tradeoff (even if tiny) ---
            parts.append(f"trade{router.tradeoff}")

        # join all segments with hyphens
        return "Router_" + "-".join(parts)

    def _expert_config_to_filename(self, eid, finetune=False) -> str:
        expert = self.conf.experts[eid]

        parts = [
            f"eid{expert.expert_id}",
            f"ep{expert.epoch}",
            f"lr{expert.lr}",
            expert.name.upper(),
            f"hd{expert.hidden_dim}",
            f"init{expert.init_layers}",
            f"conv{expert.conv_layers}"
        ]

        # normalizations
        if getattr(expert, "norm", None):
            parts.append(expert.norm)
        if getattr(expert, "out_norm", False):
            parts.append("outnorm")

        # dropouts
        if getattr(expert, "init_dropout", 0) > 0:
            parts.append(f"initdrop{expert.init_dropout}")
        if getattr(expert, "dropout", 0) > 0:
            parts.append(f"drop{expert.dropout}")

        # optional flags
        if getattr(expert, "jk", None) is not None:
            parts.append(f"jk{expert.jk}")
        if getattr(expert, "residual", None) is not None:
            parts.append(f"res{expert.residual}")

        # adjacency normalization
        if getattr(expert, "adj_norm", None):
            parts.append(expert.adj_norm)

        # fine-tuning
        if getattr(expert, "ft_epoch", 0) > 0 and finetune:
            parts.extend([
                f"fte{expert.ft_epoch}",
                f"ftlr{expert.ft_lr}",
            ])

        return "Expert_" + "-".join(parts)
