# -*- coding: utf-8 -*-
# @Time: 2024/12/12
# @File: save_results.py
# @Author: fwb
import os
import os.path as osp
import matplotlib.pyplot as plt
import torch
import scipy.io as sio
from dataproc.proc_utils import create_path


class SaveResults:
    def __init__(self, args=None, seed=None):
        self.args = args
        # Create results save path.
        self.results_path = osp.join(os.getcwd(), 'results')
        self.config_path = osp.join(self.results_path, 'config.txt')
        self.config_info = (f'seed[{seed}]'
                            f'_iad[{args.is_adaptive}]'
                            f'_be[{args.base_events / 1e+4}w]'
                            f'_ek[{args.every_k_points}]'
                            f'_ne[{args.nb_neighbors}]'
                            f'_sr[{args.std_ratio}]'
                            f'_trans[{args.transform}]'
                            f'_opt[{args.opt_name}]'
                            f'_cri[{args.criterion_name}]'
                            f'_e[{args.epochs}]_b[{args.batch_size}]_lr[{args.lr}]'
                            f'_wd[{args.weight_decay}]'
                            f'_isc[{args.is_lr_scheduler}]'
                            f'_sche[{args.lr_scheduler}]'
                            f'_ichs[{args.in_chs}]'
                            f'_gs[{"-".join(map(str, args.grid_size))}]'
                            f'_ord[{"-".join(map(str, args.order))}]'
                            f'_str[{"-".join(map(str, args.stride))}]'
                            f'_sor[{args.shuffle_orders}]'
                            f'_rpe[{args.enable_rpe}]'
                            f'_flash[{args.enable_flash}]'
                            f'_enc_dep[{"-".join(map(str, args.enc_depths))}]'
                            f'_enc_chs[{"-".join(map(str, args.enc_chs))}]'
                            f'_enc_hea[{"-".join(map(str, args.enc_heads))}]'
                            f'_enc_pat[{"-".join(map(str, args.enc_patches))}]'
                            f'_mlpr[{args.mlp_ratio}]')
        if not args.is_cls:
            self.config_info = self.config_info+(f'_dec_dep[{"-".join(map(str, args.dec_depths))}]'
                                                 f'_dec_chs[{"-".join(map(str, args.dec_chs))}]'
                                                 f'_dec_hea[{"-".join(map(str, args.dec_heads))}]'
                                                 f'_dec_pat[{"-".join(map(str, args.dec_patches))}]')
        create_path(self.results_path)

    def save_model(self, model_weight, name):
        # Save model.
        model_save_path = osp.join(self.results_path, name + '.pt')
        torch.save(model_weight, model_save_path)
        return model_save_path

    def save_results(self, results_dict):
        # Save config info.
        with open(self.config_path, 'w', encoding='utf-8') as file:
            file.write(self.config_info + '\n')
        # Save training and test results to results.mat file.
        sio.savemat(osp.join(self.results_path, 'results.mat'), results_dict)
