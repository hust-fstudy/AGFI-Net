# -*- coding: utf-8 -*-
# @Time: 2025/6/25
# @File: run_recognition.py
# @Author: fwb
import os

os.environ['CUDA_VISIBLE_DEVICES'] = "2"
import os.path as osp
import random
import numpy as np
from tqdm import tqdm
import torch
from torch_geometric.loader import DataLoader
from dataproc.create_graph_dataset import batch_create
from dataproc.proc_utils import remove_files_in_dir
from dataproc.graph_dataset import GraphDataset
from model.net import Net
from utils.parse_yaml import parse_yaml
from utils.model_performance import model_test
from utils.save_results import SaveResults


def initial_model(args):
    if args.is_cls:
        model = Net(
            in_chs=args.in_chs,
            grid_size=tuple(args.grid_size),
            num_classes=args.num_classes,
            order=tuple(args.order),
            stride=tuple(args.stride),
            enc_depths=tuple(args.enc_depths),
            enc_chs=tuple(args.enc_chs),
            enc_heads=tuple(args.enc_heads),
            enc_patches=tuple(args.enc_patches),
            mlp_ratio=args.mlp_ratio,
            shuffle_orders=args.shuffle_orders,
            enable_rpe=args.enable_rpe,
            enable_flash=args.enable_flash,
            upcast_attention=args.upcast_attention,
            upcast_softmax=args.upcast_softmax,
            is_cls=args.is_cls,
        )
    else:
        model = Net(
            in_chs=args.in_chs,
            grid_size=tuple(args.grid_size),
            num_classes=args.num_classes,
            order=tuple(args.order),
            stride=tuple(args.stride),
            enc_depths=tuple(args.enc_depths),
            enc_chs=tuple(args.enc_chs),
            enc_heads=tuple(args.enc_heads),
            enc_patches=tuple(args.enc_patches),
            dec_depths=tuple(args.dec_depths),
            dec_chs=tuple(args.dec_chs),
            dec_heads=tuple(args.dec_heads),
            dec_patches=tuple(args.dec_patches),
            mlp_ratio=args.mlp_ratio,
            shuffle_orders=args.shuffle_orders,
            enable_rpe=args.enable_rpe,
            enable_flash=args.enable_flash,
            upcast_attention=args.upcast_attention,
            upcast_softmax=args.upcast_softmax,
            is_cls=args.is_cls,
        )
    return model


class TaskMain:
    def __init__(self, cfgs_dir: str, dataset_name: str):
        # Parse configs info.
        self.dataset_name = dataset_name.lower()
        com_cfgs_path = osp.join(cfgs_dir, 'com_params' + '.yaml')
        self.com_cfgs = parse_yaml(com_cfgs_path)
        dataset_cfgs_path = osp.join(cfgs_dir, self.dataset_name + '.yaml')
        self.args = parse_yaml(dataset_cfgs_path)
        self.random_seed = self.com_cfgs.random_seed
        # Load the corresponding dataset.
        print(f"The current dataset used is {dataset_name}")
        self.data_root_dir = self.com_cfgs.data_root_dir
        self.data_path = osp.join(self.data_root_dir, self.dataset_name, 'raw')
        # Graph data storage path.
        self.save_graph_dir = osp.join(self.data_root_dir, 'to_graph', dataset_name,
                                       f'be{self.args.base_events / 1e+4}w'
                                       f'_ek{self.args.every_k_points}'
                                       f'_ne{self.args.nb_neighbors}'
                                       f'_sr{self.args.std_ratio}'
                                       f'_ic{self.args.in_chs}')
        # Fixed batch creation graph dataset.
        if self.args.is_remove_graph:
            remove_files_in_dir(self.save_graph_dir)
            batch_create(self.args, self.data_path, self.save_graph_dir, seed=self.random_seed)

    def run(self):
        # Set the random seed and initialize it.
        print(f"Random seed ID is: {self.random_seed}")
        random.seed(self.random_seed)
        np.random.seed(self.random_seed)
        torch.manual_seed(self.random_seed)
        torch.cuda.manual_seed(self.random_seed)
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.benchmark = True
        # Load graph dataset.
        test_graph_dataset = GraphDataset(self.args, self.save_graph_dir, 'test')
        test_loader = DataLoader(test_graph_dataset,
                                 batch_size=self.args.batch_size)
        # Initialization Model.
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = initial_model(self.args)
        model.to(device)
        # Model training.
        SR = SaveResults(self.args, self.random_seed)
        if self.args.is_cls:
            keys = ['train_loss', 'train_acc',
                    'test_acc', 'best_acc', 'test_epoch_acc']
        else:
            keys = ['train_epoch_loss', 'val_epoch_iou',
                    'test_epoch_iou', 'test_epoch_acc', 'test_epoch_pd', 'test_epoch_fa',
                    'test_best_loss', 'test_best_iou']
        results_dict = dict((key, []) for key in keys)
        if not self.args.is_cls:
            # best_loss_path = r'./results/best_loss_model.pt'
            best_loss_path = r'./results/best_iou_model.pt'
            model.load_state_dict(torch.load(best_loss_path))
            # Test.
            # model.load_state_dict(best_loss_model_weights)
            best_loss_iou, best_loss_seg_acc, best_loss_pd, best_loss_fa = model_test(self.args,
                                                                                      model,
                                                                                      test_loader,
                                                                                      device)
            results_dict['test_best_loss'].append(
                {'best_loss_iou': best_loss_iou,
                 'best_loss_seg_acc': best_loss_seg_acc,
                 'best_loss_pd': best_loss_pd,
                 'best_loss_fa': best_loss_fa}
            )
            # print(f"test for best loss in epoch {best_loss_epoch}: (best_loss_iou: {best_loss_iou}, "
            #       f"best_loss_seg_acc: {best_loss_seg_acc}, "
            #       f"best_loss_pd: {best_loss_pd}, best_loss_fa: {best_loss_fa}e-4)")
            print(f"test for best loss: (best_loss_iou: {best_loss_iou}, "
                  f"best_loss_seg_acc: {best_loss_seg_acc}, "
                  f"best_loss_pd: {best_loss_pd}, best_loss_fa: {best_loss_fa}e-4)")
        #     model.load_state_dict(best_iou_model_weights)
        #     best_iou_iou, best_iou_seg_acc, best_iou_pd, best_iou_fa = model_test(self.args,
        #                                                                           model,
        #                                                                           test_loader,
        #                                                                           device)
        #     results_dict['test_best_iou'].append(
        #         {'best_iou_iou': best_iou_iou,
        #          'best_iou_seg_acc': best_iou_seg_acc,
        #          'best_iou_pd': best_iou_pd,
        #          'best_iou_fa': best_iou_fa}
        #     )
        #     print(f"test for best iou in epoch {best_iou_epoch}: (best_iou_iou: {best_iou_iou}, "
        #           f"best_iou_seg_acc: {best_iou_seg_acc}, "
        #           f"best_iou_pd: {best_iou_pd}, best_iou_fa: {best_iou_fa}e-4)")
        #     # Log.
        #     SR.save_model(best_loss_model_weights, 'best_loss_model')
        #     SR.save_model(best_iou_model_weights, 'best_iou_model')
        # # Save results.
        # SR.save_results(results_dict)


if __name__ == '__main__':
    cfgs_dir = r'./configs'
    dataset_dict = {
        1: 'DVSGesture',
        2: 'PAF',
        3: 'THU',
        4: 'KTH',
        5: 'Bully',
        6: 'Daily',
        7: 'EV-UAV'
    }
    RM = TaskMain(cfgs_dir=cfgs_dir, dataset_name=dataset_dict[7])
    RM.run()
